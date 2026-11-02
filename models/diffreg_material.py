"""Sampled local, teacher-guided material regularization for Base v4.

The material kernel and joint bilateral residual follow
Diffusion-Based Material Regularization for Physics-Based Inverse Rendering,
https://arxiv.org/html/2606.31065v1 (Eqs. 5--8). Albedo follows the official code:
filter log(A), then weight the residual with detached center A. This is a deliberately bounded
adaptation, not the paper's global permutohedral filter: we sample anchors and
use a local spatial kernel plus detached depth/normal surface checks. Teachers
are the existing encoded [albedo RGB, roughness, metallic] maps. No new renderer,
network, history, or inference-time filtering is introduced.
"""
from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch
import torch.nn.functional as F


DEFAULTS = {
    "enabled": False, "w": .05, "start_step": 3000, "ramp_steps": 1000,
    "num_samples": 4096, "radius": 3, "sigma_g": .1, "sigma_xy": 2.,
    "depth_abs": .05, "depth_rel": .02, "normal_cos": .9,
    "eps": .01, "seed": 20261031,
}
_CHANNELS = (("albedo", 3), ("roughness", 1), ("metallic", 1))


def validate_diffreg_config(config=None):
    """Canonicalize an opt-in config, rejecting misspellings and invalid values."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("diffreg_material config must be a mapping")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown diffreg_material settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **dict(config)}
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("diffreg_material enabled must be boolean")
    for key in ("start_step", "ramp_steps", "num_samples", "radius", "seed"):
        value = cfg[key]
        minimum = 1 if key in ("num_samples", "radius") else 0
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"diffreg_material {key} must be an integer >= {minimum}")
        cfg[key] = int(value)
    for key in ("w", "sigma_g", "sigma_xy", "depth_abs", "depth_rel", "normal_cos", "eps"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"diffreg_material {key} must be finite and nonnegative")
        cfg[key] = float(value)
    if min(cfg["sigma_g"], cfg["sigma_xy"], cfg["eps"]) <= 0:
        raise ValueError("diffreg_material sigma_g, sigma_xy and eps must be positive")
    if not 0 < cfg["normal_cos"] <= 1 or cfg["eps"] >= 1:
        raise ValueError("diffreg_material requires 0 < normal_cos <= 1 and 0 < eps < 1")
    return cfg


def diffreg_weight(step, config=None):
    """The first active step is zero for a nonzero ramp, as in Base v4."""
    cfg = validate_diffreg_config(config)
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("diffreg_material step must be a nonnegative integer")
    if not cfg["enabled"] or step < cfg["start_step"]:
        return 0.
    ramp = min(1., (step - cfg["start_step"]) / cfg["ramp_steps"]) if cfg["ramp_steps"] else 1.
    return cfg["w"] * ramp


def _graph_zero(prediction):
    # A single element establishes a zero derivative for each map without
    # reducing full native-resolution maps. NaNs in excluded pixels stay inert.
    if isinstance(prediction, Mapping):
        tensors = [prediction[key] for key, _ in _CHANNELS
                   if isinstance(prediction.get(key), torch.Tensor)]
        if tensors:
            return sum(torch.nan_to_num(value.reshape(-1)[:1]).sum() * 0. for value in tensors)
    return torch.zeros(())


def _validate_maps(prediction, teacher, valid_mask, depth, normal):
    if not isinstance(prediction, Mapping) or not isinstance(teacher, Mapping):
        raise ValueError("Active diffreg_material requires prediction and teacher maps")
    reference = prediction.get("albedo")
    if not isinstance(reference, torch.Tensor) or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError("diffreg_material albedo must be H x W x 3")
    h, w = reference.shape[:2]
    if h < 1 or w < 1 or not reference.is_floating_point():
        raise ValueError("diffreg_material maps must be nonempty floating-point tensors")
    for label, maps in (("prediction", prediction), ("teacher", teacher)):
        for key, channels in _CHANNELS:
            value = maps.get(key)
            if (not isinstance(value, torch.Tensor) or value.shape != (h, w, channels)
                    or not value.is_floating_point() or value.device != reference.device):
                raise ValueError(f"diffreg_material {label}.{key} must be a floating H x W x {channels} map on {reference.device}")
    for label, value, shapes in (("valid_mask", valid_mask, ((h, w), (h, w, 1))),
                               ("depth", depth, ((h, w), (h, w, 1))),
                               ("normal", normal, ((h, w, 3),))):
        if value is None and label != "valid_mask":
            continue
        if not isinstance(value, torch.Tensor) or value.shape not in shapes or value.device != reference.device:
            raise ValueError(f"diffreg_material {label} has an invalid shape or device")
    return h, w, reference


def _sample_indices(h, w, count, step, sample_key, seed, device):
    if h * w <= count:
        return torch.arange(h * w, device=device)
    # Independent from training's frame sampler, topology RNG and LPIPS crops.
    generator = torch.Generator(device=device)
    generator.manual_seed((seed + 1000003 * int(step) + 9176 * int(sample_key)) % (2**63 - 1))
    return torch.randint(h * w, (count,), generator=generator, device=device)


def material_relation_loss(prediction, teacher, valid_mask, *, step, config=None,
                           sample_key=0, depth=None, normal=None):
    """Return ``(weighted_loss, detached_scalar_diagnostics)``.

    Maps use HWC layout and must already be aligned. ``valid_mask`` is the
    caller's training-only static/sky/ego/opacity mask. No mask is inferred from
    material predictions. Optional depth and normal add detached surface gates.
    Both the center and filtered prediction receive gradients. Teacher, gates,
    and random sampling never receive gradients or change the global RNG state.
    """
    cfg = validate_diffreg_config(config)
    weight = diffreg_weight(step, cfg)
    if weight == 0:
        return _graph_zero(prediction), {}
    if isinstance(sample_key, bool) or not isinstance(sample_key, Integral):
        raise ValueError("diffreg_material sample_key must be an integer")
    h, w, reference = _validate_maps(prediction, teacher, valid_mask, depth, normal)
    dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
    device = reference.device
    centers = _sample_indices(h, w, cfg["num_samples"], step, sample_key, cfg["seed"], device)
    axis = torch.arange(-cfg["radius"], cfg["radius"] + 1, device=device)
    oy, ox = torch.meshgrid(axis, axis, indexing="ij")
    offsets_y, offsets_x = oy.flatten(), ox.flatten()
    yy = centers[:, None].div(w, rounding_mode="floor") + offsets_y
    xx = centers[:, None].remainder(w) + offsets_x
    in_bounds = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    neighbors = yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)
    center_slot = offsets_x.numel() // 2

    def gather(value):
        return value.reshape(h * w, -1)[neighbors].to(dtype=dtype)

    # Only N x K stencils are materialized; no H x W x K feature tensor.
    predicted = torch.cat([gather(prediction[key]) for key, _ in _CHANNELS], dim=-1)
    with torch.no_grad():
        guide = torch.cat([gather(teacher[key].detach()) for key, _ in _CHANNELS], dim=-1)
        mask = valid_mask.detach().reshape(h * w)[neighbors]
        valid = in_bounds & torch.isfinite(mask) & (mask > .99)
        valid &= torch.isfinite(guide).all(-1) & torch.isfinite(predicted.detach()).all(-1)
        if depth is not None:
            z = gather(depth.detach())[..., 0]
            zc = z[:, center_slot:center_slot + 1]
            valid &= torch.isfinite(z) & (z > 0)
            valid &= (z - zc).abs() <= cfg["depth_abs"] + cfg["depth_rel"] * zc.abs()
        if normal is not None:
            n = gather(normal.detach())
            lengths = n.norm(dim=-1)
            valid &= torch.isfinite(n).all(-1) & (lengths > 1e-6)
            n = F.normalize(torch.nan_to_num(n), dim=-1)
            # Gaussian normals may flip towards the view. Sign does not
            # identify a different physical plane for this spatial gate.
            valid &= (n * n[:, center_slot:center_slot + 1]).sum(-1).abs() >= cfg["normal_cos"]
        valid &= valid[:, center_slot:center_slot + 1].clone()
        clean_guide = torch.nan_to_num(guide)
        distance = (clean_guide - clean_guide[:, center_slot:center_slot + 1]).square().sum(-1)
        spatial = (offsets_x.square() + offsets_y.square()).to(dtype) / (2 * cfg["sigma_xy"]**2)
        weights = torch.exp(-distance / (2 * cfg["sigma_g"]**2) - spatial) * valid
        nonself = (offsets_x != 0) | (offsets_y != 0)
        nonself_mass = (weights * nonself).sum(-1)
        supported = valid[:, center_slot] & (nonself_mass > 1e-8)
        normalized = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)

    # Sanitize before log/arithmetic: multiplying NaN by a zero mask is unsafe.
    safe_prediction = torch.where(valid[..., None], torch.nan_to_num(predicted), 0.)
    # Match the official implementation's operation order, not a literal
    # pointwise reading of Eq. 8: A*log(A) is non-injective. Filter log(A),
    # and compensate its gradient with detached center A AFTER subtraction.
    # https://github.com/gerwang/DiffReg-PBIR/blob/main/diffusion_reg/models/losses.py
    values = torch.cat((safe_prediction[..., :3].clamp_min(cfg["eps"]).log(),
                        safe_prediction[..., 3:]), dim=-1)
    center = values[:, center_slot]
    # Centered accumulation is algebraically JBF(values)-center, but ensures
    # that a constant material gives exactly zero even after roundoff.
    residual = (normalized[..., None] * (values - center[:, None])).sum(1).abs()
    support = supported.to(dtype)
    denominator = support.sum().clamp_min(1)
    albedo_error = (residual[:, :3] * safe_prediction[:, center_slot, :3].detach()).mean(-1)
    component_losses = torch.stack(((albedo_error * support).sum(),
                                    (residual[:, 3] * support).sum(),
                                    (residual[:, 4] * support).sum())) / denominator
    unweighted = component_losses.mean()
    loss = weight * unweighted
    diagnostics = {
        "weight": reference.new_tensor(weight).detach(),
        "sample_count": centers.new_tensor(centers.numel()).detach(),
        "valid_sample_count": supported.sum().detach(),
        "support_fraction": support.mean().detach(),
        "neighbor_mass": ((nonself_mass * support).sum() / denominator).detach(),
        "unweighted": unweighted.detach(),
        "albedo": component_losses[0].detach(),
        "roughness": component_losses[1].detach(),
        "metallic": component_losses[2].detach(),
    }
    return loss, diagnostics
