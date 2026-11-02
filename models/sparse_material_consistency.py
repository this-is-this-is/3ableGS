"""Sparse train-only material consistency between two rendered training views.

This borrows MaterialRefGS's multi-view material constraint, but uses sampled
surface points, camera-Z reprojection, one detached neighbor and balanced L1.
It is not the paper's 7x7 plane-homography patch/MSE implementation. Normals
are world-space (orientation sign is ignored); albedo inputs are encoded sRGB.
Only O(HW) validity masks and O(num_samples) correspondences are constructed.
There is no renderer, history fusion, inference filtering or image teacher here.
"""
from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch
import torch.nn.functional as F


_MATERIALS = (("albedo", 3), ("roughness", 1), ("metallic", 1))
_MAPS = (*_MATERIALS, ("normal", 3), ("depth", 1), ("opacity", 1))


def _graph_zero(current):
    if isinstance(current, Mapping):
        values = [current[key] for key, _ in _MATERIALS
                  if isinstance(current.get(key), torch.Tensor)]
        if values:
            # A single element connects each map without a full-image sum.
            return sum(torch.nan_to_num(v[(0,) * v.ndim] if v.numel() else v.sum(),
                                        nan=0., posinf=0., neginf=0.) * 0.
                       for v in values)
    return torch.zeros(())


def _options(weight, num_samples, seed, step, sample_key, depth_abs, depth_rel,
             normal_cos, cycle_px, min_opacity, erode_radius, depth_edge_rel):
    for name, value, minimum in (("num_samples", num_samples, 1), ("seed", seed, 0),
                                  ("step", step, 0), ("erode_radius", erode_radius, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"sparse material {name} must be an integer >= {minimum}")
    if isinstance(sample_key, bool) or not isinstance(sample_key, Integral):
        raise ValueError("sparse material sample_key must be an integer")
    for name, value in (("weight", weight), ("depth_abs", depth_abs), ("depth_rel", depth_rel),
                        ("normal_cos", normal_cos), ("cycle_px", cycle_px),
                        ("min_opacity", min_opacity), ("depth_edge_rel", depth_edge_rel)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"sparse material {name} must be finite and nonnegative")
    if not 0 < normal_cos <= 1 or not 0 <= min_opacity <= 1:
        raise ValueError("sparse material requires 0 < normal_cos <= 1 and 0 <= min_opacity <= 1")


def _validate_frame(maps, mask, camera, intrinsics, label, *, device=None):
    if not isinstance(maps, Mapping):
        raise ValueError(f"sparse material {label} maps must be a mapping")
    reference = maps.get("albedo")
    if (not isinstance(reference, torch.Tensor) or reference.ndim != 3
            or reference.shape[-1] != 3 or not reference.is_floating_point()):
        raise ValueError(f"sparse material {label}.albedo must be floating H x W x 3")
    h, w = reference.shape[:2]
    if min(h, w) < 1 or (device is not None and reference.device != device):
        raise ValueError("sparse material frames must be nonempty and share a device")
    for name, channels in _MAPS:
        value = maps.get(name)
        if (not isinstance(value, torch.Tensor) or value.shape != (h, w, channels)
                or not value.is_floating_point() or value.device != reference.device):
            raise ValueError(f"sparse material {label}.{name} has invalid shape, type or device")
    if (not isinstance(mask, torch.Tensor) or mask.shape != (h, w)
            or mask.dtype != torch.bool or mask.device != reference.device):
        raise ValueError(f"sparse material {label} static mask must be boolean H x W on the map device")
    for name, value, shape in (("camera", camera, (4, 4)), ("intrinsics", intrinsics, (3, 3))):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError(f"sparse material {label} {name} must be finite floating {shape}")
    return h, w, reference


@torch.no_grad()
def _surface_mask(maps, static, min_opacity, erode_radius, depth_edge_rel, dtype):
    z = maps["depth"].detach()[..., 0].to(dtype)
    opacity = maps["opacity"].detach()[..., 0]
    valid = static.detach() & torch.isfinite(z) & (z > 0)
    valid &= torch.isfinite(opacity) & (opacity >= min_opacity)
    for key, _ in _MATERIALS:
        valid &= torch.isfinite(maps[key].detach()).all(-1)
    normal = maps["normal"].detach()
    valid &= torch.isfinite(normal).all(-1) & (normal.square().sum(-1) > 1e-12)
    if erode_radius:
        nonstatic = F.pad((~static).to(dtype)[None, None], (erode_radius,) * 4, value=1.)
        valid &= F.max_pool2d(nonstatic, 2 * erode_radius + 1, 1)[0, 0] == 0
    if depth_edge_rel:
        safe_z = torch.nan_to_num(z, nan=0., posinf=0., neginf=0.)
        padded = F.pad(safe_z[None, None], (1,) * 4, mode="replicate")
        highest = F.max_pool2d(padded, 3, 1)[0, 0]
        lowest = -F.max_pool2d(-padded, 3, 1)[0, 0]
        valid &= highest - lowest <= depth_edge_rel * safe_z
    return valid


def _linear(value):
    value = value.clamp_min(0)
    return torch.where(value <= .04045, value / 12.92, ((value + .055) / 1.055).pow(2.4))


def _safe_ratio(numerator, denominator):
    return numerator / torch.where(denominator.abs() > 1e-12, denominator, torch.ones_like(denominator))


def _bilinear_layout(uv, height, width):
    """Pixel centers are x+.5/y+.5, equivalent to align_corners=False."""
    positions = torch.nan_to_num(uv - .5, nan=0., posinf=0., neginf=0.)
    x = positions[:, 0].clamp(0, width - 1)
    y = positions[:, 1].clamp(0, height - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    fx, fy = x - x0, y - y0
    indices = torch.stack((y0 * width + x0, y0 * width + x1,
                           y1 * width + x0, y1 * width + x1), dim=1)
    weights = torch.stack(((1 - fx) * (1 - fy), fx * (1 - fy),
                           (1 - fx) * fy, fx * fy), dim=1)
    return indices, weights


def _bilinear(value, indices, weights, dtype, *, linear=False):
    # Convert albedo at the four sparse corners before interpolation, avoiding
    # both gamma-space interpolation and a native-resolution linearized copy.
    gathered = value.detach().reshape(-1, value.shape[-1])[indices].to(dtype)
    gathered = torch.nan_to_num(gathered, nan=0., posinf=0., neginf=0.)
    if linear:
        gathered = _linear(gathered)
    # Anchoring interpolation preserves constant maps exactly, avoiding a tiny
    # nonzero L1 gradient when floating-point projection perturbs pixel centers.
    anchor = gathered[:, 0]
    return anchor + ((gathered - anchor[:, None]) * weights[..., None]).sum(1)


def _empty_diagnostics(zero, weight, pair_available):
    names = ("available_count", "sample_count", "valid_sample_count", "support_fraction",
             "projected_count", "footprint_count", "depth_count", "normal_count", "cycle_count",
             "cycle_error_px", "unweighted", "albedo", "roughness", "metallic")
    result = {key: zero.detach().new_zeros(()) for key in names}
    result.update(weight=zero.detach().new_tensor(weight),
                  pair_available=zero.detach().new_tensor(float(pair_available)))
    return result


def sparse_material_temporal_loss(current, current_static, current_camera, current_intrinsics,
                                  previous, *, weight=.05, num_samples=4096, seed=20261101,
                                  step=0, sample_key=0, depth_abs=.05, depth_rel=.02,
                                  normal_cos=.9, cycle_px=1., min_opacity=.95,
                                  erode_radius=1, depth_edge_rel=.1):
    """Return a weighted loss and detached scalar diagnostics.

    Uniform sampling uses all valid current pixels when there are at most
    ``num_samples``; otherwise it samples with replacement using an isolated
    per-step/image RNG. No gradients enter the previous material, geometry,
    normals, poses, intrinsics or validity. Current rendered materials retain
    their gradients, which can flow through the renderer into scene parameters.

    The caller must acquire ``previous`` from the training split. This function
    receives only model-generated G-buffers and masks, never RGB/teacher images.
    """
    _options(weight, num_samples, seed, step, sample_key, depth_abs, depth_rel,
             normal_cos, cycle_px, min_opacity, erode_radius, depth_edge_rel)
    zero = _graph_zero(current)
    if previous is None or weight == 0:
        return zero, _empty_diagnostics(zero, weight, previous is not None)
    h, w, reference = _validate_frame(current, current_static, current_camera,
                                       current_intrinsics, "current")
    if not isinstance(previous, Mapping) or not isinstance(previous.get("gbuffer"), Mapping):
        raise ValueError("sparse material previous must contain a gbuffer mapping")
    prior = {**previous["gbuffer"], "depth": previous.get("depth"), "opacity": previous.get("opacity")}
    ph, pw, _ = _validate_frame(prior, previous.get("static_mask"), previous.get("camera_to_world"),
                                 previous.get("intrinsics"), "previous", device=reference.device)
    dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
    device = reference.device
    with torch.no_grad():
        camera = current_camera.detach().to(device=device, dtype=dtype)
        k = current_intrinsics.detach().to(device=device, dtype=dtype)
        prior_camera = previous["camera_to_world"].detach().to(device=device, dtype=dtype)
        prior_k = previous["intrinsics"].detach().to(device=device, dtype=dtype)
        try:
            inv_k, inv_prior_k = torch.linalg.inv(k), torch.linalg.inv(prior_k)
            w2c, prior_w2c = torch.linalg.inv(camera), torch.linalg.inv(prior_camera)
        except RuntimeError as exc:
            raise ValueError("sparse material cameras and intrinsics must be invertible") from exc
        if not all(bool(torch.isfinite(v).all()) for v in (inv_k, inv_prior_k, w2c, prior_w2c)):
            raise ValueError("sparse material camera inverses must be finite")
        valid = _surface_mask(current, current_static, min_opacity, erode_radius, depth_edge_rel, dtype)
        prior_valid = _surface_mask(prior, previous["static_mask"], min_opacity, erode_radius, depth_edge_rel, dtype)
        available = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        diagnostics = _empty_diagnostics(zero, weight, True)
        diagnostics["available_count"] = available.new_tensor(available.numel())
        if available.numel() == 0:
            return zero, diagnostics
        if available.numel() <= num_samples:
            selected = available
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed((int(seed) + 1000003 * int(step) + 9176 * int(sample_key)) % (2**63 - 1))
            selected = available[torch.randint(available.numel(), (num_samples,), generator=generator, device=device)]
        n = selected.numel()
        pixels = torch.stack((selected.remainder(w).to(dtype) + .5,
                               selected.div(w, rounding_mode="floor").to(dtype) + .5,
                               torch.ones(n, device=device, dtype=dtype)), -1)
        current_z = current["depth"].detach().reshape(-1, 1)[selected].to(dtype)
        rays = pixels @ inv_k.T
        points = _safe_ratio(rays, rays[:, 2:3]) * current_z
        transform = prior_w2c @ camera
        prior_points = points @ transform[:3, :3].T + transform[:3, 3]
        projection = prior_points @ prior_k.T
        uv = _safe_ratio(projection[:, :2], projection[:, 2:3])
        expected_z = prior_points[:, 2:3]
        projected = (torch.isfinite(uv).all(-1) & torch.isfinite(expected_z[:, 0])
                     & (expected_z[:, 0] > 0) & (projection[:, 2].abs() > 1e-12)
                     & (rays[:, 2].abs() > 1e-12)
                     & (uv[:, 0] >= .5) & (uv[:, 0] <= pw - .5)
                     & (uv[:, 1] >= .5) & (uv[:, 1] <= ph - .5))
        indices, weights = _bilinear_layout(uv, ph, pw)
        # Every contributing bilinear tap must belong to the previous surface.
        # Zero-weight corners at exact pixel centers need no valid support.
        footprint = projected & (prior_valid.reshape(-1)[indices] | (weights == 0)).all(-1)
        prior_z = _bilinear(prior["depth"], indices, weights, dtype)
        depth_valid = footprint & ((prior_z - expected_z).abs()[:, 0] <= depth_abs + depth_rel * expected_z.abs()[:, 0])
        n_current = current["normal"].detach().reshape(-1, 3)[selected].to(dtype)
        n_previous = _bilinear(prior["normal"], indices, weights, dtype)
        normal_valid = depth_valid & (n_previous.norm(dim=-1) > 1e-6)
        normal_valid &= (F.normalize(n_current, dim=-1) * F.normalize(n_previous, dim=-1)).sum(-1).abs() >= normal_cos
        # Backproject the sampled previous depth, not the original current
        # point, so the cycle check independently rejects occlusion/misalignment.
        safe_uv = torch.nan_to_num(uv, nan=0., posinf=0., neginf=0.)
        previous_pixels = torch.cat((safe_uv, torch.ones(n, 1, device=device, dtype=dtype)), -1)
        previous_rays = previous_pixels @ inv_prior_k.T
        previous_surface = _safe_ratio(previous_rays, previous_rays[:, 2:3]) * prior_z
        inverse_transform = w2c @ prior_camera
        returned = previous_surface @ inverse_transform[:3, :3].T + inverse_transform[:3, 3]
        returned_projection = returned @ k.T
        returned_uv = _safe_ratio(returned_projection[:, :2], returned_projection[:, 2:3])
        cycle_error = (returned_uv - pixels[:, :2]).norm(dim=-1)
        accepted = normal_valid & torch.isfinite(cycle_error) & (cycle_error <= cycle_px)
        accepted &= (returned[:, 2] > 0) & (returned_projection[:, 2].abs() > 1e-12)
        accepted &= previous_rays[:, 2].abs() > 1e-12
        accepted &= (returned[:, 2:3] - current_z).abs()[:, 0] <= depth_abs + depth_rel * current_z[:, 0].abs()
        # Only sparse target values are materialized, all detached.
        targets = {key: _bilinear(prior[key], indices, weights, dtype, linear=key == "albedo")
                   for key, _ in _MATERIALS}
        diagnostics.update(
            sample_count=selected.new_tensor(n), valid_sample_count=accepted.sum(),
            support_fraction=accepted.to(dtype).mean(), projected_count=projected.sum(),
            footprint_count=footprint.sum(), depth_count=depth_valid.sum(), normal_count=normal_valid.sum(),
            cycle_count=accepted.sum(),
            cycle_error_px=torch.where(accepted, cycle_error, 0.).sum() / accepted.sum().clamp_min(1),
        )
    denominator = accepted.sum().clamp_min(1)
    components = []
    for key, channels in _MATERIALS:
        value = current[key].reshape(-1, channels)[selected].to(dtype)
        value = torch.nan_to_num(value, nan=0., posinf=0., neginf=0.)
        if key == "albedo":
            value = _linear(value)
        difference = torch.where(accepted[:, None], (value - targets[key]).abs(), 0.)
        component = difference.mean(-1).sum() / denominator
        components.append(component)
        diagnostics[key] = component.detach()
    unweighted = sum(components) / len(components)
    diagnostics["unweighted"] = unweighted.detach()
    return float(weight) * unweighted, {key: value.detach() for key, value in diagnostics.items()}
