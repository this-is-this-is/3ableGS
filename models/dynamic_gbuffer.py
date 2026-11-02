"""D1: instance-balanced, visibility-aware auxiliary G-buffer supervision.

The renderer must accumulate each instance's indicator with the *full scene's*
transmittance. Isolated object alpha and alpha-blended integer IDs are not valid
inputs. Ownership, purity and masks are detached; only supplied residual maps
receive gradients. This module has no renderer, history, RNG, or learned state.
"""
from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch


DEFAULTS = {
    "enabled": False, "mode": "balanced", "w": .25,
    "start_step": 5000, "ramp_steps": 1000, "interval": 1,
    "chunk_size": 8, "visibility_downscale": 2, "min_opacity": .1, "min_pixels": 16,
    "purity_floor": .1, "purity_power": 1.,
}


def validate_dynamic_gbuffer_config(config=None):
    """Return a new, fully specified opt-in configuration; reject silent typos."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("dynamic_gbuffer config must be a mapping")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown dynamic_gbuffer settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **dict(config)}
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("dynamic_gbuffer enabled must be boolean")
    if cfg["mode"] not in ("balanced", "purity"):
        raise ValueError("dynamic_gbuffer mode must be balanced or purity")
    for name, minimum in (("start_step", 0), ("ramp_steps", 0), ("interval", 1),
                          ("chunk_size", 1), ("visibility_downscale", 1), ("min_pixels", 1)):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"dynamic_gbuffer {name} must be an integer >= {minimum}")
        cfg[name] = int(value)
    for name in ("w", "min_opacity", "purity_floor", "purity_power"):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"dynamic_gbuffer {name} must be finite and nonnegative")
        cfg[name] = float(value)
    if cfg["min_opacity"] > 1 or not 0 < cfg["purity_floor"] <= 1:
        raise ValueError("dynamic_gbuffer requires min_opacity in [0, 1] and purity_floor in (0, 1]")
    if cfg["purity_power"] <= 0:
        raise ValueError("dynamic_gbuffer purity_power must be positive")
    return cfg


def dynamic_gbuffer_weight(step, config=None):
    """Scheduled additive loss multiplier, without inverse-interval rescaling.

    A nonzero ramp starts at zero at start_step, matching the existing Base v5
    schedule convention. Every enabled call depends on absolute training step,
    so resuming cannot change which updates receive auxiliary supervision.
    """
    cfg = validate_dynamic_gbuffer_config(config)
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("dynamic_gbuffer step must be a nonnegative integer")
    elapsed = int(step) - cfg["start_step"]
    if not cfg["enabled"] or elapsed < 0 or elapsed % cfg["interval"]:
        return 0.
    ramp = min(1., elapsed / cfg["ramp_steps"]) if cfg["ramp_steps"] else 1.
    return cfg["w"] * ramp


def _scalar_map(value, name, *, shape=None, device=None, floating=False):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"dynamic_gbuffer {name} must be a tensor")
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"dynamic_gbuffer {name} must be nonempty H x W or H x W x 1")
    if shape is not None and value.shape != shape:
        raise ValueError(f"dynamic_gbuffer {name} must have spatial shape {tuple(shape)}")
    if device is not None and value.device != device:
        raise ValueError(f"dynamic_gbuffer {name} must be on {device}")
    if floating and not value.is_floating_point():
        raise ValueError(f"dynamic_gbuffer {name} must have floating dtype")
    return value


class VisibleInstanceAccumulator:
    """Stream full-scene visible masses using O(HW), rather than O(HWK), state.

    Add every dynamic instance exactly once with a positive, namespaced ID.
    ID zero is reserved for the aggregate background. All channels must already
    be cropped/resized identically to the final supervision maps: interpolate
    continuous masses before ownership selection, never interpolate integer IDs.
    Ties select the smallest instance ID, with background winning an exact tie.
    """

    @torch.no_grad()
    def __init__(self, opacity):
        opacity = _scalar_map(opacity, "opacity", floating=True).detach()
        dtype = torch.float64 if opacity.dtype == torch.float64 else torch.float32
        self.opacity = opacity.to(dtype=dtype).clone()
        self.valid = torch.isfinite(self.opacity) & (self.opacity >= 0) & (self.opacity <= 1 + 1e-4)
        self.opacity = torch.nan_to_num(self.opacity, nan=0., posinf=0., neginf=0.).clamp(0, 1)
        self.dynamic_mass = torch.zeros_like(self.opacity)
        self.dominant_mass = torch.zeros_like(self.opacity)
        self.dominant_id = torch.zeros_like(self.opacity, dtype=torch.long)
        self._seen = set()
        self._finalized = False

    @torch.no_grad()
    def add(self, visible_masses, instance_ids):
        """Add H x W x K accumulated indicator channels; return self."""
        if self._finalized:
            raise RuntimeError("dynamic_gbuffer accumulator is already finalized")
        if (not isinstance(visible_masses, torch.Tensor) or visible_masses.ndim != 3
                or visible_masses.shape[:2] != self.opacity.shape
                or not visible_masses.is_floating_point()
                or visible_masses.device != self.opacity.device):
            raise ValueError("dynamic_gbuffer visible masses must be floating H x W x K on the opacity device")
        ids = torch.as_tensor(instance_ids, device=self.opacity.device)
        if ids.ndim != 1 or ids.shape[0] != visible_masses.shape[-1] or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("dynamic_gbuffer instance IDs must be an integer vector with one ID per channel")
        raw_ids = ids.tolist()
        if any(value <= 0 for value in raw_ids) or len(set(raw_ids)) != len(raw_ids) or self._seen.intersection(raw_ids):
            raise ValueError("dynamic_gbuffer instance IDs must be positive and appear exactly once")
        self._seen.update(raw_ids)
        if not raw_ids:
            return self
        masses = visible_masses.detach().to(dtype=self.opacity.dtype)
        self.valid &= (torch.isfinite(masses) & (masses >= -1e-5)).all(-1)
        masses = torch.nan_to_num(masses, nan=0., posinf=0., neginf=0.).clamp_min(0)
        self.dynamic_mass.add_(masses.sum(-1))
        order = ids.argsort()
        mass, index = masses[..., order].max(-1)
        candidate = ids[order][index].long()
        better = (mass > self.dominant_mass) | ((mass == self.dominant_mass) & (candidate < self.dominant_id))
        self.dominant_mass = torch.where(better, mass, self.dominant_mass)
        self.dominant_id = torch.where(better, candidate, self.dominant_id)
        return self

    @torch.no_grad()
    def finalize(self):
        """Return detached maps; nonphysical or nonfinite pixels are invalid."""
        if self._finalized:
            raise RuntimeError("dynamic_gbuffer accumulator is already finalized")
        self._finalized = True
        self.valid &= self.dynamic_mass <= self.opacity + 1e-4
        background = (self.opacity - self.dynamic_mass).clamp_min(0)
        dynamic_wins = self.valid & (self.dominant_mass > background) & (self.dominant_mass > 0)
        owner = torch.where(dynamic_wins, self.dominant_id, 0)
        purity = torch.where(dynamic_wins, self.dominant_mass / self.opacity.clamp_min(1e-12), 0.).clamp(0, 1)
        return {"owner": owner, "purity": purity, "opacity": self.opacity,
                "valid": self.valid.clone()}


def _graph_zero(residual):
    if isinstance(residual, torch.Tensor):
        return torch.nan_to_num(residual.reshape(-1)[:1], nan=0., posinf=0., neginf=0.).sum() * 0.
    return torch.zeros(())


def balanced_instance_loss(residual, ownership, valid_mask, config=None):
    """Return an *unscaled* instance-average residual and detached diagnostics.

    The caller supplies the existing per-pixel A/R/M or robust normal residual,
    multiplies the result by that component's existing loss weight, and applies
    dynamic_gbuffer_weight exactly once. Residual channels are averaged first.
    In purity mode the denominator remains the unweighted eligible pixel count;
    normalizing by confidence would cancel uniform uncertainty attenuation.

    Nothing runs or requires maps for a disabled config. Valid sky/ego masks,
    ownership, confidence and finite-value decisions never receive gradients.
    """
    cfg = validate_dynamic_gbuffer_config(config)
    if not cfg["enabled"]:
        return _graph_zero(residual), {}
    if (not isinstance(residual, torch.Tensor) or residual.ndim not in (2, 3)
            or min(residual.shape) < 1 or not residual.is_floating_point()):
        raise ValueError("dynamic_gbuffer residual must be floating H x W or H x W x C")
    shape, device = residual.shape[:2], residual.device
    if not isinstance(ownership, Mapping) or set(ownership) != {"owner", "purity", "opacity", "valid"}:
        raise ValueError("dynamic_gbuffer ownership must contain owner, purity, opacity and valid maps")
    maps = {name: _scalar_map(value, name, shape=shape, device=device)
            for name, value in ownership.items()}
    if maps["owner"].dtype != torch.long or maps["valid"].dtype != torch.bool:
        raise ValueError("dynamic_gbuffer ownership IDs must be int64 and valid must be boolean")
    if not maps["purity"].is_floating_point() or not maps["opacity"].is_floating_point():
        raise ValueError("dynamic_gbuffer purity and opacity must be floating maps")
    valid_mask = _scalar_map(valid_mask, "valid_mask", shape=shape, device=device)
    dtype = torch.float64 if residual.dtype == torch.float64 else torch.float32
    with torch.no_grad():
        owner, purity, opacity = (maps[name].detach() for name in ("owner", "purity", "opacity"))
        finite_residual = torch.isfinite(residual.detach())
        if residual.ndim == 3:
            finite_residual = finite_residual.all(-1)
        mask = valid_mask.detach()
        eligible = (maps["valid"].detach() & torch.isfinite(mask) & (mask > .99)
                    & (owner > 0) & torch.isfinite(opacity) & (opacity >= cfg["min_opacity"])
                    & torch.isfinite(purity) & (purity > 0) & (purity <= 1) & finite_residual)
        support_owner = owner[eligible]
        ids, inverse, counts = support_owner.unique(sorted=True, return_inverse=True, return_counts=True)
        accepted = counts >= cfg["min_pixels"]
        factors = (purity[eligible].to(dtype).pow(cfg["purity_power"]).clamp_min(cfg["purity_floor"])
                   if cfg["mode"] == "purity" else torch.ones(support_owner.shape, device=device, dtype=dtype))

    # Sanitize before channel averaging so a masked NaN cannot poison backward.
    clean = torch.where(eligible[..., None] if residual.ndim == 3 else eligible,
                        torch.nan_to_num(residual, nan=0., posinf=0., neginf=0.), 0.).to(dtype)
    error = clean.mean(-1) if clean.ndim == 3 else clean
    if ids.numel():
        totals = error.new_zeros(ids.numel()).scatter_add(0, inverse, error[eligible] * factors)
        means = totals / counts.to(dtype).clamp_min(1)
        # Avoid a CUDA scalar synchronization and data-dependent accepted-index
        # allocation for each component. A zero numerator remains connected to
        # every residual even when all candidate instances are too small.
        loss = (means * accepted).sum() / accepted.sum().clamp_min(1)
    else:
        loss = _graph_zero(residual)
    with torch.no_grad():
        accepted_pixels = accepted[inverse] if inverse.numel() else inverse.bool()
        pixel_count = accepted_pixels.sum()
        denominator = pixel_count.clamp_min(1).to(dtype)
        diagnostics = {
            "candidate_instances": ids.new_tensor(ids.numel()),
            "eligible_instances": accepted.sum(),
            "eligible_pixels": pixel_count,
            "rejected_small_instances": (~accepted).sum(),
            "support_fraction": pixel_count.to(dtype) / (shape[0] * shape[1]),
            "mean_purity": (purity[eligible].to(dtype) * accepted_pixels).sum() / denominator,
            "mean_weight": (factors * accepted_pixels).sum() / denominator,
            "weighted_coverage": (factors * accepted_pixels).sum() / (shape[0] * shape[1]),
            "unweighted": loss.detach(),
        }
    return loss, diagnostics
