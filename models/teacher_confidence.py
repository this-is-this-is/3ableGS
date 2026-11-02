"""Frozen training-teacher confidence from geometry-checked temporal moments.

SVGF/NRD motivate history rejection and variance estimates; this is not an
implementation of either denoiser. It never filters teacher values or rendered
images. A caller streams only training views from one frozen geometry snapshot,
then stores the returned scalar confidence maps for subsequent supervision.
"""
from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch

from models.temporal_material import TemporalMaterialResolver


DEFAULTS = {
    "enabled": False, "build_step": 3000, "ramp_steps": 1000,
    "downscale": 4, "max_gap": 2, "max_history": 4,
    "confidence_floor": .25, "albedo_sigma": .1,
    "roughness_sigma": .1, "metallic_sigma": .1,
    "min_opacity": .95, "depth_abs": .05, "depth_rel": .02,
    "normal_cos": .9, "cycle_px": 1., "erode_radius": 1,
    "depth_edge_rel": .1, "max_cache_mb": 256.,
}


def validate_teacher_confidence_config(config=None):
    """Validate settings, including a float16-normal confidence floor [2^-14, 1]."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("teacher_confidence config must be a mapping")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown teacher_confidence options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **dict(config)}
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("teacher_confidence enabled must be boolean")
    integers = (("build_step", 0), ("ramp_steps", 0), ("downscale", 1),
                ("max_gap", 1), ("max_history", 1), ("erode_radius", 0))
    for name, minimum in integers:
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"teacher_confidence {name} must be an integer >= {minimum}")
        cfg[name] = int(value)
    numeric = set(DEFAULTS) - {"enabled", *(name for name, _ in integers)}
    for name in numeric:
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"teacher_confidence {name} must be finite and nonnegative")
        cfg[name] = float(value)
    for name in ("albedo_sigma", "roughness_sigma", "metallic_sigma", "max_cache_mb"):
        if cfg[name] <= 0:
            raise ValueError(f"teacher_confidence {name} must be positive")
    if not torch.finfo(torch.float16).tiny <= cfg["confidence_floor"] <= 1:
        raise ValueError("teacher_confidence confidence_floor must be in [2^-14, 1] for float16 cache storage")
    if not 0 < cfg["normal_cos"] <= 1:
        raise ValueError("teacher_confidence normal_cos must be in (0, 1]")
    if cfg["min_opacity"] > 1:
        raise ValueError("teacher_confidence min_opacity must be in [0, 1]")
    return cfg


def teacher_confidence_strength(step, config=None):
    cfg = validate_teacher_confidence_config(config)
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("teacher_confidence step must be a nonnegative integer")
    if not cfg["enabled"] or step < cfg["build_step"]:
        return 0.
    if not cfg["ramp_steps"]:
        return 1.
    return min(1., (step - cfg["build_step"]) / cfg["ramp_steps"])


def _index(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"teacher_confidence {name} must be a nonnegative integer")
    return int(value)


def _resolver(config):
    # The resolver is used only as a two-view geometry correspondence utility.
    # Actual frame IDs and allowed temporal gaps are checked by update().
    return TemporalMaterialResolver({
        "mode": "normal", "max_history": 2, "history_weight": .5,
        "min_opacity": config["min_opacity"],
        "depth_abs_tolerance": config["depth_abs"],
        "depth_rel_tolerance": config["depth_rel"],
        "depth_edge_rel_tolerance": config["depth_edge_rel"],
        "normal_cos_min": config["normal_cos"], "erode_radius": config["erode_radius"],
    })


def _packet(packet):
    if not isinstance(packet, Mapping):
        raise ValueError("teacher_confidence packet must be a mapping")
    names = ("gbuffer", "depth", "opacity", "static_mask", "camera_to_world", "intrinsics")
    if any(name not in packet for name in names):
        raise ValueError("teacher_confidence packet is missing geometry or material fields")
    TemporalMaterialResolver._validate_inputs(**{name: packet[name] for name in names})
    reference = packet["gbuffer"]["albedo"]
    dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
    result = {name: packet[name].detach().to(device=reference.device, dtype=dtype).clone()
              for name in ("depth", "opacity", "camera_to_world", "intrinsics")}
    result["gbuffer"] = {name: packet["gbuffer"][name].detach().to(dtype=dtype).clone()
                         for name in ("albedo", "roughness", "metallic", "normal")}
    valid_teacher = packet["static_mask"].detach().clone()
    for name in ("albedo", "roughness", "metallic"):
        value = result["gbuffer"][name]
        valid_teacher &= (torch.isfinite(value) & (value >= 0) & (value <= 1)).all(-1)
    result["static_mask"] = valid_teacher
    return result


def _safe_ratio(numerator, denominator):
    return numerator / torch.where(denominator.abs() > 1e-12, denominator, torch.ones_like(denominator))


def _cycle(current, previous, grid, config):
    """Backproject prior DEPTH so the round trip is independent of current Z."""
    sample = TemporalMaterialResolver._sample
    h, w = current["depth"].shape[:2]
    ph, pw = previous["depth"].shape[:2]
    dtype, device = grid.dtype, grid.device
    uv = torch.stack(((grid[..., 0] + 1) * pw * .5,
                      (grid[..., 1] + 1) * ph * .5), -1)
    pixels = torch.cat((uv, torch.ones_like(uv[..., :1])), -1)
    rays = pixels @ torch.linalg.inv(previous["intrinsics"]).T
    sampled_z = sample(torch.nan_to_num(previous["depth"], nan=0., posinf=0., neginf=0.), grid)
    previous_surface = _safe_ratio(rays, rays[..., 2:3]) * sampled_z
    transform = torch.linalg.inv(current["camera_to_world"]) @ previous["camera_to_world"]
    returned = previous_surface @ transform[:3, :3].T + transform[:3, 3]
    projected = returned @ current["intrinsics"].T
    returned_uv = _safe_ratio(projected[..., :2], projected[..., 2:3])
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype) + .5,
                            torch.arange(w, device=device, dtype=dtype) + .5, indexing="ij")
    original_uv = torch.stack((xx, yy), -1)
    error = (returned_uv - original_uv).norm(dim=-1, keepdim=True)
    tolerance = config["depth_abs"] + config["depth_rel"] * current["depth"].abs()
    valid = torch.isfinite(error) & (error <= config["cycle_px"])
    valid &= torch.isfinite(returned).all(-1, keepdim=True) & (returned[..., 2:3] > 0)
    valid &= projected[..., 2:3].abs() > 1e-12
    valid &= rays[..., 2:3].abs() > 1e-12
    valid &= (returned[..., 2:3] - current["depth"]).abs() <= tolerance
    return valid, error


class TeacherConfidenceMoments:
    """Keep one preceding view and its first/second material moments.

    A separate instance per camera avoids cross-camera state. Switching camera,
    changing layout or exceeding max_gap resets history. Gaps use actual frame
    indices, so crossing a held-out frame can be allowed without pretending the
    two training views were consecutive. Split enforcement belongs to the caller.
    The effective history count is capped; this is an EMA after reaching the cap,
    not a finite sliding window. Bilinear transport of second moments includes
    interpolation uncertainty, which can lower confidence at texture boundaries.
    """
    def __init__(self, config=None):
        self.config = validate_teacher_confidence_config(config)
        self.reset()

    def reset(self):
        self._previous = None
        self._frame_id = None
        self._camera_id = None

    @torch.no_grad()
    def update(self, packet, frame_id, camera_id):
        frame_id, camera_id = _index(frame_id, "frame_id"), _index(camera_id, "camera_id")
        if self._previous is not None and camera_id == self._camera_id and frame_id <= self._frame_id:
            raise ValueError("teacher_confidence frames must increase within a camera stream")
        current = _packet(packet)
        z = current["depth"]
        previous = self._previous
        layout = (tuple(z.shape), z.device, z.dtype)
        if (previous is not None and (camera_id != self._camera_id
                or frame_id - self._frame_id > self.config["max_gap"]
                or layout != previous["layout"])):
            previous = None
        reset = previous is None
        resolver = _resolver(self.config)
        use_history = self.config["enabled"] and previous is not None and self.config["max_history"] > 1
        if use_history:
            resolver.resolve(**previous["packet"], frame_id=0, stream_id="confidence-pair")
        _, routing = resolver.resolve(**current, frame_id=int(use_history), stream_id="confidence-pair")
        current_valid = resolver._history["valid"].bool()
        material = torch.cat((TemporalMaterialResolver._linear(current["gbuffer"]["albedo"]),
                              current["gbuffer"]["roughness"], current["gbuffer"]["metallic"]), -1)
        material = torch.where(current_valid, torch.nan_to_num(material, nan=0., posinf=0., neginf=0.), 0.)
        mean, second = material, material.square()
        count = current_valid.to(z.dtype)
        accepted = torch.zeros_like(current_valid)
        cycle_error = torch.zeros_like(z)
        if use_history:
            grid = routing["temporal_grid"]
            cycle_valid, cycle_error = _cycle(current, previous["packet"], grid, self.config)
            sampled_count = resolver._sample(previous["count"], grid)
            accepted = routing["temporal_valid"] & cycle_valid & (sampled_count >= 1 - 1e-6)
            history_weight = (sampled_count / (sampled_count + 1)).clamp(
                max=(self.config["max_history"] - 1) / self.config["max_history"])
            history_weight = torch.where(accepted, history_weight, 0.)
            sampled_mean = resolver._sample(previous["mean"], grid)
            sampled_second = resolver._sample(previous["second"], grid)
            mean = (1 - history_weight) * mean + history_weight * sampled_mean
            second = (1 - history_weight) * second + history_weight * sampled_second
            count = torch.where(accepted, (sampled_count + 1).clamp(max=self.config["max_history"]), count)
        variance = (second - mean.square()).clamp_min(0.)
        score = (variance[..., :3].mean(-1, keepdim=True) / self.config["albedo_sigma"] ** 2
                 + variance[..., 3:4] / self.config["roughness_sigma"] ** 2
                 + variance[..., 4:5] / self.config["metallic_sigma"] ** 2) / 3
        confidence = self.config["confidence_floor"] + (1 - self.config["confidence_floor"]) * torch.exp(-.5 * score)
        confidence = torch.where(accepted, confidence, torch.ones_like(confidence))
        stats = {
            "support_fraction": accepted.to(z.dtype).mean(),
            "confidence_mean": confidence.mean(),
            "variance_score_mean": torch.where(accepted, score, 0.).sum() / accepted.sum().clamp_min(1),
            "history_length_mean": count.sum() / current_valid.sum().clamp_min(1),
            "reset": z.new_tensor(float(reset)),
            "valid_fraction": current_valid.to(z.dtype).mean(),
            "cycle_error_px": torch.where(accepted, cycle_error, 0.).sum() / accepted.sum().clamp_min(1),
        }
        self._previous = {"packet": current, "layout": layout,
                          "mean": mean.detach(), "second": second.detach(), "count": count.detach()}
        self._frame_id, self._camera_id = frame_id, camera_id
        return confidence.detach(), {name: value.detach() for name, value in stats.items()}


@torch.no_grad()
def normalized_teacher_confidence(confidence, static, surface_mask, strength):
    """Mean-one weights on static supervised pixels, exactly one elsewhere.

    Raw confidence lies in (0,1]. After ramping, normalization retains the
    original static loss mass, original denominator and dynamic-object weights.
    Effective weights may exceed one, bounded by 1 / raw-confidence minimum.
    All routing and confidence values are detached from the training graph.
    """
    if isinstance(strength, bool) or not isinstance(strength, Real) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("teacher_confidence strength must lie in [0, 1]")
    if (not torch.is_tensor(confidence) or confidence.ndim != 3 or confidence.shape[-1] != 1
            or min(confidence.shape[:2]) < 1 or not confidence.is_floating_point()):
        raise ValueError("teacher_confidence confidence must be floating H x W x 1")
    if (not torch.is_tensor(static) or static.shape != confidence.shape[:2]
            or static.dtype != torch.bool or static.device != confidence.device):
        raise ValueError("teacher_confidence static must be boolean H x W on the confidence device")
    if (not torch.is_tensor(surface_mask) or surface_mask.shape != confidence.shape
            or surface_mask.device != confidence.device):
        raise ValueError("teacher_confidence surface mask must have shape H x W x 1 on the confidence device")
    if not bool((torch.isfinite(confidence) & (confidence > 0) & (confidence <= 1)).all()):
        raise ValueError("teacher_confidence confidence must be finite and in (0, 1]")
    if not bool((torch.isfinite(surface_mask) & (surface_mask >= 0) & (surface_mask <= 1)).all()):
        raise ValueError("teacher_confidence surface mask must be finite and in [0, 1]")
    dtype = torch.float64 if confidence.dtype == torch.float64 else torch.float32
    ones = torch.ones_like(confidence, dtype=dtype)
    if not strength:
        return ones
    values = 1 - strength + strength * confidence.detach().to(dtype)
    eligible = static[..., None] & (surface_mask > .99)
    count = eligible.sum().to(dtype)
    total = torch.where(eligible, values, 0.).sum()
    scale = count / total.clamp_min(torch.finfo(dtype).tiny)
    return torch.where(eligible, values * scale, ones).detach()
