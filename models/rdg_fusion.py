"""Small, geometry-aligned G-buffer attention inspired by RDG.

This is a material-fusion adaptation, not RDG's super-resolution network.
Normal/BRDF appearance and geometric reliability enter separate branches.
One generated history supplies a bounded low-frequency correction to native
linear albedo. Only the current albedo value and learned head are differentiable;
geometry, routing features, and all history values are detached.

Packets follow ``ego_temporal`` conventions. ``rgb`` is an albedo proxy needed
by that shared projector, never a shaded RGB guide or an image target.
"""

from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch
from torch import nn
import torch.nn.functional as F

from models.ego_temporal import (
    _clean, _surface_valid, _validate_packet, linear_to_srgb,
    project_history, srgb_to_linear,
)


DEFAULT_CONFIG = {
    "enabled": False, "feature_downscale": 4, "width": 16,
    "start_step": 5000, "ramp_steps": 1000, "interval": 8, "max_gap": 2,
    "max_blend": .35, "max_delta_linear": .05,
    "anchor_weight": .01, "temporal_weight": .02,
    "lr": .001, "weight_decay": 0., "grad_clip": 1.,
}


def validate_rdg_config(config=None):
    """Return a canonical opt-in configuration; reject misspelled options."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("rdg_fusion config must be a mapping")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown rdg_fusion options: {sorted(unknown)}")
    cfg = {**DEFAULT_CONFIG, **dict(config)}
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("rdg_fusion enabled must be boolean")
    for key, minimum in (("feature_downscale", 1), ("width", 4),
                         ("start_step", 0), ("ramp_steps", 0),
                         ("interval", 1), ("max_gap", 1)):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"rdg_fusion {key} must be an integer >= {minimum}")
        cfg[key] = int(value)
    for key in ("max_blend", "max_delta_linear", "anchor_weight", "temporal_weight",
                "lr", "weight_decay", "grad_clip"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"rdg_fusion {key} must be finite and nonnegative")
        cfg[key] = float(value)
    if cfg["lr"] <= 0 or cfg["grad_clip"] <= 0:
        raise ValueError("rdg_fusion lr and grad_clip must be positive")
    if cfg["max_blend"] > 1 or cfg["max_delta_linear"] > 1:
        raise ValueError("rdg_fusion blend and linear-delta bounds must be <= 1")
    return cfg


def rdg_strength(config, step, training=True):
    """Saved-step warmup/ramp, with intermittent processing during training."""
    cfg = validate_rdg_config(config)
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("rdg_fusion step must be a nonnegative integer")
    if not isinstance(training, bool):
        raise ValueError("rdg_fusion training must be boolean")
    if not cfg["enabled"] or step < cfg["start_step"]:
        return 0.
    if training and (step - cfg["start_step"]) % cfg["interval"]:
        return 0.
    return min(1., (step - cfg["start_step"]) / cfg["ramp_steps"]) if cfg["ramp_steps"] else 1.


def _nchw(value):
    return value.permute(2, 0, 1).unsqueeze(0)


def _resize(value, size, mode):
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(_nchw(value), size=size, mode=mode, **kwargs)[0].permute(1, 2, 0)


def _native_support(packet):
    return (_surface_valid(packet)
            & torch.isfinite(packet["viewdirs"].detach()).all(-1, keepdim=True)
            & (packet["normal"].detach().square().sum(-1, keepdim=True) > 1e-12)
            & ((packet["albedo"].detach() >= 0) & (packet["albedo"].detach() <= 1)).all(-1, keepdim=True))


def downsample_packet(packet, factor):
    """Reduce current material/geometry, scaling pixel-center intrinsics.

    Albedo is area-averaged in linear light, preserving its gradient. Geometry
    uses the corresponding pixel centers and is detached. Every source pixel
    in a reduced support footprint must be valid; no majority-mask voting.
    """
    h, w = _validate_packet(packet)
    if isinstance(factor, bool) or not isinstance(factor, Integral) or factor < 1:
        raise ValueError("factor must be a positive integer")
    size = (math.ceil(h / factor), math.ceil(w / factor))
    current_linear = srgb_to_linear(_clean(packet["albedo"]))
    albedo = linear_to_srgb(_resize(current_linear, size, "area"))
    result = {**packet, "albedo": albedo, "rgb": albedo}
    for key in ("depth", "normal", "opacity", "viewdirs", "roughness", "metallic"):
        result[key] = _resize(_clean(packet[key].detach()), size, "bilinear")
    result["normal"] = F.normalize(result["normal"], dim=-1)
    result["viewdirs"] = F.normalize(result["viewdirs"], dim=-1)
    support = _native_support(packet).to(current_linear.dtype)
    result["static_mask"] = (_resize(support, size, "area") >= 1 - 1e-6)[..., 0]
    result["camera_to_world"] = packet["camera_to_world"].detach()
    intrinsics = packet["intrinsics"].detach().clone()
    intrinsics[0] *= size[1] / w
    intrinsics[1] *= size[0] / h
    result["intrinsics"] = intrinsics
    return result


class RDGFusion(nn.Module):
    """Two-candidate attention over current and one reprojected history.

    The feature/geometry branches run exclusively at reduced resolution. A
    bounded linear-albedo delta is upsampled, keeping native detail outside
    that low-frequency correction. Losses returned here already include their
    configured weights and the supplied warmup strength.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = validate_rdg_config(config)
        width = self.config["width"]
        # Shared appearance features: linear albedo(3), world normal(3), R, M.
        self.material_encoder = nn.Sequential(
            nn.Conv2d(8, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 1), nn.SiLU())
        self.query = nn.Conv2d(width, width, 1)
        self.key = nn.Conv2d(width, width, 1)
        # Depth disagreement, cycle error, view delta, normal cosine, validity.
        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(5, width, 1), nn.SiLU(), nn.Conv2d(width, 2, 1))
        nn.init.zeros_(self.geometry_encoder[-1].weight)
        nn.init.zeros_(self.geometry_encoder[-1].bias)
        self.logit_prior = nn.Parameter(torch.tensor([0., -2.]))

    @staticmethod
    def _identity(current):
        albedo = current["albedo"]
        zero = _clean(albedo).sum() * 0
        shape = (*albedo.shape[:2], 1)
        return {"albedo": albedo, "history_weight": albedo.new_zeros(shape),
                "valid": torch.zeros(shape, device=albedo.device, dtype=torch.bool),
                "residual_linear": torch.zeros_like(albedo),
                "loss_anchor": zero, "loss_temporal": zero}

    def forward(self, current, previous=None, strength=1.):
        h, w = _validate_packet(current)
        if isinstance(strength, bool) or not isinstance(strength, Real) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("rdg_fusion strength must be finite and in [0, 1]")
        if (not self.config["enabled"] or previous is None or strength == 0
                or self.config["max_blend"] == 0 or self.config["max_delta_linear"] == 0):
            return self._identity(current)

        low = downsample_packet(current, self.config["feature_downscale"])
        projection = project_history(low, previous)
        support = projection["valid"]
        if not bool(support.any()):
            return self._identity(current)

        current_linear = srgb_to_linear(_clean(low["albedo"]))
        history_linear = projection["warped"]["albedo_linear"].detach()
        with torch.no_grad():
            warped = projection["warped"]
            current_appearance = torch.cat((current_linear.detach(), low["normal"],
                                            low["roughness"], low["metallic"]), -1)
            history_appearance = torch.cat((history_linear, F.normalize(warped["normal"], dim=-1),
                                            warped["roughness"], warped["metallic"]), -1)
            geometry = torch.cat((
                projection["depth_residual"] / (.05 + .02 * projection["expected_depth"].abs()),
                projection["cycle_error"] / 1.5, projection["view_delta"],
                projection["normal_similarity"], support.to(current_linear.dtype)), -1)
            current_appearance = _clean(current_appearance).clamp(-1, 1)
            history_appearance = _clean(history_appearance).clamp(-1, 1)
            geometry = _clean(geometry).clamp(0, 4)
        cf = self.material_encoder(_nchw(current_appearance))
        pf = self.material_encoder(_nchw(history_appearance))
        query = self.query(cf)
        current_logit = (query * self.key(cf)).sum(1, keepdim=True)
        history_logit = (query * self.key(pf)).sum(1, keepdim=True)
        logits = torch.cat((current_logit, history_logit), 1) / math.sqrt(self.config["width"])
        logits = logits + self.geometry_encoder(_nchw(geometry)) + self.logit_prior[None, :, None, None]
        valid_nchw = _nchw(support)
        history_logit = logits[:, 1:2].masked_fill(~valid_nchw, float("-inf"))
        weights = torch.cat((logits[:, :1], history_logit), 1).softmax(1)
        low_weight = weights[:, 1:2][0].permute(1, 2, 0) * (self.config["max_blend"] * strength)
        limit = self.config["max_delta_linear"] * strength
        low_delta = (low_weight * (history_linear - current_linear)).clamp(-limit, limit)

        native_support = (_resize(support.to(current_linear.dtype), (h, w), "bilinear") >= 1 - 1e-6)
        native_support &= _native_support(current)
        delta = _resize(low_delta, (h, w), "bilinear")
        delta = torch.where(native_support, delta, torch.zeros_like(delta))
        native_linear = srgb_to_linear(_clean(current["albedo"]))
        corrected_linear = (native_linear + delta).clamp(0, 1)
        albedo = torch.where(native_support, linear_to_srgb(corrected_linear), current["albedo"])
        residual = torch.where(native_support, corrected_linear - native_linear, torch.zeros_like(delta))
        history_weight = torch.where(native_support, _resize(low_weight, (h, w), "bilinear"),
                                     torch.zeros_like(native_support, dtype=low_weight.dtype))

        anchor = residual.abs().mean(-1, keepdim=True).sum() / native_support.sum().clamp_min(1)
        low_corrected = (current_linear + low_delta).clamp(0, 1)
        error = ((low_corrected - history_linear).square() + 1e-6).sqrt().sub(.001).mean(-1, keepdim=True)
        temporal = (error * support).sum() / support.sum().clamp_min(1)
        return {"albedo": albedo, "history_weight": history_weight,
                "valid": native_support.detach(), "residual_linear": residual,
                "loss_anchor": anchor * (self.config["anchor_weight"] * strength),
                "loss_temporal": temporal * (self.config["temporal_weight"] * strength)}
