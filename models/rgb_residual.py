"""Bounded display-RGB correction for joint Base-v4 reconstruction.

Only the rendered feature input differentiates through CNN conditioning. The
PBR operand in the final composition remains trainable. External-HDR bypass
and camera-update scheduling belong to the trainer.
"""
import math

import torch
from torch import nn


def _bound(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


class BasicBlock(nn.Module):
    """Two 3x3 convolutions and a residual skip, without batch normalization."""
    def __init__(self, in_dim, dim):
        super().__init__()
        self.res_branch = nn.Conv2d(in_dim, dim, 1) if in_dim != dim else nn.Identity()
        self.main_branch = nn.Sequential(nn.Conv2d(in_dim, dim, 3, padding=1),
                                         nn.ReLU(inplace=True), nn.Conv2d(dim, dim, 3, padding=1))
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, value):
        return self.final_activation(self.res_branch(value) + self.main_branch(value))


class RGBResidual(nn.Module):
    """Feature8 + normal3 + roughness1 + depth1 + alpha1 + view3 + RGB3 -> head6."""
    def __init__(self, feature_dim=8, hidden_dim=32, max_log_gain=.3,
                 max_additive=.1, additive_enabled=True):
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("hidden_dim", hidden_dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        _bound("max_log_gain", max_log_gain)
        _bound("max_additive", max_additive)
        if type(additive_enabled) is not bool:
            raise ValueError("additive_enabled must be boolean")
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.max_log_gain, self.max_additive = float(max_log_gain), float(max_additive)
        self.additive_enabled = additive_enabled
        # Adding the branch must not advance Base-v4's scene/sampler RNG.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            head = nn.Conv2d(hidden_dim, 6, 1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.net = nn.Sequential(BasicBlock(feature_dim + 12, hidden_dim),
                                     BasicBlock(hidden_dim, hidden_dim), head)

    def forward(self, features, normal, roughness, depth, alpha, view, pbr_rgb, *, strength=1.0):
        if isinstance(strength, bool) or not isinstance(strength, (float, int)) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be a finite number in [0, 1]")
        inputs = (("features", features, self.feature_dim), ("normal", normal, 3),
                  ("roughness", roughness, 1), ("depth", depth, 1), ("alpha", alpha, 1),
                  ("view", view, 3), ("pbr_rgb", pbr_rgb, 3))
        for name, value, channels in inputs:
            if value.ndim != 3 or value.shape[-1] != channels or min(value.shape[:2]) < 1:
                raise ValueError(f"{name} must have nonempty shape [H, W, {channels}]")
            if value.shape[:2] != features.shape[:2]:
                raise ValueError(f"{name} must share feature image dimensions")
            if not value.is_floating_point() or value.device != features.device or value.dtype != features.dtype:
                raise ValueError(f"{name} must share the features' floating dtype and device")
        coverage = alpha.detach().clamp(0, 1)
        condition = torch.cat([features, normal.detach(), roughness.detach(),
                               torch.log1p(depth.detach().clamp_min(0)) / 10,
                               coverage, view.detach(), pbr_rgb.detach().clamp(0, 1)], dim=-1)
        raw = self.net(condition.permute(2, 0, 1)[None])[0].permute(1, 2, 0)
        log_gain = self.max_log_gain * raw[..., :3].tanh()
        additive = self.max_additive * raw[..., 3:].tanh()
        if not self.additive_enabled:
            additive = additive * 0
        # expm1(0) is exactly zero, preserving PBR at zero initialization.
        residual = strength * coverage * (log_gain.expm1() * pbr_rgb + additive)
        return {"rgb_hybrid": pbr_rgb + residual, "rgb_gain": log_gain.exp(),
                "rgb_log_gain": log_gain, "rgb_additive": additive, "rgb_residual": residual}


def normalized_rgb_residual_penalties(log_gain, additive, max_log_gain=.3, max_additive=.1):
    """RGB-mean normalized squared penalties [H,W,1], before support weighting."""
    _bound("max_log_gain", max_log_gain)
    _bound("max_additive", max_additive)
    if log_gain.ndim != 3 or log_gain.shape[-1] != 3 or additive.shape != log_gain.shape:
        raise ValueError("log_gain and additive require matching [H, W, 3] shapes")
    gain = (log_gain / max_log_gain).square() if max_log_gain else log_gain * 0
    add = (additive / max_additive).square() if max_additive else additive * 0
    return gain.mean(-1, keepdim=True), add.mean(-1, keepdim=True)
