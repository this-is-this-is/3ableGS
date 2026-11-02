"""Bounded training-view correction of existing G-buffer material teachers.

This is a lightweight adaptation of IIF's low-dimensional material ambiguity
model, not its multi-candidate, per-object distribution matching pipeline.
Each training image has five channel gains and five biases. Camera-wise
centering fixes their mean to identity; only valid static pixels use them.
The normal teacher and renderer are outside this module and remain unchanged.
"""
from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch
from torch import nn


DEFAULTS = {
    "enabled": False, "start_step": 3000, "ramp_steps": 1000,
    "lr": .001, "max_gain_delta": .15, "max_bias": .05,
    "identity_w": .01, "smooth_w": .01, "max_gap": 2, "min_opacity": .95,
}
CHANNELS = (("albedo", 3), ("roughness", 1), ("metallic", 1))


def validate_teacher_calibration_config(config=None):
    """Canonicalize an opt-in configuration without silently accepting typos."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("teacher_calibration config must be a mapping")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown teacher_calibration options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **dict(config)}
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("teacher_calibration enabled must be boolean")
    for key, minimum in (("start_step", 0), ("ramp_steps", 0), ("max_gap", 1)):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"teacher_calibration {key} must be an integer >= {minimum}")
        cfg[key] = int(value)
    for key in ("lr", "max_gain_delta", "max_bias", "identity_w", "smooth_w", "min_opacity"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"teacher_calibration {key} must be finite and nonnegative")
        cfg[key] = float(value)
    if cfg["lr"] <= 0:
        raise ValueError("teacher_calibration lr must be positive")
    if not 0 <= cfg["max_gain_delta"] < 1:
        raise ValueError("teacher_calibration max_gain_delta must be in [0, 1)")
    if cfg["max_bias"] > 1 or cfg["min_opacity"] > 1:
        raise ValueError("teacher_calibration max_bias and min_opacity must be in [0, 1]")
    return cfg


def teacher_calibration_strength(step, config=None):
    """Zero at the start of a nonzero ramp; full strength after ramp_steps."""
    cfg = validate_teacher_calibration_config(config)
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("teacher_calibration step must be a nonnegative integer")
    if not cfg["enabled"] or step < cfg["start_step"]:
        return 0.
    return min(1., (step - cfg["start_step"]) / cfg["ramp_steps"]) if cfg["ramp_steps"] else 1.


def _training_ids(train_indices, num_cams):
    if isinstance(num_cams, bool) or not isinstance(num_cams, Integral) or num_cams < 1:
        raise ValueError("teacher_calibration num_cams must be a positive integer")
    if isinstance(train_indices, torch.Tensor):
        if train_indices.ndim != 1 or train_indices.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("teacher_calibration train_indices must contain integers")
        train_indices = train_indices.tolist()
    if isinstance(train_indices, (str, bytes, Mapping)):
        raise ValueError("teacher_calibration train_indices must be a nonempty iterable")
    try:
        ids = list(train_indices)
    except TypeError as exc:
        raise ValueError("teacher_calibration train_indices must be a nonempty iterable") from exc
    if not ids or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in ids):
        raise ValueError("teacher_calibration train_indices must contain nonnegative integers")
    ids = sorted(int(value) for value in ids)
    if len(set(ids)) != len(ids):
        raise ValueError("teacher_calibration train_indices must be unique")
    return ids


class TeacherCalibration(nn.Module):
    """Ten deterministic, train-only parameters per training image.

    ``raw[:, 0]`` controls channel gain; ``raw[:, 1]`` controls bias. Channels
    are albedo RGB, roughness, metallic. For each camera, half of the centered
    tanh parameters lies in [-1, 1] and has mean zero. Thus gain is positive,
    bounded and camera-wise mean one, while bias is bounded with mean zero.
    These statements apply before output clipping, which is logged separately.
    """

    def __init__(self, train_indices, num_cams, config=None):
        super().__init__()
        self.config = validate_teacher_calibration_config(config)
        ids = _training_ids(train_indices, num_cams)
        self.num_cams = int(num_cams)
        self._image_to_row = {image_id: row for row, image_id in enumerate(ids)}
        self.register_buffer("train_indices", torch.tensor(ids, dtype=torch.long))
        self.raw = nn.Parameter(torch.zeros(len(ids), 2, 5))
        camera_ids = sorted({value % self.num_cams for value in ids})
        groups = {camera: group for group, camera in enumerate(camera_ids)}
        group_ids = [groups[value % self.num_cams] for value in ids]
        self.register_buffer("camera_groups", torch.tensor(group_ids, dtype=torch.long), persistent=False)
        self.register_buffer("camera_counts", torch.tensor(
            [group_ids.count(group) for group in range(len(camera_ids))], dtype=torch.long), persistent=False)
        left, right, gaps = [], [], []
        for camera in camera_ids:
            rows = [row for row, value in enumerate(ids) if value % self.num_cams == camera]
            for before, after in zip(rows, rows[1:]):
                gap = (ids[after] - ids[before]) // self.num_cams
                if gap <= self.config["max_gap"]:
                    left.append(before)
                    right.append(after)
                    gaps.append(gap)
        self.register_buffer("edge_left", torch.tensor(left, dtype=torch.long), persistent=False)
        self.register_buffer("edge_right", torch.tensor(right, dtype=torch.long), persistent=False)
        self.register_buffer("edge_gaps", torch.tensor(gaps, dtype=torch.long), persistent=False)

    def metadata(self):
        """Config and split identity for strict checkpoint compatibility checks."""
        return {"config": dict(self.config), "train_indices": sorted(self._image_to_row),
                "num_cams": self.num_cams}

    def _normalized(self):
        bounded = self.raw.tanh()
        totals = bounded.new_zeros((self.camera_counts.numel(), 2, 5))
        totals = totals.index_add(0, self.camera_groups, bounded)
        means = totals / self.camera_counts[:, None, None]
        return bounded, .5 * (bounded - means[self.camera_groups])

    def coefficients(self, step):
        """Return all train-view gains/biases, preserving camera-wise identity."""
        strength = teacher_calibration_strength(step, self.config)
        if not strength:
            return self.raw.new_ones((len(self._image_to_row), 5)), self.raw.new_zeros((len(self._image_to_row), 5))
        _, normalized = self._normalized()
        return (1. + strength * self.config["max_gain_delta"] * normalized[:, 0],
                strength * self.config["max_bias"] * normalized[:, 1])

    def _validate_teacher(self, teacher, valid):
        if not isinstance(teacher, Mapping):
            raise ValueError("teacher_calibration teacher must be a material mapping")
        reference = teacher.get("albedo")
        if (not torch.is_tensor(reference) or reference.ndim != 3 or reference.shape[-1] != 3
                or not reference.is_floating_point() or min(reference.shape[:2]) < 1):
            raise ValueError("teacher_calibration albedo must be a nonempty floating H x W x 3 tensor")
        height, width = reference.shape[:2]
        if reference.device != self.raw.device:
            raise ValueError("teacher_calibration head and teacher must share a device")
        for name, channels in CHANNELS:
            value = teacher.get(name)
            if (not torch.is_tensor(value) or value.shape != (height, width, channels)
                    or value.dtype != reference.dtype or value.device != reference.device):
                raise ValueError(f"teacher_calibration {name} has an invalid shape, dtype or device")
        if (not torch.is_tensor(valid) or valid.shape != (height, width)
                or valid.dtype != torch.bool or valid.device != reference.device):
            raise ValueError("teacher_calibration valid must be a boolean H x W tensor on the teacher device")
        return reference

    def correct(self, teacher, valid, image_idx, step):
        """Correct static pixels and return global parameter priors/diagnostics.

        Geometry and annotations decide ``valid`` in the caller. Excluded
        teacher pixels are preserved verbatim, including invalid values, while
        parameter arithmetic uses sanitized values there to avoid NaN gradients.
        No parameter is indexed by a held-out image, even during warmup.
        """
        if isinstance(image_idx, bool) or not isinstance(image_idx, Integral) or image_idx not in self._image_to_row:
            raise ValueError("teacher_calibration image must belong to the training split")
        strength = teacher_calibration_strength(step, self.config)
        reference = self._validate_teacher(teacher, valid)
        target = {name: teacher[name].detach() for name, _ in CHANNELS}
        if not strength or not self.training:
            return target, {}, {}
        with torch.no_grad():
            for name, _ in CHANNELS:
                value = target[name]
                invalid = (~torch.isfinite(value) | (value < 0.) | (value > 1.))
                if bool((invalid & valid[..., None]).any()):
                    raise ValueError(f"teacher_calibration valid {name} must be finite and in [0, 1]")
        bounded, normalized = self._normalized()
        gain = 1. + strength * self.config["max_gain_delta"] * normalized[:, 0]
        bias = strength * self.config["max_bias"] * normalized[:, 1]
        row = self._image_to_row[int(image_idx)]
        corrected, offset = {}, 0
        mask = valid.detach()[..., None]
        count = valid.sum().clamp_min(1)
        clipped_count = reference.new_zeros(())
        difference_sum = reference.new_zeros(())
        for name, channels in CHANNELS:
            value = target[name]
            safe_value = torch.where(mask, value, 0.)
            # A residual expression gives bit-exact identity at initialization.
            delta = (gain[row, offset:offset + channels].to(value) - 1.) * safe_value
            delta = delta + bias[row, offset:offset + channels].to(value)
            candidate = safe_value + delta
            transformed = candidate.clamp(0., 1.)
            corrected[name] = torch.where(mask, transformed, value)
            with torch.no_grad():
                clipped_count += (((candidate < 0.) | (candidate > 1.)) & mask).sum()
                difference_sum += torch.where(mask, (transformed - safe_value).abs(), 0.).sum()
            offset += channels
        identity = bounded.square().mean()
        if self.edge_left.numel():
            differences = normalized[self.edge_right] - normalized[self.edge_left]
            smooth = (differences / self.edge_gaps[:, None, None]).square().mean()
        else:
            smooth = self.raw[0, 0, 0] * 0.
        losses = {
            "teacher_calibration_identity_loss": strength * self.config["identity_w"] * identity,
            "teacher_calibration_smooth_loss": strength * self.config["smooth_w"] * smooth,
        }
        with torch.no_grad():
            gain_delta = gain - 1.
            stats = {
                "strength": reference.new_tensor(strength),
                "valid_fraction": valid.to(reference.dtype).mean(),
                "train_view_count": reference.new_tensor(len(self._image_to_row)),
                "parameter_count": reference.new_tensor(self.raw.numel()),
                "gain_abs_mean": gain_delta.abs().mean(), "gain_abs_max": gain_delta.abs().max(),
                "bias_abs_mean": bias.abs().mean(), "bias_abs_max": bias.abs().max(),
                "clamp_fraction": clipped_count / (count * 5),
                "correction_abs_mean": difference_sum / (count * 5),
                "identity_unweighted": identity.detach(), "smooth_unweighted": smooth.detach(),
            }
        return corrected, losses, stats
