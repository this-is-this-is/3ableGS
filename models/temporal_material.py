"""Inference-only, geometry-guided temporal fusion of rendered PBR materials.

Only static surfaces are supported. Depth is camera-Z depth, normals are in
world coordinates, and albedo is sRGB. The current normal is never filtered.
``max_history`` caps the effective sample count of an exponential history;
it is not a strict sliding window. State contains only the preceding frame.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch
import torch.nn.functional as F


class TemporalMaterialResolver:
    """Reproject material history before shading, without changing model weights.

    ``resolve`` consumes HWC tensors (depth/opacity have one channel) and a HW
    boolean static mask. Its output is detached and preserves current values at
    every rejected pixel. Call ``reset`` at an independent sequence boundary.
    Nonconsecutive frame IDs, changed streams and changed tensor layouts also
    reset history automatically.
    """

    DEFAULTS = {
        "mode": "material",
        "max_history": 8,
        "history_weight": 0.85,
        "min_opacity": 0.95,
        "depth_abs_tolerance": 0.05,
        "depth_rel_tolerance": 0.02,
        "depth_edge_rel_tolerance": 0.05,
        "normal_cos_min": 0.9,
        "albedo_sigma": 0.2,
        "roughness_sigma": 0.2,
        "metallic_sigma": 0.2,
        "erode_radius": 1,
        "history_source": "resolved",
        "clamp_history": False,
        "guide_ablation": "none",
        "ablation_seed": 0,
    }

    def __init__(self, config=None):
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Temporal material config must be a mapping.")
        self.config = dict(self.DEFAULTS)
        if config is not None:
            unknown = set(config) - set(self.DEFAULTS)
            if unknown:
                raise ValueError(f"Unknown temporal material settings: {sorted(unknown)}")
            self.config.update(config)
        self._validate_config()
        self.reset()

    def _validate_config(self):
        c = self.config
        if c["mode"] not in ("depth", "normal", "material"):
            raise ValueError("mode must be 'depth', 'normal', or 'material'.")
        if c["history_source"] not in ("resolved", "current"):
            raise ValueError("history_source must be 'resolved' or 'current'.")
        if not isinstance(c["clamp_history"], bool):
            raise ValueError("clamp_history must be boolean.")
        if c["guide_ablation"] not in (
                "none", "shuffle_normal", "shuffle_roughness",
                "constant_normal", "constant_roughness"):
            raise ValueError("Unknown guide_ablation.")
        for name, minimum in (("max_history", 1), ("erode_radius", 0), ("ablation_seed", 0)):
            value = c[name]
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        numeric_names = set(self.DEFAULTS) - {
            "mode", "max_history", "erode_radius", "history_source",
            "clamp_history", "guide_ablation", "ablation_seed",
        }
        for name in numeric_names:
            value = c[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        for name in ("history_weight", "min_opacity", "normal_cos_min"):
            if c[name] > 1:
                raise ValueError(f"{name} must lie in [0, 1].")
        for name in ("albedo_sigma", "roughness_sigma", "metallic_sigma"):
            if c[name] <= 0:
                raise ValueError(f"{name} must be positive.")

    def reset(self):
        """Release all previous sequence state."""
        self._history = None
        self._frame_id = None
        self._stream_id = None
        self._layout = None

    @staticmethod
    def _linear(srgb):
        x = srgb.clamp_min(0)
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))

    @staticmethod
    def _srgb(linear):
        x = linear.clamp_min(0)
        return torch.where(x <= 0.0031308, x * 12.92,
                           1.055 * x.clamp_min(0.0031308).pow(1 / 2.4) - 0.055)

    @staticmethod
    def _sample(value, grid):
        return F.grid_sample(
            value.permute(2, 0, 1).unsqueeze(0), grid.unsqueeze(0),
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )[0].permute(1, 2, 0)

    @staticmethod
    def _clamp_to_neighborhood(history, current):
        """Clamp each channel to its finite current 3 x 3 neighborhood."""
        channel_first = current.permute(2, 0, 1).unsqueeze(0)
        finite = torch.isfinite(channel_first)
        for_min = torch.where(finite, channel_first, float("inf"))
        for_max = torch.where(finite, channel_first, -float("inf"))
        lower = -F.max_pool2d(-F.pad(for_min, (1,) * 4, mode="replicate"), 3, stride=1)
        upper = F.max_pool2d(F.pad(for_max, (1,) * 4, mode="replicate"), 3, stride=1)
        fallback = torch.nan_to_num(channel_first, nan=0., posinf=0., neginf=0.)
        lower = torch.where(torch.isfinite(lower), lower, fallback)[0].permute(1, 2, 0)
        upper = torch.where(torch.isfinite(upper), upper, fallback)[0].permute(1, 2, 0)
        return torch.maximum(lower, torch.minimum(history, upper))

    def _routing_guides(self, current, frame_id):
        """Alter routing maps only, preserving every material/shading payload.

        ``shuffle_*`` is a cyclic spatial permutation, not an IID shuffle.
        Each nonsingleton dimension d rolls by
        1 + ((d // 3 + 17 * frame_id + 31 * ablation_seed) % (d - 1)).
        The offset changes with frame ID, is always nonzero for d > 1, and
        leaves the global random-number generators untouched.
        """
        guides = dict(current)
        ablation = self.config["guide_ablation"]
        if ablation.startswith("shuffle_"):
            key = ablation.removeprefix("shuffle_")
            shifts = tuple(
                0 if size == 1 else
                1 + ((size // 3 + 17 * frame_id + 31 * self.config["ablation_seed"]) % (size - 1))
                for size in current[key].shape[:2]
            )
            guides[key] = torch.roll(current[key], shifts=shifts, dims=(0, 1))
        elif ablation == "constant_normal":
            guides["normal"] = torch.zeros_like(current["normal"])
            guides["normal"][..., 2] = 1.
        elif ablation == "constant_roughness":
            guides["roughness"] = torch.full_like(current["roughness"], .5)
        return guides

    def _surface_valid(self, depth, opacity, static_mask, gbuffer):
        valid = static_mask & torch.isfinite(depth[..., 0]) & (depth[..., 0] > 0)
        valid &= torch.isfinite(opacity[..., 0]) & (opacity[..., 0] >= self.config["min_opacity"])
        for key in ("albedo", "roughness", "metallic"):
            valid &= torch.isfinite(gbuffer[key]).all(dim=-1)
        if self.config["mode"] != "depth":
            normal = gbuffer["normal"]
            valid &= torch.isfinite(normal).all(dim=-1) & (normal.square().sum(-1) > 1e-12)
        radius = self.config["erode_radius"]
        if radius:
            # Treat outside the image as nonstatic, including at image borders.
            nonstatic = (~static_mask).to(depth.dtype)[None, None]
            nonstatic = F.pad(nonstatic, (radius,) * 4, value=1)
            valid &= F.max_pool2d(nonstatic, 2 * radius + 1, stride=1)[0, 0] == 0
        edge_tolerance = self.config["depth_edge_rel_tolerance"]
        if edge_tolerance:
            # Expected depth can interpolate foreground and background. Reject
            # even static pixels neighboring a substantial depth discontinuity.
            safe_depth = torch.nan_to_num(depth[..., 0], nan=0., posinf=0., neginf=0.)
            padded = F.pad(safe_depth[None, None], (1,) * 4, mode="replicate")
            local_max = F.max_pool2d(padded, 3, stride=1)[0, 0]
            local_min = -F.max_pool2d(-padded, 3, stride=1)[0, 0]
            valid &= (local_max - local_min) <= edge_tolerance * safe_depth
        return valid

    @staticmethod
    def _validate_inputs(gbuffer, depth, opacity, static_mask, camera_to_world, intrinsics):
        if not isinstance(gbuffer, Mapping):
            raise ValueError("gbuffer must be a mapping.")
        albedo = gbuffer.get("albedo")
        if not isinstance(albedo, torch.Tensor) or albedo.ndim != 3 or albedo.shape[-1] != 3:
            raise ValueError("gbuffer albedo must have shape H x W x 3.")
        h, w = albedo.shape[:2]
        if min(h, w) < 1 or not albedo.is_floating_point():
            raise ValueError("gbuffer must have nonempty floating-point spatial dimensions.")
        for name, channels in (("albedo", 3), ("roughness", 1), ("metallic", 1), ("normal", 3)):
            value = gbuffer.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != (h, w, channels):
                raise ValueError(f"gbuffer {name} must have shape {(h, w, channels)}.")
            if value.device != albedo.device or value.dtype != albedo.dtype:
                raise ValueError("All G-buffer channels must share device and dtype.")
        for name, value in (("depth", depth), ("opacity", opacity)):
            if not isinstance(value, torch.Tensor) or value.shape != (h, w, 1):
                raise ValueError(f"{name} must have shape {(h, w, 1)}.")
            if value.device != albedo.device or value.dtype != albedo.dtype:
                raise ValueError(f"{name} must share the G-buffer device and dtype.")
        if not isinstance(static_mask, torch.Tensor) or static_mask.shape != (h, w):
            raise ValueError(f"static_mask must have shape {(h, w)}.")
        if static_mask.dtype != torch.bool or static_mask.device != albedo.device:
            raise ValueError("static_mask must be boolean on the G-buffer device.")
        for name, value, shape in (("camera_to_world", camera_to_world, (4, 4)),
                                   ("intrinsics", intrinsics, (3, 3))):
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating-point values.")
        return h, w

    @torch.no_grad()
    def resolve(self, gbuffer, depth, opacity, static_mask, camera_to_world,
                intrinsics, frame_id, stream_id, signal=None):
        """Return ``(resolved_gbuffer, diagnostics)`` for one chronological frame.

        Diagnostic maps are H x W x 1: ``temporal_weight`` is the actual history
        contribution, ``temporal_valid`` marks accepted correspondences, and
        ``temporal_history_length`` is the capped effective sample count.
        A valid first observation has length one; invalid pixels have zero.
        ``temporal_grid`` is H x W x 2 in grid_sample's align_corners=False
        convention (all -2 when projection was not attempted).

        Optional ``signal`` is H x W x 3 linear RGB on the G-buffer's device
        and dtype. ``temporal_signal`` then contains its fusion using exactly
        the same reprojection, validity, age and weights as the materials.
        Nonfinite signal pixels reject history; their original values remain.
        Enabling/disabling the signal resets history. No RGB state or output is
        retained when it is omitted. ``history_source='current'`` retains raw
        observations instead of the recursively resolved values, but uses the
        same confidence and effective-count schedule for a controlled ablation.
        """
        if isinstance(frame_id, bool) or not isinstance(frame_id, Integral):
            raise ValueError("frame_id must be an integer.")
        try:
            hash(stream_id)
        except TypeError as exc:
            raise ValueError("stream_id must be hashable.") from exc
        h, w = self._validate_inputs(gbuffer, depth, opacity, static_mask, camera_to_world, intrinsics)
        dtype, device = depth.dtype, depth.device
        if signal is not None:
            if not isinstance(signal, torch.Tensor) or signal.shape != (h, w, 3):
                raise ValueError(f"signal must have shape {(h, w, 3)}.")
            if signal.device != device or signal.dtype != dtype:
                raise ValueError("signal must share the G-buffer device and dtype.")
        layout = (h, w, dtype, device, signal is not None)
        if (self._history is not None and
                (layout != self._layout or stream_id != self._stream_id or frame_id != self._frame_id + 1)):
            self.reset()
        # CPU grid_sample does not support half types; float32 projection is
        # also preferable for accurate subpixel coordinates on the GPU.
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        current = {key: value.detach().to(work_dtype) for key, value in gbuffer.items()
                   if key in ("albedo", "roughness", "metallic", "normal")}
        guides = self._routing_guides(current, frame_id)
        current_signal = None if signal is None else signal.detach().to(work_dtype)
        resolved_signal = None if signal is None else signal.detach().clone()
        z = depth.detach().to(work_dtype)
        camera = camera_to_world.detach().to(device=device, dtype=work_dtype)
        k = intrinsics.detach().to(device=device, dtype=work_dtype)
        try:
            inv_k = torch.linalg.inv(k)
            world_to_camera = torch.linalg.inv(camera)
        except RuntimeError as exc:
            raise ValueError("Camera transform and intrinsics must be invertible.") from exc
        if not torch.isfinite(inv_k).all() or not torch.isfinite(world_to_camera).all():
            raise ValueError("Camera transform and intrinsics must have finite inverses.")
        current_valid = self._surface_valid(z, opacity.detach().to(work_dtype), static_mask, current)
        if current_signal is not None:
            current_valid &= torch.isfinite(current_signal).all(dim=-1)
        resolved = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                    for key, value in gbuffer.items()}
        weight = torch.zeros_like(z)
        grid = torch.full((h, w, 2), -2., device=device, dtype=work_dtype)
        accepted = torch.zeros_like(z, dtype=torch.bool)
        age = current_valid[..., None].to(work_dtype)
        previous = self._history
        if previous is not None and self.config["max_history"] > 1 and self.config["history_weight"] > 0:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device, dtype=work_dtype) + 0.5,
                torch.arange(w, device=device, dtype=work_dtype) + 0.5, indexing="ij",
            )
            pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
            rays = pixels @ inv_k.T
            safe_z = torch.nan_to_num(z, nan=0., posinf=0., neginf=0.)
            points = rays / rays[..., 2:3] * safe_z
            transform = previous["world_to_camera"] @ camera
            previous_points = points @ transform[:3, :3].T + transform[:3, 3]
            projected = previous_points @ previous["intrinsics"].T
            uv = projected[..., :2] / projected[..., 2:3]
            expected_z = previous_points[..., 2:3]
            # Pixel coordinates refer to centers at x + .5. This expression
            # matches grid_sample's align_corners=False convention exactly.
            grid = torch.stack((2 * uv[..., 0] / w - 1, 2 * uv[..., 1] / h - 1), dim=-1)
            projected_valid = (torch.isfinite(grid).all(-1, keepdim=True)
                               & torch.isfinite(expected_z) & (expected_z > 0)
                               & (uv[..., 0:1] >= .5) & (uv[..., 0:1] <= w - .5)
                               & (uv[..., 1:2] >= .5) & (uv[..., 1:2] <= h - .5))
            grid = torch.nan_to_num(grid, nan=-2., posinf=-2., neginf=-2.)
            footprint = self._sample(previous["valid"], grid)
            history_depth = self._sample(previous["depth"], grid)
            tolerance = (self.config["depth_abs_tolerance"]
                         + self.config["depth_rel_tolerance"] * expected_z.abs())
            accepted = (current_valid[..., None] & projected_valid & (footprint >= 1 - 1e-6)
                        & ((history_depth - expected_z).abs() <= tolerance))
            if self.config["mode"] != "depth":
                history_normal = self._sample(previous["normal"], grid)
                dot = (F.normalize(torch.nan_to_num(guides["normal"], nan=0., posinf=0., neginf=0.), dim=-1)
                       * F.normalize(history_normal, dim=-1)).sum(-1, keepdim=True).abs()
                accepted &= dot >= self.config["normal_cos_min"]
            previous_age = self._sample(previous["age"], grid)
            weight = torch.minimum(
                previous_age / (previous_age + 1),
                torch.full_like(previous_age, self.config["history_weight"]),
            )
            # max_history=8 permits at most 7/8 prior history in each update.
            weight.clamp_(max=(self.config["max_history"] - 1) / self.config["max_history"])
            if self.config["mode"] == "material":
                distance = torch.zeros_like(z)
                for key, sigma_key in (("albedo", "albedo_sigma"), ("roughness", "roughness_sigma"),
                                       ("metallic", "metallic_sigma")):
                    guide = self._sample(previous["guide_" + key], grid)
                    difference = (torch.nan_to_num(guides[key], nan=0., posinf=0., neginf=0.)
                                  - guide) / self.config[sigma_key]
                    distance += difference.square().mean(-1, keepdim=True)
                weight *= torch.exp(-0.5 * distance)
            weight = torch.where(accepted, weight, torch.zeros_like(weight))
            accepted &= weight > 0
            age = torch.where(accepted, (previous_age + 1).clamp(max=self.config["max_history"]), age)
            for key in ("albedo", "roughness", "metallic"):
                history_value = self._sample(previous[key], grid)
                value = torch.nan_to_num(current[key])
                blend_value = self._linear(value) if key == "albedo" else value
                if self.config["clamp_history"]:
                    clamp_guide = self._linear(current[key]) if key == "albedo" else current[key]
                    history_value = self._clamp_to_neighborhood(history_value, clamp_guide)
                if key == "albedo":
                    fused = self._srgb((1 - weight) * blend_value + weight * history_value)
                else:
                    fused = (1 - weight) * blend_value + weight * history_value
                # Preserve exact current dtype/values when rejecting history,
                # including NaNs in unsupported pixels. Never modify inputs.
                resolved[key] = torch.where(accepted, fused.to(dtype), resolved[key])
            if current_signal is not None:
                history_signal = self._sample(previous["signal"], grid)
                if self.config["clamp_history"]:
                    history_signal = self._clamp_to_neighborhood(history_signal, current_signal)
                fused_signal = ((1 - weight) * torch.nan_to_num(current_signal)
                                + weight * history_signal)
                resolved_signal = torch.where(accepted, fused_signal.to(dtype), resolved_signal)

        def clean(value):
            return torch.nan_to_num(value, nan=0., posinf=0., neginf=0.).detach().clone()

        self._history = {
            "depth": clean(z),
            "valid": current_valid[..., None].to(work_dtype).detach().clone(),
            "normal": clean(guides["normal"]),
            "age": clean(age),
            "intrinsics": k.clone(),
            "world_to_camera": world_to_camera.clone(),
        }
        for key in ("albedo", "roughness", "metallic"):
            value = (resolved[key].to(work_dtype) if self.config["history_source"] == "resolved"
                     else current[key])
            self._history[key] = clean(self._linear(value) if key == "albedo" else value)
            self._history["guide_" + key] = clean(guides[key])
        if current_signal is not None:
            self._history["signal"] = clean(
                resolved_signal.to(work_dtype) if self.config["history_source"] == "resolved"
                else current_signal)
        self._frame_id, self._stream_id, self._layout = frame_id, stream_id, layout
        diagnostics = {
            "temporal_weight": weight.to(dtype),
            "temporal_valid": accepted,
            "temporal_history_length": age.to(dtype),
            "temporal_grid": grid,
        }
        if resolved_signal is not None:
            diagnostics["temporal_signal"] = resolved_signal
        return resolved, diagnostics
