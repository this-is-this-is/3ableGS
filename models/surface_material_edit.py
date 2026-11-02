"""Frozen world-space anchors for sparse, static-surface material edits.

This is a controlled edit propagation experiment, not a learned weather model.
Camera-Z depth and world-space normals follow the renderer's conventions.
Build the bank from training keyframes, finalize once, then query any order.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch
from scipy.spatial import cKDTree


class SurfaceMaterialEditor:
    """Diffuse a seed edit over a fixed surface graph, then shade it elsewhere.

    ``geometry`` uses distance alone, ``rgb`` adds rendered color similarity,
    and ``gbuffer`` adds normal and intrinsic material similarity. Graph and
    query weights retain absolute affinity, so weak edges cannot become strong
    merely because they are a surface's only neighbors.

    All inputs are detached. Add keyframes only before ``finalize``; ``apply``
    never changes the bank. Provide ``allowed_frame_ids`` to enforce the train
    split at the API boundary in addition to the runner's split checks.
    """

    DEFAULTS = {
        "affinity": "gbuffer",
        "allowed_frame_ids": None,
        "voxel_size": 0.2,
        "max_anchors": 20000,
        "sample_stride": 8,
        "knn": 8,
        "neighbor_radius": 0.75,
        "steps": 20,
        "diffusion_rate": 0.8,
        "min_opacity": 0.95,
        "normal_cos_min": 0.9,
        "albedo_sigma": 0.15,
        "roughness_sigma": 0.15,
        "metallic_sigma": 0.15,
        "rgb_sigma": 0.15,
        "delta_albedo_logit": [-0.4, -0.4, -0.4],
        "delta_roughness_logit": -1.0,
        "query_chunk_size": 16384,
    }

    def __init__(self, config=None):
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Surface edit config must be a mapping.")
        self.config = dict(self.DEFAULTS)
        self.config["delta_albedo_logit"] = list(self.DEFAULTS["delta_albedo_logit"])
        if config is not None:
            unknown = set(config) - set(self.DEFAULTS)
            if unknown:
                raise ValueError(f"Unknown surface edit settings: {sorted(unknown)}")
            self.config.update(config)
        self._validate_config()
        self._observations = {}
        self._finalized = False
        self._summary = None

    def _validate_config(self):
        c = self.config
        if c["affinity"] not in ("geometry", "rgb", "gbuffer"):
            raise ValueError("affinity must be geometry, rgb, or gbuffer.")
        for key in ("max_anchors", "sample_stride", "knn", "query_chunk_size"):
            if isinstance(c[key], bool) or not isinstance(c[key], Integral) or c[key] < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if isinstance(c["steps"], bool) or not isinstance(c["steps"], Integral) or c["steps"] < 0:
            raise ValueError("steps must be a nonnegative integer.")
        for key in ("voxel_size", "neighbor_radius", "albedo_sigma",
                    "roughness_sigma", "metallic_sigma", "rgb_sigma"):
            if (isinstance(c[key], bool) or not isinstance(c[key], Real)
                    or not math.isfinite(c[key]) or c[key] <= 0):
                raise ValueError(f"{key} must be finite and positive.")
        for key in ("min_opacity", "normal_cos_min", "diffusion_rate"):
            if (isinstance(c[key], bool) or not isinstance(c[key], Real)
                    or not math.isfinite(c[key]) or not 0 <= c[key] <= 1):
                raise ValueError(f"{key} must lie in [0, 1].")
        delta = np.asarray(c["delta_albedo_logit"], dtype=np.float64)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("delta_albedo_logit must contain three finite numbers.")
        c["delta_albedo_logit"] = delta.tolist()
        if (not isinstance(c["delta_roughness_logit"], Real)
                or not math.isfinite(c["delta_roughness_logit"])):
            raise ValueError("delta_roughness_logit must be finite.")
        if c["allowed_frame_ids"] is not None:
            ids = c["allowed_frame_ids"]
            if (not isinstance(ids, (list, tuple, set, range))
                    or any(isinstance(i, bool) or not isinstance(i, Integral) for i in ids)):
                raise ValueError("allowed_frame_ids must contain integer frame IDs.")
            c["allowed_frame_ids"] = tuple(sorted(set(ids)))

    def _inputs(self, gbuffer, depth, opacity, static_mask,
                camera_to_world, intrinsics, raw_rgb):
        from models.temporal_material import TemporalMaterialResolver

        h, w = TemporalMaterialResolver._validate_inputs(
            gbuffer, depth, opacity, static_mask, camera_to_world, intrinsics)
        if self.config["affinity"] == "rgb" and raw_rgb is None:
            raise ValueError("RGB affinity requires raw_rgb for keyframes and queries.")
        if raw_rgb is not None:
            if (not isinstance(raw_rgb, torch.Tensor) or raw_rgb.shape != (h, w, 3)
                    or raw_rgb.device != depth.device or not raw_rgb.is_floating_point()):
                raise ValueError("raw_rgb must be a floating-point H x W x 3 tensor on the input device.")
        # Work on CPU float64 for stable geometry/voxel aggregation. Querying
        # is chunked below; no pixels-by-all-anchors array is ever allocated.
        def array(tensor):
            return tensor.detach().to(device="cpu", dtype=torch.float64).numpy()

        z, op = array(depth)[..., 0], array(opacity)[..., 0]
        material = np.concatenate([array(gbuffer[k]) for k in
                                   ("albedo", "roughness", "metallic")], axis=-1)
        normal = array(gbuffer["normal"]).copy()
        lengths = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal /= np.maximum(lengths, 1e-12)
        # Renderer normals face the camera; choose a deterministic hemisphere
        # for averaging, and use abs(dot) for all affinity comparisons.
        major = np.argmax(np.abs(normal), axis=-1)
        sign = np.take_along_axis(normal, major[..., None], axis=-1)
        normal *= np.where(sign < 0, -1, 1)
        rgb = array(raw_rgb) if raw_rgb is not None else np.zeros((h, w, 3))
        features = np.concatenate((material, normal, rgb), axis=-1)
        valid = (static_mask.detach().cpu().numpy().copy() & np.isfinite(z) & (z > 0)
                 & np.isfinite(op) & (op > 0) & (op >= self.config["min_opacity"])
                 & np.isfinite(features).all(axis=-1) & (lengths[..., 0] > 1e-12))
        camera, k = array(camera_to_world), array(intrinsics)
        try:
            inverse_k = np.linalg.inv(k)
            inverse_camera = np.linalg.inv(camera)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Camera and intrinsics must be invertible.") from exc
        if not np.isfinite(inverse_k).all() or not np.isfinite(inverse_camera).all():
            raise ValueError("Camera and intrinsics must have finite inverses.")
        return h, w, z, op, features, valid, camera, inverse_k

    @staticmethod
    def _world_points(indices, width, depth, camera, inverse_k):
        y, x = np.divmod(indices, width)
        pixels = np.stack((x + 0.5, y + 0.5, np.ones_like(x)), axis=-1)
        rays = pixels @ inverse_k.T
        with np.errstate(divide="ignore", invalid="ignore"):
            local = rays / rays[:, 2:3] * depth.reshape(-1)[indices, None]
        return local @ camera[:3, :3].T + camera[:3, 3]

    @torch.no_grad()
    def add_keyframe(self, gbuffer, depth, opacity, static_mask, camera_to_world,
                     intrinsics, frame_id, seed_mask, raw_rgb=None):
        """Record sampled *training* observations and a user-controlled seed.

        ``seed_mask`` is HW boolean. It is used only to construct this bank;
        query/evaluation frames must never be added. Frame IDs are unique.
        Sampling always includes at least one valid seed if one exists.
        """
        if self._finalized:
            raise RuntimeError("The finalized surface bank is immutable.")
        if isinstance(frame_id, bool) or not isinstance(frame_id, Integral):
            raise ValueError("frame_id must be an integer.")
        allowed = self.config["allowed_frame_ids"]
        if allowed is not None and frame_id not in allowed:
            raise ValueError(f"Frame {frame_id} is not an allowed training keyframe.")
        if frame_id in self._observations:
            raise ValueError(f"Frame {frame_id} has already been added.")
        h, w, z, op, features, valid, camera, inverse_k = self._inputs(
            gbuffer, depth, opacity, static_mask, camera_to_world, intrinsics, raw_rgb)
        if (not isinstance(seed_mask, torch.Tensor) or seed_mask.shape != (h, w)
                or seed_mask.dtype != torch.bool or seed_mask.device != depth.device):
            raise ValueError("seed_mask must be HW boolean on the input device.")
        seed = seed_mask.detach().cpu().numpy() & valid
        stride = self.config["sample_stride"]
        sample = np.zeros((h, w), dtype=bool)
        sample[::stride, ::stride] = True
        sample &= valid
        sample.reshape(-1)[np.flatnonzero(seed)[::stride ** 2]] = True
        indices = np.flatnonzero(sample)
        points = self._world_points(indices, w, z, camera, inverse_k)
        finite = np.isfinite(points).all(axis=-1)
        indices, points = indices[finite], points[finite]
        self._observations[int(frame_id)] = (
            points, features.reshape(-1, 11)[indices].copy(),
            op.reshape(-1)[indices].copy(), seed.reshape(-1)[indices].copy())
        return {"frame_id": int(frame_id), "samples": len(indices),
                "seed_samples": int(seed.reshape(-1)[indices].sum())}

    def _affinity(self, query_features, neighbor_features):
        c = self.config
        if c["affinity"] == "geometry":
            return np.ones(neighbor_features.shape[:-1], dtype=np.float64)
        if c["affinity"] == "rgb":
            distance = np.mean(((query_features[..., 8:11]
                                 - neighbor_features[..., 8:11]) / c["rgb_sigma"]) ** 2, axis=-1)
            return np.exp(-0.5 * distance)
        delta = query_features[..., :5] - neighbor_features[..., :5]
        distance = (np.mean((delta[..., :3] / c["albedo_sigma"]) ** 2, axis=-1)
                    + (delta[..., 3] / c["roughness_sigma"]) ** 2
                    + (delta[..., 4] / c["metallic_sigma"]) ** 2)
        normal_dot = np.abs(np.sum(query_features[..., 5:8] * neighbor_features[..., 5:8], axis=-1))
        return np.exp(-0.5 * distance) * (normal_dot >= c["normal_cos_min"])

    def finalize(self):
        """Freeze a deterministic voxel bank and propagate clamped seed edits."""
        if self._finalized:
            return dict(self._summary)
        if not self._observations:
            raise ValueError("No training keyframes have been added.")
        ordered = [self._observations[k] for k in sorted(self._observations)]
        points, features, confidence, seeds = [
            np.concatenate([observation[i] for observation in ordered], axis=0) for i in range(4)]
        if not len(points):
            raise ValueError("Training keyframes contain no valid static surface samples.")
        if not seeds.any():
            raise ValueError("Training keyframes contain no valid seed; adjust the seed ROI.")
        voxels = np.floor(points / self.config["voxel_size"])
        _, inverse = np.unique(voxels, axis=0, return_inverse=True)
        n = int(inverse.max()) + 1
        mass = np.bincount(inverse, weights=confidence, minlength=n)
        self._points = np.stack([np.bincount(inverse, weights=points[:, j] * confidence,
                                            minlength=n) / mass for j in range(3)], axis=-1)
        self._features = np.stack([np.bincount(inverse, weights=features[:, j] * confidence,
                                              minlength=n) / mass for j in range(11)], axis=-1)
        self._seeds = np.bincount(inverse, weights=seeds.astype(float), minlength=n) > 0
        normals = self._features[:, 5:8]
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
        # Even sampling over lexicographic voxels is deterministic across all
        # affinity variants. Retain seeds first, without exceeding the cap.
        cap = self.config["max_anchors"]
        if n > cap:
            seed_ids, other_ids = np.flatnonzero(self._seeds), np.flatnonzero(~self._seeds)
            def pick(values, count):
                return values[np.linspace(0, len(values) - 1, count, dtype=int)] if count else values[:0]
            if len(seed_ids) >= cap:
                selected = pick(seed_ids, cap)
            else:
                selected = np.sort(np.concatenate((seed_ids, pick(other_ids, cap - len(seed_ids)))))
            self._points = self._points[selected]
            self._features = self._features[selected]
            self._seeds = self._seeds[selected]
        self._tree = cKDTree(self._points)
        count = len(self._points)
        k = min(self.config["knn"] + 1, count)
        distances, neighbors = self._tree.query(
            self._points, k=k, distance_upper_bound=self.config["neighbor_radius"], workers=1)
        if k == 1:
            distances, neighbors = distances[:, None], neighbors[:, None]
        valid = np.isfinite(distances) & (neighbors != np.arange(count)[:, None])
        safe = np.minimum(neighbors, count - 1)
        geometry = np.exp(-0.5 * (distances / (self.config["neighbor_radius"] / 2)) ** 2) * valid
        weights = geometry * self._affinity(self._features[:, None], self._features[safe])
        denominator = np.maximum(geometry.sum(axis=1), 1e-12)
        strength = self._seeds.astype(np.float64)
        for _ in range(self.config["steps"]):
            change = (weights * (strength[safe] - strength[:, None])).sum(axis=1) / denominator
            strength = np.clip(strength + self.config["diffusion_rate"] * change, 0, 1)
            strength[self._seeds] = 1.
        self._strength = strength
        self._summary = {
            "affinity": self.config["affinity"],
            "keyframe_ids": sorted(self._observations),
            "sample_count": len(points),
            "voxel_count_before_cap": n,
            "anchor_count": count,
            "seed_anchor_count": int(self._seeds.sum()),
            "edited_anchor_count": int((strength > 0.01).sum()),
            "steps": self.config["steps"],
            "frozen": True,
        }
        self._observations.clear()
        self._finalized = True
        return dict(self._summary)

    @property
    def summary(self):
        return None if self._summary is None else dict(self._summary)

    @torch.no_grad()
    def apply(self, gbuffer, depth, opacity, static_mask, camera_to_world,
              intrinsics, raw_rgb=None):
        """Return edited G-buffer and HW1 ``edit_weight``/``edit_support`` maps."""
        if not self._finalized:
            raise RuntimeError("Call finalize before applying surface edits.")
        h, w, z, _, features, valid, camera, inverse_k = self._inputs(
            gbuffer, depth, opacity, static_mask, camera_to_world, intrinsics, raw_rgb)
        strength = np.zeros(h * w, dtype=np.float64)
        support = np.zeros(h * w, dtype=bool)
        indices = np.flatnonzero(valid)
        chunk = self.config["query_chunk_size"]
        count = len(self._points)
        k = min(self.config["knn"], count)
        for start in range(0, len(indices), chunk):
            selected = indices[start:start + chunk]
            points = self._world_points(selected, w, z, camera, inverse_k)
            finite = np.isfinite(points).all(axis=-1)
            selected, points = selected[finite], points[finite]
            if not len(selected):
                continue
            distances, neighbors = self._tree.query(
                points, k=k, distance_upper_bound=self.config["neighbor_radius"], workers=1)
            if k == 1:
                distances, neighbors = distances[:, None], neighbors[:, None]
            near = np.isfinite(distances)
            safe = np.minimum(neighbors, count - 1)
            # Inverse distance makes an exact anchor observation reproduce its
            # edit and retains continuous interpolation between nearby anchors.
            geometry = near / np.maximum(distances, 1e-6) ** 2
            affinity = self._affinity(features.reshape(-1, 11)[selected, None], self._features[safe])
            weights = geometry * affinity
            numerator = (weights * self._strength[safe]).sum(axis=-1)
            strength[selected] = numerator / np.maximum(geometry.sum(axis=-1), 1e-12)
            support[selected] = weights.sum(axis=-1) > 1e-12
        weight = torch.as_tensor(strength.reshape(h, w, 1), device=depth.device, dtype=depth.dtype)
        output = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                  for key, value in gbuffer.items()}
        delta_a = depth.new_tensor(self.config["delta_albedo_logit"])
        for key, delta in (("albedo", delta_a), ("roughness", self.config["delta_roughness_logit"])):
            original = output[key]
            # Avoid float16 rounding 1 - eps to exactly 1 during logit.
            work = original.double() if original.dtype == torch.float64 else original.float()
            edited = torch.sigmoid(torch.logit(work.clamp(1e-6, 1 - 1e-6)) + weight * delta)
            output[key] = torch.where(weight > 0, edited.to(original.dtype), original)
        info = {
            "edit_weight": weight,
            "edit_support": torch.as_tensor(support.reshape(h, w, 1), device=depth.device),
        }
        return output, info
