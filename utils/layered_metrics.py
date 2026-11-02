"""Reference-only boundary and temporal metrics for the layered G-buffer study.

Dynamic masks define *evaluation regions*, never flow validity. Object boundary
bands include both sides and are not eroded out of temporal evaluation. Their
radius is measured with a square (Chebyshev) neighbourhood. Interior regions
exclude the smallest configured boundary band. Dataset masks only identify the
union of moving objects: touching actors require an instance-aware mask for a
future evaluator and are not claimed to be separate boundaries here.

All methods share RGB-derived flow, in-bounds and forward/backward cycle checks.
Only ``evaluation_mask`` removes invalid/ego pixels. ``flow_rejected_l1`` is
current RGB error where flow was rejected, NOT an occlusion metric. Optional
depth edges require two adjacent measured depths; missing sparse depths are
never treated as foreground/background edges. The names explicitly retain
``sparse_reference_depth`` because these edges need not cover all scene edges.

Flat records contain sums/counts for pixel-pooled summaries. RGB counts include
three channels; gradient counts include two Sobel components. PSNR is calculated
from pooled MSE and capped at 120 dB for exact matches to remain JSON-safe.
"""

import os
import tempfile

import cv2
import numpy as np

from utils.temporal_metrics import (
    TemporalEvaluator, _FLOW_CYCLE_TOLERANCE, _LUMA, _gradient, _mean_stats,
    _remap, _rgb,
)


_PSNR_CEILING_DB = 120.0


def _mask(value, shape, label):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"{label} must be a bool H x W array matching reference")
    return value.copy()


def _boundary_band(mask, radius):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    source = mask.astype(np.uint8)
    outer = cv2.dilate(source, kernel, borderType=cv2.BORDER_REPLICATE)
    inner = cv2.erode(source, kernel, borderType=cv2.BORDER_REPLICATE)
    return outer != inner


def _depth_edges(depth):
    valid = np.isfinite(depth) & (depth > 0)
    result = np.zeros(depth.shape, bool)
    # Mark both measured samples on an actual adjacent depth discontinuity.
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = first.copy()
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(first), tuple(second)
        jointly_valid = valid[a] & valid[b]
        difference = np.zeros(depth[a].shape, np.float64)
        np.subtract(depth[a], depth[b], out=difference, where=jointly_valid)
        scale = np.minimum(np.where(valid[a], depth[a], 1), np.where(valid[b], depth[b], 1))
        edge = jointly_valid & (np.abs(difference) > 0.05 * scale)
        result[a] |= edge
        result[b] |= edge
    return result


def _psnr(mse, count):
    if not count:
        return None
    return float(-10 * np.log10(max(float(mse), 10 ** (-_PSNR_CEILING_DB / 10))))


class LayeredBoundaryEvaluator(TemporalEvaluator):
    """Sequential single-camera metrics with a separate content-addressed cache.

    The inherited flow backend/cache-key implementation is shared with the older
    evaluator, but this class deliberately does not inherit its static erosion
    or correspondence mask construction. Nonconsecutive IDs reset history.
    """

    def __init__(self, cache_dir=None, metadata=None, boundary_radii=(3, 5, 9)):
        boundary_radii = tuple(boundary_radii)
        if (not boundary_radii or any(isinstance(r, (bool, np.bool_))
                or not isinstance(r, (int, np.integer)) or r < 1 for r in boundary_radii)):
            raise ValueError("boundary_radii must contain positive integers")
        self.boundary_radii = tuple(sorted(set(int(r) for r in boundary_radii)))
        super().__init__(cache_dir, metadata)
        self.config.update({
            "schema": "layered_boundary_v1",
            "static_erosion_px": 0,
            "mask_semantics": "evaluation_valid_only_no_dynamic_filter",
            "boundary_radii": list(self.boundary_radii),
            "depth_relative_threshold": 0.05,
            "psnr_ceiling_db": _PSNR_CEILING_DB,
        })

    def _correspondence(self, reference, mask, frame_id):
        key = self._cache_key(reference, mask, frame_id)
        path = self.cache_dir / (key + ".npz") if self.cache_dir is not None else None
        height, width = mask.shape
        if path is not None and path.exists():
            try:
                with np.load(path, allow_pickle=False) as cached:
                    backward, valid = cached["backward"], cached["valid"]
                    if (str(cached["key"].item()) == key
                            and backward.shape == (height, width, 2)
                            and backward.dtype == np.float32 and np.isfinite(backward).all()
                            and valid.shape == (height, width) and valid.dtype == np.bool_):
                        return backward, valid, True
            except (ValueError, OSError, KeyError):
                pass
        previous = self._previous
        current_gray = np.rint((reference @ _LUMA) * 255).astype(np.uint8)
        previous_gray = np.rint((previous["reference"] @ _LUMA) * 255).astype(np.uint8)
        backward = np.asarray(self._flow(current_gray, previous_gray), np.float32)
        forward = np.asarray(self._flow(previous_gray, current_gray), np.float32)
        if (backward.shape != (height, width, 2) or forward.shape != backward.shape
                or not np.isfinite(backward).all() or not np.isfinite(forward).all()):
            raise ValueError("Optical flow returned an invalid or non-finite array")
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        map_x, map_y = xx + backward[..., 0], yy + backward[..., 1]
        bounds = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
        previous_valid = _remap(previous["mask"].astype(np.float32), map_x, map_y) >= 0.999
        reverse = _remap(forward, map_x, map_y)
        cycle = np.linalg.norm(backward + reverse, axis=-1)
        valid = mask & bounds & previous_valid & (cycle <= _FLOW_CYCLE_TOLERANCE)
        if path is not None:
            descriptor, temporary = tempfile.mkstemp(prefix=key + ".", suffix=".tmp", dir=self.cache_dir)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.savez_compressed(handle, key=np.array(key), backward=backward, valid=valid)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return backward, valid, False

    def evaluate(self, reference, predictions, dynamic_mask, frame_id,
                 evaluation_mask=None, depth=None):
        """Evaluate all methods on common reference masks and advance history.

        ``depth`` must be reference/independently measured depth, never a method's
        rendered prediction. Zeros and nonfinite depth entries mean unavailable.
        Prediction names must remain fixed within each consecutive segment.
        """
        reference = _rgb(reference, "reference")
        if not isinstance(predictions, dict) or not predictions:
            raise ValueError("predictions must be a nonempty dict of method names to RGB")
        for name in predictions:
            if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
                raise ValueError("prediction names must contain only letters, numbers, or underscores")
        predictions = {name: _rgb(value, name) for name, value in predictions.items()}
        if any(value.shape != reference.shape for value in predictions.values()):
            raise ValueError("Every prediction must have the reference shape")
        shape = reference.shape[:2]
        dynamic_mask = _mask(dynamic_mask, shape, "dynamic_mask")
        mask = (np.ones(shape, bool) if evaluation_mask is None else
                _mask(evaluation_mask, shape, "evaluation_mask"))
        if isinstance(frame_id, (bool, np.bool_)) or not isinstance(frame_id, (int, np.integer)):
            raise ValueError("frame_id must be an integer")
        frame_id = int(frame_id)
        previous = self._previous
        has_pair = (previous is not None and frame_id == previous["frame_id"] + 1
                    and previous["reference"].shape == reference.shape)
        if has_pair and set(predictions) != set(previous["predictions"]):
            raise ValueError("Prediction names must stay the same within a consecutive sequence")

        bands = {radius: _boundary_band(dynamic_mask, radius) for radius in self.boundary_radii}
        inner_boundary = bands[self.boundary_radii[0]]
        regions = {
            "global": mask,
            "dynamic_interior": mask & dynamic_mask & ~inner_boundary,
            "static_interior": mask & ~dynamic_mask & ~inner_boundary,
        }
        regions.update({f"object_boundary_r{radius}": mask & band for radius, band in bands.items()})
        if depth is not None:
            depth = np.asarray(depth)
            if depth.shape != shape or not np.issubdtype(depth.dtype, np.number):
                raise ValueError("depth must be a numeric H x W reference array")
            depth = depth.astype(np.float64)
            depth = np.where(mask, depth, 0)
            edges = _depth_edges(depth)
            for radius in self.boundary_radii:
                # Edge seeds already mark both adjacent samples, so radius 1
                # is the undilated two-sided edge, consistent with mask bands.
                kernel = np.ones((2 * radius - 1, 2 * radius - 1), np.uint8)
                band = cv2.dilate(edges.astype(np.uint8), kernel).astype(bool)
                regions[f"sparse_reference_depth_boundary_r{radius}"] = mask & band

        valid = np.zeros(shape, bool)
        cache_hit = None
        if has_pair:
            backward, valid, cache_hit = self._correspondence(reference, mask, frame_id)
            yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
            map_x, map_y = xx + backward[..., 0], yy + backward[..., 1]
            reference_delta = reference - _remap(previous["reference"], map_x, map_y)
        rejected = mask & ~valid if has_pair else np.zeros(shape, bool)
        result = {
            "frame_id": frame_id,
            "temporal_has_pair": bool(has_pair),
            "temporal_cache_hit": cache_hit,
            "psnr_ceiling_db": _PSNR_CEILING_DB,
        }
        for region, region_mask in regions.items():
            pixels = int(region_mask.sum())
            support = pixels if has_pair else 0
            accepted = int((region_mask & valid).sum())
            result[region + "_pixels"] = pixels
            result[region + "_temporal_reference_pixels"] = support
            result[region + "_temporal_valid_pixels"] = accepted
            result[region + "_temporal_valid_coverage"] = accepted / support if support else None
            result[region + "_flow_rejected_pixels"] = int((region_mask & rejected).sum())

        reference_gradient = _gradient(reference)
        for name, prediction in predictions.items():
            absolute_error = np.abs(prediction - reference)
            squared_error = np.square(prediction - reference)
            gradient_error = np.abs(_gradient(prediction) - reference_gradient)
            if has_pair:
                prediction_delta = prediction - _remap(previous["predictions"][name], map_x, map_y)
                temporal_error = np.abs(prediction_delta - reference_delta)
                raw_warp_error = np.abs(prediction_delta)
            for region, region_mask in regions.items():
                prefix = name + "_" + region
                _mean_stats(result, prefix + "_rgb_mae", absolute_error[region_mask])
                _mean_stats(result, prefix + "_rgb_mse", squared_error[region_mask])
                result[prefix + "_rgb_psnr"] = _psnr(
                    result[prefix + "_rgb_mse"], result[prefix + "_rgb_mse_count"])
                _mean_stats(result, prefix + "_gradient_l1", gradient_error[region_mask])
                temporal_mask = region_mask & valid
                _mean_stats(result, prefix + "_temporal_residual_l1",
                            temporal_error[temporal_mask] if has_pair else np.empty(0))
                _mean_stats(result, prefix + "_raw_warp_l1",
                            raw_warp_error[temporal_mask] if has_pair else np.empty(0))
                _mean_stats(result, prefix + "_flow_rejected_l1", absolute_error[region_mask & rejected])
        self._previous = dict(reference=reference, predictions=predictions, mask=mask, frame_id=frame_id)
        return result


def summarize_layered_records(records):
    """Pool sufficient statistics over frames; never average per-frame PSNR."""
    records = list(records)
    result = {"frames": len(records),
              "temporal_pairs": sum(bool(r.get("temporal_has_pair")) for r in records),
              "psnr_ceiling_db": _PSNR_CEILING_DB}
    keys = set().union(*(record.keys() for record in records)) if records else set()
    for key in sorted(keys):
        if key.endswith("_pixels"):
            result[key] = sum(int(record.get(key, 0)) for record in records)
        elif key.endswith("_sum") and key[:-4] + "_count" in keys:
            base = key[:-4]
            count_key = base + "_count"
            total = sum(float(record.get(key, 0)) for record in records)
            count = sum(int(record.get(count_key, 0)) for record in records)
            result[key], result[count_key] = total, count
            result[base] = total / count if count else None
            if base.endswith("_rgb_mse"):
                result[base[:-len("_mse")] + "_psnr"] = _psnr(result[base], count)
    for key in sorted(keys):
        if key.endswith("_temporal_valid_coverage"):
            prefix = key[:-len("_temporal_valid_coverage")]
            count = result.get(prefix + "_temporal_reference_pixels", 0)
            result[key] = result.get(prefix + "_temporal_valid_pixels", 0) / count if count else None
    return result
