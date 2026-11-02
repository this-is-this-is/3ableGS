"""GT-only temporal correspondences shared by every rendering variant.

This evaluator receives display RGB in [0, 1] and dataset static masks. It never
uses a method's depth, material buffers, or history-acceptance decisions. Optical
flow on real images is an estimate, so ``flow_rejected_l1`` measures current-frame
error in rejected correspondence regions; it is not an occlusion metric.

``TemporalEvaluator.evaluate(reference, predictions, static_mask, frame_id)``
returns flat JSON-safe statistics. Pass all methods together, or reuse the same
cache directory and metadata across separate sequence runs. ``reset()`` clears
sequence history. ``summarize_metric_records(records)`` aggregates sums and
counts over pixels/channels, rather than averaging per-frame means.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np


_SCHEMA = 1
_FLOW_CYCLE_TOLERANCE = 1.5
_EDGE_THRESHOLD = 0.02
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_ERODE_KERNEL = np.ones((3, 3), np.uint8)


def _rgb(value, label):
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{label} must have shape H x W x 3")
    if min(value.shape[:2]) < 3:
        raise ValueError(f"{label} must be at least 3 x 3")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{label} must be floating RGB in [0, 1]")
    if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f"{label} must be finite RGB in [0, 1]")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _erode(mask):
    return cv2.erode(
        mask.astype(np.uint8), _ERODE_KERNEL,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)


def _gradient(image):
    luminance = image @ _LUMA
    # Divide the standard 3x3 Sobel operator by 8 for unit-range inputs.
    gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    return np.stack([gx, gy], axis=-1)


def _mean_stats(result, name, values):
    values = np.asarray(values)
    count = int(values.size)
    total = float(values.sum(dtype=np.float64))
    result[name] = total / count if count else None
    result[name + "_sum"] = total
    result[name + "_count"] = count


def _remap(image, map_x, map_y):
    return cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


class TemporalEvaluator:
    """Sequential, single-camera evaluator with optional content-addressed cache.

    ``metadata`` should identify the dataset, camera, and resolution settings.
    Input RGB/mask bytes, flow configuration, OpenCV version, and frame IDs are
    also included in each cache key, so old or incompatible references cannot
    silently reuse cached correspondences. Nonconsecutive IDs reset history.
    """

    def __init__(self, cache_dir=None, metadata=None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.metadata = json.loads(json.dumps(metadata or {}, sort_keys=True, allow_nan=False))
        self.flow_backend = "dis_medium" if hasattr(cv2, "DISOpticalFlow_create") else "farneback"
        self.config = {
            "schema": _SCHEMA,
            "flow": self.flow_backend,
            "opencv": cv2.__version__,
            "cycle_tolerance_px": _FLOW_CYCLE_TOLERANCE,
            "static_erosion_px": 1,
            "previous_mask_bilinear_min": 0.999,
            "edge_threshold": _EDGE_THRESHOLD,
            "edge_sobel_scale": 0.125,
            "luma": _LUMA.tolist(),
            "remap": "opencv_linear",
            "farneback": [0.5, 4, 21, 5, 7, 1.5, 0],
        }
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        self._previous = None

    def _flow(self, source_gray, target_gray):
        """Return source -> target flow; kept separate for known-flow tests."""
        if self.flow_backend == "dis_medium" and min(source_gray.shape) >= 12:
            estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            return estimator.calc(source_gray, target_gray, None)
        # DIS requires a minimum image size; tiny evaluation/test images use the
        # same explicitly configured Farneback fallback on every method.
        return cv2.calcOpticalFlowFarneback(
            source_gray, target_gray, None, 0.5, 4, 21, 5, 7, 1.5, 0,
        )

    def _cache_key(self, reference, mask, frame_id):
        previous = self._previous
        digest = hashlib.sha256()
        digest.update(json.dumps({
            "config": self.config, "metadata": self.metadata,
            "frames": [previous["frame_id"], frame_id],
            "shape": list(reference.shape),
        }, sort_keys=True, allow_nan=False).encode("utf-8"))
        for value in (previous["reference"], reference, previous["mask"], mask):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def _correspondence(self, reference, mask, frame_id):
        key = self._cache_key(reference, mask, frame_id)
        path = self.cache_dir / (key + ".npz") if self.cache_dir is not None else None
        height, width = mask.shape
        if path is not None and path.exists():
            try:
                with np.load(path, allow_pickle=False) as cached:
                    backward = cached["backward"]
                    valid = cached["valid"]
                    matches = (
                        str(cached["key"].item()) == key
                        and backward.shape == (height, width, 2)
                        and backward.dtype == np.float32
                        and np.isfinite(backward).all()
                        and valid.shape == (height, width)
                        and valid.dtype == np.bool_
                    )
                    if matches:
                        return backward, valid, True
            except (ValueError, OSError, KeyError):
                # An incomplete/corrupt entry is safe to recompute.
                pass
        previous = self._previous
        current_gray = np.rint((reference @ _LUMA) * 255).astype(np.uint8)
        previous_gray = np.rint((previous["reference"] @ _LUMA) * 255).astype(np.uint8)
        backward = np.asarray(self._flow(current_gray, previous_gray), dtype=np.float32)
        forward = np.asarray(self._flow(previous_gray, current_gray), dtype=np.float32)
        if (backward.shape != (height, width, 2) or forward.shape != backward.shape
                or not np.isfinite(backward).all() or not np.isfinite(forward).all()):
            raise ValueError("Optical flow returned an invalid or non-finite array")
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        map_x, map_y = xx + backward[..., 0], yy + backward[..., 1]
        bounds = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
        previous_valid = _remap(_erode(previous["mask"]).astype(np.float32), map_x, map_y) >= 0.999
        reverse = _remap(forward, map_x, map_y)
        cycle = np.linalg.norm(backward + reverse, axis=-1)
        valid = _erode(mask) & bounds & previous_valid & (cycle <= _FLOW_CYCLE_TOLERANCE)
        if path is not None:
            # Readers only observe complete files, including concurrent suite runs.
            descriptor, temporary = tempfile.mkstemp(prefix=key + ".", suffix=".tmp", dir=self.cache_dir)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.savez_compressed(handle, key=np.array(key), backward=backward, valid=valid)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return backward, valid, False

    def evaluate(self, reference, predictions, static_mask, frame_id):
        """Measure one frame and advance history; no method-specific masks.

        All prediction names must remain identical within a consecutive segment.
        A first frame, gap, or resolution change has null temporal metrics.
        Edge metrics are measured on every frame. Counts are RGB channel counts
        for RGB L1 metrics and gradient component counts for edge-gradient L1.
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
        mask = np.asarray(static_mask)
        if mask.shape != reference.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("static_mask must be a bool H x W array matching reference")
        mask = mask.copy()
        if isinstance(frame_id, (bool, np.bool_)) or not isinstance(frame_id, (int, np.integer)):
            raise ValueError("frame_id must be an integer")
        frame_id = int(frame_id)
        previous = self._previous
        has_pair = (
            previous is not None and frame_id == previous["frame_id"] + 1
            and previous["reference"].shape == reference.shape
        )
        if has_pair and set(predictions) != set(previous["predictions"]):
            raise ValueError("Prediction names must stay the same within a consecutive sequence")

        interior = _erode(mask)
        gt_gradient = _gradient(reference)
        gt_strength = np.linalg.norm(gt_gradient, axis=-1)
        edges = interior & (gt_strength > _EDGE_THRESHOLD)
        edge_pixels = int(edges.sum())
        valid = np.zeros(mask.shape, dtype=bool)
        rejected = np.zeros(mask.shape, dtype=bool)
        cache_hit = None
        if has_pair:
            backward, valid, cache_hit = self._correspondence(reference, mask, frame_id)
            yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]].astype(np.float32)
            map_x, map_y = xx + backward[..., 0], yy + backward[..., 1]
            reference_delta = reference - _remap(previous["reference"], map_x, map_y)
            rejected = interior & ~valid

        valid_pixels = int(valid.sum())
        static_pixels = int(interior.sum()) if has_pair else 0
        result = {
            "frame_id": frame_id,
            "temporal_has_pair": bool(has_pair),
            "temporal_cache_hit": cache_hit,
            "temporal_valid_pixels": valid_pixels,
            "temporal_static_pixels": static_pixels,
            "temporal_valid_coverage": valid_pixels / static_pixels if static_pixels else None,
            "flow_rejected_pixels": int(rejected.sum()),
            "edge_pixels": edge_pixels,
        }
        for name, prediction in predictions.items():
            if has_pair:
                prediction_delta = prediction - _remap(previous["predictions"][name], map_x, map_y)
                residual = np.abs(prediction_delta - reference_delta)[valid]
                raw_warp = np.abs(prediction_delta)[valid]
            else:
                residual = raw_warp = np.empty((0, 3), dtype=np.float32)
            _mean_stats(result, name + "_temporal_residual_l1", residual)
            _mean_stats(result, name + "_raw_warp_l1", raw_warp)
            _mean_stats(result, name + "_flow_rejected_l1", np.abs(prediction - reference)[rejected])
            predicted_gradient = _gradient(prediction)
            _mean_stats(result, name + "_edge_gradient_l1",
                        np.abs(predicted_gradient - gt_gradient)[edges])
            numerator = float(np.linalg.norm(predicted_gradient, axis=-1)[edges].sum(dtype=np.float64))
            denominator = float(gt_strength[edges].sum(dtype=np.float64))
            result[name + "_edge_strength_ratio"] = numerator / denominator if denominator else None
            result[name + "_edge_strength_ratio_numerator"] = numerator
            result[name + "_edge_strength_ratio_denominator"] = denominator
        self._previous = dict(reference=reference, predictions=predictions, mask=mask, frame_id=frame_id)
        return result


def summarize_metric_records(records):
    """Return global means/ratios and their summed sufficient statistics.

    Unrelated fields in renderer records are ignored. An empty record list
    returns zero counts and null coverage; zero-support method metrics are null.
    """
    records = list(records)
    result = {
        "temporal_pairs": sum(bool(record.get("temporal_has_pair")) for record in records),
    }
    for key in ("temporal_valid_pixels", "temporal_static_pixels", "flow_rejected_pixels", "edge_pixels"):
        result[key] = sum(int(record.get(key, 0)) for record in records)
    denominator = result["temporal_static_pixels"]
    result["temporal_valid_coverage"] = (
        result["temporal_valid_pixels"] / denominator if denominator else None
    )
    keys = set().union(*(record.keys() for record in records)) if records else set()
    metric_suffixes = (
        "_temporal_residual_l1_sum", "_raw_warp_l1_sum",
        "_flow_rejected_l1_sum", "_edge_gradient_l1_sum",
    )
    for key in sorted(keys):
        if key.endswith(metric_suffixes):
            base = key[:-4]
            count_key = base + "_count"
            total = sum(float(record.get(key, 0)) for record in records)
            count = sum(int(record.get(count_key, 0)) for record in records)
            result[key] = total
            result[count_key] = count
            result[base] = total / count if count else None
        elif key.endswith("_edge_strength_ratio_numerator"):
            base = key[:-len("_numerator")]
            numerator = sum(float(record.get(key, 0)) for record in records)
            denominator = sum(float(record.get(base + "_denominator", 0)) for record in records)
            result[key] = numerator
            result[base + "_denominator"] = denominator
            result[base] = numerator / denominator if denominator else None
    return result
