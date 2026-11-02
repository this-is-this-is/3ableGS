"""Reference-mask diagnostics supplementing, never replacing, native RGB metrics.

Connected components below are image regions, not tracked actors. Every support,
crop and temporal correspondence depends on the reference data alone. Model
opacity, instance predictions and teacher G-buffers are deliberately absent.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity


REGION_KEYS = {"dynamic": "dynamic_masks", "human": "human_masks", "vehicle": "vehicle_masks"}
_SSIM_WINDOW = 7
_MIN_COMPONENT_PIXELS = 16
_CROP_PADDING = 8
_MIN_LPIPS_SIDE = 64


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def reference_mask(images, key, shape):
    """Reference class mask at the evaluated resolution, excluding sky/ego."""
    if key not in images or images[key] is None:
        raise ValueError(f"Dynamic evaluation requires reference {key}")

    def mask(value, name):
        array = _numpy(value)
        if array.shape == (*shape, 1):
            array = array[..., 0]
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Reference {name} must be finite with shape {shape}")
        return array > .5

    result = mask(images[key], key)
    for excluded in ("sky_masks", "egocar_masks"):
        if images.get(excluded) is not None:
            result &= ~mask(images[excluded], excluded)
    return result


def _erode(mask, radius):
    return cv2.erode(mask.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def component_crops(mask):
    """Deterministic padded boxes, shared across prediction branches."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    height, width = mask.shape
    crops = []
    for x, y, w, h, area in stats[1:]:
        if int(area) >= _MIN_COMPONENT_PIXELS:
            crops.append({"box_xyxy": [max(0, int(x) - _CROP_PADDING), max(0, int(y) - _CROP_PADDING),
                                        min(width, int(x + w) + _CROP_PADDING),
                                        min(height, int(y + h) + _CROP_PADDING)],
                          "mask_pixels": int(area)})
    return crops, count - 1


def _psnr(squared_error_sum, rgb_values):
    # Finite, documented ceiling lets exact matches remain JSON-safe.
    return -10 * math.log10(max(squared_error_sum / rgb_values, 1e-12)) if rgb_values else None


def _crop_lpips(prediction, target, crop, metric):
    x0, y0, x1, y1 = crop["box_xyxy"]
    pred = prediction[y0:y1, x0:x1].permute(2, 0, 1)[None]
    truth = target[y0:y1, x0:x1].permute(2, 0, 1)[None]
    height, width = pred.shape[-2:]
    if min(height, width) < _MIN_LPIPS_SIDE:
        scale = _MIN_LPIPS_SIDE / min(height, width)
        size = (max(_MIN_LPIPS_SIDE, round(height * scale)),
                max(_MIN_LPIPS_SIDE, round(width * scale)))
        pred = torch.nn.functional.interpolate(pred, size=size, mode="bilinear", align_corners=False)
        truth = torch.nn.functional.interpolate(truth, size=size, mode="bilinear", align_corners=False)
    metric.reset()
    try:
        value = float(metric(pred, truth))
    finally:
        metric.reset()
    if not math.isfinite(value):
        raise ValueError("Non-finite dynamic crop LPIPS")
    return value


def evaluate_dynamic_frame(predictions, target, images, lpips):
    """Return per-region sums/counts for two or more rendered RGB branches."""
    target = target.detach()
    if target.ndim != 3 or target.shape[-1] != 3 or min(target.shape[:2]) < _SSIM_WINDOW:
        raise ValueError("Dynamic RGB evaluation requires H x W x 3, H/W >= 7")
    truth = _numpy(target)
    if not np.isfinite(truth).all() or (truth < 0).any() or (truth > 1).any():
        raise ValueError("Reference RGB must be finite in [0, 1]")
    masks = {region: reference_mask(images, key, truth.shape[:2]) for region, key in REGION_KEYS.items()}
    result = {}
    for region, mask in masks.items():
        crops, component_count = component_crops(mask)
        support = _erode(mask, _SSIM_WINDOW // 2)
        # Inside the reference dynamic boundary; fixed support for all methods.
        boundary = mask & ~_erode(mask, 2)
        result[region] = {"pixels": int(mask.sum()), "image_pixels": int(mask.size),
                          "ssim_valid_pixels": int(support.sum()),
                          "boundary_pixels": int(boundary.sum()),
                          "component_count": component_count, "eligible_crop_count": len(crops),
                          "crops": crops, "branches": {}}
    if not predictions:
        raise ValueError("At least one RGB prediction is required")
    for name, prediction in predictions.items():
        if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError("Dynamic metric predictions must be finite and match the reference")
        prediction = prediction.detach().clamp(0, 1)
        pred = _numpy(prediction)
        delta = pred.astype(np.float64) - truth.astype(np.float64)
        _, ssim_map = structural_similarity(pred, truth, data_range=1., channel_axis=-1,
                                            win_size=_SSIM_WINDOW, full=True)
        for region, mask in masks.items():
            row = result[region]
            support = _erode(mask, _SSIM_WINDOW // 2)
            boundary = mask & ~_erode(mask, 2)
            squared = float(np.square(delta[mask]).sum())
            ssim_sum = float(ssim_map[support].sum(dtype=np.float64))
            boundary_squared = float(np.square(delta[boundary]).sum())
            crop_values = [_crop_lpips(prediction, target.to(prediction), crop, lpips) for crop in row["crops"]]
            rgb_count, ssim_count = row["pixels"] * 3, row["ssim_valid_pixels"] * 3
            row["branches"][name] = {
                "squared_error_sum": squared, "rgb_value_count": rgb_count,
                "psnr": _psnr(squared, rgb_count),
                "ssim_sum": ssim_sum, "ssim_value_count": ssim_count,
                "ssim_valid_window": ssim_sum / ssim_count if ssim_count else None,
                "boundary_squared_error_sum": boundary_squared,
                "boundary_rgb_value_count": row["boundary_pixels"] * 3,
                "boundary_psnr": _psnr(boundary_squared, row["boundary_pixels"] * 3),
                "component_crop_lpips": crop_values,
            }
    return result


def summarize_dynamic_frames(frames):
    """Pixel-weighted quality and component-averaged perceptual diagnostics."""
    result = {}
    for region in REGION_KEYS:
        rows = [frame[region] for frame in frames]
        if not rows:
            continue
        counts = {key: sum(row[key] for row in rows) for key in
                  ("pixels", "image_pixels", "ssim_valid_pixels", "boundary_pixels",
                   "component_count", "eligible_crop_count")}
        summary = {**counts, "frames": len(rows), "nonempty_frames": sum(row["pixels"] > 0 for row in rows),
                   "pixel_coverage": counts["pixels"] / counts["image_pixels"], "branches": {}}
        for name in rows[0]["branches"]:
            branch = [row["branches"][name] for row in rows]
            totals = {key: sum(item[key] for item in branch) for key in
                      ("squared_error_sum", "rgb_value_count", "ssim_sum", "ssim_value_count",
                       "boundary_squared_error_sum", "boundary_rgb_value_count")}
            values = [value for item in branch for value in item["component_crop_lpips"]]
            frame_psnr = [item["psnr"] for item in branch if item["psnr"] is not None]
            summary["branches"][name] = {
                **totals, "pixel_weighted_psnr": _psnr(totals["squared_error_sum"], totals["rgb_value_count"]),
                "nonempty_frame_mean_psnr": float(np.mean(frame_psnr)) if frame_psnr else None,
                "ssim_valid_window": totals["ssim_sum"] / totals["ssim_value_count"] if totals["ssim_value_count"] else None,
                "boundary_psnr": _psnr(totals["boundary_squared_error_sum"], totals["boundary_rgb_value_count"]),
                "component_mean_crop_lpips": float(np.mean(values)) if values else None,
                "crop_count": len(values),
            }
        result[region] = summary
    return result


def dynamic_metric_protocol():
    return {
        "support": "reference dynamic/human/vehicle masks > 0.5; reference sky/ego excluded if available",
        "psnr": "pixel-weighted RGB MSE; exact-zero MSE floored at 1e-12 (120 dB ceiling)",
        "ssim": "7x7 uniform skimage SSIM map; only windows entirely inside reference region and image",
        "boundary": "reference region minus its 2-pixel erosion, inside boundary only",
        "crop_lpips": "unmasked reference connected-component bounding boxes, 8-pixel padding; components >=16 pixels; "
                      "bilinear aspect-preserving upsample to minimum side 64 when needed; independent LPIPS reset",
        "aggregation": "pixel-weighted PSNR/SSIM, nonempty-frame mean PSNR, connected-component mean crop LPIPS",
        "limitation": "Connected components may merge touching actors or split one actor; these are not track/instance averages.",
        "model_dependent_support": False, "teacher_inputs": False,
    }


def _dynamic_temporal_names(row):
    return {key.replace("temporal_static_pixels", "temporal_dynamic_pixels"): value for key, value in row.items()}


class DynamicTemporalDiagnostic:
    """Reference-flow temporal reconstruction diagnostic, including train views."""
    def __init__(self, train_frames, test_frames, metadata=None):
        self.train_frames = set(map(int, train_frames))
        self.test_frames = set(map(int, test_frames))
        if self.train_frames & self.test_frames:
            raise ValueError("Train and test frame IDs must be disjoint")
        self.metadata = metadata or {}
        self.evaluators = {}
        self.records = []

    def update(self, reference, predictions, images, frame_id, camera_id):
        from utils.temporal_metrics import TemporalEvaluator
        camera_id, frame_id = int(camera_id), int(frame_id)
        if frame_id not in self.train_frames | self.test_frames:
            raise ValueError("Temporal frame is outside the saved train/test split")
        reference = np.clip(_numpy(reference), 0, 1).astype(np.float32)
        mask = reference_mask(images, "dynamic_masks", reference.shape[:2])
        if camera_id not in self.evaluators:
            self.evaluators[camera_id] = TemporalEvaluator(metadata={**self.metadata, "camera_id": camera_id,
                                                                   "support": "reference_dynamic"})
        row = self.evaluators[camera_id].evaluate(reference,
                {name: np.clip(_numpy(value), 0, 1).astype(np.float32) for name, value in predictions.items()},
                mask, frame_id)
        row["camera_id"] = camera_id
        current = "train" if frame_id in self.train_frames else "test"
        previous = "train" if frame_id - 1 in self.train_frames else "test"
        row["pair_split"] = f"{previous}_to_{current}" if row["temporal_has_pair"] else None
        self.records.append(row)
        return _dynamic_temporal_names(row)

    def summary(self):
        from utils.temporal_metrics import summarize_metric_records
        subsets = {"all_adjacent": self.records}
        for name in ("train_to_train", "train_to_test", "test_to_train", "test_to_test"):
            subsets[name] = [row for row in self.records if row["pair_split"] == name]
        return {
            "kind": "train_inclusive_temporal_reconstruction_diagnostic",
            "heldout_temporal_benchmark": False,
            "protocol": {
                "correspondence": "reference RGB optical flow, forward/backward cycle <=1.5 px; same flow/masks for all branches",
                "support": "reference dynamic masks, sky/ego excluded, 1-pixel erosion, warped previous mask required",
                "residual": "abs((prediction_t - warped_prediction_previous) - (reference_t - warped_reference_previous))",
                "flow_rejected_l1": "current-frame RGB error on dynamic interior rejected by correspondence",
                "limitations": "Estimated flow is not ground-truth correspondence; sequence includes training frames. "
                               "No model opacity, depth or confidence is used to choose evaluation pixels.",
            },
            "subsets": {name: _dynamic_temporal_names(summarize_metric_records(rows)) for name, rows in subsets.items()},
            "frames": [_dynamic_temporal_names(row) for row in self.records],
        }
