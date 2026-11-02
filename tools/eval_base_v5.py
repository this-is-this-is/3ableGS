#!/usr/bin/env python3
"""Paired held-out RGB evaluation of a Base v5 checkpoint.

Both outputs share the current learned PBR scene. Their difference measures the
display residual at this checkpoint, not a comparison with the original Base-v4
scene. The saved native image resolution, held-out split and affine policy are
retained. Only trusted local config.yaml / PyTorch checkpoints should be used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Only camera/ray identifiers enter inference; GT RGB, masks, depth and teacher
# material maps remain available separately for evaluation.
RENDER_IMAGE_KEYS = frozenset({
    "origins", "viewdirs", "direction_norm", "pixel_coords",
    "normed_time", "img_idx", "frame_idx",
})
METRIC_KEYS = ("psnr", "ssim", "lpips")
MAP_KEYS = ("rgb_gain", "rgb_log_gain", "rgb_additive", "rgb_residual")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--resume_from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to <checkpoint directory>/rgb_joint_eval; replaces evaluation outputs")
    parser.add_argument("--save-images", action="store_true",
                        help="Save GT/pure/hybrid comparisons and signed float residual maps")
    parser.add_argument("--dynamic-metrics", action="store_true",
                        help="Add reference-mask vehicle/human/dynamic and connected-component crop diagnostics")
    parser.add_argument("--dynamic-temporal", action="store_true",
                        help="Also render the full chronological sequence for train-inclusive dynamic temporal diagnostics")
    return parser


def validate_args(args):
    args.checkpoint = args.checkpoint.expanduser().resolve()
    for path in (args.checkpoint, args.checkpoint.parent / "config.yaml"):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir = (args.output_dir or args.checkpoint.parent / "rgb_joint_eval").expanduser().resolve()
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError("--output-dir must be a directory")
    return args


def read_model_state(checkpoint, device):
    """Map the archive on CPU, discard training state, then move model tensors.

    Moving the complete archive straight to CUDA would materialize Adam moments
    and training caches that evaluation never uses. Native node loaders assign
    some non-buffer tensors directly, so the remaining model tree must still be
    moved recursively rather than relying only on Module.load_state_dict.
    """
    import torch
    from utils.weather_training import model_state

    archive = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    archive.pop("rgb_joint_training_state", None)
    archive.pop("optimizer", None)
    state = model_state(archive, device)
    del archive
    return state


def load_scene(checkpoint):
    """Load a saved native scene without loading any material teacher buffers."""
    import torch
    from omegaconf import OmegaConf
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    saved = OmegaConf.load(checkpoint.parent / "config.yaml")
    if not bool(saved.trainer.get("rgb_joint", {}).get("enabled", False)):
        raise ValueError("Paired RGB joint evaluation requires trainer.rgb_joint.enabled=true")
    if str(saved.trainer.type) not in (
        "models.trainers.MultiTrainer_pbr", "models.trainers.scene_graph_pbr.MultiTrainer_pbr",
    ):
        raise ValueError("This evaluator requires the raw Base-v4 MultiTrainer_pbr")
    if not torch.cuda.is_available():
        raise RuntimeError("Native Gaussian rendering requires CUDA and the project's gsplat environment")
    config = deepcopy(saved)
    # Storage/teacher loading changes do not change split, calibration or resolution.
    config.data.preload_device = "cpu"
    config.data.pixel_source.load_weather = []
    config.data.pixel_source.load_gbuffer = False
    dataset = DrivingDataset(data_cfg=config.data)
    trainer = import_str(config.trainer.type)(
        **deepcopy(config.trainer), dataset=dataset, weathers=["raw"],
        vgg_config=config.get("vgg"), model_config=deepcopy(config.model), device=torch.device("cuda"))
    state = read_model_state(checkpoint, trainer.device)
    trainer.load_state_dict(state, load_only_model=True, strict=True)
    del state
    trainer.set_eval()
    trainer._suppress_eval_layers = True
    return saved, dataset, trainer


def image_metrics(prediction, target, lpips):
    """Native models/video_utils.py definitions, evaluated on float RGB [0,1]."""
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity

    prediction = prediction.detach().clamp(0, 1)
    target = target.detach().to(prediction)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("Metric inputs must have matching [H,W,3] shapes")
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise ValueError("Non-finite RGB metric inputs")
    # Isolate the current image/branch from torchmetrics' accumulated state.
    lpips.reset()
    perceptual = lpips(prediction.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None])
    lpips.reset()
    result = {
        "psnr": float(-10 * torch.log10(torch.nn.functional.mse_loss(prediction, target))),
        "ssim": float(structural_similarity(prediction.cpu().numpy(), target.cpu().numpy(),
                                            data_range=1.0, channel_axis=-1)),
        "lpips": float(perceptual),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(f"Non-finite image metrics: {result}")
    return result


def aggregate_metrics(records, branch):
    import numpy as np
    if not records:
        raise ValueError("Evaluation requires at least one held-out image")
    result = {}
    for key in METRIC_KEYS:
        values = np.asarray([row[branch][key] for row in records], dtype=np.float64)
        result[key] = float(values.mean())
        result[key + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def budget_metadata(config, trainer):
    """The saved completed-camera counter is authoritative, not step + 1."""
    joint = config.trainer.rgb_joint
    count = int(trainer.rgb_joint_camera_updates)
    if count < 0:
        raise ValueError("Completed camera updates cannot be negative")
    return {
        "initialization_mode": "scratch_rgb_joint", "initialization_checkpoint": None,
        "checkpoint_step": int(trainer.step), "completed_camera_updates": count,
        "cumulative_training_iterations": count,
        "configured_num_iters": int(config.trainer.optim.num_iters),
        "warmup_camera_updates": int(joint.get("warmup_camera_updates", 2000)),
        "ramp_camera_updates": int(joint.get("ramp_camera_updates", 1000)),
        "pure_pbr_lambda": float(joint.get("pure_pbr_lambda", .3)),
        "base_model_frozen": False,
        "note": "3ableGS uses camera training iterations with the native Base-v4 objectives, including "
                "projected LiDAR depth supervision when configured. It has no alternating SplatAD LiDAR batch "
                "stage. The saved completed-camera counter determines the budget and residual schedule; the "
                "checkpoint step is recorded separately. Pure PBR is the current jointly learned scene.",
    }


class ResidualStatistics:
    """Pixel-weighted diagnostics on rendered opacity > 0.1, without teachers."""
    def __init__(self, max_log_gain=.3, max_additive=.1):
        import numpy as np
        self.bounds = (float(max_log_gain), float(max_additive))
        if not all(np.isfinite(v) and v >= 0 for v in self.bounds):
            raise ValueError("Residual bounds must be finite and nonnegative")
        self.sums = np.zeros((7, 3), dtype=np.float64)
        self.count = self.active_frames = self.bypassed_frames = 0

    def update(self, outputs):
        import numpy as np
        opacity = outputs["opacity"].detach().cpu().numpy().squeeze(-1)
        mask = opacity > .1
        active = bool(outputs["rgb_residual_active"])
        self.active_frames += int(active)
        self.bypassed_frames += int(not active)
        arrays = [outputs[name].detach().float().cpu().numpy()[mask] for name in MAP_KEYS]
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Non-finite residual maps on valid opacity support")
        gain, log_gain, additive, residual = arrays
        self.count += int(mask.sum())
        self.sums += np.stack([
            gain.sum(0, dtype=np.float64), np.abs(additive).sum(0, dtype=np.float64),
            np.abs(residual).sum(0, dtype=np.float64), np.square(residual, dtype=np.float64).sum(0),
            (np.abs(log_gain) >= .95 * self.bounds[0]).sum(0) if self.bounds[0] else np.zeros(3),
            (np.abs(additive) >= .95 * self.bounds[1]).sum(0) if self.bounds[1] else np.zeros(3),
            additive.sum(0, dtype=np.float64),
        ])

    def summary(self):
        import math
        channels = {}
        if self.count:
            for channel, values in zip("rgb", self.sums.T / self.count):
                channels[channel] = {
                    "mean_gain": float(values[0]), "mean_abs_additive": float(values[1]),
                    "mean_abs_residual": float(values[2]), "rms_residual": math.sqrt(float(values[3])),
                    "gain_fraction_at_least_95pct_bound": float(values[4]) if self.bounds[0] else None,
                    "additive_fraction_at_least_95pct_bound": float(values[5]) if self.bounds[1] else None,
                    "mean_additive": float(values[6]),
                }
        return {"valid_pixels": self.count, "active_frames": self.active_frames,
                "bypassed_frames": self.bypassed_frames, "channels": channels,
                "support": "rendered opacity > 0.1; no target material/confidence mask",
                "aggregation": "global pixel-weighted float64 sums",
                "note": "These are display RGB reconstruction terms, not identified physical lighting components."}


def save_images(directory, stem, target, outputs, max_log_gain, max_additive):
    import numpy as np
    from PIL import Image

    def array(value):
        return value.detach().float().cpu().numpy()

    def png(name, value):
        Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)).save(directory / f"{stem}_{name}.png")

    gt, pure, hybrid = array(target), array(outputs["rgb_pure_pbr"]), array(outputs["rgb_hybrid"])
    png("gt", gt)
    png("pure_pbr", pure)
    png("hybrid", hybrid)
    png("paired", np.concatenate([gt, pure, hybrid], axis=1))
    maps = {name: array(outputs[name]) for name in MAP_KEYS}
    maps["rgb_residual_active"] = np.asarray(bool(outputs["rgb_residual_active"]))
    maps["rgb_residual_strength"] = np.asarray(float(outputs["rgb_residual_strength"]), dtype=np.float32)
    maps["rgb_visible_residual"] = np.clip(hybrid, 0, 1) - np.clip(pure, 0, 1)
    residual_scale = max(float(np.abs(maps["rgb_residual"]).max()), 1e-6)
    maps["rgb_residual_png_scale"] = np.asarray(residual_scale, dtype=np.float32)
    maps["rgb_log_gain_png_scale"] = np.asarray(max_log_gain, dtype=np.float32)
    maps["rgb_additive_png_scale"] = np.asarray(max_additive, dtype=np.float32)
    for name, values, bound in (("gain", maps["rgb_log_gain"], max_log_gain),
                                ("additive", maps["rgb_additive"], max_additive),
                                ("residual", maps["rgb_residual"], residual_scale)):
        png(name, .5 + values / (2 * bound) if bound else np.full_like(values, .5))
    np.savez_compressed(directory / f"{stem}_residual.npz", **maps)


def write_report(result, path):
    budget, protocol = result["training_budget"], result["protocol"]
    hybrid, pure = result["image_metrics"], result["pure_pbr_image_metrics"]
    delta = result["delta_hybrid_minus_pure_pbr"]
    rows = ["# Base v5 학습 평가", ""]
    if result.get("smoke", False):
        rows += ["**실행 정확성 확인용 smoke 결과입니다. 최종 학습 성능으로 해석하면 안 됩니다.**", ""]
    rows += [
            f"- Checkpoint: `{result['checkpoint']}`; step {result['checkpoint_step']}.",
            f"- 완료된 카메라 학습: {budget['completed_camera_updates']:,}회. 초기화: dataset seed points.",
            f"- Scene {protocol['scene']}, heldout 이미지 {len(result['frames'])}장; 저장된 해상도·split·affine 정책 사용.",
            f"- 카메라 업데이트 기준 워밍업 {budget['warmup_camera_updates']:,}회 / ramp {budget['ramp_camera_updates']:,}회; "
            f"pure-PBR 보조 loss λ={budget['pure_pbr_lambda']}.",
            "- 3ableGS의 카메라 학습이며 SplatAD의 camera/LiDAR 혼합 스케줄과 업데이트 예산이 다릅니다.", "",
            "| 출력 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |", "|---|---:|---:|---:|"]
    for label, metrics in (("Pure PBR (현재 공동 학습 모델, residual off)", pure), ("Hybrid", hybrid)):
        rows.append(f"| {label} | {metrics['psnr']:.4f} | {metrics['ssim']:.4f} | {metrics['lpips']:.4f} |")
    rows += [f"| Hybrid − Pure PBR | {delta['psnr']:+.4f} | {delta['ssim']:+.4f} | {delta['lpips']:+.4f} |", "",
             "Pure PBR는 현재 체크포인트의 재질·기하·조명·카메라 보정을 사용합니다. "
             "이전 Base v4의 고정 출력과 동일한 값이 아닙니다. 두 출력은 같은 렌더에서 얻으며 차이는 display RGB residual입니다.", "",
             "SSIM은 3ableGS의 skimage 기본 7×7 uniform window입니다. "
             "SplatAD의 pytorch_msssim 11×11 Gaussian SSIM과 직접 같은 열로 비교하지 마세요. "
             "PSNR/SSIM/LPIPS는 [0,1]로 clamp한 float RGB 전체 이미지에서 계산하고 이미지별 값을 평균합니다.", "",
             "RGB residual은 camera response·affine 이후의 재구성 보정입니다. "
             "외부 HDR 또는 raw 외 조명에서는 이 경로를 우회하며, 이 RGB 지표는 재조명 품질을 검증하지 않습니다.", "",
             "[원본 지표와 프레임별 결과](evaluation.json)"]
    if result["images_saved"]:
        rows += ["", "[렌더링 폴더](renders/)의 paired PNG는 GT / 현재 Pure PBR / Hybrid 순서입니다. "
                 "Gain·additive·residual PNG는 중간 회색이 무보정이며, NPZ에는 signed float 값과 PNG scale이 저장됩니다. "
                 "Residual PNG scale은 프레임별로 달라 색 강도를 프레임 간 직접 비교할 수 없습니다."]
    if "dynamic_object_metrics" in result:
        rows += ["", "## 동적 영역 보조 평가", "",
                 "기존 전체 이미지 지표와 별도입니다. Reference mask만 사용하며 빈 영역은 평균에서 제외합니다. "
                 "SSIM은 7×7 window 전체가 해당 영역 안에 있는 픽셀만 사용합니다. "
                 "Crop LPIPS는 reference 연결 성분의 padding된 사각형을 사용하며, actor/track 평균이 아닙니다.", "",
                 "| 영역 | 출력 | 비어 있지 않은 프레임 | PSNR ↑ | SSIM ↑ | Crop LPIPS ↓ |",
                 "|---|---|---:|---:|---:|---:|"]
        for region, summary in result["dynamic_object_metrics"]["summary"].items():
            for branch, metrics in summary["branches"].items():
                values = [metrics[key] for key in ("pixel_weighted_psnr", "ssim_valid_window", "component_mean_crop_lpips")]
                formatted = [f"{value:.4f}" if value is not None else "—" for value in values]
                rows.append(f"| {region} | {branch} | {summary['nonempty_frames']} | " + " | ".join(formatted) + " |")
    if "dynamic_temporal_diagnostic" in result:
        temporal = result["dynamic_temporal_diagnostic"]["subsets"]["all_adjacent"]
        rows += ["", "## 동적 temporal 재구성 진단", "",
                 "학습 프레임을 포함한 전체 연속 영상의 진단이며 heldout temporal benchmark가 아닙니다. "
                 "모든 출력에 reference RGB optical flow와 동일한 dynamic mask를 적용합니다. "
                 "Train→test 등 구간별 결과와 유효 영역 비율은 evaluation.json에 있습니다.", "",
                 f"- 인접 프레임 쌍: {temporal['temporal_pairs']}; 유효 동적 픽셀: {temporal['temporal_valid_pixels']}."]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(args):
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    args = validate_args(args)
    os.chdir(ROOT)
    config, dataset, trainer = load_scene(args.checkpoint)
    split = dataset.test_image_set
    if split is None or len(split) == 0:
        raise ValueError("Saved data protocol has no held-out test images")
    budget = budget_metadata(config, trainer)
    joint = config.trainer.rgb_joint
    bounds = (float(joint.get("max_log_gain", .3)), float(joint.get("max_additive", .1)))
    stats = ResidualStatistics(*bounds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "renders"
    if args.save_images:
        image_dir.mkdir(exist_ok=True)
    records = []
    dynamic_records = []
    expected_frames = set(map(int, dataset.test_timesteps))
    expected_cameras = set(map(int, config.data.pixel_source.cameras))
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(len(split)):
            image_by_weather, camera_by_weather = split.get_image(index, trainer._get_downscale_factor(), ["raw"])
            images, cameras = image_by_weather["raw"], camera_by_weather["raw"]
            frame_id, camera_id = int(images["frame_idx"].flatten()[0]), int(cameras["cam_id"].flatten()[0])
            if frame_id not in expected_frames or camera_id not in expected_cameras:
                raise ValueError("Evaluation split returned an unexpected frame or camera")
            inputs = {key: value.to(trainer.device) if torch.is_tensor(value) else value
                      for key, value in images.items() if key in RENDER_IMAGE_KEYS}
            camera_inputs = {key: value.to(trainer.device) if torch.is_tensor(value) else value
                             for key, value in cameras.items()}
            outputs = trainer(inputs, camera_inputs, "raw")
            required = {"rgb", "rgb_pure_pbr", "rgb_hybrid", "rgb_residual_strength", "rgb_residual_active", "opacity", *MAP_KEYS}
            if not required <= set(outputs):
                raise ValueError(f"Joint renderer omitted outputs: {sorted(required - set(outputs))}")
            target = images["pixels"].to(trainer.device)
            pure = image_metrics(outputs["rgb_pure_pbr"], target, trainer.lpips)
            hybrid = image_metrics(outputs["rgb_hybrid"], target, trainer.lpips)
            if not torch.equal(outputs["rgb"].clamp(0, 1), outputs["rgb_hybrid"].clamp(0, 1)):
                raise ValueError("Main RGB output differs from hybrid output")
            records.append({"frame_id": frame_id, "source_frame": frame_id + int(config.data.start_timestep),
                            "camera_id": camera_id, "height": target.shape[0], "width": target.shape[1],
                            "pure_pbr": pure, "hybrid": hybrid,
                            "residual_strength": float(outputs["rgb_residual_strength"]),
                            "residual_active": bool(outputs["rgb_residual_active"]),
                            "delta_hybrid_minus_pure_pbr": {key: hybrid[key] - pure[key] for key in METRIC_KEYS}})
            if getattr(args, "dynamic_metrics", False):
                from utils.dynamic_actor_metrics import evaluate_dynamic_frame
                dynamic = evaluate_dynamic_frame(
                    {"pure_pbr": outputs["rgb_pure_pbr"], "hybrid": outputs["rgb_hybrid"]},
                    target, images, trainer.lpips)
                dynamic_records.append(dynamic)
                records[-1]["dynamic_regions"] = dynamic
            stats.update(outputs)
            if args.save_images:
                save_images(image_dir, f"frame_{frame_id:03d}_cam_{camera_id}", target, outputs, *bounds)
            print(f"Frame {frame_id:03d} camera {camera_id}: hybrid {hybrid['psnr']:.4f} dB, "
                  f"pure {pure['psnr']:.4f} dB", flush=True)
    observed = [(row["frame_id"], row["camera_id"]) for row in records]
    expected = {(frame, camera) for frame in expected_frames for camera in expected_cameras}
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("Evaluated frame/camera pairs do not exactly match the held-out split")
    if int(trainer.rgb_joint_camera_updates) != budget["completed_camera_updates"]:
        raise ValueError("Evaluation changed the training camera-update counter")
    hybrid, pure = aggregate_metrics(records, "hybrid"), aggregate_metrics(records, "pure_pbr")
    result = {
        "schema_version": 1, "evaluation_kind": "base_v5", "checkpoint": str(args.checkpoint),
        "smoke": budget["completed_camera_updates"] < 100 or "smoke" in args.checkpoint.parent.name.lower(),
        "checkpoint_step": int(trainer.step), "config": str(args.checkpoint.parent / "config.yaml"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "training_budget": budget,
        "protocol": {"scene": int(config.data.scene_idx), "weather": "raw", "split": "test",
            "source_start_timestep": int(config.data.start_timestep),
            "test_timesteps": sorted(expected_frames), "cameras": sorted(expected_cameras),
            "saved_data": OmegaConf.to_container(config.data, resolve=True),
            "saved_renderer": OmegaConf.to_container(config.trainer.render, resolve=True),
            "affine_policy": "Base v5 inherited native Base-v4 heldout mean-embedding policy",
            "teacher_inputs": False, "aggregation": "arithmetic mean over held-out images; sample standard deviations",
            "metrics": {"psnr": "-10*log10(full-image mean squared error)",
                        "ssim": "skimage structural_similarity defaults: 7x7 uniform window, channel_axis=-1, data_range=1",
                        "lpips": "native frozen AlexNet LPIPS, normalize=True; reset between images/branches"}},
        "image_metrics": hybrid, "pure_pbr_image_metrics": pure,
        "delta_hybrid_minus_pure_pbr": {key: hybrid[key] - pure[key] for key in METRIC_KEYS},
        "pure_pbr_definition": "Current jointly trained PBR with only the display RGB residual bypassed",
        "comparison_kind": "within_checkpoint_display_rgb_residual_contribution",
        "rgb_residual_statistics": stats.summary(), "frames": records,
        "images_saved": bool(args.save_images), "elapsed_seconds": time.perf_counter() - started,
    }
    if getattr(args, "dynamic_metrics", False):
        from utils.dynamic_actor_metrics import dynamic_metric_protocol, summarize_dynamic_frames
        result["dynamic_object_metrics"] = {"protocol": dynamic_metric_protocol(),
                                             "summary": summarize_dynamic_frames(dynamic_records)}
    if getattr(args, "dynamic_temporal", False):
        result["dynamic_temporal_diagnostic"] = evaluate_dynamic_sequence(config, dataset, trainer)
        if int(trainer.rgb_joint_camera_updates) != budget["completed_camera_updates"]:
            raise ValueError("Temporal evaluation changed the training camera-update counter")
    result["elapsed_seconds"] = time.perf_counter() - started
    destination = args.output_dir / "evaluation.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result, args.output_dir / "report.md")
    return result


def evaluate_dynamic_sequence(config, dataset, trainer):
    """Render every saved frame; reference RGB/masks never enter inference."""
    import torch
    from utils.dynamic_actor_metrics import DynamicTemporalDiagnostic

    split = dataset.full_image_set
    if split is None or len(split) == 0:
        raise ValueError("Dynamic temporal diagnostics require the full saved sequence")
    expected_frames = set(map(int, dataset.train_timesteps)) | set(map(int, dataset.test_timesteps))
    expected_cameras = set(map(int, config.data.pixel_source.cameras))
    diagnostic = DynamicTemporalDiagnostic(dataset.train_timesteps, dataset.test_timesteps,
                                            metadata={"scene": int(config.data.scene_idx)})
    observed, previous = set(), {}
    with torch.no_grad():
        for index in range(len(split)):
            image_by_weather, camera_by_weather = split.get_image(index, trainer._get_downscale_factor(), ["raw"])
            images, cameras = image_by_weather["raw"], camera_by_weather["raw"]
            frame_id = int(images["frame_idx"].flatten()[0])
            camera_id = int(cameras["cam_id"].flatten()[0])
            pair = (frame_id, camera_id)
            if pair in observed or frame_id not in expected_frames or camera_id not in expected_cameras:
                raise ValueError("Unexpected or duplicate temporal frame/camera pair")
            if camera_id in previous and frame_id != previous[camera_id] + 1:
                raise ValueError("Full sequence must be consecutive and chronological for each camera")
            observed.add(pair)
            previous[camera_id] = frame_id
            inputs = {key: value.to(trainer.device) if torch.is_tensor(value) else value
                      for key, value in images.items() if key in RENDER_IMAGE_KEYS}
            camera_inputs = {key: value.to(trainer.device) if torch.is_tensor(value) else value
                             for key, value in cameras.items()}
            outputs = trainer(inputs, camera_inputs, "raw")
            diagnostic.update(images["pixels"],
                              {"pure_pbr": outputs["rgb_pure_pbr"], "hybrid": outputs["rgb_hybrid"]},
                              images, frame_id, camera_id)
            print(f"Dynamic temporal frame {frame_id:03d} camera {camera_id}", flush=True)
    expected = {(frame, camera) for frame in expected_frames for camera in expected_cameras}
    if observed != expected:
        raise ValueError("Temporal evaluation did not cover the complete saved sequence")
    return diagnostic.summary()


def main():
    result = run(make_parser().parse_args())
    print(json.dumps({"checkpoint": result["checkpoint"], "image_metrics": result["image_metrics"],
                      "smoke": result["smoke"],
                      "pure_pbr_image_metrics": result["pure_pbr_image_metrics"],
                      "completed_camera_updates": result["training_budget"]["completed_camera_updates"]}, indent=2))


if __name__ == "__main__":
    main()
