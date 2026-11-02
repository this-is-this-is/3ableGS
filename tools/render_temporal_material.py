#!/usr/bin/env python3
"""Compare frozen raw PBR transport and sparse surface material edits.

The video shows baseline | variant | history/edit weight. Evaluation includes
reference-flow temporal residual and edge preservation. PSNR is a spatial
reconstruction measure, not an independent temporal metric. Use
--evaluation-split test with a contiguous-holdout checkpoint for held-out scores.
"""
import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logger = logging.getLogger(__name__)

# Ground-truth RGB/masks, LiDAR and teacher buffers are evaluation data only.
RENDER_IMAGE_KEYS = frozenset({
    "origins", "viewdirs", "direction_norm", "pixel_coords",
    "normed_time", "img_idx", "frame_idx",
})


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Raw MultiTrainer_pbr checkpoint; config.yaml must be beside it")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty directory; existing results are never overwritten")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame index relative to the checkpoint's complete training sequence")
    parser.add_argument("--num-frames", type=int,
                        help="Consecutive camera-0 frames; defaults to the remaining sequence")
    parser.add_argument("--downscale", type=float, default=2.0,
                        help="Additional divisor of the saved config's rendering resolution (default: 2)")
    parser.add_argument("--mode", choices=("depth", "normal", "material"), default="material",
                        help="History gate: depth; plus normals; plus material similarity (default)")
    parser.add_argument("--history-weight", type=float, default=0.85)
    parser.add_argument("--max-history", type=int, default=8,
                        help="Effective EMA sample-count cap, not a strict frame window (default: 8)")
    parser.add_argument("--domain", choices=("material", "rgb"), default="material")
    parser.add_argument("--history-source", choices=("resolved", "current"), default="resolved")
    parser.add_argument("--clamp-history", action="store_true")
    parser.add_argument("--guide-ablation", default="none", choices=(
        "none", "shuffle_normal", "shuffle_roughness", "constant_normal", "constant_roughness"))
    parser.add_argument("--normal-cos-min", type=float, default=0.9)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.02)
    parser.add_argument("--evaluation-cache", type=Path,
                        help="Shared GT optical-flow cache, independent of transport settings")
    parser.add_argument("--evaluation-split", choices=("all", "test"), default="all")
    parser.add_argument("--edit-affinity", choices=("geometry", "rgb", "gbuffer"))
    parser.add_argument("--edit-keyframes", type=int, default=3,
                        help="Number of train-only keyframes for the controlled surface edit")
    parser.add_argument("--edit-roi", nargs=4, type=float, default=[.35, .65, .65, .90],
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--save-images", action="store_true",
                        help="Also save lossless baseline, transport and weight PNGs")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip MP4 encoding; metrics are always written")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="gsplat scene rendering requires cuda; cpu reports a clear unsupported error")
    return parser


def validate_args(args):
    if args.start_frame < 0:
        raise ValueError("--start-frame must be nonnegative")
    if args.num_frames is not None and args.num_frames < 1:
        raise ValueError("--num-frames must be positive")
    if not math.isfinite(args.downscale) or args.downscale < 1:
        raise ValueError("--downscale must be finite and >= 1")
    if not math.isfinite(args.history_weight) or not 0 <= args.history_weight < 1:
        raise ValueError("--history-weight must be finite and in [0, 1)")
    if args.max_history < 1:
        raise ValueError("--max-history must be positive")
    if not 0 <= args.normal_cos_min <= 1 or not math.isfinite(args.normal_cos_min):
        raise ValueError("--normal-cos-min must be in [0,1]")
    if not math.isfinite(args.depth_rel_tolerance) or args.depth_rel_tolerance < 0:
        raise ValueError("--depth-rel-tolerance must be nonnegative")
    if args.edit_keyframes < 1:
        raise ValueError("--edit-keyframes must be positive")
    x0, y0, x1, y1 = args.edit_roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("--edit-roi must be a nonempty normalized rectangle")
    if args.edit_affinity and args.domain != "material":
        raise ValueError("Surface edit comparisons require --domain material")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be finite and positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not (checkpoint.parent / "config.yaml").is_file():
        raise FileNotFoundError(f"Saved config not found: {checkpoint.parent / 'config.yaml'}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output is not empty: {output}; choose a new --output-dir")
    return checkpoint, output


def inference_config(saved_cfg, downscale=1.0):
    """Keep trained IDs, timestamps, object topology and embedding dimensions."""
    cfg = copy.deepcopy(saved_cfg)
    if str(cfg.trainer.type) not in {
        "models.trainers.MultiTrainer_pbr",
        "models.trainers.scene_graph_pbr.MultiTrainer_pbr",
    }:
        raise ValueError("This first experiment requires a raw MultiTrainer_pbr checkpoint")
    if 0 not in list(cfg.data.pixel_source.cameras):
        raise ValueError("This first experiment requires camera 0 in the saved config")
    cfg.data.preload_device = "cpu"
    cfg.data.pixel_source.load_gbuffer = False
    cfg.data.pixel_source.load_weather = []
    # Downsample before CPU preloading to bound RAM; camera loaders scale K to
    # the resulting dimensions. Do not additionally divide runtime K below.
    cfg.data.pixel_source.downscale_when_loading = [
        float(value) * downscale for value in cfg.data.pixel_source.downscale_when_loading
    ]
    return cfg


def frame_indices(start, count, total):
    if start < 0 or start >= total:
        raise ValueError(f"--start-frame {start} is outside the saved sequence [0, {total})")
    end = total if count is None else start + count
    if end <= start or end > total:
        raise ValueError(f"Requested frames [{start}, {end}) exceed saved sequence length {total}")
    return range(start, end)


def psnr_from_sums(squared_error, values):
    """A finite PSNR for JSON, with None for an empty evaluation mask."""
    if values == 0:
        return None
    return -10.0 * math.log10(max(float(squared_error) / values, 1e-12))


def quality_stats(baseline, transported, reference, static_mask):
    """Use one prediction-defined static mask for both methods, never history acceptance."""
    import numpy as np

    if baseline.shape != transported.shape or baseline.shape != reference.shape:
        raise ValueError("Baseline, transported RGB and reference shapes differ")
    mask = np.asarray(static_mask, dtype=bool).reshape(baseline.shape[:2])
    result = {"static_pixels": int(mask.sum()), "pixels": int(mask.size)}
    for method, predicted in (("baseline", baseline), ("transport", transported)):
        squared = (predicted.astype(np.float64) - reference.astype(np.float64)) ** 2
        for region, values in (("full", squared), ("static", squared[mask])):
            total = float(values.sum())
            size = int(values.size)
            prefix = f"{method}_{region}"
            result[f"{prefix}_squared_error"] = total
            result[f"{prefix}_values"] = size
            result[f"{prefix}_psnr"] = psnr_from_sums(total, size)
    result["rgb_delta_mae"] = float(np.abs(transported - baseline).mean())
    result["static_rgb_delta_mae"] = (
        float(np.abs(transported[mask] - baseline[mask]).mean()) if mask.any() else None
    )
    return result


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def to_device(values, device, torch):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in values.items()}


def render_inputs(image_infos, device, torch):
    return to_device({key: value for key, value in image_infos.items() if key in RENDER_IMAGE_KEYS},
                     device, torch)


def rgb_array(tensor):
    return tensor.detach().clamp(0, 1).cpu().numpy()


def image_u8(array):
    import numpy as np
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def evaluation_static_mask(image_infos):
    """Reference-only static masks; these never enter model inference."""
    import numpy as np
    shape = image_infos["pixels"].shape[:2]
    valid = np.ones(shape, dtype=bool)
    for key in ("sky_masks", "dynamic_masks"):
        if key not in image_infos:
            raise ValueError(f"Reference temporal evaluation requires {key}")
    for key in ("sky_masks", "dynamic_masks", "egocar_masks"):
        if key in image_infos:
            valid &= image_infos[key].detach().cpu().numpy().reshape(shape) < .5
    return valid


def edit_keyframe_ids(train_frames, count):
    import numpy as np
    values = sorted({int(frame) for frame in train_frames})
    if count > len(values):
        raise ValueError("Requested more edit keyframes than available train frames")
    return [values[index] for index in np.linspace(0, len(values) - 1, count, dtype=int)]


def build_surface_editor(trainer, camera, dataset, args, device, torch):
    from models.surface_material_edit import SurfaceMaterialEditor
    frames = edit_keyframe_ids(dataset.train_timesteps, args.edit_keyframes)
    editor = SurfaceMaterialEditor({
        "affinity": args.edit_affinity,
        "allowed_frame_ids": [int(value) for value in dataset.train_timesteps],
    })
    trainer._export_temporal_inputs = True
    with torch.no_grad():
        for frame in frames:
            infos, cameras = camera.get_image(frame, "raw")
            output = trainer(render_inputs(infos, device, torch),
                             to_device(cameras, device, torch), "raw")
            h, w = output["depth"].shape[:2]
            seed = torch.zeros((h, w), dtype=torch.bool, device=device)
            x0, y0, x1, y1 = args.edit_roi
            seed[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = True
            gbuf = {key: output[f"unfiltered_{key}"]
                    for key in ("albedo", "metallic", "roughness", "normal")}
            editor.add_keyframe(
                gbuffer=gbuf, depth=output["depth"], opacity=output["opacity"],
                static_mask=output["temporal_static_mask"][..., 0],
                camera_to_world=trainer._render_c2w,
                intrinsics=trainer._render_intrinsics, frame_id=frame,
                seed_mask=seed, raw_rgb=output["unfiltered_surface_rgb"],
            )
    summary = editor.finalize()
    trainer._export_temporal_inputs = False
    trainer.configure_surface_editor(editor)
    return summary


def comparison_frame(baseline, transported, weight, label, *, domain="material", edit=False, titles=None):
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = baseline.shape[:2]
    panel = np.concatenate((image_u8(baseline), image_u8(transported),
                            image_u8(np.repeat(weight.reshape(height, width, 1), 3, axis=-1))),
                           axis=1)
    canvas = Image.new("RGB", (3 * width, height + 28))
    canvas.paste(Image.fromarray(panel), (0, 28))
    draw = ImageDraw.Draw(canvas)
    if titles is None:
        titles = ("Baseline", "Surface edit" if edit else f"{domain.capitalize()} transport",
                  "Edit strength [0, 1]" if edit else "History weight [0, 1]")
    for column, title in enumerate(titles):
        draw.text((column * width + 6, 7), f"{title}  |  {label}", fill=(255, 255, 255))
    pixels = np.asarray(canvas)
    # libx264 with yuv420p requires even dimensions; pad without resizing.
    return np.pad(pixels, ((0, pixels.shape[0] % 2), (0, pixels.shape[1] % 2), (0, 0)),
                  mode="edge")


def summarize(records):
    summary = {
        "frames": len(records),
        "psnr_definition": "Global RGB MSE over the stated region; RGB clamped to [0,1].",
        "static_mask_definition": "Current predicted opacity/dynamic-opacity gate, independent of history acceptance.",
        "latency_definition": "Synchronized combined baseline+transport forward, including dynamic guard and two PBR shadings; excludes input transfer and output I/O.",
        "temporal_metric": None,
        "limitations": [
            "Chronological reconstruction path includes train frames; this is not a held-out temporal benchmark.",
            "PSNR, history coverage and RGB change do not establish improved temporal consistency.",
            "Static PSNR uses predicted geometry/opacity; reference temporal and edge metrics use separate GT masks.",
            "Both outputs use mean affine embeddings; these PSNRs are not directly comparable to image-specific affine evaluation.",
            "First-frame latency includes initialization and history is empty.",
        ],
    }
    for method in ("baseline", "transport"):
        for region in ("full", "static"):
            prefix = f"{method}_{region}"
            summary[f"{prefix}_psnr"] = psnr_from_sums(
                sum(row[f"{prefix}_squared_error"] for row in records),
                sum(row[f"{prefix}_values"] for row in records),
            )
    for key in ("history_coverage_full", "history_coverage_static", "mean_history_weight",
                "rgb_delta_mae", "static_rgb_delta_mae", "forward_ms"):
        values = [row[key] for row in records if row[key] is not None]
        summary[f"mean_{key}"] = sum(values) / len(values) if values else None
    rest = [row["forward_ms"] for row in records[1:]]
    summary["mean_forward_ms_excluding_first"] = sum(rest) / len(rest) if rest else None
    summary["train_frames"] = sum(row["split"] == "train" for row in records)
    summary["test_frames"] = sum(row["split"] == "test" for row in records)
    if records and any(key.startswith("evaluation_") for key in records[0]):
        from utils.temporal_metrics import summarize_metric_records
        metrics = summarize_metric_records([
            {key[len("evaluation_"):]: value for key, value in row.items()
             if key.startswith("evaluation_")} for row in records
        ])
        summary["reference_evaluation"] = metrics
        summary["temporal_metric"] = "GT RGB DIS optical flow, forward/backward consistency, shared GT static mask"
        summary["limitations"] = [
            text for text in summary["limitations"]
            if not text.startswith("PSNR, history coverage")
        ] + ["Reference optical flow is an estimate, not ground-truth motion."]
    return summary


def run(args):
    checkpoint, output = validate_args(args)
    import torch

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Scene rendering requires CUDA and the project's gsplat environment. "
                           "CUDA is unavailable or --device cpu was requested. "
                           "The transport unit tests and --help can run without a GPU.")
    import imageio.v2 as imageio
    import numpy as np
    from omegaconf import OmegaConf

    saved_cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
    cfg = inference_config(saved_cfg, args.downscale)
    device = torch.device("cuda")
    # Saved dataset paths and SMPL assets follow the repository's existing convention.
    os.chdir(ROOT)
    random.seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))
    torch.manual_seed(int(cfg.get("seed", 0)))
    output.mkdir(parents=True, exist_ok=True)
    # Claim this empty directory before expensive dataset/model construction.
    with (output / "manifest.json").open("x") as handle:
        manifest = {
            "status": "initializing",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "source_config": str(checkpoint.parent / "config.yaml"),
            "source_sequence": {"start": int(cfg.data.start_timestep), "end": int(cfg.data.end_timestep)},
            "camera_id": 0,
            "args": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
            "shared_affine": True,
            "reference_usage": "GT RGB/masks are used only for post-render quality, flow and edge evaluation; teacher G-buffers are not loaded.",
            "video_columns": [
                "baseline", "surface_edit" if args.edit_affinity else f"{args.domain}_transport",
                "edit_strength" if args.edit_affinity else "history_weight",
            ],
        }
        json.dump(manifest, handle, indent=2)
    OmegaConf.save(cfg, output / "inference_config.yaml")
    try:
        from datasets.driving_dataset import DrivingDataset
        from utils.misc import import_str

        logger.info("Loading the complete saved sequence on CPU to preserve trained timestamps and IDs")
        dataset = DrivingDataset(data_cfg=cfg.data)
        camera = dataset.pixel_source.camera_data[0]
        frames = list(frame_indices(args.start_frame, args.num_frames, len(camera)))
        test_frames = {int(value) for value in dataset.test_timesteps}
        if args.evaluation_split == "test":
            frames = [frame for frame in frames if frame in test_frames]
            if not frames:
                raise ValueError("Requested interval contains no test frames")
        camera.set_downscale_factor(dataset.pixel_source.downscale_factor)
        trainer = import_str(cfg.trainer.type)(
            **copy.deepcopy(cfg.trainer), dataset=dataset, weathers=["raw"],
            vgg_config=cfg.get("vgg"), model_config=copy.deepcopy(cfg.model), device=device,
        )
        # Existing node loaders directly install some non-parameter tensors;
        # map them to the renderer device as well as the learned parameters.
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        trainer.load_state_dict(state, load_only_model=True, strict=True)
        del state
        trainer.set_eval()
        for model in trainer.models.values():
            model.requires_grad_(False)
        trainer.requires_grad_(False)
        trainer._suppress_eval_layers = True
        transport_cfg = {"mode": args.mode, "history_weight": args.history_weight,
                         "max_history": args.max_history,
                         "history_source": args.history_source,
                         "clamp_history": args.clamp_history,
                         "guide_ablation": args.guide_ablation,
                         "normal_cos_min": args.normal_cos_min,
                         "depth_rel_tolerance": args.depth_rel_tolerance}
        if args.edit_affinity:
            transport_cfg["history_weight"] = 0.
        trainer.configure_temporal_material(transport_cfg, shared_affine=True, domain=args.domain)
        trainer.reset_temporal_material()
        edit_summary = None
        if args.edit_affinity:
            edit_summary = build_surface_editor(trainer, camera, dataset, args, device, torch)
        from utils.temporal_metrics import TemporalEvaluator
        evaluator = TemporalEvaluator(
            cache_dir=args.evaluation_cache,
            metadata={"scene": int(cfg.data.scene_idx), "camera": 0,
                      "sequence_start": int(cfg.data.start_timestep)},
        )
        manifest.update(status="rendering", relative_frames=list(frames),
                        full_sequence_frames=len(camera),
                        transport=dict(trainer._temporal_material.config),
                        max_dynamic_opacity=0.01,
                        saved_pixel_downscale=float(cfg.data.pixel_source.downscale),
                        runtime_camera_scale=float(camera.downscale_factor),
                        load_size_hw=list(camera.load_size))
        manifest["domain"] = args.domain
        manifest["edit_bank"] = edit_summary
        manifest["evaluation_split"] = args.evaluation_split
        write_json(output / "manifest.json", manifest)
        if args.save_images:
            for name in ("baseline", "transport", "weight"):
                (output / name).mkdir()
        records = []
        with ExitStack() as stack, torch.no_grad():
            metrics = stack.enter_context((output / "frames.jsonl").open("x"))
            writer = None
            if not args.no_video:
                writer = imageio.get_writer(
                    str(output / "comparison.mp4"), fps=args.fps, codec="libx264",
                    quality=8, macro_block_size=1, pixelformat="yuv420p",
                )
                stack.callback(writer.close)
            for frame in frames:
                image_infos, cam_infos = camera.get_image(frame, "raw")
                # References/masks never enter the forward call.
                inputs = render_inputs(image_infos, device, torch)
                cams = to_device(cam_infos, device, torch)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                result = trainer(inputs, cams, "raw")
                torch.cuda.synchronize(device)
                forward_ms = (time.perf_counter() - started) * 1000
                baseline, transported = rgb_array(result["rgb_unfiltered"]), rgb_array(result["rgb"])
                reference = rgb_array(image_infos["pixels"])
                static = result["temporal_static_mask"].detach().cpu().numpy().astype(bool).squeeze(-1)
                weight = result["temporal_weight"].detach().cpu().numpy()
                accepted = weight.reshape(static.shape) > 0
                evaluation = evaluator.evaluate(
                    reference=reference,
                    predictions={"baseline": baseline, "transport": transported},
                    static_mask=evaluation_static_mask(image_infos), frame_id=frame,
                )
                record = {
                    "relative_frame": frame,
                    "source_frame": dataset.start_timestep + frame,
                    "split": "test" if frame in test_frames else "train",
                    "forward_ms": forward_ms,
                    **quality_stats(baseline, transported, reference, static),
                    "history_coverage_full": float(accepted.mean()),
                    "history_coverage_static": float(accepted[static].mean()) if static.any() else None,
                    "mean_history_weight": float(weight.mean()),
                    "max_history_length": float(result["temporal_history_length"].max().item()),
                    **{f"evaluation_{key}": value for key, value in evaluation.items()},
                }
                if args.edit_affinity:
                    weight = result["edit_weight"].detach().cpu().numpy()
                    record["edit_coverage"] = float((weight > 0).mean())
                    record["edit_mean_strength"] = float(weight.mean())
                records.append(record)
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
                metrics.flush()
                if writer is not None:
                    writer.append_data(comparison_frame(
                        baseline, transported, weight, f"frame {frame:04d}",
                        domain=args.domain, edit=bool(args.edit_affinity),
                    ))
                if args.save_images:
                    name = f"{frame:06d}.png"
                    imageio.imwrite(output / "baseline" / name, image_u8(baseline))
                    imageio.imwrite(output / "transport" / name, image_u8(transported))
                    imageio.imwrite(output / "weight" / name, image_u8(weight.squeeze(-1)))
                logger.info("Frame %d [%s]: PSNR %.3f -> %.3f, history %.1f%%, forward %.1f ms",
                            frame, record["split"], record["baseline_full_psnr"],
                            record["transport_full_psnr"], 100 * record["history_coverage_full"], forward_ms)
                del result, inputs, cams, image_infos, cam_infos
        summary = summarize(records)
        summary["task"] = "controlled_surface_edit" if args.edit_affinity else "temporal_transport"
        summary["evaluation_split"] = args.evaluation_split
        summary["edit_bank"] = edit_summary
        if args.evaluation_split == "test":
            summary["limitations"][0] = (
                "Only saved test frames are scored; history resets across nonconsecutive blocks. "
                "LiDAR, calibrated cameras and object annotations remain full-sequence side information."
            )
        if args.edit_affinity:
            summary["limitations"].append(
                "This is a controlled seed-ROI material edit, not learned weather transfer. "
                "Raw-reference PSNR/temporal residual are diagnostics, not edit quality; no edited GT is available."
            )
            summary["mean_edit_coverage"] = sum(r["edit_coverage"] for r in records) / len(records)
            summary["mean_edit_strength"] = sum(r["edit_mean_strength"] for r in records) / len(records)
        write_json(output / "summary.json", summary)
        manifest.update(status="completed", rendered_frames=len(records))
        write_json(output / "manifest.json", manifest)
        logger.info("Saved comparison and reconstruction diagnostics to %s", output)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = make_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
