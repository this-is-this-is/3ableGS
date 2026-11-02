#!/usr/bin/env python3
"""Frozen, visibility-preserving G-buffer renderer comparisons.

The default matrix is B0 (original renderer), B1 (single-layer linear albedo),
L0 (surface-layered sRGB albedo), L1 (surface-layered linear albedo). No fitting,
temporal loss, rendered history, or reference-buffer model input is used.
Reference RGB and masks enter only the common CPU evaluator. Camera 0 and the
runtime's shared mean affine are used by every variant, including B0.

Existing output is never silently reused. --restart requires identical input,
code and CLI identities and reruns every frame so flow history is reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test", "heldout", "all"), default="heldout")
    parser.add_argument("--diagnostic-all", action="store_true", help=(
        "Explicitly allow a checkpoint without clean block holdouts and/or --split all. "
        "Results are labelled exploratory, even when selecting saved held-out frames."))
    parser.add_argument("--frames", help="Local frame IDs, e.g. 20:30,60:70 or 20,21; ends exclusive.")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--downscale", type=float, default=1.)
    parser.add_argument("--antialiased", action="store_true", help="Off by default; a separate AA audit.")
    parser.add_argument("--grouping", choices=("dynamic", "instance"), default="instance")
    parser.add_argument("--layer-budgets", default="0", help=(
        "Comma-separated 0,2,4. Zero keeps all groups; capped variants include a residual slot."))
    parser.add_argument("--ablations", default="", help=(
        "Additional L1 comparisons: normal,brdf,albedo (comma separated)."))
    parser.add_argument("--trained-only", action="store_true", help=(
        "Evaluate only the renderer recorded in the trained checkpoint's config; "
        "do not swap renderers on a jointly trained model."))
    parser.add_argument("--save-every", type=int, default=0, help=(
        "Save native-resolution RGB PNGs and a labelled contact sheet every N frames; 0 disables."))
    parser.add_argument("--no-perceptual", action="store_true", help=(
        "Explicit debug mode: omit full-frame SSIM/LPIPS; these metrics are enabled by default."))
    parser.add_argument("--restart", action="store_true", help=(
        "Rerun all frames in an existing output directory with exactly matching provenance."))
    return parser


def csv_choices(value, allowed, label, *, cast=str, allow_empty=False):
    if not value and allow_empty:
        return []
    try:
        choices = [cast(part.strip()) for part in value.split(",")]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not choices or any(choice not in allowed for choice in choices):
        raise ValueError(f"{label} must contain only {', '.join(map(str, allowed))}")
    if len(set(choices)) != len(choices):
        raise ValueError(f"{label} contains duplicate values")
    return choices


def parse_frames(value):
    """Parse local IDs with exclusive-end ranges; reject silent truncation."""
    if value is None:
        return None
    frames = []
    for part in value.split(","):
        token = part.strip()
        try:
            if ":" in token:
                pieces = token.split(":")
                if len(pieces) != 2:
                    raise ValueError
                start, end = map(int, pieces)
                if start < 0 or end <= start:
                    raise ValueError
                frames.extend(range(start, end))
            else:
                frame = int(token)
                if frame < 0:
                    raise ValueError
                frames.append(frame)
        except ValueError as exc:
            raise ValueError(f"Invalid --frames token {token!r}; use nonnegative IDs or start:end") from exc
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("--frames must be nonempty and contain no overlapping/duplicate IDs")
    return sorted(frames)


def validate_args(args):
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not (args.checkpoint.parent / "config.yaml").is_file():
        raise FileNotFoundError(args.checkpoint.parent / "config.yaml")
    if args.split == "all" and not args.diagnostic_all:
        raise ValueError("--split all requires --diagnostic-all; it includes training observations")
    if not math.isfinite(args.downscale) or args.downscale < 1:
        raise ValueError("--downscale must be finite and >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    if args.save_every < 0:
        raise ValueError("--save-every must be nonnegative")
    args.layer_budgets = csv_choices(args.layer_budgets, (0, 2, 4), "--layer-budgets", cast=int)
    args.ablations = csv_choices(args.ablations, ("normal", "brdf", "albedo"),
                                 "--ablations", allow_empty=True)
    args.frames = parse_frames(args.frames)
    if args.trained_only and (args.ablations or args.layer_budgets != [0]
                             or args.grouping != "instance"):
        raise ValueError("--trained-only uses the saved renderer; omit grouping/budget/ablation overrides")
    return args


def variant_configs(grouping="instance", layer_budgets=(0,), ablations=()):
    def config(group, color, budget=0, ablation="none"):
        return dict(grouping=group, color_space=color, max_layers=budget,
                    ablation=ablation, diagnostics=True)

    result = {"B0": None, "B1": config("single", "linear")}
    for budget in layer_budgets:
        suffix = "" if budget == 0 else f"_k{budget}"
        result[f"L0{suffix}"] = config(grouping, "srgb", budget)
        result[f"L1{suffix}"] = config(grouping, "linear", budget)
        for ablation in ablations:
            result[f"L1{suffix}_global_{ablation}"] = config(grouping, "linear", budget, ablation)
    return result


def trained_variant_configs(checkpoint):
    """Recover the actual training renderer, failing closed on missing metadata."""
    from omegaconf import OmegaConf
    directory = Path(checkpoint).parent
    cfg = OmegaConf.load(directory / "config.yaml")
    trainer = cfg.get("trainer", {})
    if "layered_gbuffer" in trainer:
        renderer = trainer["layered_gbuffer"]
        renderer = OmegaConf.to_container(renderer, resolve=True) if renderer is not None else None
    elif (directory / "layered_config.json").is_file():
        metadata = json.loads((directory / "layered_config.json").read_text())
        if "renderer" not in metadata:
            raise ValueError("Saved layered_config.json has no renderer")
        renderer = metadata["renderer"]
    else:
        raise ValueError("--trained-only requires an explicit saved trainer.layered_gbuffer or layered_config.json")
    if renderer is not None:
        if not isinstance(renderer, dict):
            raise ValueError("Saved layered renderer must be a mapping or null")
        renderer = dict(renderer, diagnostics=True)
    return {"trained": renderer}


def select_frames(runtime, args):
    available = {
        "val": runtime.val_frames,
        "test": runtime.test_frames,
        "heldout": sorted(set(runtime.val_frames) | set(runtime.test_frames)),
        "all": runtime.frame_ids,
    }[args.split]
    allowed = set(map(int, available))
    frames = sorted(allowed) if args.frames is None else list(args.frames)
    outside = set(frames) - allowed
    if outside:
        raise ValueError(f"Requested frames are outside the selected {args.split} split: {sorted(outside)}")
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    if not frames:
        raise ValueError(f"No frames in selected {args.split} split")
    return frames


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def experiment_identity(args, variants):
    sources = (
        "tools/layered_gbuffer_experiment.py", "tools/ego_motion_runtime.py",
        "models/trainers/scene_graph_pbr.py", "models/pbr.py",
        "models/layered_gbuffer.py", "utils/layered_metrics.py",
        "utils/temporal_metrics.py", "utils/temporal_split.py",
        "tools/render_temporal_material.py",
    )
    return {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "config_sha256": digest(args.checkpoint.parent / "config.yaml"),
        "source_sha256": {name: digest(ROOT / name) for name in sources if (ROOT / name).is_file()},
        "split": args.split, "diagnostic_all": args.diagnostic_all,
        "requested_frames": args.frames, "max_frames": args.max_frames,
        "downscale": args.downscale, "antialiased": args.antialiased,
        "save_every": args.save_every, "variants": variants,
        "camera": 0, "affine_policy": "shared_mean_for_all_variants",
        "optimization": "none", "renderer_history": "none",
        "renderer_policy": "saved_trained_renderer" if args.trained_only else "frozen_renderer_matrix",
        "metric_rgb_policy": "display_rgb_clamped_to_0_1_all_variants",
        "component_preview_policy": "linear_radiance_Reinhard_then_srgb_no_camera_affine",
        "perceptual_metrics": not args.no_perceptual,
        "perceptual_mask_policy": "full_frame_unmasked_frame_mean",
        "conservation_max_abs_tolerance": 5e-5,
    }


def prepare_output(output, identity, restart):
    manifest = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not restart:
            raise FileExistsError(f"Output is not empty: {output}; use a new directory or --restart")
        if not manifest.is_file():
            raise ValueError("--restart requires the existing manifest.json")
        previous = json.loads(manifest.read_text())
        if previous.get("identity") != identity:
            raise ValueError("Output provenance differs (checkpoint/config/code/flags); use a new directory")
    elif restart:
        raise ValueError("--restart requires an existing experiment")
    output.mkdir(parents=True, exist_ok=True)


def as_numpy(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_diagnostics(values):
    """Do not accidentally dump full-resolution buffers into JSON."""
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            nested = scalar_diagnostics(value)
            result.update({f"{key}.{subkey}": item for subkey, item in nested.items()})
            continue
        if value is None or isinstance(value, (str, bool)):
            result[key] = value
            continue
        try:
            array = as_numpy(value)
            if array.size != 1:
                continue
            item = array.item()
        except (TypeError, ValueError):
            continue
        if isinstance(item, (int, float)):
            if not math.isfinite(float(item)):
                raise ValueError(f"Nonfinite renderer diagnostic: {key}")
            result[key] = item
    return result


def assert_conservation(diagnostics, tolerance=5e-5):
    for key in ("feature_reconstruction_max_abs", "mass_reconstruction_max_abs", "raster_mass_max_abs"):
        if key not in diagnostics:
            raise ValueError(f"Missing conservation diagnostic: {key}")
        error = float(diagnostics[key])
        if not math.isfinite(error) or error < 0 or error > tolerance:
            raise ValueError(f"G-buffer conservation failed: {key}={error:.6g}, tolerance={tolerance:.6g}")


def perceptual_metrics(trainer, packet, name):
    import torch
    result = {}
    with torch.no_grad():
        prediction = packet["rgb"].clamp(0, 1).permute(2, 0, 1)[None]
        reference = packet["target_rgb"].clamp(0, 1).permute(2, 0, 1)[None]
        for label in ("ssim", "lpips"):
            metric = getattr(trainer, label)
            if hasattr(metric, "reset"):
                metric.reset()
            value = float(metric(prediction, reference))
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite {label} for {name}")
            key = f"{name}_fullframe_{label}"
            result.update({key: value, key + "_sum": value, key + "_count": 1})
    return result


def save_comparison(output, frame_id, reference, predictions, diagnostic_previews=None):
    import numpy as np
    from PIL import Image, ImageDraw

    directory = output / "images" / f"{frame_id:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    rgb = {"reference": reference, **predictions}
    width = min(640, reference.shape[1])
    height = round(reference.shape[0] * width / reference.shape[1])
    columns = min(3, len(rgb))
    rows = math.ceil(len(rgb) / columns)
    sheet = Image.new("RGB", (width * columns, (height + 24) * rows), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (name, value) in enumerate(rgb.items()):
        image = Image.fromarray(np.round(np.clip(value, 0, 1) * 255).astype(np.uint8))
        image.save(directory / f"{name}.png")
        preview = image.resize((width, height), Image.Resampling.LANCZOS)
        x, y = index % columns * width, index // columns * (height + 24)
        draw.text((x + 4, y + 4), name, fill="white")
        sheet.paste(preview, (x, y + 24))
    sheet.save(directory / "comparison.png")
    for name, value in (diagnostic_previews or {}).items():
        Image.fromarray(np.round(np.clip(value, 0, 1) * 255).astype(np.uint8)).save(directory / f"{name}.png")


def component_preview(value):
    """A fixed visualization transform; not the scene's camera response."""
    import numpy as np
    linear = np.maximum(as_numpy(value), 0)
    compressed = linear / (1 + linear)
    return np.where(compressed <= 0.0031308, compressed * 12.92,
                    1.055 * np.power(compressed, 1 / 2.4) - 0.055)


def diagnostic_summary(records, variants):
    result = {}
    for name in variants:
        rows = [row["rendering"][name] for row in records]
        keys = sorted({key for row in rows for key, value in row.items()
                       if isinstance(value, (int, float)) and not isinstance(value, bool)})
        result[name] = {}
        for key in keys:
            values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
            if values:
                result[name][key] = {"mean": sum(values) / len(values),
                                     "max": max(values), "count": len(values)}
        raw_count = sum(row.get("preclip_value_count", 0) for row in rows)
        raw_sum = sum(row.get("preclip_squared_error_sum", 0) for row in rows)
        result[name]["preclip_global_mse"] = raw_sum / raw_count if raw_count else None
    return result


def run_experiment(args, *, runtime_factory=None, evaluator_factory=None, summarize=None):
    """Injectable factories keep orchestration tests independent of CUDA."""
    if runtime_factory is None:
        from tools.ego_motion_runtime import SceneRuntime
        runtime_factory = SceneRuntime
    if evaluator_factory is None or summarize is None:
        from utils.layered_metrics import LayeredBoundaryEvaluator, summarize_layered_records
        evaluator_factory = evaluator_factory or LayeredBoundaryEvaluator
        summarize = summarize or summarize_layered_records
    import numpy as np

    variants = (trained_variant_configs(args.checkpoint) if args.trained_only
                else variant_configs(args.grouping, args.layer_budgets, args.ablations))
    identity = experiment_identity(args, variants)
    prepare_output(args.output, identity, args.restart)
    started = time.time()
    manifest = {
        "identity": identity, "status": "initializing", "completed_frames": 0,
        "evaluation_claim": ("exploratory_diagnostic_not_clean_holdout" if args.diagnostic_all
                             else "saved_block_holdout_exploratory_scene_evaluation"),
        "limitations": [
            "B0 uses the same shared mean affine as every variant; it is not an exact replay of per-image affine.",
            "Instance grouping separates objects, not surfaces within one object or the background.",
            "Optical-flow temporal scores are reference-derived proxies with reported valid coverage.",
            "Runtime includes diagnostic auxiliary passes and is not an optimized-kernel speed benchmark.",
            "No retraining or material adaptation is performed by this command.",
        ],
    }
    atomic_json(args.output / "manifest.json", manifest)
    runtime = None
    records = []
    try:
        runtime = runtime_factory(args.checkpoint, downscale=args.downscale,
                                  antialiased=args.antialiased,
                                  require_holdout=not args.diagnostic_all)
        frames = select_frames(runtime, args)
        manifest.update(status="running", frames=frames,
                        train_frames=list(map(int, runtime.train_frames)),
                        val_frames=list(map(int, runtime.val_frames)),
                        test_frames=list(map(int, runtime.test_frames)),
                        selected_training_frames=sorted(set(frames) & set(runtime.train_frames)))
        atomic_json(args.output / "manifest.json", manifest)
        evaluator = evaluator_factory()
        for index, frame_id in enumerate(frames):
            predictions, rendering = {}, {}
            perceptual = {}
            save_images = bool(args.save_every and index % args.save_every == 0)
            diagnostic_previews = {}
            reference = dynamic = valid = reference_depth = None
            for name, config in variants.items():
                runtime.trainer.configure_layered_gbuffer(config)
                runtime.reset()
                packet = runtime.render(frame_id)
                prediction = as_numpy(packet["rgb"]).astype(np.float32, copy=True)
                if prediction.ndim != 3 or prediction.shape[-1] != 3 or not np.isfinite(prediction).all():
                    raise ValueError(f"Invalid RGB shape/values for {name} frame {frame_id}")
                predictions[name] = np.clip(prediction, 0, 1)
                rendering[name] = scalar_diagnostics(packet.get("layered_diagnostics", {}))
                if config is not None:
                    assert_conservation(rendering[name], identity["conservation_max_abs_tolerance"])
                rendering[name]["forward_ms"] = float(packet["forward_ms"])
                rendering[name]["unclipped_rgb_min"] = float(prediction.min())
                rendering[name]["unclipped_rgb_max"] = float(prediction.max())
                rendering[name]["clipped_rgb_fraction"] = float(((prediction < 0) | (prediction > 1)).mean())
                if reference is None:
                    reference = as_numpy(packet["target_rgb"]).astype(np.float32, copy=True)
                    # Fail closed: a missing mask must not silently label all
                    # pixels static or reuse a renderer-derived prediction mask.
                    for key in ("evaluation_dynamic", "evaluation_valid"):
                        if packet.get(key) is None:
                            raise ValueError(f"Missing reference {key}; boundary evaluation cannot use predicted masks")
                        value = as_numpy(packet[key])
                        if value.shape != reference.shape[:2] or value.dtype != np.bool_:
                            raise ValueError(f"Reference {key} must be a bool H x W mask")
                    dynamic = as_numpy(packet["evaluation_dynamic"]).copy()
                    valid = as_numpy(packet["evaluation_valid"]).copy()
                    if packet.get("evaluation_depth") is not None:
                        reference_depth = as_numpy(packet["evaluation_depth"]).copy()
                        if reference_depth.shape == (*reference.shape[:2], 1):
                            reference_depth = reference_depth[..., 0]
                if prediction.shape != reference.shape:
                    raise ValueError("All renderer variants must share the reference resolution")
                raw_difference = prediction[valid] - reference[valid]
                rendering[name]["preclip_squared_error_sum"] = float(np.square(raw_difference).sum(dtype=np.float64))
                rendering[name]["preclip_value_count"] = int(raw_difference.size)
                if not args.no_perceptual:
                    perceptual.update(perceptual_metrics(runtime.trainer, packet, name))
                if save_images:
                    mixing = packet.get("layered_diagnostic_maps", {}).get("mixing_fraction")
                    if mixing is not None:
                        diagnostic_previews[f"{name}_mixing_fraction"] = as_numpy(mixing).reshape(reference.shape[:2])
                    for component in ("diffuse", "specular"):
                        value = packet.get(f"layered_{component}_linear")
                        if value is not None:
                            diagnostic_previews[f"{name}_{component}_preview"] = component_preview(value)
                del packet
            evaluation_kwargs = {"evaluation_mask": valid}
            if reference_depth is not None:
                evaluation_kwargs["depth"] = reference_depth
            metric_row = evaluator.evaluate(reference, predictions, dynamic, frame_id,
                                            **evaluation_kwargs)
            metric_row.update(perceptual)
            row = {"frame_id": frame_id, "metrics": metric_row, "rendering": rendering}
            records.append(row)
            atomic_json(args.output / "frames" / f"{frame_id:06d}.json", row)
            if save_images:
                save_comparison(args.output, frame_id, reference, predictions, diagnostic_previews)
            summary = {
                "identity": identity, "status": "running", "frames": len(records),
                "metrics": summarize([record["metrics"] for record in records]),
                "rendering": diagnostic_summary(records, variants),
                "evaluation_claim": manifest["evaluation_claim"],
            }
            atomic_json(args.output / "summary.json", summary)
            manifest.update(completed_frames=len(records), last_completed_frame=frame_id,
                            elapsed_seconds=time.time() - started)
            atomic_json(args.output / "manifest.json", manifest)
            print(f"layered G-buffer: {index + 1}/{len(frames)} frames, frame={frame_id}, "
                  f"variants={len(variants)}", flush=True)
        if experiment_identity(args, variants) != identity:
            raise RuntimeError("Source checkpoint, config or implementation changed during evaluation")
        manifest.update(status="completed", elapsed_seconds=time.time() - started)
        summary["status"] = "completed"
        summary["elapsed_seconds"] = manifest["elapsed_seconds"]
        atomic_json(args.output / "summary.json", summary)
        atomic_json(args.output / "manifest.json", manifest)
        return summary
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                        elapsed_seconds=time.time() - started)
        atomic_json(args.output / "manifest.json", manifest)
        raise
    finally:
        if runtime is not None:
            runtime.trainer.configure_layered_gbuffer(None)
            runtime.reset()


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        run_experiment(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
