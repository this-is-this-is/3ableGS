#!/usr/bin/env python3
"""Supplemental temporal diagnostics using the original Base v4 render policy.

This full trajectory includes training observations. Official held-out image
metrics remain the responsibility of tools/eval.py. No pooled PSNR here is
presented as the original Base v4 test PSNR.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import os
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_gbuffer_attention import atomic_json, code_identity, file_digest, load_scene


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flow-cache", type=Path)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--streaming-cache", action="store_true",
                        help="Enable the trainer's bounded generated-context cache; report this timing mode")
    return parser


def validate_args(args):
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.baseline_checkpoint is not None:
        args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    for checkpoint in (args.checkpoint, args.baseline_checkpoint):
        if checkpoint is not None:
            for path in (checkpoint, checkpoint.parent / "config.yaml"):
                if not path.is_file():
                    raise FileNotFoundError(path)
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f"Output is not empty: {args.output_dir}; choose a new directory")
    if args.flow_cache is not None:
        args.flow_cache = args.flow_cache.expanduser().resolve()
    return args


def protocol_signature(config):
    """Fields that must match in paired temporal evaluation."""
    from omegaconf import OmegaConf
    data = OmegaConf.to_container(config.data, resolve=True)
    # Storage/preloading/teacher availability never defines inference quality.
    data.pop("preload_device", None)
    pixel = data["pixel_source"]
    for key in ("load_gbuffer", "gbuffer_path", "gbuffer_seq_id", "load_weather"):
        pixel.pop(key, None)
    return {"data": data,
            "renderer": OmegaConf.to_container(config.trainer.render, resolve=True),
            "affine_policy": "original frame-specific training / mean-embedding test",
            "resolution": "saved native resolution; additional downscale=1"}


def train_to_test_records(records, train_frames, test_frames):
    train, test = set(map(int, train_frames)), set(map(int, test_frames))
    return [record for record in records if record.get("temporal_has_pair")
            and int(record["frame_id"]) in test and int(record["frame_id"]) - 1 in train]


def summarize(records, train_frames, test_frames):
    from utils.temporal_metrics import summarize_metric_records
    result = {
        "scope": "full-sequence reconstruction diagnostic; includes training frames",
        "official_image_metrics": "Run tools/eval.py on the saved test split",
        "temporal": summarize_metric_records(records),
        "train_to_test": summarize_metric_records(train_to_test_records(records, train_frames, test_frames)),
    }
    for key in ("prediction_render_ms", "baseline_render_ms", "history_coverage", "history_weight"):
        values = [record[key] for record in records if record.get(key) is not None]
        result[f"mean_{key}"] = sum(values) / len(values) if values else None
    return result


def _frame(trainer, dataset, index):
    import torch
    from tools.render_temporal_material import render_inputs, to_device
    image_by_weather, camera_by_weather = dataset.full_image_set.get_image(index, 1., ["raw"])
    info, camera = image_by_weather["raw"], camera_by_weather["raw"]
    frame_id = int(info["frame_idx"].flatten()[0])
    inputs = render_inputs(info, trainer.device, torch)
    cameras = to_device(camera, trainer.device, torch)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        outputs = trainer(inputs, cameras, "raw")
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    return frame_id, info, outputs, elapsed


def _freeze(trainer, streaming):
    trainer.set_eval()
    trainer.requires_grad_(False)
    for model in trainer.models.values():
        model.requires_grad_(False)
    # Skip visual class passes only; full RGB and its original transforms stay identical.
    trainer._suppress_eval_layers = True
    trainer.set_gbuffer_attention_cache(enabled=streaming)


def run(args):
    import numpy as np
    import torch
    from tools.render_temporal_material import evaluation_static_mask, rgb_array
    from utils.temporal_metrics import TemporalEvaluator

    args = validate_args(args)
    os.chdir(ROOT)
    from omegaconf import OmegaConf
    right = OmegaConf.load(args.checkpoint.parent / "config.yaml")
    configs = [right]
    if args.baseline_checkpoint is not None:
        left = OmegaConf.load(args.baseline_checkpoint.parent / "config.yaml")
        configs.append(left)
        if protocol_signature(left) != protocol_signature(right):
            raise ValueError("Baseline and prediction data/render protocols differ")
    if args.streaming_cache and any((cfg.trainer.get("rdg_fusion") or {}).get("enabled") for cfg in configs):
        raise ValueError("RDG fusion generates fresh history and does not support --streaming-cache")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {"schema": 1, "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": file_digest(args.checkpoint),
                "config_sha256": file_digest(args.checkpoint.parent / "config.yaml"),
                "code": {**code_identity(), "tools/eval_gbuffer_attention.py": file_digest(__file__)},
                "streaming_cache": args.streaming_cache}
    if args.baseline_checkpoint is not None:
        identity.update(baseline_checkpoint=str(args.baseline_checkpoint),
                        baseline_sha256=file_digest(args.baseline_checkpoint),
                        baseline_config_sha256=file_digest(args.baseline_checkpoint.parent / "config.yaml"))
    manifest = {"identity": identity, "status": "rendering",
                "timing_mode": "streaming generated-context cache" if args.streaming_cache
                               else "random access including history re-rendering",
                "timing_inclusions": "Forward, including generated-context loading, resizing, device transfers and rendering",
                "timing_exclusions": "Current-view loading/device transfers, temporary baseline npz I/O, flow and PNG encoding",
                "limitations": ["Full trajectory contains training frames; not a contiguous held-out benchmark",
                                "Reference optical flow is an estimate; masks never use learned acceptance"]}
    atomic_json(args.output_dir / "manifest.json", manifest)
    records, baseline_protocol = [], None
    # Baseline and prediction models are never simultaneously on the GPU.
    # Temporary float32 packets preserve metrics without PNG quantization.
    with tempfile.TemporaryDirectory(prefix=".baseline_", dir=args.output_dir) as directory:
        temporary = Path(directory)
        if args.baseline_checkpoint is not None:
            cfg, dataset, trainer = load_scene(args.baseline_checkpoint)
            baseline_protocol = protocol_signature(cfg)
            _freeze(trainer, args.streaming_cache)
            for index in range(len(dataset.full_image_set)):
                frame_id, infos, outputs, elapsed = _frame(trainer, dataset, index)
                reference = rgb_array(infos["pixels"])
                np.savez_compressed(temporary / f"{frame_id:06d}.npz",
                                    rgb=rgb_array(outputs["rgb"]), reference=reference,
                                    render_ms=np.asarray(elapsed))
            del cfg, dataset, trainer, infos, outputs, reference
            gc.collect()
            torch.cuda.empty_cache()
        cfg, dataset, trainer = load_scene(args.checkpoint)
        protocol = protocol_signature(cfg)
        if baseline_protocol is not None and baseline_protocol != protocol:
            raise ValueError("Baseline and prediction data/render protocols differ")
        _freeze(trainer, args.streaming_cache)
        torch.cuda.reset_peak_memory_stats()
        train_frames = list(map(int, dataset.train_timesteps))
        test_frames = list(map(int, dataset.test_timesteps))
        manifest.update(protocol=protocol, train_frames=train_frames, test_frames=test_frames)
        atomic_json(args.output_dir / "manifest.json", manifest)
        evaluator = TemporalEvaluator(
            cache_dir=args.flow_cache or args.output_dir / "flow_cache",
            metadata={"protocol": protocol, "camera": 0},
        )
        with (args.output_dir / "frames.jsonl").open("w") as stream:
            for index in range(len(dataset.full_image_set)):
                frame_id, infos, outputs, elapsed = _frame(trainer, dataset, index)
                reference = rgb_array(infos["pixels"])
                predictions = {"prediction": rgb_array(outputs["rgb"])}
                baseline_ms = None
                if args.baseline_checkpoint is not None:
                    with np.load(temporary / f"{frame_id:06d}.npz", allow_pickle=False) as prior:
                        if not np.array_equal(reference, prior["reference"]):
                            raise ValueError(f"Paired reference RGB differs at frame {frame_id}")
                        predictions["baseline"] = prior["rgb"].copy()
                        baseline_ms = float(prior["render_ms"])
                record = evaluator.evaluate(reference, predictions, evaluation_static_mask(infos), frame_id)
                valid = outputs.get("attention_valid", outputs.get("rdg_valid"))
                weights = outputs.get("attention_history_weight", outputs.get("rdg_history_weight"))
                record.update(prediction_render_ms=elapsed, baseline_render_ms=baseline_ms,
                              split="test" if frame_id in set(test_frames) else "train",
                              history_coverage=float(valid.float().mean()) if valid is not None else 0.,
                              history_weight=float(weights.mean()) if weights is not None else 0.,
                              context_ids=list(outputs.get("attention_context_ids", outputs.get("rdg_context_ids", []))))
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if args.save_images:
                    from PIL import Image
                    for name, values in {**predictions, "reference": reference}.items():
                        folder = args.output_dir / "images" / name
                        folder.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(np.rint(values.clip(0, 1) * 255).astype(np.uint8)).save(folder / f"{frame_id:06d}.png")
                print(json.dumps({"frame_id": frame_id, "prediction_render_ms": elapsed,
                                  "temporal": record.get("prediction_temporal_residual_l1")}), flush=True)
        summary = summarize(records, train_frames, test_frames)
        summary.update(protocol=protocol, timing_mode=manifest["timing_mode"],
                       peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                       flow_config=evaluator.config)
        atomic_json(args.output_dir / "summary.json", summary)
    manifest.update(status="complete", frames=len(records),
                    summary_sha256=file_digest(args.output_dir / "summary.json"),
                    frames_sha256=file_digest(args.output_dir / "frames.jsonl"))
    atomic_json(args.output_dir / "manifest.json", manifest)
    return summary


def main(argv=None):
    args = make_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
