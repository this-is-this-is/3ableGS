#!/usr/bin/env python3
"""Preview fresh Base v4 / frozen teacher-confidence experiments.

No child process or experiment directory is created without --execute. V builds
a train-only confidence cache once from its own warmup geometry, then reweights
the original material supervision. No pretrained geometry checkpoint is used.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_diffreg_experiments as paired

DEFAULT_OUTPUT = ROOT / "output/pbr/teacher_confidence/scene788_from_scratch_v1"
OVERLAYS = ROOT / "configs/teacher_confidence"


def make_parser():
    parser = paired.make_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=DEFAULT_OUTPUT)
    return parser


def source_crop_fraction(config):
    """Match CameraData.gbuffer_crop_frac without constructing a dataset."""
    data = config["data"]
    top = data["pixel_source"].get("gbuffer_crop_top", 0)
    if isinstance(top, bool) or not isinstance(top, int) or top < 0:
        raise ValueError("gbuffer_crop_top must be a nonnegative integer")
    if top == 0:
        return 0.
    from datasets.dataset_meta import DATASETS_CONFIG
    try:
        height = DATASETS_CONFIG[data["dataset"]][0]["original_size"][0]
    except (KeyError, TypeError) as exc:
        raise ValueError("Nonzero teacher crop requires known dataset/camera metadata") from exc
    if top >= height / 2:
        raise ValueError("Teacher crop must leave a nonempty image")
    return top / height


def _reject_other_methods(trainer):
    losses = trainer.get("losses", {})
    if (trainer.get("rgb_joint") or trainer.get("gbuffer_attention") or trainer.get("layered_gbuffer")
            or (trainer.get("teacher_calibration") or {}).get("enabled")
            or (losses.get("diffreg_material") or {}).get("enabled")
            or (losses.get("temporal_material") or {}).get("w", 0)):
        raise ValueError("Teacher confidence comparison requires Base v4 without other experimental adapters")


def training_config(source, variant, steps, seed):
    from omegaconf import OmegaConf
    from models.teacher_confidence import validate_teacher_confidence_config
    if variant not in ("B0", "V"):
        raise ValueError("Expected B0 or V")
    cfg = copy.deepcopy(source)
    cfg.pop("log_dir", None)
    cfg["seed"] = seed
    cfg["trainer"]["optim"]["num_iters"] = steps
    overlay = OmegaConf.to_container(OmegaConf.load(OVERLAYS / f"{variant}.yaml"), resolve=True)
    cfg["trainer"]["teacher_confidence"] = validate_teacher_confidence_config(
        overlay["trainer"]["teacher_confidence"])
    cfg["logging"]["saveckpt_freq"] = steps
    cfg["render"].update(render_test=False, render_full=False, render_novel=None)
    return cfg


def build_plan(args):
    from omegaconf import OmegaConf
    from models.teacher_confidence import validate_teacher_confidence_config
    if args.steps < 1 or not args.seeds or min(args.seeds) < 0:
        raise ValueError("Positive final step and nonnegative seeds are required")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise ValueError("min-free-gb must be finite and nonnegative")
    args.source_config = args.source_config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    source = OmegaConf.to_container(OmegaConf.load(args.source_config), resolve=True)
    count = paired.validate_source(source)
    _reject_other_methods(source["trainer"])
    if validate_teacher_confidence_config(source["trainer"].get("teacher_confidence"))["enabled"]:
        raise ValueError("Source already enables teacher confidence")
    if source["render"].get("render_full") or source["render"].get("render_novel"):
        raise ValueError("Expected original recorded-camera test-only evaluation")
    crop_frac = source_crop_fraction(source)
    test_ids = list(range(8, count, 8))
    train_ids = [frame for frame in range(count) if frame not in set(test_ids)]
    stages, configurations = [], {}
    for seed in args.seeds:
        for variant in ("B0", "V"):
            arm = f"{variant}_s{seed}"
            train_id = arm + "_train"
            configurations[train_id] = training_config(source, variant, args.steps, seed)
            stages.extend([
                {"id": train_id, "kind": "train", "variant": variant, "seed": seed,
                 "steps": args.steps},
                {"id": arm + "_images", "kind": "images", "variant": variant,
                 "seed": seed, "checkpoint_from": train_id},
            ])
        if not args.skip_temporal:
            stages.append({"id": f"V_s{seed}_temporal", "kind": "temporal", "variant": "V",
                           "seed": seed, "checkpoint_from": f"V_s{seed}_train",
                           "baseline_from": f"B0_s{seed}_train"})
    source_paths = [args.source_config, OVERLAYS / "B0.yaml", OVERLAYS / "V.yaml"]
    recipe = configurations[f"V_s{args.seeds[0]}_train"]["trainer"]["teacher_confidence"]
    identity = {
        "schema_version": 1,
        "options": {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key not in ("execute", "status")},
        "source_config": source,
        "source_sha256": {str(path): paired.digest(path) for path in source_paths},
        "code_sha256": paired.code_identity(), "input_stat": paired.input_identity(source),
        "training_configs": configurations, "python": sys.executable,
        "initialization": "fresh dataset initialization; no scene checkpoint loaded",
        "train_ids": train_ids, "test_ids": test_ids,
        "updates_per_training": args.steps + 1,
        "confidence_cache": {
            "geometry_source": f"same fresh model before optimizer update {recipe['build_step']}; no pretrained geometry",
            "build_step": recipe["build_step"], "downscale": recipe["downscale"],
            "train_views": len(train_ids), "crop_frac": crop_frac,
            "planned_extra_renders": len(train_ids) if args.steps >= recipe["build_step"] else 0,
            "additional_learned_parameters": 0,
            "storage": "CPU float16 HxWx1 maps embedded in final checkpoint",
        },
        "common_execution_changes": {
            "saveckpt_freq": args.steps,
            "evaluation_schedule": "final checkpoint then original tools.eval in an isolated stage",
        },
    }
    return {
        "identity": identity, "stages": stages, "comparison_variant": "V",
        "training_diagnostic_prefixes": ["confidence_stats/"],
        "training_diagnostic_loss_keys": ["losses/raw/albedo_loss", "losses/raw/roughness_loss",
                                          "losses/raw/metallic_loss"],
        "interpretation": "Exploratory scene comparison. Full-path temporal diagnostics include training frames; no test-set checkpoint selection.",
        "restart_policy": "Completed stages are verified and skipped; unfinished stages restart in a new attempt from scratch.",
        "method": "SVGF/NRD-inspired frozen train-teacher confidence from geometric history and material moments; normalized static material weights, original inference renderer",
    }


def stage_artifacts(stage, directory, *, inspect_checkpoint=True):
    from omegaconf import OmegaConf
    from models.teacher_confidence import validate_teacher_confidence_config
    if stage["kind"] not in ("train", "images"):
        return paired.stage_artifacts(stage, directory, inspect_checkpoint=inspect_checkpoint)
    cfg = OmegaConf.to_container(OmegaConf.load(directory / "config.yaml"), resolve=True)
    confidence = validate_teacher_confidence_config(cfg["trainer"].get("teacher_confidence"))
    if stage["variant"] not in ("B0", "V"):
        raise ValueError("Unknown teacher confidence arm")
    if confidence["enabled"] != (stage["variant"] == "V"):
        raise ValueError("Saved teacher confidence differs from planned arm")
    _reject_other_methods(cfg["trainer"])
    artifacts = paired.stage_artifacts(stage, directory, inspect_checkpoint=False)
    if stage["kind"] == "train" and inspect_checkpoint:
        import torch
        from models.teacher_confidence_runtime import validate_teacher_confidence_state
        checkpoint = torch.load(directory / "checkpoint_final.pth", map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not checkpoint.get("models"):
            raise ValueError("Invalid scene checkpoint")
        if checkpoint.get("step") != stage["steps"]:
            raise ValueError("Checkpoint did not reach the requested final step")
        data = cfg["data"]
        count = int(data["end_timestep"]) - int(data["start_timestep"]) + 1
        train_ids = [frame for frame in range(count) if frame == 0 or frame % 8]
        validate_teacher_confidence_state(
            checkpoint.get("_teacher_confidence"), config=confidence,
            train_indices=train_ids, num_cams=1, crop_frac=source_crop_fraction(cfg),
            checkpoint_step=checkpoint["step"],
        )
    return artifacts


def run_queue(args, plan, *, child_runner=paired.run_child, artifact_reader=stage_artifacts):
    return paired.run_queue(args, plan, child_runner=child_runner, artifact_reader=artifact_reader)


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.status:
        path = args.output_dir.expanduser().resolve() / "state.json"
        if not path.is_file():
            parser.error(f"No saved queue state: {path}")
        print(path.read_text(), end="")
        return

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Queue interrupted; repeat the same command to continue")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        run_queue(args, build_plan(args))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
