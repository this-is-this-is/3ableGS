#!/usr/bin/env python3
"""Preview fresh Base v4 / bounded teacher-calibration experiments.

The default is read-only. --execute trains B0 and C independently from dataset
initialization, then runs the original Base v4 image and temporal evaluation.
C is a small IIF-inspired teacher correction, not full Intrinsic Image Fusion.
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

DEFAULT_OUTPUT = ROOT / "output/pbr/teacher_calibration/scene788_from_scratch_v1"
OVERLAYS = ROOT / "configs/teacher_calibration"


def make_parser():
    parser = paired.make_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=DEFAULT_OUTPUT)
    return parser


def training_config(source, variant, steps, seed):
    from omegaconf import OmegaConf
    from models.teacher_calibration import validate_teacher_calibration_config
    if variant not in ("B0", "C"):
        raise ValueError("Expected B0 or C")
    cfg = copy.deepcopy(source)
    cfg.pop("log_dir", None)
    cfg["seed"] = seed
    cfg["trainer"]["optim"]["num_iters"] = steps
    overlay = OmegaConf.to_container(OmegaConf.load(OVERLAYS / f"{variant}.yaml"), resolve=True)
    cfg["trainer"]["teacher_calibration"] = validate_teacher_calibration_config(
        overlay["trainer"]["teacher_calibration"])
    cfg["logging"]["saveckpt_freq"] = steps
    cfg["render"].update(render_test=False, render_full=False, render_novel=None)
    return cfg


def build_plan(args):
    from omegaconf import OmegaConf
    from models.teacher_calibration import validate_teacher_calibration_config
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
    trainer = source["trainer"]
    calibration = validate_teacher_calibration_config(trainer.get("teacher_calibration"))
    if (trainer.get("rgb_joint") or calibration["enabled"]
            or (trainer["losses"].get("diffreg_material") or {}).get("enabled")):
        raise ValueError("Expected the original Base v4 source without RGB, DiffReg or teacher calibration")
    if source["render"].get("render_full") or source["render"].get("render_novel"):
        raise ValueError("Expected original recorded-camera test-only evaluation")
    test_ids = list(range(8, count, 8))
    train_ids = [frame for frame in range(count) if frame not in set(test_ids)]
    stages, configurations = [], {}
    for seed in args.seeds:
        for variant in ("B0", "C"):
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
            stages.append({"id": f"C_s{seed}_temporal", "kind": "temporal", "variant": "C",
                           "seed": seed, "checkpoint_from": f"C_s{seed}_train",
                           "baseline_from": f"B0_s{seed}_train"})
    source_paths = [args.source_config, OVERLAYS / "B0.yaml", OVERLAYS / "C.yaml"]
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
        "calibration_parameters": len(train_ids) * 10,
        "common_execution_changes": {
            "saveckpt_freq": args.steps,
            "evaluation_schedule": "final checkpoint then original tools.eval in an isolated stage",
        },
    }
    return {
        "identity": identity, "stages": stages, "comparison_variant": "C",
        "training_diagnostic_prefixes": ["calibration_stats/"],
        "training_diagnostic_loss_keys": ["losses/raw/teacher_calibration_identity_loss",
                                          "losses/raw/teacher_calibration_smooth_loss"],
        "interpretation": "Exploratory scene comparison. Full-path temporal diagnostics include training frames; no test-set checkpoint selection.",
        "restart_policy": "Completed stages are verified and skipped; unfinished stages restart in a new attempt from scratch.",
        "method": "IIF-inspired bounded per-training-view teacher gain/bias with camera-mean gauge, identity and neighboring-parameter priors; no additional render or inference correction",
    }


def stage_artifacts(stage, directory, *, inspect_checkpoint=True):
    from omegaconf import OmegaConf
    from models.teacher_calibration import validate_teacher_calibration_config
    if stage["kind"] not in ("train", "images"):
        return paired.stage_artifacts(stage, directory, inspect_checkpoint=inspect_checkpoint)
    cfg = OmegaConf.to_container(OmegaConf.load(directory / "config.yaml"), resolve=True)
    calibration = validate_teacher_calibration_config(cfg["trainer"].get("teacher_calibration"))
    if stage["variant"] not in ("B0", "C"):
        raise ValueError("Unknown teacher calibration arm")
    if calibration["enabled"] != (stage["variant"] == "C"):
        raise ValueError("Saved teacher calibration differs from planned arm")
    # Parent validation checks artifacts and rejects DiffReg contamination.
    # Load a large checkpoint only once, adding the calibration checks below.
    artifacts = paired.stage_artifacts(stage, directory, inspect_checkpoint=False)
    if stage["kind"] == "train" and inspect_checkpoint:
        import torch
        checkpoint = torch.load(directory / "checkpoint_final.pth", map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not checkpoint.get("models"):
            raise ValueError("Invalid scene checkpoint")
        if checkpoint.get("step") != stage["steps"]:
            raise ValueError("Checkpoint did not reach the requested final step")
        metadata = checkpoint.get("_teacher_calibration")
        prefix = "teacher_calibration_head."
        if calibration["enabled"]:
            data = cfg["data"]
            count = int(data["end_timestep"]) - int(data["start_timestep"]) + 1
            train_ids = [frame for frame in range(count) if frame == 0 or frame % 8]
            if (not isinstance(metadata, dict) or metadata.get("config") != calibration
                    or metadata.get("train_indices") != train_ids or metadata.get("num_cams") != 1):
                raise ValueError("Checkpoint teacher calibration config or training split mismatch")
            parameters = checkpoint.get(prefix + "raw")
            if (not torch.is_tensor(parameters) or parameters.shape != (len(train_ids), 2, 5)
                    or not parameters.is_floating_point() or not bool(torch.isfinite(parameters).all())):
                raise ValueError("Invalid teacher calibration parameters")
            from models.teacher_calibration import TeacherCalibration
            expected = TeacherCalibration(train_ids, 1, calibration).state_dict()
            actual_keys = {key[len(prefix):] for key in checkpoint if key.startswith(prefix)}
            if actual_keys != set(expected):
                raise ValueError("Incomplete teacher calibration state")
            for key, value in expected.items():
                if key != "raw":
                    saved = checkpoint[prefix + key]
                    if (not torch.is_tensor(saved) or saved.dtype != value.dtype
                            or not torch.equal(saved, value)):
                        raise ValueError("Teacher calibration structural buffers differ from saved split")
        elif metadata is not None or any(key.startswith(prefix) for key in checkpoint):
            raise ValueError("Baseline checkpoint contains teacher calibration")
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
