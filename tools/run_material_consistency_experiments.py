#!/usr/bin/env python3
"""Preview paired fresh Base v4 / sparse material-consistency experiments.

No process or output directory is created without --execute. B0 and T use
dataset initialization, original Base v4 image evaluation and paired temporal
diagnostics. T is a sampled train-only adaptation of MaterialRefGS, not the
original paper's complete renderer or patch-homography objective.
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

DEFAULT_OUTPUT = ROOT / "output/pbr/material_consistency/scene788_from_scratch_v1"
OVERLAYS = ROOT / "configs/material_consistency"


def make_parser():
    parser = paired.make_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=DEFAULT_OUTPUT)
    return parser


def training_config(source, variant, steps, seed):
    from omegaconf import OmegaConf
    from models.temporal_training import validate_temporal_training_config
    if variant not in ("B0", "T"):
        raise ValueError("Expected B0 or T")
    cfg = copy.deepcopy(source)
    cfg.pop("log_dir", None)
    cfg["seed"] = seed
    cfg["trainer"]["optim"]["num_iters"] = steps
    overlay = OmegaConf.to_container(OmegaConf.load(OVERLAYS / f"{variant}.yaml"), resolve=True)
    temporal = overlay["trainer"]["losses"]["temporal_material"]
    cfg["trainer"]["losses"]["temporal_material"] = validate_temporal_training_config(temporal)
    cfg["logging"]["saveckpt_freq"] = steps
    cfg["render"].update(render_test=False, render_full=False, render_novel=None)
    return cfg


def build_plan(args):
    from omegaconf import OmegaConf
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
    if trainer.get("rgb_joint") or (trainer["losses"].get("diffreg_material") or {}).get("enabled"):
        raise ValueError("Expected the unmodified Base v4 source without RGB or DiffReg adapters")
    if source["render"].get("render_full") or source["render"].get("render_novel"):
        raise ValueError("Expected the original recorded-camera test-only evaluation configuration")
    test_ids = list(range(8, count, 8))
    stages, configurations = [], {}
    for seed in args.seeds:
        for variant in ("B0", "T"):
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
            stages.append({"id": f"T_s{seed}_temporal", "kind": "temporal", "variant": "T",
                           "seed": seed, "checkpoint_from": f"T_s{seed}_train",
                           "baseline_from": f"B0_s{seed}_train"})
    source_paths = [args.source_config, OVERLAYS / "B0.yaml", OVERLAYS / "T.yaml"]
    identity = {
        "schema_version": 1,
        "options": {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key not in ("execute", "status")},
        "source_config": source,
        "source_sha256": {str(path): paired.digest(path) for path in source_paths},
        "code_sha256": paired.code_identity(), "input_stat": paired.input_identity(source),
        "training_configs": configurations, "python": sys.executable,
        "initialization": "fresh dataset initialization; no scene checkpoint loaded",
        "train_ids": [frame for frame in range(count) if frame not in set(test_ids)],
        "test_ids": test_ids, "updates_per_training": args.steps + 1,
        "common_execution_changes": {
            "saveckpt_freq": args.steps,
            "evaluation_schedule": "final checkpoint then original tools.eval in an isolated stage",
        },
        "pairing": "same-camera previous training view within two frame indices; no held-out targets",
    }
    return {
        "identity": identity, "stages": stages, "comparison_variant": "T",
        "training_diagnostic_prefixes": ["temporal_stats/"],
        "training_diagnostic_loss_keys": ["losses/raw/temporal_material_loss"],
        "interpretation": "Exploratory scene comparison. Full-path temporal diagnostics include training frames; no test-set checkpoint selection.",
        "restart_policy": "Completed stages are verified and skipped; unfinished stages restart in a new attempt from scratch.",
        "method": "MaterialRefGS-inspired sparse train-only material reprojection consistency; one detached neighbor every eight eligible steps, no inference modification",
    }


def stage_artifacts(stage, directory, *, inspect_checkpoint=True):
    if stage["kind"] in ("train", "images"):
        from omegaconf import OmegaConf
        from models.temporal_training import validate_temporal_training_config
        cfg = OmegaConf.to_container(OmegaConf.load(directory / "config.yaml"), resolve=True)
        temporal = validate_temporal_training_config(cfg["trainer"]["losses"].get("temporal_material"))
        if stage["variant"] not in ("B0", "T"):
            raise ValueError("Unknown material consistency arm")
        if (temporal["w"] > 0) != (stage["variant"] == "T"):
            raise ValueError("Saved temporal regularizer differs from planned arm")
        if stage["variant"] == "T" and (temporal["num_samples"] <= 0 or temporal["interval"] != 8):
            raise ValueError("T requires the sampled every-eight-step objective")
    # This also rejects an accidentally enabled DiffReg loss in either arm.
    return paired.stage_artifacts(stage, directory, inspect_checkpoint=inspect_checkpoint)


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
