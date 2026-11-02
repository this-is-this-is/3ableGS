#!/usr/bin/env python3
"""Preview fresh Base v4 / lightweight RDG-inspired G-buffer fusion experiments.

Without --execute this only prints the plan. B0 and D start independently from
dataset initialization and keep the original Base v4 image evaluation protocol.
This is a material-fusion adaptation, not a reproduction of the RDG SR network.
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

DEFAULT_OUTPUT = ROOT / "output/pbr/rdg_fusion/scene788_from_scratch_v1"
OVERLAYS = ROOT / "configs/rdg_fusion"


def make_parser():
    parser = paired.make_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=DEFAULT_OUTPUT)
    return parser


def _reject_other_methods(trainer):
    losses = trainer.get("losses", {})
    if (trainer.get("rgb_joint") or trainer.get("gbuffer_attention") or trainer.get("layered_gbuffer")
            or (trainer.get("teacher_calibration") or {}).get("enabled")
            or (trainer.get("teacher_confidence") or {}).get("enabled")
            or (losses.get("diffreg_material") or {}).get("enabled")
            or (losses.get("temporal_material") or {}).get("w", 0)):
        raise ValueError("RDG fusion comparison requires Base v4 without other experimental adapters")


def training_config(source, variant, steps, seed):
    from omegaconf import OmegaConf
    from models.rdg_fusion import validate_rdg_config
    if variant not in ("B0", "D"):
        raise ValueError("Expected B0 or D")
    cfg = copy.deepcopy(source)
    cfg.pop("log_dir", None)
    cfg["seed"] = seed
    cfg["trainer"]["optim"]["num_iters"] = steps
    overlay = OmegaConf.to_container(OmegaConf.load(OVERLAYS / f"{variant}.yaml"), resolve=True)
    cfg["trainer"]["rdg_fusion"] = validate_rdg_config(overlay["trainer"]["rdg_fusion"])
    cfg["logging"]["saveckpt_freq"] = steps
    cfg["render"].update(render_test=False, render_full=False, render_novel=None)
    return cfg


def build_plan(args):
    from omegaconf import OmegaConf
    from models.rdg_fusion import validate_rdg_config
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
    if validate_rdg_config(source["trainer"].get("rdg_fusion"))["enabled"]:
        raise ValueError("Source already enables RDG fusion")
    if source["render"].get("render_full") or source["render"].get("render_novel"):
        raise ValueError("Expected original recorded-camera test-only evaluation")
    test_ids = list(range(8, count, 8))
    train_ids = [frame for frame in range(count) if frame not in set(test_ids)]
    stages, configurations = [], {}
    for seed in args.seeds:
        for variant in ("B0", "D"):
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
            stages.append({"id": f"D_s{seed}_temporal", "kind": "temporal", "variant": "D",
                           "seed": seed, "checkpoint_from": f"D_s{seed}_train",
                           "baseline_from": f"B0_s{seed}_train"})
    source_paths = [args.source_config, OVERLAYS / "B0.yaml", OVERLAYS / "D.yaml"]
    recipe = configurations[f"D_s{args.seeds[0]}_train"]["trainer"]["rdg_fusion"]
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
        "fusion": {
            "history_frames": 1, "feature_downscale": recipe["feature_downscale"],
            "start_step": recipe["start_step"], "ramp_steps": recipe["ramp_steps"],
            "training_interval": recipe["interval"],
            "train_history": "nearest preceding train frame in same camera, bounded by max_gap",
            "evaluation_history": "actual preceding recorded timestamp; independent of test stride",
            "history_source": "detached generated material; no RGB reference or material teacher",
            "inference_overhead": "one low-resolution context plus geometry, attention and static-mask rendering; unmeasured",
        },
        "common_execution_changes": {
            "saveckpt_freq": args.steps,
            "evaluation_schedule": "final checkpoint then original tools.eval in an isolated stage",
        },
    }
    return {
        "identity": identity, "stages": stages, "comparison_variant": "D",
        "training_diagnostic_prefixes": ["rdg_stats/"],
        "training_diagnostic_loss_keys": ["losses/raw/rdg_anchor", "losses/raw/rdg_temporal"],
        "interpretation": "Exploratory scene comparison. Full-path temporal diagnostics include training frames; no test-set checkpoint selection.",
        "restart_policy": "Completed stages are verified and skipped; unfinished stages restart in a new attempt from scratch.",
        "method": "RDG-inspired decoupled geometry/material attention over one low-resolution history; bounded low-frequency albedo fusion before current-view shading",
    }


def stage_artifacts(stage, directory, *, inspect_checkpoint=True):
    from omegaconf import OmegaConf
    from models.rdg_fusion import validate_rdg_config
    if stage["kind"] not in ("train", "images"):
        return paired.stage_artifacts(stage, directory, inspect_checkpoint=inspect_checkpoint)
    cfg = OmegaConf.to_container(OmegaConf.load(directory / "config.yaml"), resolve=True)
    fusion = validate_rdg_config(cfg["trainer"].get("rdg_fusion"))
    if stage["variant"] not in ("B0", "D"):
        raise ValueError("Unknown RDG fusion arm")
    if fusion["enabled"] != (stage["variant"] == "D"):
        raise ValueError("Saved RDG fusion differs from planned arm")
    _reject_other_methods(cfg["trainer"])
    artifacts = paired.stage_artifacts(stage, directory, inspect_checkpoint=False)
    if stage["kind"] == "train" and inspect_checkpoint:
        import torch
        from models.rdg_fusion import RDGFusion
        checkpoint = torch.load(directory / "checkpoint_final.pth", map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not checkpoint.get("models"):
            raise ValueError("Invalid scene checkpoint")
        if checkpoint.get("step") != stage["steps"]:
            raise ValueError("Checkpoint did not reach the requested final step")
        metadata, prefix = checkpoint.get("_rdg_fusion"), "rdg_fusion_head."
        if fusion["enabled"]:
            if (not isinstance(metadata, dict) or set(metadata) != {"config", "rdg_step"}
                    or validate_rdg_config(metadata.get("config")) != fusion
                    or metadata.get("rdg_step") != checkpoint["step"]):
                raise ValueError("Checkpoint RDG fusion config or step mismatch")
            # Construction for shape validation must not perturb caller RNG.
            with torch.random.fork_rng(devices=[]):
                expected = RDGFusion(fusion).state_dict()
            actual_keys = {key[len(prefix):] for key in checkpoint if key.startswith(prefix)}
            if actual_keys != set(expected):
                raise ValueError("Incomplete RDG fusion head state")
            for key, value in expected.items():
                saved = checkpoint[prefix + key]
                if (not torch.is_tensor(saved) or saved.shape != value.shape or saved.dtype != value.dtype
                        or not bool(torch.isfinite(saved).all())):
                    raise ValueError("Invalid RDG fusion head tensor")
        elif metadata is not None or any(key.startswith(prefix) for key in checkpoint):
            raise ValueError("Baseline checkpoint contains RDG fusion")
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
