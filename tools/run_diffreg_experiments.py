#!/usr/bin/env python3
"""Train paired Base v4 / local DiffReg scenes from scratch and evaluate them.

Default invocation is a read-only preview. --execute runs B0 and R with the
same seed and original Base v4 data/optimization settings. No pretrained scene
weights are read. Repeating the command verifies and skips completed stages;
an unfinished stage restarts in a new attempt, not an optimizer-state resume.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_gbuffer_attention_experiments import (
    atomic_json, code_identity, digest, input_identity, run_child, validate_source,
    verify_artifacts,
)

DEFAULT_SOURCE = ROOT / "output/pbr/rebuild/scene_788_100f/base_v4_raw/config.yaml"
DEFAULT_OUTPUT = ROOT / "output/pbr/diffreg_material/scene788_from_scratch_v1"
OVERLAYS = ROOT / "configs/diffreg_material"

# These directories are produced inside a queue-owned stage attempt and are
# never inputs to the final image/temporal evaluations.  Keep the names
# explicit: cleanup must not walk the source teacher directory or infer targets
# from broad filename patterns such as "normal" or "material".
INTERMEDIATE_GBUFFER_DIRS = (
    "buffer_maps",
    "gbuffer_cache",
    "gbuffer_caches",
    "gbuffer_frames",
    "gbuffers",
    "generated_gbuffers",
    "intermediate_gbuffers",
)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=30000,
                        help="Inclusive final step: 30000 means 30001 optimizer updates")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--skip-temporal", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=20.,
                        help="Require this many GiB free before starting each stage")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status", action="store_true", help="Read saved queue state without executing")
    return parser


def training_config(source, variant, steps, seed):
    from omegaconf import OmegaConf
    if variant not in ("B0", "R"):
        raise ValueError("Expected B0 or R")
    cfg = copy.deepcopy(source)
    cfg.pop("log_dir", None)
    cfg["seed"] = seed
    cfg["trainer"]["optim"]["num_iters"] = steps
    overlay = OmegaConf.to_container(OmegaConf.load(OVERLAYS / f"{variant}.yaml"), resolve=True)
    cfg["trainer"]["losses"]["diffreg_material"] = overlay["trainer"]["losses"]["diffreg_material"]
    # Keep storage bounded and separate failures in evaluation from training.
    # These common changes do not change the training updates or test protocol.
    cfg["logging"]["saveckpt_freq"] = steps
    cfg["render"]["render_test"] = False
    cfg["render"]["render_full"] = False
    cfg["render"]["render_novel"] = None
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
    count = validate_source(source)
    trainer = source["trainer"]
    if trainer.get("rgb_joint") or trainer["losses"].get("diffreg_material", {}).get("enabled"):
        raise ValueError("Expected the unmodified Base v4 source, without RGB or DiffReg adapters")
    if source["render"].get("render_full") or source["render"].get("render_novel"):
        raise ValueError("Expected the original recorded-camera test-only evaluation configuration")
    test_ids = list(range(8, count, 8))
    stages, configurations = [], {}
    for seed in args.seeds:
        for variant in ("B0", "R"):
            arm = f"{variant}_s{seed}"
            train = arm + "_train"
            configurations[train] = training_config(source, variant, args.steps, seed)
            stages.extend([
                {"id": train, "kind": "train", "variant": variant, "seed": seed,
                 "steps": args.steps},
                {"id": arm + "_images", "kind": "images", "variant": variant, "seed": seed,
                 "checkpoint_from": train},
            ])
        if not args.skip_temporal:
            stages.append({"id": f"R_s{seed}_temporal", "kind": "temporal", "variant": "R",
                           "seed": seed, "checkpoint_from": f"R_s{seed}_train",
                           "baseline_from": f"B0_s{seed}_train"})
    source_paths = [args.source_config, OVERLAYS / "B0.yaml", OVERLAYS / "R.yaml"]
    identity = {
        "schema_version": 1,
        "options": {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key not in ("execute", "status")},
        "source_config": source,
        "source_sha256": {str(path): digest(path) for path in source_paths},
        "code_sha256": code_identity(), "input_stat": input_identity(source),
        "training_configs": configurations, "python": sys.executable,
        "initialization": "fresh dataset initialization; no scene checkpoint loaded",
        "train_ids": [frame for frame in range(count) if frame not in set(test_ids)],
        "test_ids": test_ids,
        "updates_per_training": args.steps + 1,
        "common_execution_changes": {
            "saveckpt_freq": args.steps,
            "evaluation_schedule": "final checkpoint then original tools.eval in an isolated stage",
        },
    }
    return {
        "identity": identity, "stages": stages,
        "interpretation": "Exploratory scene comparison. Full-path temporal diagnostics include training frames; no test-set checkpoint selection.",
        "restart_policy": "Completed stages are verified and skipped; unfinished stages restart in a new attempt from scratch.",
        "method": "DiffReg-inspired sampled local material relation regularizer; no temporal attention or inference modification",
    }


def stage_directory(output, stage_id, attempt):
    return Path(output) / "stages" / stage_id / f"attempt_{attempt:03d}"


def checkpoint_for(stage_id, plan, state, preview=False):
    attempts = state.get("stages", {}).get(stage_id, {}).get("attempts", [])
    if not attempts:
        if not preview:
            raise ValueError(f"Dependency has no completed checkpoint: {stage_id}")
        return stage_directory(plan["identity"]["options"]["output_dir"], stage_id, 1) / "checkpoint_final.pth"
    if attempts[-1]["status"] != "completed" and not preview:
        raise ValueError(f"Dependency is incomplete: {stage_id}")
    return Path(attempts[-1]["directory"]) / "checkpoint_final.pth"


def stage_command(stage, directory, plan, state, *, preview=False):
    output = Path(plan["identity"]["options"]["output_dir"])
    python = plan["identity"]["python"]
    if stage["kind"] == "train":
        return [python, "-u", "-m", "tools.train", "--config_file",
                str(output / "configs" / (stage["id"] + ".yaml")), "--output_root",
                str(output / "stages"), "--project", stage["id"], "--run_name", directory.name]
    if stage["kind"] == "images":
        return [python, "-u", "-m", "tools.eval", "--resume_from",
                str(directory / "checkpoint_final.pth"), "--eval_weathers"]
    return [python, "-u", "-m", "tools.eval_gbuffer_attention", "--checkpoint",
            str(checkpoint_for(stage["checkpoint_from"], plan, state, preview)),
            "--baseline-checkpoint", str(checkpoint_for(stage["baseline_from"], plan, state, preview)),
            "--output-dir", str(directory), "--flow-cache", str(output / "reference_flow")]


def prepare_image_snapshot(checkpoint, directory):
    from omegaconf import OmegaConf
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "checkpoint_final.pth"
    try:
        os.link(checkpoint, target)
    except OSError:
        shutil.copy2(checkpoint, target)
    cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
    cfg.log_dir = str(directory)
    cfg.render.render_test = True
    OmegaConf.save(cfg, directory / "config.yaml")


def original_metrics(directory):
    files = sorted((directory / "metrics_eval/raw").glob("images_test_*.json"))
    if len(files) != 1:
        raise ValueError("Expected exactly one original test metric JSON")
    metrics = json.loads(files[0].read_text())
    for key in ("psnr", "ssim", "lpips"):
        value = metrics.get("image_metrics/raw/test/" + key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid original image metric {key}")
    return files[0], metrics


def stage_artifacts(stage, directory, *, inspect_checkpoint=True):
    from omegaconf import OmegaConf
    if stage["kind"] in ("train", "images"):
        files = [directory / "checkpoint_final.pth", directory / "config.yaml"]
        if any(not path.is_file() for path in files):
            raise ValueError("Stage did not produce a scene checkpoint and configuration")
        config = OmegaConf.to_container(OmegaConf.load(files[1]), resolve=True)
        enabled = config["trainer"]["losses"].get("diffreg_material", {}).get("enabled", False)
        if enabled != (stage["variant"] == "R"):
            raise ValueError("Saved regularizer differs from the planned arm")
        if stage["kind"] == "train" and inspect_checkpoint:
            import torch
            checkpoint = torch.load(files[0], map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict) or not checkpoint.get("models"):
                raise ValueError("Invalid scene checkpoint")
            if checkpoint.get("step") != stage["steps"]:
                raise ValueError("Checkpoint did not reach the requested final step")
            del checkpoint
        if stage["kind"] == "images":
            files.append(original_metrics(directory)[0])
        elif (directory / "metrics.json").is_file():
            files.append(directory / "metrics.json")
    else:
        files = [directory / name for name in ("summary.json", "frames.jsonl", "manifest.json")]
        if any(not path.is_file() for path in files):
            raise ValueError("Temporal diagnostic is missing required artifacts")
        manifest = json.loads(files[2].read_text())
        if manifest.get("status") not in ("complete", "completed"):
            raise ValueError("Temporal diagnostic is incomplete")
        summary = json.loads(files[0].read_text())
        temporal = summary.get("temporal", {})
        for key in ("baseline_temporal_residual_l1", "prediction_temporal_residual_l1"):
            value = temporal.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid paired temporal metric: {key}")
    return {str(path.relative_to(directory)): digest(path) for path in files}


def verify_saved_configuration(stage, directory, plan):
    from omegaconf import OmegaConf
    if stage["kind"] not in ("train", "images"):
        return
    train_id = stage["id"] if stage["kind"] == "train" else stage["checkpoint_from"]
    expected = copy.deepcopy(plan["identity"]["training_configs"][train_id])
    if stage["kind"] == "images":
        expected["render"]["render_test"] = True
    saved = OmegaConf.to_container(OmegaConf.load(directory / "config.yaml"), resolve=True)
    saved.pop("log_dir", None)
    if saved != expected:
        raise ValueError("Saved configuration changed the planned training/evaluation protocol")


def verify_inputs(plan):
    identity = plan["identity"]
    if code_identity() != identity["code_sha256"]:
        raise ValueError("Code changed; use a new output directory")
    if any(not Path(name).is_file() or digest(name) != expected
           for name, expected in identity["source_sha256"].items()):
        raise ValueError("Source configuration or loss overlay changed")
    if input_identity(identity["source_config"]) != identity["input_stat"]:
        raise ValueError("Scene or teacher inputs changed")


def _finite_or_none(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _tree_usage(path):
    """Return file count and apparent bytes without following symlinks."""
    if path.is_symlink() or path.is_file():
        return 1, path.lstat().st_size
    files = bytes_used = 0
    for item in path.rglob("*"):
        if item.is_symlink() or item.is_file():
            files += 1
            bytes_used += item.lstat().st_size
    return files, bytes_used


def cleanup_intermediate_gbuffers(output, state):
    """Remove queue-owned G-buffer caches after every evaluation has passed.

    Only explicitly named direct children of recorded attempt directories are
    eligible.  In particular, source teacher G-buffers, checkpoints, rendered
    results, metric files and the shared reference-flow cache are out of scope.
    """
    output = Path(output).resolve()
    stages_root = (output / "stages").resolve()
    removed, file_count, bytes_used = [], 0, 0
    for stage_state in state.get("stages", {}).values():
        for attempt in stage_state.get("attempts", []):
            directory = Path(attempt["directory"]).resolve()
            try:
                directory.relative_to(stages_root)
            except ValueError as exc:
                raise ValueError(f"Refusing cleanup outside the experiment stages: {directory}") from exc
            for name in INTERMEDIATE_GBUFFER_DIRS:
                candidate = directory / name
                if not candidate.exists() and not candidate.is_symlink():
                    continue
                files, size = _tree_usage(candidate)
                if candidate.is_symlink() or candidate.is_file():
                    candidate.unlink()
                else:
                    shutil.rmtree(candidate)
                removed.append(str(candidate.relative_to(output)))
                file_count += files
                bytes_used += size
    return {
        "status": "completed",
        "completed_at": time.time(),
        "directory_names": list(INTERMEDIATE_GBUFFER_DIRS),
        "removed_directories": removed,
        "removed_files": file_count,
        "removed_bytes": bytes_used,
        "preserved": [
            "source teacher G-buffers",
            "checkpoints and configurations",
            "rendered images and videos",
            "image and temporal metrics",
            "reference flow cache",
        ],
    }


def write_summary(plan, state, output):
    comparison_variant = plan.get("comparison_variant", "R")
    diagnostic_prefixes = tuple(plan.get("training_diagnostic_prefixes", ["diffreg_stats/"]))
    diagnostic_loss_keys = set(plan.get("training_diagnostic_loss_keys", ["losses/raw/diffreg_material_loss"]))
    rows, temporal, durations, diagnostics = [], {}, {}, {}
    for stage in plan["stages"]:
        attempts = state["stages"][stage["id"]]["attempts"]
        if not attempts or attempts[-1]["status"] != "completed":
            continue
        attempt = attempts[-1]
        directory = Path(attempt["directory"])
        durations[stage["id"]] = attempt["completed_at"] - attempt["started_at"]
        if stage["kind"] == "images":
            path, metrics = original_metrics(directory)
            rows.append({"variant": stage["variant"], "seed": stage["seed"],
                         **{key.split("/")[-1]: _finite_or_none(value) for key, value in metrics.items()},
                         "metrics_file": str(path)})
        elif stage["kind"] == "temporal":
            temporal[str(stage["seed"])] = json.loads((directory / "summary.json").read_text())
        elif (directory / "metrics.json").is_file():
            last = {}
            with (directory / "metrics.json").open() as handle:
                for line in handle:
                    if line.strip():
                        last = json.loads(line)
            diagnostics[stage["id"]] = {
                "last_logged_iteration": last.get("iteration"),
                "last_logged_values": {key: _finite_or_none(value) for key, value in last.items()
                                       if key.startswith(diagnostic_prefixes)
                                       or key in diagnostic_loss_keys},
                "note": "Values use the original training logger's averaging window; they are not a new full-run average.",
                "metrics_file": str(directory / "metrics.json"),
            }
    comparisons = []
    for seed in plan["identity"]["options"]["seeds"]:
        arms = {row["variant"]: row for row in rows if row["seed"] == seed}
        if set(arms) != {"B0", comparison_variant}:
            continue
        comparison = {"seed": seed, f"image_delta_{comparison_variant}_minus_B0": {
            key: arms[comparison_variant][key] - arms["B0"][key] for key in ("psnr", "ssim", "lpips")}}
        before, after = durations.get(f"B0_s{seed}_train"), durations.get(f"{comparison_variant}_s{seed}_train")
        if before and after:
            comparison[f"training_stage_wall_time_ratio_{comparison_variant}_over_B0"] = after / before
        if str(seed) in temporal:
            values = temporal[str(seed)]["temporal"]
            old, new = values["baseline_temporal_residual_l1"], values["prediction_temporal_residual_l1"]
            comparison["temporal_residual_reduction_percent"] = 100 * (old - new) / old if old else None
        comparisons.append(comparison)
    atomic_json(output / "summary.json", {
        "status": state["status"], "image_metrics": rows, "paired_comparisons": comparisons,
        "temporal_diagnostics": temporal, "stage_wall_seconds": durations,
        "training_diagnostics": diagnostics,
        "intermediate_gbuffer_cleanup": state.get("intermediate_gbuffer_cleanup"),
        "timing_note": "Training-stage wall time includes initialization and checkpoint serialization; temporal evaluation is reported separately.",
        "test_ids": plan["identity"]["test_ids"], "interpretation": plan["interpretation"],
        "method": plan["method"], "initialization": plan["identity"]["initialization"],
    })
    if rows:
        fields = ["variant", "seed"] + sorted(set().union(*(row.keys() for row in rows)) - {"variant", "seed"})
        temporary = output / "summary.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output / "summary.csv")


def run_queue(args, plan, *, child_runner=run_child, artifact_reader=stage_artifacts):
    from omegaconf import OmegaConf
    plan["post_completion_cleanup"] = {
        "when": "after all requested rendering and performance evaluation stages complete",
        "scope": "explicit queue-owned direct children of stage attempt directories",
        "directory_names": list(INTERMEDIATE_GBUFFER_DIRS),
        "on_failure": "preserve intermediates for diagnosis and retry",
    }
    if not args.execute:
        commands = {stage["id"]: stage_command(stage,
                    stage_directory(args.output_dir, stage["id"], 1), plan, {}, preview=True)
                    for stage in plan["stages"]}
        preview = {"status": "planned", "output_dir": str(args.output_dir),
                   "source_config": str(args.source_config), "commands": commands,
                   "initialization": plan["identity"]["initialization"],
                   "train_frames": len(plan["identity"]["train_ids"]),
                   "test_ids": plan["identity"]["test_ids"],
                   "updates_per_training": plan["identity"]["updates_per_training"],
                   "restart_policy": plan["restart_policy"], "method": plan["method"],
                   "post_completion_cleanup": plan["post_completion_cleanup"]}
        print(json.dumps(preview, indent=2), flush=True)
        return preview
    output = args.output_dir
    if output.exists() and any(output.iterdir()) and not (output / "manifest.json").exists():
        if set(path.name for path in output.iterdir()) != {".queue.lock"}:
            raise ValueError("Output is not an initialized experiment queue; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This queue or a child is already running") from exc
        manifest_path, state_path = output / "manifest.json", output / "state.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != plan:
                raise ValueError("Code, configuration, input, or options changed; use a new output directory")
            state = json.loads(state_path.read_text())
            for stage_id, cfg in plan["identity"]["training_configs"].items():
                saved = OmegaConf.to_container(OmegaConf.load(output / "configs" / f"{stage_id}.yaml"), resolve=True)
                if saved != cfg:
                    raise ValueError("Planned configuration was modified")
        else:
            if any(path.name != ".queue.lock" for path in output.iterdir()):
                raise ValueError("Output contains unrecognized files")
            (output / "configs").mkdir()
            for stage_id, cfg in plan["identity"]["training_configs"].items():
                OmegaConf.save(OmegaConf.create(cfg), output / "configs" / f"{stage_id}.yaml")
            state = {"status": "planned", "stages": {s["id"]: {"attempts": []} for s in plan["stages"]}}
            atomic_json(manifest_path, plan)
        state.update(status="running", pid=os.getpid(), updated_at=time.time())
        state.pop("error", None)
        state.pop("error_type", None)
        state.pop("intermediate_gbuffer_cleanup", None)
        atomic_json(state_path, state)
        try:
            for stage in plan["stages"]:
                verify_inputs(plan)
                attempts = state["stages"][stage["id"]]["attempts"]
                if attempts and attempts[-1]["status"] == "completed":
                    verify_artifacts(attempts[-1])
                    print(f"{stage['id']}: verified; skipping", flush=True)
                    continue
                free = shutil.disk_usage(output).free / 1024 ** 3
                if free < args.min_free_gb:
                    raise RuntimeError(f"Only {free:.1f} GiB free; need {args.min_free_gb:.1f} GiB before {stage['id']}")
                if attempts:
                    attempts[-1]["status"] = "interrupted_or_failed_preserved"
                directory = stage_directory(output, stage["id"], len(attempts) + 1)
                if directory.exists():
                    raise ValueError(f"Fresh attempt directory already exists: {directory}")
                command = stage_command(stage, directory, plan, state)
                attempt = {"status": "running", "directory": str(directory), "command": command,
                           "started_at": time.time()}
                attempts.append(attempt)
                state.update(active_stage=stage["id"], updated_at=time.time())
                atomic_json(state_path, state)
                if stage["kind"] == "images":
                    prepare_image_snapshot(checkpoint_for(stage["checkpoint_from"], plan, state), directory)
                print(f"{stage['id']}: starting {directory.name}", flush=True)
                child_runner(command, output / "logs" / f"{stage['id']}_{directory.name}.log", lock.fileno())
                verify_saved_configuration(stage, directory, plan)
                attempt["artifacts"] = artifact_reader(stage, directory)
                attempt.update(status="completed", completed_at=time.time())
                atomic_json(state_path, state)
                write_summary(plan, state, output)
            verify_inputs(plan)
            cleanup = cleanup_intermediate_gbuffers(output, state)
            state.update(status="completed", active_stage=None, completed_at=time.time())
            state["intermediate_gbuffer_cleanup"] = cleanup
            atomic_json(state_path, state)
            write_summary(plan, state, output)
            return state
        except BaseException as exc:
            active = state.get("active_stage")
            if active:
                attempt = state["stages"][active]["attempts"][-1]
                if attempt["status"] == "running":
                    attempt.update(status="failed", error=str(exc), ended_at=time.time())
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=str(exc), error_type=type(exc).__name__, updated_at=time.time())
            atomic_json(state_path, state)
            write_summary(plan, state, output)
            raise


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
