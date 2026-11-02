#!/usr/bin/env python3
"""Preview or execute matched Base v4 G-buffer attention experiments.

The default is read-only: no output directories or GPU processes are created.
--execute runs one stage at a time. Repeating the identical command verifies and
skips completed stages; interrupted stages restart in a fresh attempt, preserving
all earlier files. This is queue continuation, not an optimizer-state resume.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT = ROOT / "output/pbr/rebuild/scene_788_100f/base_v4_raw/checkpoint_final.pth"
PILOT_VARIANTS = ("B0", "H0", "C0", "C1", "A0", "A1", "A2")
JOINT_VARIANTS = ("B0", "H0", "C0", "A0", "A1", "A2")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output/pbr/gbuffer_attention/scene788_v1")
    parser.add_argument("--mode", choices=("pilot", "joint", "both"), default="pilot")
    parser.add_argument("--variants", nargs="+", choices=PILOT_VARIANTS,
                        help="Subset; B0 is always included as a paired comparator")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--pilot-steps", type=int, default=2000)
    parser.add_argument("--joint-steps", type=int, default=30000,
                        help="Original inclusive final step: N means N+1 optimizer updates")
    parser.add_argument("--skip-temporal", action="store_true",
                        help="Keep original image evaluation but omit full-path temporal diagnostics")
    parser.add_argument("--execute", action="store_true")
    return parser


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def code_identity():
    return {str(path.relative_to(ROOT)): digest(path)
            for folder in ("models", "datasets", "utils", "tools")
            for path in sorted((ROOT / folder).rglob("*.py"))}


def input_identity(source):
    """Record exact source file metadata; this is explicitly not a content hash."""
    data = source["data"]
    scene = Path(data["data_root"]) / str(data["scene_idx"])
    teacher = Path(data["pixel_source"]["gbuffer_path"])
    scene = scene if scene.is_absolute() else ROOT / scene
    teacher = teacher if teacher.is_absolute() else ROOT / teacher
    if not scene.is_dir() or not teacher.is_dir():
        raise FileNotFoundError("Source scene and original G-buffer teacher directories must exist")
    folders = [scene / name for name in ("images/raw", "sky_masks", "dynamic_masks", "fine_dynamic_masks",
               "ego_pose", "intrinsics", "extrinsics", "instances", "humanpose", "lidar")]
    folders.append(teacher)
    files = sorted({path for folder in folders if folder.exists()
                    for path in folder.rglob("*") if path.is_file()})
    return {str(path.resolve()): [path.stat().st_size, path.stat().st_mtime_ns] for path in files}


def validate_source(source):
    """Refuse to silently replace the native Base v4 stride evaluation protocol."""
    trainer = source["trainer"]
    if trainer["type"] not in ("models.trainers.MultiTrainer_pbr",
                               "models.trainers.scene_graph_pbr.MultiTrainer_pbr"):
        raise ValueError("Expected a raw Base v4 PBR source")
    pixel = source["data"]["pixel_source"]
    if list(pixel.get("cameras", [])) != [0] or pixel.get("load_weather", []):
        raise ValueError("Expected camera 0 and raw-only source")
    if pixel.get("test_frame_ranges") is not None or pixel.get("test_image_stride") != 8:
        raise ValueError("Preserve Base v4 stride 8; contiguous holdouts are a different benchmark")
    if pixel.get("downscale", 1) != 1 or list(pixel.get("downscale_when_loading", [1])) != [1]:
        raise ValueError("Native loading and evaluation resolution are required")
    if not pixel.get("load_gbuffer") or not pixel.get("gbuffer_path"):
        raise ValueError("Original G-buffer teacher supervision must remain enabled")
    if trainer["render"].get("antialiased", False) or trainer["render"].get("packed", False):
        raise ValueError("Base v4 evaluation requires AA off and packed false")
    if trainer.get("layered_gbuffer") or trainer.get("gbuffer_attention"):
        raise ValueError("Source must be the unmodified Base v4 renderer")
    if (trainer.get("teacher_calibration") or {}).get("enabled"):
        raise ValueError("Source already enables teacher calibration; expected unmodified Base v4")
    if (trainer.get("teacher_confidence") or {}).get("enabled"):
        raise ValueError("Source already enables teacher confidence; expected unmodified Base v4")
    if (trainer.get("rdg_fusion") or {}).get("enabled"):
        raise ValueError("Source already enables RDG fusion; expected unmodified Base v4")
    if trainer.get("losses", {}).get("temporal_material", {}).get("w", 0):
        raise ValueError("Source already enables a separate temporal-material loss")
    if not source.get("render", {}).get("render_test"):
        raise ValueError("Source must enable original test-image evaluation")
    count = int(source["data"]["end_timestep"]) - int(source["data"]["start_timestep"]) + 1
    if count <= 8:
        raise ValueError("Source needs at least one stride-8 test frame")
    return count


def joint_config(source, variant, steps, seed):
    if variant not in PILOT_VARIANTS or variant == "H0":
        raise ValueError("H0 is inference-only transport on B0, not a separate joint training")
    cfg = copy.deepcopy(source)
    cfg.pop("log_dir", None)
    cfg["seed"] = seed
    cfg["trainer"]["optim"]["num_iters"] = steps
    cfg["trainer"]["gbuffer_attention"] = None if variant == "B0" else {
        "variant": variant, "start_step": 5000, "ramp_steps": 1000,
        "temporal_start": 6000, "temporal_ramp": 1000,
        "temporal_w": 0. if variant == "A1" else .05,
    }
    return cfg


def build_plan(args):
    from omegaconf import OmegaConf
    if args.pilot_steps < 1 or args.joint_steps < 1 or min(args.seeds) < 0:
        raise ValueError("Training steps must be positive and seeds nonnegative")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Duplicate seeds are not allowed")
    if args.variants and len(args.variants) != len(set(args.variants)):
        raise ValueError("Duplicate variants are not allowed")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    config_path = args.checkpoint.parent / "config.yaml"
    source = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    count = validate_source(source)
    test_ids = list(range(8, count, 8))
    test_set = set(test_ids)
    stages, configurations = [], {}
    source_seed = int(source.get("seed", 0))

    def add_arm(mode, variant, seed, source_stage=None):
        arm = f"{mode}_{variant}_s{seed}"
        train_id = arm + "_train"
        kind = "prepare" if (variant in ("B0", "H0") and mode == "pilot") or variant == "H0" else (
            "joint_train" if mode == "joint" else "pilot_train")
        stages.append({"id": train_id, "kind": kind, "mode": mode, "variant": variant,
                       "seed": seed, "checkpoint_from": source_stage,
                       "steps": args.joint_steps if kind == "joint_train" else (
                           args.pilot_steps if kind == "pilot_train" else 0)})
        if kind == "joint_train":
            configurations[train_id] = joint_config(source, variant, args.joint_steps, seed)
        baseline = f"joint_B0_s{seed}_train" if mode == "joint" else None
        stages.append({"id": arm + "_images", "kind": "image_eval", "mode": mode,
                       "variant": variant, "seed": seed, "checkpoint_from": train_id})
        if not args.skip_temporal:
            stages.append({"id": arm + "_temporal", "kind": "temporal_eval", "mode": mode,
                           "variant": variant, "seed": seed, "checkpoint_from": train_id,
                           "baseline_from": baseline})

    if args.mode in ("pilot", "both"):
        requested = set(args.variants or PILOT_VARIANTS) | {"B0"}
        # Frozen scene weights are identical across head seeds; fixed controls are shared.
        for variant in ("B0", "H0"):
            if variant in requested:
                add_arm("pilot", variant, source_seed)
        for seed in args.seeds:
            for variant in PILOT_VARIANTS[2:]:
                if variant in requested:
                    add_arm("pilot", variant, seed)
    if args.mode in ("joint", "both"):
        requested = set(args.variants or JOINT_VARIANTS) | {"B0"}
        for seed in args.seeds:
            for variant in PILOT_VARIANTS:
                if variant in requested:
                    add_arm("joint", variant, seed,
                            f"joint_B0_s{seed}_train" if variant == "H0" else None)
    options = {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items() if key != "execute"}
    identity = {
        "schema_version": 1, "options": options, "source_config": source,
        "checkpoint_sha256": digest(args.checkpoint), "config_sha256": digest(config_path),
        "source_stat": {str(path): [path.stat().st_size, path.stat().st_mtime_ns]
                        for path in (args.checkpoint, config_path)},
        "code_sha256": code_identity(), "input_stat": input_identity(source),
        "joint_configs": configurations, "python": sys.executable,
        "evaluation": "original_test_images_native_original_affine_original_metric_aggregation",
        "train_ids": [frame for frame in range(count) if frame not in test_set],
        "test_ids": test_ids, "context_offsets": [1, 2, 4],
    }
    return {
        "identity": identity, "stages": stages,
        "interpretation": "Exploratory scene comparison; no test-set selection. Full-path temporal diagnostics include training frames.",
        "seed_interpretation": "Pilot seeds vary head optimization only; joint seeds vary the scene and head together.",
        "restart_policy": "Completed artifacts are verified and skipped; unfinished stages restart in a new directory.",
        "joint_updates_per_training": args.joint_steps + 1,
    }


def stage_directory(output, stage_id, attempt):
    return Path(output) / "stages" / stage_id / f"attempt_{attempt:03d}"


def checkpoint_for(stage_id, plan, state, preview=False):
    if stage_id is None:
        return Path(plan["identity"]["options"]["checkpoint"])
    attempts = state.get("stages", {}).get(stage_id, {}).get("attempts", [])
    if not attempts:
        if not preview:
            raise ValueError(f"Dependency has no completed checkpoint: {stage_id}")
        return stage_directory(plan["identity"]["options"]["output_dir"], stage_id, 1) / "checkpoint_final.pth"
    if attempts[-1]["status"] != "completed" and not preview:
        raise ValueError(f"Dependency is incomplete: {stage_id}")
    return Path(attempts[-1]["directory"]) / "checkpoint_final.pth"


def stage_command(stage, directory, plan, state, *, preview=False):
    python = plan["identity"]["python"]
    checkpoint = checkpoint_for(stage.get("checkpoint_from"), plan, state, preview)
    if stage["kind"] == "joint_train":
        output = Path(plan["identity"]["options"]["output_dir"])
        return [python, "-u", "-m", "tools.train", "--config_file",
                str(output / "configs" / (stage["id"] + ".yaml")), "--output_root", str(output / "stages"),
                "--project", stage["id"], "--run_name", directory.name]
    if stage["kind"] in ("prepare", "pilot_train"):
        return [python, "-u", "-m", "tools.train_gbuffer_attention", "--checkpoint", str(checkpoint),
                "--output-dir", str(directory), "--variant", stage["variant"],
                "--steps", str(stage["steps"]), "--seed", str(stage["seed"])]
    if stage["kind"] == "image_eval":
        return [python, "-u", "-m", "tools.eval", "--resume_from",
                str(directory / "checkpoint_final.pth"), "--eval_weathers"]
    baseline = checkpoint_for(stage.get("baseline_from"), plan, state, preview)
    output = Path(plan["identity"]["options"]["output_dir"])
    return [python, "-u", "-m", "tools.eval_gbuffer_attention", "--checkpoint", str(checkpoint),
            "--baseline-checkpoint", str(baseline), "--output-dir", str(directory),
            "--flow-cache", str(output / "reference_flow")]


def prepare_image_snapshot(checkpoint, directory):
    """Keep the ordinary evaluator's writes away from source/training outputs."""
    from omegaconf import OmegaConf
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "checkpoint_final.pth"
    try:
        os.link(checkpoint, destination)
    except OSError:
        shutil.copy2(checkpoint, destination)
    cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
    cfg.log_dir = str(directory)
    OmegaConf.save(cfg, directory / "config.yaml")


def stage_artifacts(stage, directory, *, inspect_checkpoint=True):
    """Check required semantic outputs before accepting a stage as complete."""
    from omegaconf import OmegaConf
    kind = stage["kind"]
    if kind in ("prepare", "pilot_train", "joint_train"):
        files = [directory / "checkpoint_final.pth", directory / "config.yaml"]
        if any(not path.is_file() for path in files):
            raise ValueError(f"Stage {stage['id']} did not produce its checkpoint and configuration")
        config = OmegaConf.to_container(OmegaConf.load(files[1]), resolve=True)
        attention = config["trainer"].get("gbuffer_attention")
        variant = attention.get("variant", "A2") if attention else "B0"
        if variant != stage["variant"]:
            raise ValueError(f"Saved renderer differs from stage {stage['id']}")
        if inspect_checkpoint:
            import torch
            checkpoint = torch.load(files[0], map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict) or not checkpoint.get("models"):
                raise ValueError(f"Invalid scene checkpoint for {stage['id']}")
            if kind == "joint_train" and checkpoint.get("step") != stage["steps"]:
                raise ValueError("Joint checkpoint did not reach the requested final step")
            del checkpoint
        manifest = directory / "manifest.json"
        if manifest.is_file():
            record = json.loads(manifest.read_text())
            if record.get("status") not in ("complete", "completed"):
                raise ValueError("Training manifest is incomplete")
            files.append(manifest)
    elif kind == "image_eval":
        files = sorted((directory / "metrics_eval" / "raw").glob("images_test_*.json"))
        if len(files) != 1:
            raise ValueError("Expected exactly one original test metric JSON in a fresh evaluation attempt")
        metrics = json.loads(files[0].read_text())
        import math
        for key in ("psnr", "ssim", "lpips"):
            value = metrics.get("image_metrics/raw/test/" + key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid original image metric {key}")
        files += [directory / "config.yaml", directory / "checkpoint_final.pth"]
    else:
        files = [directory / name for name in ("summary.json", "frames.jsonl", "manifest.json")]
        if any(not path.is_file() for path in files):
            raise ValueError("Temporal diagnostic did not produce all expected artifacts")
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("status") not in ("complete", "completed"):
            raise ValueError("Temporal diagnostic manifest is incomplete")
        json.loads((directory / "summary.json").read_text())
    return {str(path.relative_to(directory)): digest(path) for path in files}


def verify_artifacts(attempt):
    for relative, checksum in attempt["artifacts"].items():
        path = Path(attempt["directory"]) / relative
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Completed artifact is missing or changed: {path}")


def verify_saved_configuration(stage, directory, plan):
    """Assert the child preserved the paired scene, teacher, loss and split setup."""
    if stage["kind"] not in ("prepare", "pilot_train", "joint_train"):
        return
    from omegaconf import OmegaConf
    saved = OmegaConf.to_container(OmegaConf.load(directory / "config.yaml"), resolve=True)
    if stage["kind"] == "joint_train":
        expected = copy.deepcopy(plan["identity"]["joint_configs"][stage["id"]])
    elif stage["checkpoint_from"] is not None:
        expected = copy.deepcopy(plan["identity"]["joint_configs"][stage["checkpoint_from"]])
    else:
        expected = copy.deepcopy(plan["identity"]["source_config"])
    if saved.get("seed") != stage["seed"]:
        raise ValueError("Saved seed differs from the planned seed")
    for config in (saved, expected):
        config.pop("log_dir", None)
        config.pop("seed", None)
        config["trainer"].pop("gbuffer_attention", None)
    if saved != expected:
        raise ValueError("Saved configuration changed the original paired training/evaluation protocol")


def verify_inputs(plan):
    identity = plan["identity"]
    if code_identity() != identity["code_sha256"]:
        raise ValueError("Code changed; use a new output directory")
    for name, expected in identity["source_stat"].items():
        path = Path(name)
        if not path.is_file() or [path.stat().st_size, path.stat().st_mtime_ns] != expected:
            raise ValueError("Source checkpoint/config changed during execution")
    if input_identity(identity["source_config"]) != identity["input_stat"]:
        raise ValueError("Source scene or teacher inputs changed during execution")


def run_child(command, log_path, lock_fd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        log.write(("\nCOMMAND " + shlex.join(command) + "\n").encode())
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", "/tmp/3ablegs-mpl")
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_fd,))
        try:
            result = child.wait()
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    if result:
        raise RuntimeError(f"Child exited {result}; see {log_path}")


def write_summary(plan, state, output):
    rows = []
    temporal = {}
    for stage in plan["stages"]:
        attempts = state["stages"][stage["id"]]["attempts"]
        if not attempts or attempts[-1]["status"] != "completed":
            continue
        directory = Path(attempts[-1]["directory"])
        if stage["kind"] == "image_eval":
            metric = next((directory / "metrics_eval/raw").glob("images_test_*.json"))
            values = json.loads(metric.read_text())
            rows.append({"mode": stage["mode"], "variant": stage["variant"], "seed": stage["seed"],
                         **{key.split("/")[-1]: value for key, value in values.items()},
                         "metrics_file": str(metric)})
        elif stage["kind"] == "temporal_eval":
            temporal[stage["id"]] = json.loads((directory / "summary.json").read_text())
    atomic_json(output / "summary.json", {"status": state["status"], "image_metrics": rows,
                "temporal_diagnostics": temporal, "test_ids": plan["identity"]["test_ids"],
                "interpretation": plan["interpretation"], "seed_interpretation": plan["seed_interpretation"]})
    if rows:
        fields = ["mode", "variant", "seed"] + sorted(set().union(*(row.keys() for row in rows)) - {"mode", "variant", "seed"})
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def run_queue(args, plan, *, child_runner=run_child, artifact_reader=stage_artifacts):
    from omegaconf import OmegaConf
    if not args.execute:
        preview = {"status": "planned", **plan}
        preview["commands"] = {stage["id"]: stage_command(stage,
            stage_directory(args.output_dir, stage["id"], 1), plan, {}, preview=True)
            for stage in plan["stages"]}
        identity = plan["identity"]
        printable = {
            "status": "planned", "output_dir": str(args.output_dir),
            "checkpoint": str(args.checkpoint), "mode": args.mode, "seeds": args.seeds,
            "train_frames": len(identity["train_ids"]), "test_ids": identity["test_ids"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "config_sha256": identity["config_sha256"],
            "code_files": len(identity["code_sha256"]), "input_files": len(identity["input_stat"]),
            "joint_updates_per_training": plan["joint_updates_per_training"],
            "interpretation": plan["interpretation"], "seed_interpretation": plan["seed_interpretation"],
            "restart_policy": plan["restart_policy"], "commands": preview["commands"],
        }
        print(json.dumps(printable, indent=2), flush=True)
        return preview
    output = args.output_dir
    # Reject unrelated contents before creating a lock file in the directory.
    if output.exists() and any(output.iterdir()) and not (output / "manifest.json").exists():
        if set(path.name for path in output.iterdir()) != {".queue.lock"}:
            raise ValueError("Output is not an initialized attention queue; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This queue or a child is already running") from exc
        manifest_path, state_path = output / "manifest.json", output / "state.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest != plan:
                raise ValueError("Code, config, inputs or options changed; choose a new output directory")
            state = json.loads(state_path.read_text())
            for stage_id, cfg in plan["identity"]["joint_configs"].items():
                if OmegaConf.to_container(OmegaConf.load(output / "configs" / f"{stage_id}.yaml"), resolve=True) != cfg:
                    raise ValueError("Planned joint configuration was modified")
        else:
            if any(path.name != ".queue.lock" for path in output.iterdir()):
                raise ValueError("Output contains unrecognized files")
            (output / "configs").mkdir()
            for stage_id, cfg in plan["identity"]["joint_configs"].items():
                OmegaConf.save(OmegaConf.create(cfg), output / "configs" / f"{stage_id}.yaml")
            state = {"status": "planned", "stages": {stage["id"]: {"attempts": []} for stage in plan["stages"]}}
            atomic_json(manifest_path, plan)
            atomic_json(state_path, state)
        state.update(status="running", pid=os.getpid(), updated_at=time.time())
        atomic_json(state_path, state)
        try:
            for stage in plan["stages"]:
                verify_inputs(plan)
                attempts = state["stages"][stage["id"]]["attempts"]
                if attempts and attempts[-1]["status"] == "completed":
                    verify_artifacts(attempts[-1])
                    print(f"{stage['id']}: verified; skipping", flush=True)
                    continue
                if attempts:
                    attempts[-1]["status"] = "interrupted_or_failed_preserved"
                directory = stage_directory(output, stage["id"], len(attempts) + 1)
                if directory.exists():
                    raise ValueError(f"Fresh attempt directory already exists: {directory}")
                command = stage_command(stage, directory, plan, state)
                attempt = {"status": "running", "directory": str(directory), "command": command,
                           "started_at": time.time()}
                attempts.append(attempt)
                atomic_json(state_path, state)
                if stage["kind"] == "image_eval":
                    prepare_image_snapshot(checkpoint_for(stage["checkpoint_from"], plan, state), directory)
                print(f"{stage['id']}: starting {directory.name}", flush=True)
                child_runner(command, output / "logs" / f"{stage['id']}_{directory.name}.log", lock.fileno())
                attempt["artifacts"] = artifact_reader(stage, directory)
                verify_saved_configuration(stage, directory, plan)
                attempt.update(status="completed", completed_at=time.time())
                atomic_json(state_path, state)
                write_summary(plan, state, output)
            verify_inputs(plan)
            state.update(status="completed", completed_at=time.time())
            atomic_json(state_path, state)
            write_summary(plan, state, output)
            return state
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=str(exc), error_type=type(exc).__name__, updated_at=time.time())
            atomic_json(state_path, state)
            raise


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Queue interrupted; repeat the same command to continue")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run_queue(args, build_plan(args))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
