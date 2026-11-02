#!/usr/bin/env python3
"""Train the base and instance-layered renderers jointly from dataset initialization.

Both arms use the original full optimizer, losses, teachers and resolution
schedule. There is no checkpoint initialization or decoder-only adaptation.
The default writes a reviewable plan; --execute runs the pair sequentially.
Use external nohup for terminal-independent execution. --resume continues the
queue, restarting interrupted training in a new attempt: tools.train does not
save optimizer state, so this command never claims optimizer continuation.
"""
from __future__ import annotations

import argparse
import copy
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

DEFAULT_SOURCE = ROOT / "output/pbr/temporal/suite_scene788_v1/train/static_teacher_seed0/attempt_001/config.yaml"
LAYERED_CONFIG = dict(grouping="instance", color_space="srgb", max_layers=0,
                      ablation="none", diagnostics=False)
HOLDOUTS = [[20, 30], [60, 70]]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000,
                        help="Original trainer's final step, inclusive; steps+1 optimizer updates.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order", choices=("base-first", "layered-first"), default="base-first")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a matching queue; restart incomplete training from scratch.")
    return parser


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def code_identity():
    return {str(path.relative_to(ROOT)): digest(path)
            for folder in ("models", "datasets", "utils", "tools")
            for path in sorted((ROOT / folder).rglob("*.py"))}


def input_identity(config):
    """Track read-only scene and teacher input metadata, excluding other jobs' outputs."""
    data = config["data"]
    scene = Path(data["data_root"]) / str(data["scene_idx"])
    teacher = Path(data["pixel_source"]["gbuffer_path"])
    scene = scene if scene.is_absolute() else ROOT / scene
    teacher = teacher if teacher.is_absolute() else ROOT / teacher
    folders = [scene / name for name in ("images/raw", "sky_masks", "dynamic_masks",
               "ego_pose", "intrinsics", "extrinsics", "instances", "humanpose", "lidar")]
    folders.append(teacher)
    files = sorted({path for folder in folders if folder.exists()
                    for path in folder.rglob("*") if path.is_file()})
    return {str(path.resolve()): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in files}


def build_configs(source, *, steps, seed):
    if steps < 1 or seed < 0:
        raise ValueError("--steps must be positive and --seed nonnegative")
    cfg = copy.deepcopy(source)
    cfg.pop("log_dir", None)
    pixel = cfg["data"]["pixel_source"]
    if pixel.get("test_frame_ranges") != HOLDOUTS or pixel.get("test_image_stride", 0) != 0:
        raise ValueError("Source must exclude contiguous blocks [20,30) and [60,70), with stride 0")
    if int(cfg["data"]["end_timestep"]) - int(cfg["data"]["start_timestep"]) + 1 != 100:
        raise ValueError("This paired scene protocol requires exactly 100 frames")
    if list(pixel.get("cameras", [])) != [0] or pixel.get("load_weather", []):
        raise ValueError("This experiment requires camera 0 and raw-only training")
    if not pixel.get("load_gbuffer") or not pixel.get("gbuffer_path"):
        raise ValueError("The original G-buffer teacher must remain enabled")
    if cfg["trainer"]["type"] not in ("models.trainers.MultiTrainer_pbr",
                                       "models.trainers.scene_graph_pbr.MultiTrainer_pbr"):
        raise ValueError("The source must use the raw PBR trainer")
    cfg["seed"] = seed
    cfg["trainer"]["optim"]["num_iters"] = steps
    cfg["trainer"]["render"]["antialiased"] = False
    cfg["trainer"]["losses"].setdefault("temporal_material", {})["w"] = 0.
    cfg["logging"]["print_freq"] = 100
    result = {}
    for name, renderer in (("base", None), ("layered", LAYERED_CONFIG)):
        result[name] = copy.deepcopy(cfg)
        result[name]["trainer"]["layered_gbuffer"] = copy.deepcopy(renderer)
    return result


def training_command(output, arm, attempt, python=sys.executable):
    return [python, "-u", "-m", "tools.train", "--config_file", str(output / "configs" / f"{arm}.yaml"),
            "--output_root", str(output / "train"), "--project", arm, "--run_name", attempt]


def evaluation_command(checkpoint, output, restart=False, python=sys.executable):
    command = [python, "-u", "-m", "tools.layered_gbuffer_experiment", "--checkpoint", str(checkpoint),
               "--output", str(output), "--trained-only", "--split", "heldout", "--save-every", "1"]
    return command + (["--restart"] if restart else [])


def build_plan(args):
    from omegaconf import OmegaConf
    from utils.config import load_training_config
    args.source_config = args.source_config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    source = OmegaConf.to_container(load_training_config(args.source_config), resolve=True)
    configs = build_configs(source, steps=args.steps, seed=args.seed)
    order = ["base", "layered"] if args.order == "base-first" else ["layered", "base"]
    identity = {
        "schema_version": 1, "source_config": str(args.source_config),
        "source_config_sha256": digest(args.source_config), "source_sha256": code_identity(),
        "input_stat": input_identity(source), "output": str(args.output_dir),
        "configs": configs, "order": order, "python": sys.executable,
        "initialization": "dataset_from_scratch", "optimizer": "original_full_optimizer",
        "final_step": args.steps, "optimizer_updates_per_arm": args.steps + 1,
        "evaluation": "trained_renderer_only_native_shared_mean_affine_AA_off",
        "holdouts": {"val": list(range(20, 30)), "test": list(range(60, 70))},
    }
    commands = {}
    for arm in order:
        checkpoint = args.output_dir / "train" / arm / "attempt_001" / "checkpoint_final.pth"
        commands[arm] = {
            "train": training_command(args.output_dir, arm, "attempt_001"),
            "evaluate": evaluation_command(checkpoint, args.output_dir / "evaluate" / arm),
        }
    return {"identity": identity, "commands": commands,
            "interpretation": "Exploratory single-scene paired training; val and test are reported separately.",
            "restart_policy": "Incomplete training restarts in a fresh attempt; no optimizer resume."}


def verify_checkpoint(checkpoint, expected_config, final_step):
    """Read a local trusted model, require the requested final step and saved config."""
    import torch
    from omegaconf import OmegaConf
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    saved = OmegaConf.to_container(OmegaConf.load(checkpoint.parent / "config.yaml"), resolve=True)
    saved.pop("log_dir", None)
    if saved != expected_config:
        raise ValueError(f"Saved training configuration changed: {checkpoint.parent}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("step") != final_step or not state.get("models"):
        raise ValueError(f"Checkpoint is not a completed step {final_step}: {checkpoint}")
    del state
    return {"checkpoint_sha256": digest(checkpoint),
            "config_sha256": digest(checkpoint.parent / "config.yaml"), "final_step": final_step}


def verified_training(attempt):
    expected = attempt.get("training_artifacts")
    if not expected:
        return False
    checkpoint = Path(attempt["checkpoint"])
    for key, path in (("checkpoint_sha256", checkpoint),
                      ("config_sha256", checkpoint.parent / "config.yaml")):
        if not path.is_file() or digest(path) != expected[key]:
            raise ValueError(f"Completed training artifact is missing or changed: {path}")
    return True


def split_evaluation(directory, checkpoint, expected_config, *, summarize=None):
    """Validate one renderer's evaluation, then pool sufficient statistics per block."""
    if summarize is None:
        from utils.layered_metrics import summarize_layered_records
        summarize = summarize_layered_records
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if summary.get("status") != "completed" or manifest.get("status") != "completed":
        raise ValueError("Held-out evaluation is not complete")
    identity = summary["identity"]
    if manifest.get("identity") != identity:
        raise ValueError("Evaluation summary and manifest identities differ")
    if identity.get("checkpoint_sha256") != digest(checkpoint):
        raise ValueError("Evaluation checkpoint hash differs from trained checkpoint")
    if identity.get("config_sha256") != digest(checkpoint.parent / "config.yaml"):
        raise ValueError("Evaluation config hash differs from trained config")
    if identity.get("split") != "heldout" or identity.get("downscale") != 1. or identity.get("antialiased"):
        raise ValueError("Expected native-resolution held-out evaluation with AA off")
    variants = identity.get("variants", {})
    if len(variants) != 1:
        raise ValueError("Expected only the renderer actually used during training")
    variant, renderer = next(iter(variants.items()))
    # Evaluation may enable diagnostic statistics, which do not change rendering.
    renderer = copy.deepcopy(renderer)
    trained = copy.deepcopy(expected_config["trainer"]["layered_gbuffer"])
    if renderer is not None:
        renderer.pop("diagnostics", None)
    if trained is not None:
        trained.pop("diagnostics", None)
    if renderer != trained:
        raise ValueError("Evaluation renderer differs from trained renderer")
    splits = {"val": list(range(20, 30)), "test": list(range(60, 70))}
    expected_frames = splits["val"] + splits["test"]
    files = sorted((directory / "frames").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in files]
    if sorted(row["frame_id"] for row in rows) != expected_frames:
        raise ValueError("Evaluation must contain exactly the 20 saved holdout frames")
    if manifest.get("frames") != expected_frames or summary.get("frames") != 20:
        raise ValueError("Evaluation frame counts do not agree")
    metrics = {}
    for split, frames in splits.items():
        selected = [row["metrics"] for row in rows if row["frame_id"] in frames]
        pooled = summarize(selected)
        if pooled.get("frames") != 10 or pooled.get("temporal_pairs") != 9:
            raise ValueError(f"Incomplete temporal pairs in {split}")
        metrics[split] = {key[len(variant) + 1:] if key.startswith(variant + "_") else key: value
                          for key, value in pooled.items()}
    artifacts = {str(path.relative_to(directory)): digest(path)
                 for path in [directory / "summary.json", directory / "manifest.json", *files]}
    return {"variant": variant, "metrics": metrics, "evaluation_artifacts": artifacts}


def verify_evaluation_files(directory, expected):
    for relative, checksum in expected.items():
        path = directory / relative
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Completed evaluation artifact is missing or changed: {path}")


def aggregate_pair(state):
    result = {"status": "completed", "interpretation": "Exploratory single scene; no test-set selection.",
              "initialization": "dataset_from_scratch", "arms": {}, "delta_layered_minus_base": {}}
    for arm in ("base", "layered"):
        result["arms"][arm] = state["arms"][arm]["attempts"][-1]["evaluation"]["metrics"]
    scores = ("global_rgb_psnr", "fullframe_lpips", "fullframe_ssim", "global_temporal_residual_l1",
              "object_boundary_r5_temporal_residual_l1", "dynamic_interior_temporal_residual_l1",
              "static_interior_temporal_residual_l1", "object_boundary_r5_rgb_mae")
    for split in ("val", "test"):
        base, layered = (result["arms"][arm][split] for arm in ("base", "layered"))
        result["delta_layered_minus_base"][split] = {
            key: {"absolute": layered[key] - base[key],
                  "relative_percent": 100 * (layered[key] - base[key]) / base[key] if base[key] else None}
            for key in scores if isinstance(base.get(key), (int, float))
            and isinstance(layered.get(key), (int, float))}
    return result


def run_child(command, log_path, lock_fd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        log.write(("\nCOMMAND " + shlex.join(command) + "\n").encode())
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=(lock_fd,))
        try:
            returncode = child.wait()
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    if returncode:
        raise RuntimeError(f"Child exited {returncode}; see {log_path}")


def run_queue(args, plan, *, child_runner=run_child, checkpoint_verifier=verify_checkpoint,
              evaluation_reader=split_evaluation):
    from omegaconf import OmegaConf
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This queue or its training child is already running") from exc
        manifest_path, state_path = output / "manifest.json", output / "state.json"
        existing = [path for path in output.iterdir() if path.name != ".queue.lock"]
        if existing:
            if not manifest_path.is_file() or not state_path.is_file():
                raise ValueError("Output is not an initialized joint-training queue; use a new directory")
            if json.loads(manifest_path.read_text())["identity"] != plan["identity"]:
                raise ValueError("Code, config, inputs or flags changed; use a new output directory")
            state = json.loads(state_path.read_text())
            if args.execute and state["status"] != "planned" and not args.resume:
                raise ValueError("Existing execution requires --resume (queue continuation, not optimizer resume)")
            for arm, cfg in plan["identity"]["configs"].items():
                if OmegaConf.to_container(OmegaConf.load(output / "configs" / f"{arm}.yaml")) != cfg:
                    raise ValueError(f"Planned {arm} configuration was modified")
        else:
            if args.resume:
                raise ValueError("--resume requires an existing initialized queue")
            (output / "configs").mkdir()
            for arm, cfg in plan["identity"]["configs"].items():
                OmegaConf.save(OmegaConf.create(cfg), output / "configs" / f"{arm}.yaml")
            atomic_json(manifest_path, plan)
            state = {"status": "planned", "arms": {name: {"attempts": []} for name in ("base", "layered")}}
            atomic_json(state_path, state)
        if not args.execute:
            print(json.dumps(plan["commands"], indent=2), flush=True)
            print(f"Plan saved to {output}; add --execute to train from scratch.", flush=True)
            return state
        free_bytes = shutil.disk_usage(output).free
        if args.steps >= 30000 and free_bytes < 20 * 1024 ** 3:
            raise RuntimeError("Full paired training needs at least 20 GiB free disk before launch; no files were removed")
        state.update(status="running", pid=os.getpid(), updated_at=time.time())
        atomic_json(state_path, state)
        try:
            for arm in plan["identity"]["order"]:
                if build_plan(args)["identity"] != plan["identity"]:
                    raise RuntimeError("Code/config/input identity changed; refusing to start the next arm")
                attempts = state["arms"][arm]["attempts"]
                attempt = attempts[-1] if attempts else None
                evaluation_dir = output / "evaluate" / arm
                if attempt and verified_training(attempt):
                    if attempt.get("status") == "completed":
                        verify_evaluation_files(evaluation_dir, attempt["evaluation"]["evaluation_artifacts"])
                        print(f"{arm}: verified completed training and evaluation; skipping.", flush=True)
                        continue
                else:
                    if attempt:
                        attempt["status"] = "interrupted_or_failed_restarted_from_scratch"
                    name = f"attempt_{len(attempts) + 1:03d}"
                    training_dir = output / "train" / arm / name
                    if training_dir.exists():
                        raise ValueError(f"Fresh training attempt already exists: {training_dir}")
                    attempt = {"name": name, "status": "training", "started_at": time.time(),
                               "checkpoint": str(training_dir / "checkpoint_final.pth")}
                    attempts.append(attempt)
                    atomic_json(state_path, state)
                    print(f"{arm}: starting {name} from dataset initialization.", flush=True)
                    child_runner(training_command(output, arm, name), output / "logs" / f"{arm}_{name}_train.log", lock.fileno())
                    attempt["training_artifacts"] = checkpoint_verifier(
                        Path(attempt["checkpoint"]), plan["identity"]["configs"][arm], plan["identity"]["final_step"])
                    attempt.update(status="trained", trained_at=time.time())
                    atomic_json(state_path, state)
                attempt["status"] = "evaluating"
                atomic_json(state_path, state)
                print(f"{arm}: evaluating its trained renderer on validation and test blocks.", flush=True)
                child_runner(evaluation_command(Path(attempt["checkpoint"]), evaluation_dir,
                             restart=evaluation_dir.exists() and any(evaluation_dir.iterdir())),
                             output / "logs" / f"{arm}_{attempt['name']}_evaluate.log", lock.fileno())
                attempt["evaluation"] = evaluation_reader(evaluation_dir, Path(attempt["checkpoint"]),
                                                           plan["identity"]["configs"][arm])
                attempt.update(status="completed", completed_at=time.time())
                atomic_json(state_path, state)
            # Refuse to report a matched comparison if code or read-only inputs changed mid-run.
            if build_plan(args)["identity"] != plan["identity"]:
                raise RuntimeError("Code/config/input identity changed during training; results cannot be pooled")
            report = aggregate_pair(state)
            report["identity"] = plan["identity"]
            atomic_json(output / "results.json", report)
            state.update(status="completed", completed_at=time.time())
            atomic_json(state_path, state)
            print(f"Both full training runs and held-out evaluations completed: {output / 'results.json'}", flush=True)
            return state
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error_type=type(exc).__name__, error=str(exc), updated_at=time.time())
            atomic_json(state_path, state)
            raise


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Queue interrupted; incomplete training will restart with --resume")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run_queue(args, build_plan(args))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
