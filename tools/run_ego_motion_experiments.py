#!/usr/bin/env python3
"""Preview or sequentially execute the high-ego-motion experiment matrix.

Defaults to a read-only plan. --execute runs one child at a time; --detach
survives terminal/VS Code closure. Completed stages are checksum-verified and
skipped, interrupted optimizers resume, and changed code/options/inputs require
a new output directory. This runner never stops an existing GPU job.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "output/pbr/temporal/suite_scene788_v1/train/static_teacher_seed0/attempt_001/checkpoint_final.pth"


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/pbr/ego_motion/scene788_v1")
    parser.add_argument("--downscale", type=float, default=2.)
    parser.add_argument("--material-steps", type=int, default=2000)
    parser.add_argument("--head-steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--temporal-weight", type=float, default=.05)
    parser.add_argument("--groups", nargs="+", choices=("diagnosis", "material", "cnn", "transformer"),
                        default=["diagnosis", "material", "cnn", "transformer"])
    parser.add_argument("--wait-for-queue", type=Path,
                        help="Wait for this existing queue status.json to reach completed or failed")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    return parser


def code_identity():
    from tools.ego_motion_experiment import digest
    return {str(path.relative_to(ROOT)): digest(path)
            for folder in ("models", "datasets", "utils", "tools")
            for path in sorted((ROOT / folder).rglob("*.py"))}


def data_identity(cfg):
    """Fingerprint read-only scene inputs, excluding weather output queues."""
    source = Path(str(cfg.data.data_root)) / str(cfg.data.scene_idx)
    if not source.is_absolute():
        source = ROOT / source
    folders = [source / name for name in ("images/raw", "sky_masks", "dynamic_masks", "ego_pose",
                "intrinsics", "extrinsics", "instances", "humanpose", "lidar")]
    files = sorted({path for folder in folders if folder.exists() for path in folder.rglob("*") if path.is_file()})
    return {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in files}


def build_plan(args):
    from omegaconf import OmegaConf
    from tools.ego_motion_runtime import experiment_splits
    from tools.ego_motion_experiment import digest
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not math.isfinite(args.downscale) or args.downscale < 1:
        raise ValueError("--downscale must be finite and >=1")
    if min(args.material_steps, args.head_steps) < 1 or min(args.seeds) < 0:
        raise ValueError("Steps must be positive and seeds nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds")
    if not math.isfinite(args.temporal_weight) or args.temporal_weight < 0:
        raise ValueError("--temporal-weight must be finite and nonnegative")
    if args.detach and not args.execute:
        raise ValueError("--detach requires --execute")
    config_path = args.checkpoint.parent / "config.yaml"
    cfg = OmegaConf.load(config_path)
    total = int(cfg.data.end_timestep) - int(cfg.data.start_timestep) + 1
    train, val, test = experiment_splits(total, cfg.data.pixel_source)
    if str(cfg.trainer.type) not in {"models.trainers.MultiTrainer_pbr", "models.trainers.scene_graph_pbr.MultiTrainer_pbr"}:
        raise ValueError("A raw PBR checkpoint is required")
    if 0 not in cfg.data.pixel_source.cameras:
        raise ValueError("Camera 0 must be present")
    output = args.output_dir.resolve()
    cache = output / "cache_base"
    motion = output / "diagnostics/motion.json"
    stages = []

    def add(name, module, parameters, kind, **metadata):
        stages.append({"id": name, "module": module, "args": list(map(str, parameters)),
                       "output": str(output / name), "kind": kind, **metadata})

    def cached(name, checkpoint, heldout=False, aa=False):
        params = ["cache", "--checkpoint", checkpoint, "--downscale", args.downscale,
                  "--output-dir", output / name, "--split", "heldout" if heldout else "all"]
        if aa:
            params += ["--antialiased"]
        add(name, "tools.ego_motion_experiment", params, "cache")

    def evaluate(name, input_cache, head=None, ablation="none", **metadata):
        params = ["evaluate", "--cache", input_cache, "--base-cache", cache,
                  "--motion-manifest", motion, "--flow-cache", output / "reference_flow",
                  "--output-dir", output / name]
        if head:
            params += ["--head", head, "--guide-ablation", ablation]
        add(name, "tools.ego_motion_experiment", params, "evaluate", **metadata)

    cached("cache_base", args.checkpoint)
    add("diagnostics", "tools.ego_motion_experiment", ["diagnose", "--cache", cache,
        "--output-dir", output / "diagnostics"], "diagnose")
    evaluate("eval_base", cache, family="base")
    if "diagnosis" in args.groups:
        cached("cache_aa", args.checkpoint, heldout=True, aa=True)
        evaluate("eval_aa", output / "cache_aa", family="aa")
    if "material" in args.groups:
        for seed in args.seeds:
            for mode in ("continue", "balanced", "legacy", "albedo", "albedo_roughness"):
                name = f"material_{mode}_s{seed}"
                add(name, "tools.finetune_ego_material", ["--checkpoint", args.checkpoint,
                    "--output-dir", output / name, "--mode", mode, "--steps", args.material_steps,
                    "--seed", seed, "--downscale", args.downscale, "--temporal-weight", args.temporal_weight,
                    "--motion-manifest", motion], "material", family=mode, seed=seed)
                cache_name = "cache_" + name
                cached(cache_name, output / name / "checkpoint_final.pth", heldout=True)
                evaluate("eval_" + name, output / cache_name, family="material_" + mode, seed=seed)
    for group, variants in (("cnn", ("current", "geometry", "full", "no_offset", "full_no_temporal")),
                            ("transformer", ("transformer",))):
        if group not in args.groups:
            continue
        for seed in args.seeds:
            for variant in variants:
                name = f"head_{variant}_s{seed}"
                add(name, "tools.ego_motion_experiment", ["train-head", "--cache", cache,
                    "--motion-manifest", motion, "--variant", "full" if variant == "full_no_temporal" else variant, "--steps", args.head_steps,
                    "--seed", seed, "--temporal-weight", 0 if variant == "full_no_temporal" else args.temporal_weight,
                    "--output-dir", output / name], "head", family=variant, seed=seed)
                head = output / name / "head_final.pt"
                evaluate("eval_" + name, cache, head, family=variant, seed=seed)
                if variant == "full":
                    for ablation in ("constant_normal", "shuffle_normal", "constant_roughness", "shuffle_roughness"):
                        evaluate(f"eval_{name}_{ablation}", cache, head, ablation,
                                 family=ablation, seed=seed)
    options = {key: value for key, value in vars(args).items()
               if key not in {"execute", "detach", "wait_for_queue"}}
    options = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in options.items()}
    identity = {"options": options, "code": code_identity(), "data_stat": data_identity(cfg), "checkpoint_sha256": digest(args.checkpoint),
                "config_sha256": digest(config_path), "source_stat": {
                    str(path.resolve()): [path.stat().st_size, path.stat().st_mtime_ns]
                    for path in (args.checkpoint, config_path)}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"identity": identity, "signature": signature, "scene": int(cfg.data.scene_idx),
            "train_frames": train, "val_frames": val, "test_frames": test, "stages": stages,
            "interpretation": "Exploratory single-scene pilot; validation-only selection, no confirmatory claim"}


def artifacts(stage):
    output = Path(stage["output"])
    kind = stage["kind"]
    if kind == "cache":
        names = ["manifest.json"] + [str(path.relative_to(output)) for path in sorted((output / "frames").glob("*.pt"))]
    elif kind == "diagnose":
        names = ["motion.json"] + [str(path.relative_to(output)) for path in sorted((output / "regions").glob("*.pt"))]
    elif kind == "material":
        names = ["checkpoint_final.pth", "config.yaml", "manifest.json"]
    elif kind == "head":
        names = ["head_final.pt", "manifest.json"]
    else:
        names = ["summary.json", "frames.jsonl", "manifest.json", "comparison.mp4"] + [
            str(path.relative_to(output)) for path in sorted((output / "predictions").glob("*.npz"))]
    return [output / name for name in names]


def stage_hashes(stage):
    from tools.ego_motion_experiment import digest
    paths = artifacts(stage)
    if any(not path.is_file() for path in paths):
        raise ValueError(f"Missing completed output in {stage['id']}")
    if stage["kind"] != "diagnose":
        manifest = json.loads((Path(stage["output"]) / "manifest.json").read_text())
        if manifest.get("status") != "completed":
            raise ValueError(f"Stage did not finish: {stage['id']}")
    return {str(path.relative_to(stage["output"])): digest(path) for path in paths}


def wait_for_queue(path):
    if path is None:
        return
    while True:
        state = json.loads(Path(path).read_text())
        if state.get("stage") in {"completed", "failed"}:
            print(f"Prior queue ended: {state['stage']}", flush=True)
            return
        pid = state.get("pid")
        if not isinstance(pid, int) or not Path(f"/proc/{pid}").exists():
            raise RuntimeError("Prior queue is unfinished but its worker is absent; resume that queue first")
        print(f"Waiting for queue PID {pid}: {state.get('active_weather', state.get('stage'))}", flush=True)
        time.sleep(30)


def verify_live_inputs(plan):
    if code_identity() != plan["identity"]["code"]:
        raise ValueError("Code changed during this run; use a new output directory")
    for name, expected in plan["identity"].get("source_stat", {}).items():
        path = Path(name)
        if not path.is_file() or [path.stat().st_size, path.stat().st_mtime_ns] != expected:
            raise ValueError("Source checkpoint/config changed during this run")
    if "data_stat" in plan["identity"]:
        from omegaconf import OmegaConf
        config = Path(plan["identity"]["options"]["checkpoint"]).parent / "config.yaml"
        if data_identity(OmegaConf.load(config)) != plan["identity"]["data_stat"]:
            raise ValueError("Source scene data changed during this run")


def matched_coverage(plan):
    """Secondary quality comparison on identical history-support intersections.

    Primary reference-flow scores always use method-independent GT masks.
    This secondary subset is deliberately labelled method-dependent and cannot
    establish that a semantic guide helps independently of history acceptance.
    """
    import numpy as np
    from tools.ego_motion_experiment import FrameCache
    base = FrameCache(Path(plan["stages"][0]["output"]))
    result = {}
    seeds = sorted({s["seed"] for s in plan["stages"] if "seed" in s})
    for seed in seeds:
        stages = [s for s in plan["stages"] if s["kind"] == "evaluate" and s.get("seed") == seed and
                  s.get("family") in {"geometry", "full", "no_offset", "constant_normal", "shuffle_normal",
                                       "constant_roughness", "shuffle_roughness", "transformer"}]
        if len(stages) < 2:
            continue
        for split in ("val", "test"):
            errors = {s["id"]: 0. for s in stages}
            count = pixels = total = 0
            for frame in base.meta[split + "_frames"]:
                packet = base.get(frame)
                static = packet["evaluation_static"].numpy().astype(bool)
                predictions = {}
                mask = static.copy()
                for stage in stages:
                    with np.load(Path(stage["output"]) / "predictions" / f"{frame:06d}.npz") as file:
                        predictions[stage["id"]] = file["rgb"].copy()
                        mask &= file["valid"].astype(bool)
                pixels += int(mask.sum())
                total += int(static.sum())
                reference = packet["target_rgb"].numpy()
                for name, predicted in predictions.items():
                    value = np.square(predicted.astype(np.float64) - reference)[mask]
                    errors[name] += float(value.sum())
                count += int(mask.sum()) * 3
            result[f"s{seed}_{split}"] = {"common_support_pixels": pixels,
                "common_support_fraction": pixels / total if total else None,
                "psnr": {name: -10 * math.log10(max(value / count, 1e-12)) if count else None for name, value in errors.items()}}
    return {"comparison": result, "mask": "intersection of variant history support and GT static masks",
            "limitation": "Secondary method-dependent subset; primary temporal evaluation uses independent reference flow"}


def write_report(args, plan, state):
    from tools.ego_motion_experiment import atomic_json
    rows = []
    for stage in plan["stages"]:
        if stage["kind"] != "evaluate" or state["stages"].get(stage["id"], {}).get("status") != "completed":
            continue
        summary = json.loads((Path(stage["output"]) / "summary.json").read_text())
        row = {"stage": stage["id"], "family": stage.get("family"), "seed": stage.get("seed")}
        for split in ("val", "test"):
            for key in ("prediction_full_psnr", "lpips", "prediction_temporal_residual_l1",
                        "prediction_edge_gradient_l1", "prediction_edge_strength_ratio", "history_coverage", "head_ms", "forward_ms"):
                row[f"{split}_{key}"] = summary[split].get(key)
            row[f"{split}_high_motion_temporal"] = summary[split]["by_motion"]["high"].get("prediction_temporal_residual_l1")
        rows.append(row)
    if not rows:
        return
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    base = next(row for row in rows if row["family"] == "base")
    candidates = []
    for row in rows:
        # Prespecified validation-only quality constraints; test never ranks.
        if row["family"] in {"constant_normal", "shuffle_normal", "constant_roughness", "shuffle_roughness", "base"}:
            continue
        temporal, baseline = row.get("val_prediction_temporal_residual_l1"), base.get("val_prediction_temporal_residual_l1")
        psnr, base_psnr = row.get("val_prediction_full_psnr"), base.get("val_prediction_full_psnr")
        edge, base_edge = row.get("val_prediction_edge_gradient_l1"), base.get("val_prediction_edge_gradient_l1")
        ratio, base_ratio = row.get("val_prediction_edge_strength_ratio"), base.get("val_prediction_edge_strength_ratio")
        if (all(value is not None for value in (temporal, baseline, psnr, base_psnr, edge, base_edge, ratio, base_ratio))
                and temporal < baseline * .99 and psnr >= base_psnr - .05 and edge <= base_edge * 1.02
                and ratio >= base_ratio * .98):
            candidates.append(row)
    selected = min(candidates, key=lambda row: row["val_prediction_temporal_residual_l1"]) if candidates else base
    atomic_json(args.output_dir / "selection.json", {"selected": selected["stage"], "selection_split": "val",
        "rule": "temporal residual improves >1%, PSNR loss <=0.05dB, edge L1 <=1.02x, edge strength >=0.98x baseline",
        "status": "complete" if state.get("status") == "completed" else "provisional",
        "limitations": ["One scene/seed pilot unless explicitly repeated; no confirmatory significance claim",
                         "Previously inspected scene788 test frames are exploratory"]})
    if state.get("status") == "completed":
        atomic_json(args.output_dir / "matched_coverage.json", matched_coverage(plan))


def execute_plan(args, plan, run=subprocess.run):
    from tools.ego_motion_experiment import atomic_json
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".suite.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This ego-motion suite is already running") from exc
        state_path = output / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"signature": plan["signature"], "stages": {}}
        if state["signature"] != plan["signature"]:
            raise ValueError("Code, checkpoint or options changed; use a new --output-dir")
        atomic_json(output / "plan.json", plan)
        state.update(status="waiting", pid=os.getpid())
        atomic_json(state_path, state)
        wait_for_queue(args.wait_for_queue)
        # Recheck code after a potentially long wait; do not run silently edited code.
        for index, stage in enumerate(plan["stages"]):
            try:
                verify_live_inputs(plan)
            except BaseException as exc:
                state.update(status="failed", error=str(exc))
                atomic_json(state_path, state)
                raise
            record = state["stages"].get(stage["id"], {})
            if record.get("status") == "completed":
                if record.get("hashes") != stage_hashes(stage):
                    raise ValueError(f"Completed artifacts changed in {stage['id']}; use a new output directory")
                print(f"[{index+1}/{len(plan['stages'])}] skip {stage['id']}", flush=True)
                continue
            command = [sys.executable, "-m", stage["module"], *stage["args"]]
            if stage["kind"] == "material" and (Path(stage["output"]) / "resume_state.pt").exists():
                command.append("--resume")
            logs = output / "logs"
            logs.mkdir(exist_ok=True)
            attempt = int(record.get("attempt", 0)) + 1
            log_path = logs / f"{stage['id']}.{attempt:03d}.log"
            record = {"status": "running", "attempt": attempt, "command": command, "log": str(log_path), "started": time.time()}
            state["stages"][stage["id"]] = record
            state.update(status="running", active_stage=stage["id"])
            atomic_json(state_path, state)
            print(f"[{index+1}/{len(plan['stages'])}] {stage['id']} -> {log_path}", flush=True)
            try:
                with log_path.open("w") as log:
                    result = run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1", "MPLCONFIGDIR": "/tmp/3ablegs-mpl"})
                if result.returncode:
                    raise RuntimeError(f"{stage['id']} exited {result.returncode}; see {log_path}")
                record.update(status="completed", finished=time.time(), hashes=stage_hashes(stage))
                atomic_json(state_path, state)
                write_report(args, plan, state)
            except BaseException as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                state["status"] = "failed"
                atomic_json(state_path, state)
                raise
        state.update(status="completed", active_stage=None, finished=time.time())
        atomic_json(state_path, state)
        write_report(args, plan, state)
    return state


def main():
    args = make_parser().parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.wait_for_queue:
        args.wait_for_queue = args.wait_for_queue.expanduser().resolve()
    os.chdir(ROOT)
    plan = build_plan(args)
    print(f"scene {plan['scene']}: {len(plan['stages'])} sequential stages; "
          f"train/val/test={len(plan['train_frames'])}/{len(plan['val_frames'])}/{len(plan['test_frames'])}")
    for index, stage in enumerate(plan["stages"], 1):
        print(f"{index:02d}. {stage['id']}")
    if not args.execute:
        print("Preview only. Add --execute; optionally --detach to survive terminal closure.")
        return
    if args.detach:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / ".suite.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This ego-motion suite is already running") from exc
        state_path = args.output_dir / "state.json"
        if state_path.exists() and json.loads(state_path.read_text())["signature"] != plan["signature"]:
            raise ValueError("Code, checkpoint or options changed; use a new --output-dir")
        command = [sys.executable, "-u", "-m", "tools.run_ego_motion_experiments",
                   *[value for value in sys.argv[1:] if value != "--detach"]]
        with (args.output_dir / "runner.log").open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Spawned detached worker PID {child.pid}; log: {args.output_dir / 'runner.log'}")
        return
    execute_plan(args, plan)


if __name__ == "__main__":
    main()
