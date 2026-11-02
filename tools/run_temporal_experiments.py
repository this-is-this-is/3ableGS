#!/usr/bin/env python3
"""Preview or sequentially run frozen, teacher and held-out retraining experiments.

Preview is read-only. --execute runs one subprocess at a time and persists every
attempt. Completed artifacts are checked before reuse. Interrupted training is
restarted from scratch in a new folder, not resumed with optimizer state.
"""

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.temporal_split import resolve_test_frames


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--downscale", type=float, default=2.)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seeds", default="0", help="Comma-separated training seeds")
    parser.add_argument("--holdout-ranges", default="20:30,60:70")
    parser.add_argument("--cosmos-root", type=Path, default=ROOT.parent / "cosmos-transfer1-diffusion-renderer")
    parser.add_argument("--cosmos-env", type=Path, default=Path("/home/user/miniconda3/envs/cosmos-dr"))
    parser.add_argument("--conda", type=Path, default=Path("/home/user/miniconda3/bin/conda"))
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--only", choices=("all", "frozen"), default="all")
    parser.add_argument("--execute", action="store_true")
    return parser


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, prefix=path.name + ".") as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def local_path(value):
    value = Path(value).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def parse_ranges(text, total):
    try:
        ranges = [[int(value) for value in part.strip().split(":")] for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--holdout-ranges must be sorted start:end integer ranges.") from exc
    heldout = resolve_test_frames(total, ranges)
    if not heldout:
        raise ValueError("The all suite requires at least one held-out frame.")
    return ranges


def variants():
    """Fixed hypotheses, specified before looking at held-out scores."""
    result = [
        ("baseline", ["--history-weight", "0"]),
        ("rgb_normal_w030", ["--domain", "rgb", "--mode", "normal", "--history-weight", ".3"]),
        ("depth_w030", ["--mode", "depth", "--history-weight", ".3"]),
        ("normal_w030", ["--mode", "normal", "--history-weight", ".3"]),
    ]
    result += [(f"material_w{tag}", ["--history-weight", value])
               for tag, value in (("085", ".85"), ("050", ".5"), ("030", ".3"), ("010", ".1"))]
    result += [
        ("material_current_w030", ["--history-weight", ".3", "--history-source", "current"]),
        ("material_clamp_w030", ["--history-weight", ".3", "--clamp-history"]),
        ("material_strict_w030", ["--history-weight", ".3", "--normal-cos-min", ".98",
                                 "--depth-rel-tolerance", ".01"]),
    ]
    result += [(f"guide_{name}", ["--history-weight", ".3", "--guide-ablation", name])
               for name in ("shuffle_normal", "shuffle_roughness", "constant_normal", "constant_roughness")]
    return result


def training_config(saved, args, ranges, seed, teacher, temporal):
    cfg = copy.deepcopy(saved)
    cfg.pop("log_dir", None)
    cfg.seed = seed
    cfg.trainer.optim.num_iters = args.iterations
    cfg.trainer.optim.cache_buffer_freq = -1
    cfg.trainer.losses.temporal_material = {
        "w": .05 if temporal else 0., "start_step": min(3000, args.iterations // 10), "max_gap": 1,
    }
    cfg.data.preload_device = "cpu"
    cfg.data.pixel_source.test_frame_ranges = ranges
    cfg.data.pixel_source.test_image_stride = 0
    cfg.data.pixel_source.load_weather = []
    cfg.data.pixel_source.load_gbuffer = True
    if teacher == "video":
        cfg.data.pixel_source.gbuffer_path = str(args.output_dir / "video_teacher/data/gbuffer_frames")
        cfg.data.pixel_source.gbuffer_seq_id = 0
    cfg.logging.vis_freq = 0
    cfg.render.render_test = False
    cfg.render.render_full = False
    cfg.render.render_novel = None
    return OmegaConf.to_container(cfg, resolve=True)


def build_plan(args):
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    saved_path = args.checkpoint.parent / "config.yaml"
    saved = OmegaConf.load(saved_path)
    if str(saved.trainer.type) not in (
            "models.trainers.MultiTrainer_pbr", "models.trainers.scene_graph_pbr.MultiTrainer_pbr"):
        raise ValueError("The suite requires a raw Base v4 PBR checkpoint.")
    if list(saved.data.pixel_source.cameras) != [0]:
        raise ValueError("The controlled suite currently requires camera 0 only.")
    if args.iterations < 1 or not math.isfinite(args.downscale) or args.downscale < 1:
        raise ValueError("--iterations must be positive and --downscale finite and >= 1.")
    try:
        seeds = [int(value) for value in args.seeds.split(",")]
    except ValueError as exc:
        raise ValueError("--seeds must contain comma-separated nonnegative integers.") from exc
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("--seeds must be nonempty, unique and nonnegative.")
    scene_root = local_path(saved.data.data_root) / f"{int(saved.data.scene_idx):03d}"
    start, end = int(saved.data.start_timestep), int(saved.data.end_timestep)
    if end == -1:
        end = len(list((scene_root / "ego_pose").iterdir())) - 1
    total = end - start + 1
    count = total if args.num_frames is None else args.num_frames
    if not 1 <= count <= total:
        raise ValueError("--num-frames must lie within the saved complete sequence.")
    source_assets = {}
    for frame in range(start, end + 1):
        source = scene_root / "images/raw" / f"{frame:03d}_0.jpg"
        if not source.is_file():
            raise FileNotFoundError(source)
        source_assets[str(source)] = digest(source)
    scripts = ("render_temporal_material.py", "run_video_gbuffers.py", "train.py")
    for name in scripts:
        if not (ROOT / "tools" / name).is_file():
            raise FileNotFoundError(ROOT / "tools" / name)
    ranges, teacher_plan, teacher_assets = [], None, {}
    if args.only == "all":
        ranges = parse_ranges(args.holdout_ranges, total)
        from tools.run_video_gbuffers import build_plan as teacher_build_plan
        teacher_plan = teacher_build_plan(argparse.Namespace(
            scene=int(saved.data.scene_idx), start_frame=start, frames=total,
            holdout_ranges=args.holdout_ranges, work_dir=args.output_dir / "video_teacher/work",
            output_root=args.output_dir / "video_teacher/data", cosmos_root=args.cosmos_root,
            cosmos_env=args.cosmos_env, conda=args.conda,
        ))
        expected_source = ROOT / f"data/waymo/processed/training/{int(saved.data.scene_idx):03d}"
        if scene_root.resolve() != expected_source.resolve():
            raise ValueError("Video teacher currently requires the repository's processed Waymo data root.")
        teacher_root = local_path(saved.data.pixel_source.gbuffer_path) / "raw"
        for frame in range(start, end + 1):
            for name in ("normal", "basecolor", "roughness", "metallic"):
                path = teacher_root / f"{int(saved.data.pixel_source.gbuffer_seq_id):04d}.{frame:04d}.{name}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
                teacher_assets[str(path)] = digest(path)
    stages = [
        {"id": name, "kind": "render", "group": "frozen", "args": extra, "frames": count}
        for name, extra in variants()
    ]
    for affinity in ("geometry", "rgb", "gbuffer"):
        for keyframes in (1, 3):
            stages.append({
                "id": f"edit_{affinity}_k{keyframes}", "kind": "render", "group": "surface_edit",
                "args": ["--history-weight", "0", "--edit-affinity", affinity,
                         "--edit-keyframes", str(keyframes)], "frames": count,
            })
    if args.only == "all":
        stages.append({"id": "video_teacher", "kind": "teacher", "group": "teacher"})
        for seed in seeds:
            for teacher, temporal in (("static", False), ("static", True), ("video", False), ("video", True)):
                name = f"{teacher}_{'temporal_loss' if temporal else 'teacher'}_seed{seed}"
                stage = {
                    "id": name, "kind": "train", "group": "heldout", "seed": seed,
                    "dependencies": ["video_teacher"] if teacher == "video" else [],
                    "config": training_config(saved, args, ranges, seed, teacher, temporal),
                }
                stages.append(stage)
                for suffix, extra in (("baseline", ["--history-weight", "0"]),
                                      ("clamp_w030", ["--history-weight", ".3", "--clamp-history"])):
                    stages.append({
                        "id": name + "_" + suffix, "kind": "render", "group": "heldout",
                        "seed": seed, "dependencies": [name], "checkpoint_stage": name,
                        "args": ["--evaluation-split", "test", *extra],
                        "frames": len(resolve_test_frames(total, ranges)),
                    })
    code_paths = sorted({path for folder in ("models", "datasets", "utils", "tools")
                         for path in (ROOT / folder).rglob("*.py")})
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
               if key != "execute"}
    identity = {
        "schema": 1, "options": options, "full_frames": total, "render_frames": count,
        "source_checkpoint": {"path": str(args.checkpoint), "bytes": args.checkpoint.stat().st_size,
                              "mtime_ns": args.checkpoint.stat().st_mtime_ns},
        "source_config": {"path": str(saved_path), "sha256": digest(saved_path)},
        "source_rgb_sha256": source_assets,
        "static_teacher_sha256": teacher_assets,
        "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in code_paths},
        "teacher_signature": None if teacher_plan is None else teacher_plan["signature"],
        "stages": stages,
    }
    return {"signature": signature(identity), "identity": identity, "teacher_plan": teacher_plan,
            "scene": int(saved.data.scene_idx), "start": start, "ranges": ranges, "stages": stages}


def latest(state, stage_id):
    return state["stages"][stage_id]["attempts"][-1]


def stage_output(args, stage, attempt):
    if stage["kind"] == "teacher":
        return args.output_dir / "video_teacher/work"
    folder = "train" if stage["kind"] == "train" else "renders"
    return args.output_dir / folder / stage["id"] / f"attempt_{attempt:03d}"


def stage_command(args, plan, stage, attempt, state):
    output = stage_output(args, stage, attempt)
    python = sys.executable
    if stage["kind"] == "render":
        checkpoint = args.checkpoint
        if "checkpoint_stage" in stage:
            record = latest(state, stage["checkpoint_stage"]) if state else None
            checkpoint = ((Path(record["output"]) if record else
                           args.output_dir / "train" / stage["checkpoint_stage"] / "attempt_001")
                          / "checkpoint_final.pth")
        command = [
            python, str(ROOT / "tools/render_temporal_material.py"),
            "--checkpoint", str(checkpoint), "--output-dir", str(output),
            "--num-frames", str(plan["identity"]["full_frames"] if "checkpoint_stage" in stage
                                else plan["identity"]["render_frames"]),
            "--downscale", str(args.downscale),
            "--evaluation-cache", str(args.output_dir / "evaluation_cache"), *stage["args"],
        ]
        if args.save_images:
            command.append("--save-images")
        return command
    if stage["kind"] == "teacher":
        command = [
            python, str(ROOT / "tools/run_video_gbuffers.py"), "--scene", str(plan["scene"]),
            "--start-frame", str(plan["start"]), "--frames", str(plan["identity"]["full_frames"]),
            "--holdout-ranges", args.holdout_ranges, "--work-dir", str(output),
            "--output-root", str(args.output_dir / "video_teacher/data"),
            "--cosmos-root", str(args.cosmos_root), "--cosmos-env", str(args.cosmos_env),
            "--conda", str(args.conda), "--execute",
        ]
        if (output / "generation.json").is_file():
            command.append("--resume")
        return command
    config = args.output_dir / "configs" / stage["id"] / f"attempt_{attempt:03d}.yaml"
    return [python, "-m", "tools.train", "--config_file", str(config),
            "--output_root", str(args.output_dir / "train"), "--project", stage["id"],
            "--run_name", f"attempt_{attempt:03d}"]


def verify_artifacts(args, plan, stage, record):
    """Check completion markers and required outputs; return byte identities."""
    output = Path(record["output"])
    files = []
    if stage["kind"] == "render":
        manifest = json.loads((output / "manifest.json").read_text())
        summary = json.loads((output / "summary.json").read_text())
        if (manifest.get("status") != "completed" or summary.get("frames") != stage["frames"]
                or manifest.get("rendered_frames") != stage["frames"]):
            raise ValueError(f"Incomplete render: {output}")
        if len((output / "frames.jsonl").read_text().splitlines()) != stage["frames"]:
            raise ValueError(f"Render frame records are incomplete: {output}")
        files = [output / name for name in ("manifest.json", "summary.json", "frames.jsonl", "comparison.mp4")]
        if args.save_images:
            for name in ("baseline", "transport", "weight"):
                images = sorted((output / name).glob("*.png"))
                if len(images) != stage["frames"]:
                    raise ValueError(f"Missing saved images: {output / name}")
                files.extend(images)
    elif stage["kind"] == "train":
        saved = OmegaConf.to_container(OmegaConf.load(output / "config.yaml"), resolve=True)
        saved.pop("log_dir", None)
        if saved != stage["config"]:
            raise ValueError(f"Training configuration differs: {output}")
        files = [output / "checkpoint_final.pth", output / "config.yaml"]
    else:
        manifest = json.loads((output / "generation.json").read_text())
        expected = plan["teacher_plan"]
        if (manifest.get("stage") != "completed" or manifest.get("signature") != expected["signature"]
                or len(manifest.get("final_sha256", {})) != expected["final_png_count"]):
            raise ValueError(f"Incomplete or mismatched video teacher: {output}")
        from tools.run_video_gbuffers import valid_png, SOURCE_SIZE
        destination = args.output_dir / "video_teacher/data/gbuffer_frames/raw"
        for name, checksum in manifest["final_sha256"].items():
            path = destination / name
            if not valid_png(path, SOURCE_SIZE) or digest(path) != checksum:
                raise ValueError(f"Video teacher output is missing or changed: {path}")
        files = [output / "generation.json"]
    identities = {}
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required output is missing or empty: {path}")
        identities[str(path)] = digest(path)
    if record.get("artifacts") is not None and record["artifacts"] != identities:
        raise ValueError(f"Completed artifact bytes changed: {output}")
    return identities


def aggregate(args, plan, state):
    rows = []
    for stage in plan["stages"]:
        if stage["kind"] != "render" or stage["id"] not in state["stages"]:
            continue
        record = latest(state, stage["id"])
        if record["status"] != "completed":
            continue
        summary = json.loads((Path(record["output"]) / "summary.json").read_text())
        metrics = summary.get("reference_evaluation", {})
        for method in ("baseline", "transport"):
            edited = stage["group"] == "surface_edit"
            rows.append({
                "stage": stage["id"], "group": stage["group"], "seed": stage.get("seed"),
                "method": method, "frames": summary["frames"],
                "rgb_psnr": None if edited else summary.get(f"{method}_full_psnr"),
                "temporal_residual_l1": metrics.get(f"{method}_temporal_residual_l1"),
                "edge_gradient_l1": metrics.get(f"{method}_edge_gradient_l1"),
                "edge_strength_ratio": metrics.get(f"{method}_edge_strength_ratio"),
                "edit_coverage": summary.get("mean_edit_coverage"),
                "summary": str(Path(record["output"]) / "summary.json"),
            })
    payload = {
        "signature": plan["signature"], "rows": rows,
        "notes": [
            "No held-out score is used to select hyperparameters or choose a best result.",
            "Surface edit RGB PSNR is omitted: the raw reference is not an edited ground truth.",
            "Reference flow is an estimate; poses/LiDAR/instance annotations are fixed all-sequence priors.",
            "Interrupted training restarts from scratch in a fresh attempt directory.",
        ],
    }
    atomic_json(args.output_dir / "suite_summary.json", payload)
    destination = args.output_dir / "suite_summary.csv"
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["stage"])
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def execute_plan(args, plan, run=subprocess.run):
    output = args.output_dir
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".suite.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is already running this suite.") from exc
        state_path = output / "suite.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            if state.get("signature") != plan["signature"]:
                raise ValueError("Suite options, source checkpoint/config or code changed; choose a new output directory.")
        else:
            if any(path.name != ".suite.lock" for path in output.iterdir()):
                raise ValueError("Output contains unrelated files without a suite manifest; choose a new directory.")
            state = {"signature": plan["signature"], "identity": plan["identity"], "stages": {}}
            atomic_json(state_path, state)
        for index, stage in enumerate(plan["stages"], 1):
            dependencies = {name: latest(state, name)["artifacts"]
                            for name in stage.get("dependencies", [])}
            stage_signature = signature({"suite": plan["signature"], "stage": stage, "dependencies": dependencies})
            progress = state["stages"].setdefault(stage["id"], {"attempts": []})
            if progress["attempts"]:
                last = progress["attempts"][-1]
                if last["status"] == "completed" and last["signature"] == stage_signature:
                    try:
                        verify_artifacts(args, plan, stage, last)
                    except (OSError, ValueError, KeyError) as exc:
                        print(f"[{index}/{len(plan['stages'])}] Rebuilding invalid {stage['id']}: {exc}", flush=True)
                    else:
                        print(f"[{index}/{len(plan['stages'])}] Verified complete; skip {stage['id']}", flush=True)
                        continue
            attempt = len(progress["attempts"]) + 1
            command = stage_command(args, plan, stage, attempt, state)
            log = output / "logs" / stage["id"] / f"attempt_{attempt:03d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "status": "running", "signature": stage_signature,
                "output": str(stage_output(args, stage, attempt)), "log": str(log),
                "command": command, "started": datetime.now(timezone.utc).isoformat(),
            }
            progress["attempts"].append(record)
            atomic_json(state_path, state)
            try:
                if stage["kind"] == "train":
                    config_path = Path(command[command.index("--config_file") + 1])
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    OmegaConf.save(OmegaConf.create(stage["config"]), config_path)
                print(f"[{index}/{len(plan['stages'])}] {stage['id']} attempt {attempt}; log: {log}", flush=True)
                with log.open("x") as handle:
                    result = run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f"Stage exited with code {result.returncode}; see {log}")
                record["artifacts"] = verify_artifacts(args, plan, stage, record)
                record["status"] = "completed"
            except BaseException as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                atomic_json(state_path, state)
                aggregate(args, plan, state)
                raise
            record["finished"] = datetime.now(timezone.utc).isoformat()
            atomic_json(state_path, state)
            aggregate(args, plan, state)
        aggregate(args, plan, state)
        return state


def main(argv=None):
    args = make_parser().parse_args(argv)
    plan = build_plan(args)
    print(f"{len(plan['stages'])} sequential stages; full timeline={plan['identity']['full_frames']} frames.")
    if not args.execute:
        for index, stage in enumerate(plan["stages"], 1):
            print(f"{index:02d}. {stage['id']}\n    {shlex.join(stage_command(args, plan, stage, 1, None))}")
        print("Preview only; no files or GPU jobs created. Add --execute to run.")
        return 0
    execute_plan(args, plan)
    print(f"Completed. Summary: {args.output_dir / 'suite_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
