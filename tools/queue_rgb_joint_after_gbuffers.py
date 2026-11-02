#!/usr/bin/env python3
"""Wait for validated single-static G-buffers and an idle GPU, then train."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def generation_stage(path, scene):
    try:
        manifest = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return "pending"
    if manifest.get("scene") != scene or manifest.get("weather") != "raw":
        raise ValueError("G-buffer manifest scene/weather mismatch")
    if manifest.get("frame_ids") != list(range(100)):
        raise ValueError("Expected frames 0 through 99")
    if manifest.get("stage") == "failed":
        raise RuntimeError(f"G-buffer generation failed: {manifest.get('error')}")
    return manifest.get("stage", "pending")


def gpu_compute_pids():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    return [int(row.strip()) for row in result.stdout.splitlines() if row.strip().isdigit()]


def validate_pngs(folder):
    from PIL import Image
    for frame in range(100):
        for kind in ("normal", "basecolor", "roughness", "metallic"):
            path = folder / "raw" / f"0000.{frame:04d}.{kind}.png"
            with Image.open(path) as image:
                if image.size != (1920, 1280):
                    raise ValueError(f"Unexpected G-buffer dimensions: {path}")
                image.verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.scene < 0:
        parser.error("scene must be nonnegative")
    os.chdir(ROOT)
    scene = f"{args.scene:03d}"
    manifest = ROOT / f"output/reconstruction/scene_{scene}_raw_single_static_100f/generation.json"
    buffers = ROOT / f"data/waymo/render_png_{scene}_single_static_100f/gbuffer_frames"
    run_dir = ROOT / f"output/pbr/rebuild/scene_{scene}_100f/base_v5"
    command = ["bash", str(ROOT / "run_scene788_base_v5.sh"),
               "--run-dir", str(run_dir), f"data.scene_idx={args.scene}",
               f"data.pixel_source.gbuffer_path={buffers.relative_to(ROOT)}"]
    if args.check_only:
        print(json.dumps({"stage": generation_stage(manifest, args.scene),
                          "run_dir": str(run_dir), "command": command}, indent=2))
        return
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    status_path = logs / f"scene{scene}_rgb_joint_queue_status.json"
    training_log = logs / f"scene{scene}_base_v5.log"
    with (logs / f"scene{scene}_rgb_joint_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A queue for this scene is already running")
        previous = None

        def status(state, **details):
            nonlocal previous
            value = {"state": state, "scene": args.scene, "queue_pid": os.getpid(),
                     "run_dir": str(run_dir), "training_log": str(training_log), **details}
            signature = json.dumps(value, sort_keys=True)
            if signature != previous:
                value["updated_at"] = datetime.datetime.now().astimezone().isoformat()
                temporary = status_path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(value, indent=2) + "\n")
                temporary.replace(status_path)
                print(json.dumps(value), flush=True)
                previous = signature

        try:
            while True:
                stage = generation_stage(manifest, args.scene)
                if stage != "completed":
                    status("waiting_for_gbuffers", generation_stage=stage)
                else:
                    pids = gpu_compute_pids()
                    if pids:
                        status("waiting_for_gpu", compute_pids=pids)
                    else:
                        break
                time.sleep(30)
            status("validating_inputs")
            validate_pngs(buffers)
            # Make the total budget and automatic evaluation independent of
            # variables inherited from an interactive experiment shell.
            env = os.environ.copy()
            for key in ("ITERATIONS", "SKIP_EVAL", "RUN_TAG"):
                env.pop(key, None)
            env["ITERATIONS"] = "30001"
            env["RGB_JOINT_PYTHON"] = str(ROOT / ".conda/bin/python")
            subprocess.run(command + ["--prepare-only"], env=env, check=True)
            # Validate first, then recheck resources immediately before launch.
            while pids := gpu_compute_pids():
                status("waiting_for_gpu", compute_pids=pids)
                time.sleep(30)
            with training_log.open("a") as output:
                child = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
                status("training_and_evaluating", training_pid=child.pid)
                result = child.wait()
            if result:
                raise RuntimeError(f"Training/evaluation exited {result}; see {training_log}")
            status("completed")
        except Exception as error:
            status("failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
