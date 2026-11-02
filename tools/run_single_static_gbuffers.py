#!/usr/bin/env python3
"""Rebuild historical tiled single-static G-buffers; preview unless --execute.

The recovered 788/100-frame recipe repeats each crop 26 times, runs a 57-frame
window, selects its first output and stitches two overlapping crops. Reducing
the source frame count does not change that teacher recipe. This tool never
starts Gaussian training or deletes intermediate data.
"""

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PASSES = ("normal", "basecolor", "roughness", "metallic")
REPEATS = 26


def source_frames(source, start, count):
    if start < 0:
        raise ValueError("--start-frame must be nonnegative")
    if count < 1:
        raise ValueError("--frames must be positive")
    expected = [source / f"{i:03d}_0.jpg" for i in range(start, start + count)]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise ValueError(
            f"Expected consecutive camera-0 frames {start:03d}..{start + count - 1:03d} "
            f"in {source}; missing {missing[0].name}"
        )
    for path in expected:
        with Image.open(path) as image:
            if image.size != (1920, 1280):
                raise ValueError(f"Unexpected source size {image.size}: {path}")
    return expected


def inverse_arguments(cosmos_root, work, dataset_path=None):
    dataset_path = dataset_path or work / "static_inputs"
    return [
        str(cosmos_root / "cosmos_predict1/diffusion/inference/inference_inverse_renderer_long.py"),
        "--checkpoint_dir", str(cosmos_root / "checkpoints"),
        "--diffusion_transformer_dir", "Diffusion_Renderer_Inverse_Cosmos_7B",
        "--dataset_path", str(dataset_path),
        "--num_video_frames", "57", "--stride", "32", "--pin_mode", "hybrid",
        "--height", "704", "--width", "1280",
        "--seed", "1000", "--num_steps", "15", "--guidance", "0",
        "--inference_passes", *PASSES, "--normalize_normal", "false",
        "--save_image_format", "png", "--save_video", "false",
        "--video_save_folder", str(work / "static_results"),
    ]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_png(path, expected_size):
    try:
        with Image.open(path) as image:
            if image.size != expected_size:
                return False
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, default=788)
    parser.add_argument("--weather", choices=("raw", "rain", "snow"), default="raw")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="First original frame ID to generate (inclusive)")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--cosmos-root", type=Path, default=ROOT.parent / "cosmos-transfer1-diffusion-renderer")
    parser.add_argument("--conda", type=Path, default=Path("/home/user/miniconda3/bin/conda"))
    parser.add_argument("--cosmos-env", type=Path, default=Path("/home/user/miniconda3/envs/cosmos-dr"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output-root", type=Path, help="Directory to contain gbuffer_frames/<weather>")
    parser.add_argument("--allow-existing-output", action="store_true",
                        help="Append only missing requested frame IDs to an existing destination")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse a matching interrupted work directory and generate only incomplete clips")
    parser.add_argument("--execute", action="store_true", help="Generate G-buffers; otherwise only validate/print the plan")
    args = parser.parse_args()
    if args.scene < 0:
        parser.error("--scene must be nonnegative")
    source = ROOT / f"data/waymo/processed/training/{args.scene:03d}/images/{args.weather}"
    frames = source_frames(source, args.start_frame, args.frames)
    end_frame = args.start_frame + args.frames
    frame_ids = list(range(args.start_frame, end_frame))
    if args.start_frame == 0:
        work_name = f"scene_{args.scene:03d}_{args.weather}_single_static_{args.frames}f"
    else:
        work_name = (
            f"scene_{args.scene:03d}_{args.weather}_single_static_"
            f"{args.start_frame:04d}_{end_frame - 1:04d}"
        )
    work = (args.work_dir or ROOT / "output/reconstruction" / work_name).resolve()
    output = (args.output_root or ROOT / f"data/waymo/render_png_{args.scene:03d}_single_static_{end_frame}f").resolve()
    destination = output / "gbuffer_frames" / args.weather
    cosmos = args.cosmos_root.resolve()
    prepare = ROOT / "tools/prepare_tiled_dr_inputs.py"
    stitch = ROOT / "tools/stitch_tiled_gbuffers.py"
    inverse_args = inverse_arguments(cosmos, work)
    required = [prepare, stitch, Path(inverse_args[0]), args.conda,
                args.cosmos_env / "bin/python",
                cosmos / "checkpoints/Diffusion_Renderer_Inverse_Cosmos_7B/model.pt"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    conda_python = [str(args.conda), "run", "--no-capture-output", "--prefix", str(args.cosmos_env), "python", "-u"]
    plan = {
        "recipe": "historical_single_static_top_bottom",
        "scene": args.scene, "weather": args.weather, "camera": 0,
        "frame_ids": frame_ids, "source": str(source),
        "work_dir": str(work), "destination": str(destination),
        "static_copies_per_tile": REPEATS, "static_clips": args.frames * 2,
        "selected_clip_frame": 0, "final_png_count": args.frames * len(PASSES),
        "final_size_wh": [1920, 1280],
        "inference_command": [*conda_python, *inverse_args],
        "note": "No training. Diffusion outputs are pseudo-labels; intermediates are retained.",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Preview only; no inputs, G-buffers or model instances were created.")
        return 0
    manifest = work / "generation.json"
    if args.resume and not work.is_dir():
        raise FileNotFoundError(f"Cannot resume missing work directory: {work}")
    if not args.resume and work.exists():
        raise FileExistsError(f"Work directory already exists; nothing was overwritten: {work}")
    if destination.exists() and not (args.allow_existing_output or args.resume):
        raise FileExistsError(
            "Output already exists. Use --allow-existing-output to append non-overlapping frames; "
            "nothing was overwritten."
        )
    collisions = [
        destination / f"0000.{frame:04d}.{name}.png"
        for frame in frame_ids for name in PASSES
        if (destination / f"0000.{frame:04d}.{name}.png").exists()
    ]
    if collisions and not args.resume:
        raise FileExistsError(f"Requested output already exists; nothing was overwritten: {collisions[0]}")
    if args.resume:
        if not manifest.is_file():
            raise FileNotFoundError(f"Cannot validate resume without manifest: {manifest}")
        previous = json.loads(manifest.read_text())
        expected = {
            "scene": args.scene,
            "weather": args.weather,
            "frame_ids": frame_ids,
            "source": str(source),
            "destination": str(destination),
        }
        mismatches = {
            key: (previous.get(key), value)
            for key, value in expected.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Resume arguments do not match existing manifest: {mismatches}")
    work.mkdir(parents=True, exist_ok=args.resume)
    plan["source_sha256"] = {path.name: digest(path) for path in frames}
    plan["code_sha256"] = {str(path): digest(path) for path in (prepare, stitch, Path(inverse_args[0]), Path(__file__))}
    plan["resume"] = args.resume

    def status(stage):
        plan["stage"] = stage
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"stage={stage}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(cosmos)
    env["CUDA_HOME"] = str(args.cosmos_env)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    try:
        tiled = work / "tiled_inputs"
        expected_tiles = [
            tiled / f"{args.weather}_{position}" / f"{frame:03d}.png"
            for position in ("top", "bottom") for frame in frame_ids
        ]
        if not args.resume or not all(valid_png(path, (1280, 704)) for path in expected_tiles):
            status("prepare_tiles")
            subprocess.run([*conda_python, str(prepare), "--src", str(source), "--dst", str(tiled),
                            "--start-frame", str(args.start_frame), "--frames", str(args.frames),
                            "--label", args.weather], cwd=ROOT, env=env, check=True)
        else:
            status("resume_validate_inputs")
        for position in ("top", "bottom"):
            for frame in frame_ids:
                clip = work / "static_inputs" / f"{args.weather}_{position}_{frame:03d}"
                clip.mkdir(parents=True, exist_ok=args.resume)
                for local_frame in range(REPEATS):
                    src = tiled / f"{args.weather}_{position}/{frame:03d}.png"
                    dst = clip / f"{local_frame:03d}.png"
                    if not dst.exists():
                        os.link(src, dst)

        clip_names = [
            f"{args.weather}_{position}_{frame:03d}"
            for position in ("top", "bottom") for frame in frame_ids
        ]
        results = work / "static_results" / "gbuffer_frames"

        def clip_complete(name):
            return all(
                valid_png(results / name / f"full.00000.{pass_name}.png", (1280, 704))
                for pass_name in PASSES
            )

        completed = [name for name in clip_names if clip_complete(name)]
        pending = [name for name in clip_names if name not in completed]
        plan["resume_completed_clips"] = len(completed)
        plan["pending_clips"] = len(pending)
        if pending:
            input_root = work / f"pending_inputs_{len(completed):04d}_{os.getpid()}"
            input_root.mkdir(parents=True)
            for name in pending:
                clip = input_root / name
                clip.mkdir()
                source_clip = work / "static_inputs" / name
                for local_frame in range(REPEATS):
                    os.link(source_clip / f"{local_frame:03d}.png", clip / f"{local_frame:03d}.png")
            inverse_args = inverse_arguments(cosmos, work, dataset_path=input_root)
            plan["inference_command"] = [*conda_python, *inverse_args]
            status("inverse_rendering")
            subprocess.run([*conda_python, *inverse_args], cwd=cosmos, env=env, check=True)
        else:
            status("inverse_rendering_already_complete")

        incomplete = [name for name in clip_names if not clip_complete(name)]
        if incomplete:
            raise RuntimeError(f"Inverse rendering left incomplete clip: {incomplete[0]}")
        status("select_and_stitch")
        selected = work / "selected_tiles"
        for position in ("top", "bottom"):
            tile_dst = selected / "gbuffer_frames" / f"{args.weather}_{position}"
            tile_dst.mkdir(parents=True, exist_ok=args.resume)
            for frame in frame_ids:
                clip = work / "static_results/gbuffer_frames" / f"{args.weather}_{position}_{frame:03d}"
                for name in PASSES:
                    src = clip / f"full.00000.{name}.png"
                    dst = tile_dst / f"full.{frame:05d}.{name}.png"
                    if not dst.exists():
                        os.link(src, dst)
        final = work / "final_1920x1280" / args.weather
        subprocess.run([sys.executable, str(stitch), "--tile-root", str(selected), "--dst", str(final),
                        "--start-frame", str(args.start_frame), "--frames", str(args.frames),
                        "--condition", args.weather, "--passes", *PASSES],
                       cwd=ROOT, check=True)
        # Validate before publishing any loader-facing files.
        for frame in frame_ids:
            for name in PASSES:
                with Image.open(final / f"full.{frame:05d}.{name}.png") as image:
                    if image.size != (1920, 1280):
                        raise ValueError("Stitched output dimensions differ from the historical contract")
                    image.verify()
        destination.mkdir(parents=True, exist_ok=True)
        for frame in frame_ids:
            for name in PASSES:
                src = final / f"full.{frame:05d}.{name}.png"
                dst = destination / f"0000.{frame:04d}.{name}.png"
                if dst.exists() and args.resume:
                    if not valid_png(dst, (1920, 1280)):
                        raise ValueError(f"Existing resumed output is invalid: {dst}")
                    continue
                try:
                    os.link(src, dst)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    shutil.copyfile(src, dst)
        status("completed")
        print(f"Completed: {plan['final_png_count']} PNGs in {destination}", flush=True)
    except Exception as exc:
        plan["error"] = f"{type(exc).__name__}: {exc}"
        status("failed")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
