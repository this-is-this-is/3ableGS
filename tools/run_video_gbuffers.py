#!/usr/bin/env python3
"""Generate temporally conditioned G-buffers without crossing heldout splits.

Preview is the default. Every maximal train/test segment is an independent
Cosmos clip, with ordered video frames and endpoint-only padding inside that
segment. The four final maps follow the existing camera-0 Waymo loader naming.
Intermediate inputs/results and exact frame provenance are retained.
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
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_tiled_dr_inputs import CROPS, DR_SIZE, SOURCE_SIZE
from tools.stitch_tiled_gbuffers import blend, resize_pass

PASSES = ("normal", "basecolor", "roughness", "metallic")
WINDOW = 57
STRIDE = 32


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=int, default=788)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--holdout-ranges", default="20:30,60:70",
                        help="Comma-separated relative start:end ranges; end is exclusive")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Contains gbuffer_frames/raw")
    parser.add_argument("--cosmos-root", type=Path, default=ROOT.parent / "cosmos-transfer1-diffusion-renderer")
    parser.add_argument("--conda", type=Path, default=Path("/home/user/miniconda3/bin/conda"))
    parser.add_argument("--cosmos-env", type=Path, default=Path("/home/user/miniconda3/envs/cosmos-dr"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume only a matching manifest; existing unrelated files are never replaced")
    return parser


def parse_holdout_ranges(text, count):
    if count < 1:
        raise ValueError("--frames must be positive")
    ranges = []
    if text.strip():
        for token in text.split(","):
            try:
                start, end = (int(value) for value in token.strip().split(":"))
            except (ValueError, TypeError):
                raise ValueError("--holdout-ranges must contain start:end pairs") from None
            if not 0 <= start < end <= count:
                raise ValueError(f"Holdout range {token!r} must satisfy 0 <= start < end <= --frames")
            ranges.append((start, end))
    ranges.sort()
    if any(right[0] < left[1] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("Holdout ranges must not overlap")
    return ranges


def split_segments(start, count, ranges):
    """Return maximal same-role consecutive segments, using global frame IDs."""
    if start < 0 or count < 1:
        raise ValueError("Frame start must be nonnegative and count positive")
    heldout = {index for first, last in ranges for index in range(first, last)}
    if heldout and (min(heldout) < 0 or max(heldout) >= count):
        raise ValueError("Holdout range lies outside requested frames")
    segments = []
    for relative in range(count):
        role = "test" if relative in heldout else "train"
        frame = start + relative
        if not segments or segments[-1]["role"] != role:
            segments.append({
                "name": f"raw_{role}_{frame:06d}", "role": role, "frame_ids": [],
            })
        segments[-1]["frame_ids"].append(frame)
    return segments


def padded_frame_ids(frame_ids):
    if not frame_ids:
        raise ValueError("A clip needs at least one source frame")
    return list(frame_ids) + [frame_ids[-1]] * max(0, WINDOW - len(frame_ids))


def output_filename(frame, pass_name):
    return f"0000.{frame:04d}.{pass_name}.png"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def valid_png(path, size):
    try:
        with Image.open(path) as image:
            if image.size != size or image.format != "PNG":
                return False
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def inverse_command(args, dataset_path, result_path):
    cosmos = args.cosmos_root.expanduser().resolve()
    return [
        str(args.conda.expanduser().resolve()), "run", "--no-capture-output",
        "--prefix", str(args.cosmos_env.expanduser().resolve()), "python", "-u",
        str(cosmos / "cosmos_predict1/diffusion/inference/inference_inverse_renderer_long.py"),
        "--checkpoint_dir", str(cosmos / "checkpoints"),
        "--diffusion_transformer_dir", "Diffusion_Renderer_Inverse_Cosmos_7B",
        "--dataset_path", str(dataset_path), "--group_mode", "folder",
        "--num_video_frames", str(WINDOW), "--stride", str(STRIDE), "--pin_mode", "hybrid",
        "--height", str(DR_SIZE[1]), "--width", str(DR_SIZE[0]),
        "--seed", "1000", "--num_steps", "15", "--guidance", "0",
        "--inference_passes", *PASSES, "--normalize_normal", "false",
        "--save_image", "true", "--save_image_format", "png", "--save_video", "false",
        "--video_save_folder", str(result_path),
    ]


def build_plan(args):
    if args.scene < 0:
        raise ValueError("--scene must be nonnegative")
    ranges = parse_holdout_ranges(args.holdout_ranges, args.frames)
    segments = split_segments(args.start_frame, args.frames, ranges)
    source = ROOT / f"data/waymo/processed/training/{args.scene:03d}/images/raw"
    cosmos = args.cosmos_root.expanduser().resolve()
    inverse = cosmos / "cosmos_predict1/diffusion/inference/inference_inverse_renderer_long.py"
    checkpoint = cosmos / "checkpoints/Diffusion_Renderer_Inverse_Cosmos_7B/model.pt"
    required = [inverse, checkpoint, args.conda.expanduser().resolve(),
                args.cosmos_env.expanduser().resolve() / "bin/python"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    frame_ids = list(range(args.start_frame, args.start_frame + args.frames))
    sources = [source / f"{frame:03d}_0.jpg" for frame in frame_ids]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            if image.size != SOURCE_SIZE:
                raise ValueError(f"Unexpected source size {image.size}; expected {SOURCE_SIZE}: {path}")
            image.verify()
    code = [Path(__file__), Path(resize_pass.__code__.co_filename),
            ROOT / "tools/prepare_tiled_dr_inputs.py", inverse]
    # Track helper changes that affect Cosmos' actual grouping/padding when installed.
    helpers = cosmos / "cosmos_predict1/diffusion/inference/diffusion_renderer_utils"
    code.extend(path for path in (helpers / "dataset_inference.py", helpers / "inference_utils.py")
                if path.is_file())
    provenance, clips = [], []
    for segment in segments:
        conditioned = padded_frame_ids(segment["frame_ids"])
        for position in CROPS:
            clips.append({
                "name": segment["name"] + "_" + position,
                "segment": segment["name"], "role": segment["role"], "position": position,
                "frame_ids": segment["frame_ids"], "conditioning_frame_ids": conditioned,
            })
        for local, frame in enumerate(segment["frame_ids"]):
            provenance.append({
                "frame_id": frame, "relative_frame_id": frame - args.start_frame,
                "role": segment["role"], "segment": segment["name"], "local_output_index": local,
                "source_frame_ids": segment["frame_ids"], "conditioning_frame_ids": conditioned,
            })
    work = args.work_dir.expanduser().resolve()
    destination = args.output_root.expanduser().resolve() / "gbuffer_frames/raw"
    identity = {
        "recipe": "split_isolated_ordered_video_top_bottom_v1",
        "scene": args.scene, "weather": "raw", "camera": 0, "frame_ids": frame_ids,
        "holdout_ranges_relative_exclusive": ranges,
        "source": str(source.resolve()), "work_dir": str(work), "destination": str(destination),
        "source_sha256": {path.name: digest(path) for path in sources},
        "code_sha256": {str(path.resolve()): digest(path) for path in code},
        "model": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size,
                  "mtime_ns": checkpoint.stat().st_mtime_ns},
        "cosmos_env": str(args.cosmos_env.expanduser().resolve()),
        "conda": str(args.conda.expanduser().resolve()),
        "window": WINDOW, "stride": STRIDE, "pin_mode": "hybrid", "steps": 15,
        "seed": 1000, "guidance": 0, "passes": PASSES,
        "source_size_wh": SOURCE_SIZE, "tile_size_wh": DR_SIZE, "crops": CROPS,
        "segments": segments, "clips": clips, "frame_provenance": provenance,
        "padding": "Repeat final source within each segment to >=57; trim padded outputs",
        "note": "Train and test images never share a diffusion clip; pseudo-labels, not ground truth.",
    }
    identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    plan = {
        "signature": signature, "identity": identity, "stage": "planned",
        "final_png_count": args.frames * len(PASSES), "attempts": [],
        "inference_command_example": inverse_command(args, work / "attempts/0000/inputs",
                                                    work / "attempts/0000/results"),
    }
    return plan


def validate_resume_manifest(plan, previous):
    if previous.get("signature") != plan["signature"] or previous.get("identity") != plan["identity"]:
        raise ValueError("Resume manifest differs: source bytes, split, recipe, paths, model, or code changed")


def atomic_json(path, content):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(content, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        # Exclusive create retains the no-overwrite guarantee across filesystems.
        with target.open("xb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)


def prepare_inputs(plan):
    identity = plan["identity"]
    root = Path(identity["work_dir"]) / "inputs"
    source = Path(identity["source"])
    for clip in identity["clips"]:
        folder = root / clip["name"]
        folder.mkdir(parents=True, exist_ok=True)
        for local, frame in enumerate(clip["conditioning_frame_ids"]):
            target = folder / f"{local:05d}.png"
            if target.exists():
                if not valid_png(target, DR_SIZE):
                    raise ValueError(f"Invalid retained clip input: {target}")
                continue
            descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=folder)
            os.close(descriptor)
            try:
                with Image.open(source / f"{frame:03d}_0.jpg") as opened:
                    tile = opened.convert("RGB").crop(CROPS[clip["position"]]).resize(
                        DR_SIZE, Image.Resampling.LANCZOS)
                    tile.save(temporary, format="PNG")
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    return root


def clip_complete(results, clip):
    folder = Path(results) / "gbuffer_frames" / clip["name"]
    return all(
        valid_png(folder / f"full.{local:05d}.{pass_name}.png", DR_SIZE)
        for local in range(len(clip["conditioning_frame_ids"])) for pass_name in PASSES
    )


def discover_completed(plan):
    completed = {}
    for attempt in plan.get("attempts", []):
        results = Path(attempt["results"])
        for clip in plan["identity"]["clips"]:
            if clip["name"] in attempt["clips"] and clip_complete(results, clip):
                completed[clip["name"]] = str(results)
    return completed


def restore_pair(top, bottom, pass_name):
    return blend(resize_pass(top, pass_name), resize_pass(bottom, pass_name), pass_name)


def stage_outputs(plan, completed):
    identity = plan["identity"]
    staged = Path(identity["work_dir"]) / "final"
    staged.mkdir(exist_ok=True)
    hashes = {}
    for item in identity["frame_provenance"]:
        local = item["local_output_index"]
        for pass_name in PASSES:
            name = output_filename(item["frame_id"], pass_name)
            target = staged / name
            paths = {}
            for position in CROPS:
                clip = item["segment"] + "_" + position
                paths[position] = Path(completed[clip]) / "gbuffer_frames" / clip / f"full.{local:05d}.{pass_name}.png"
            # Always rebuild from validated complete clip outputs; interrupted staging
            # cannot promote stale or partial images.
            descriptor, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=staged)
            os.close(descriptor)
            try:
                Image.fromarray(restore_pair(paths["top"], paths["bottom"], pass_name)).save(temporary, format="PNG")
                if not valid_png(temporary, SOURCE_SIZE):
                    raise ValueError(f"Stitched output has unexpected dimensions: {temporary}")
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            hashes[name] = digest(target)
    return hashes


def publish_outputs(plan):
    identity = plan["identity"]
    staged = Path(identity["work_dir"]) / "final"
    destination = Path(identity["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    expected = plan["final_sha256"]
    extras = [path.name for path in destination.iterdir() if path.name not in expected]
    if extras:
        raise FileExistsError(f"Unrelated file in output destination: {extras[0]}")
    for name, expected_digest in expected.items():
        source, target = staged / name, destination / name
        if digest(source) != expected_digest or not valid_png(source, SOURCE_SIZE):
            raise ValueError(f"Staged output changed before publication: {source}")
        if target.exists():
            if digest(target) != expected_digest or not valid_png(target, SOURCE_SIZE):
                raise FileExistsError(f"Existing output differs; nothing was overwritten: {target}")
            continue
        # Copy to a same-filesystem temporary, then atomically hard-link into place;
        # even a concurrent publisher cannot replace another file.
        descriptor, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=destination)
        os.close(descriptor)
        try:
            shutil.copyfile(source, temporary)
            os.link(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def run(args):
    plan = build_plan(args)
    work = Path(plan["identity"]["work_dir"])
    destination = Path(plan["identity"]["destination"])
    manifest = work / "generation.json"
    if args.resume:
        if not manifest.is_file():
            raise FileNotFoundError(f"Cannot resume without manifest: {manifest}")
        previous = json.loads(manifest.read_text())
        validate_resume_manifest(plan, previous)
        plan = previous
    elif work.exists():
        raise FileExistsError(f"Work directory already exists; use a new directory or --resume: {work}")
    elif destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty; nothing was overwritten: {destination}")
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Preview only: no inputs, G-buffers, or model instances created.", flush=True)
        return 0
    work.mkdir(parents=True, exist_ok=args.resume)

    def status(stage):
        plan["stage"] = stage
        atomic_json(manifest, plan)
        print(f"stage={stage}", flush=True)

    try:
        if args.resume and plan.get("stage") == "completed":
            # Completed resumes validate all published bytes without rerunning diffusion.
            for name, expected in plan["final_sha256"].items():
                path = destination / name
                if not valid_png(path, SOURCE_SIZE) or digest(path) != expected:
                    raise ValueError(f"Completed published output is missing or changed: {path}")
            print("All completed outputs match their recorded hashes.", flush=True)
            return 0
        plan.pop("error", None)
        status("prepare_inputs")
        inputs = prepare_inputs(plan)
        completed = discover_completed(plan)
        pending = [clip for clip in plan["identity"]["clips"] if clip["name"] not in completed]
        if pending:
            # Retain every partial attempt. Retries run incomplete clips in a fresh
            # results directory, so leftovers never masquerade as a completed clip.
            attempt_index = len(plan["attempts"])
            attempt = work / "attempts" / f"{attempt_index:04d}"
            while attempt.exists():
                attempt_index += 1
                attempt = work / "attempts" / f"{attempt_index:04d}"
            attempt_inputs, results = attempt / "inputs", attempt / "results"
            for clip in pending:
                folder = attempt_inputs / clip["name"]
                folder.mkdir(parents=True)
                for local in range(len(clip["conditioning_frame_ids"])):
                    name = f"{local:05d}.png"
                    _link_or_copy(inputs / clip["name"] / name, folder / name)
            command = inverse_command(args, attempt_inputs, results)
            plan["attempts"].append({
                "directory": str(attempt), "results": str(results),
                "clips": [clip["name"] for clip in pending], "command": command,
            })
            status("inverse_rendering")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(args.cosmos_root.expanduser().resolve())
            env["CUDA_HOME"] = str(args.cosmos_env.expanduser().resolve())
            env.setdefault("MPLCONFIGDIR", "/tmp/3ablegs-mpl")
            subprocess.run(command, cwd=args.cosmos_root.expanduser().resolve(), env=env, check=True)
            completed = discover_completed(plan)
        missing = [clip["name"] for clip in plan["identity"]["clips"] if clip["name"] not in completed]
        if missing:
            raise RuntimeError(f"Inverse rendering left an incomplete clip: {missing[0]}")
        plan["clip_results"] = completed
        status("stitch")
        hashes = stage_outputs(plan, completed)
        if "final_sha256" in plan and plan["final_sha256"] != hashes:
            raise ValueError("Rebuilt final outputs differ from already recorded publication")
        plan["final_sha256"] = hashes
        status("publishing")
        publish_outputs(plan)
        status("completed")
        print(f"Completed {plan['final_png_count']} PNGs in {destination}", flush=True)
    except Exception as exc:
        plan["error"] = f"{type(exc).__name__}: {exc}"
        status("failed")
        raise
    return 0


def main(argv=None):
    args = make_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
