#!/usr/bin/env python3
"""Matched decoder-only adaptation for the four layered G-buffer conditions.

Use the checkpoint's original weighted losses and saved G-buffer teachers.
Geometry, motion, opacity, features, normals, light and camera response are
frozen. No temporal loss/history is added. Native saved resolution and AA off
are defaults. Explicit saved validation/test blocks are required unless the
run is deliberately marked diagnostic with --allow-nonblock-diagnostic.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.finetune_ego_material import (
    all_named_parameters, assert_frozen_parameters, atomic_json, atomic_torch_save,
    capture_rng, file_digest, restore_rng,
)


VARIANTS = ("B0", "B1", "L0", "L1")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--grouping", choices=("dynamic", "instance"), default="instance")
    parser.add_argument("--max-layers", type=int, default=0,
                        help="0 preserves every group; >=2 uses contribution-ranked layers plus overflow")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--downscale", type=float, default=1.,
                        help="Additional divisor of saved resolution; default 1")
    parser.add_argument("--antialiased", action="store_true")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonblock-diagnostic", action="store_true",
                        help="Explicitly permit stride/nonblock source splits; never claim clean block generalization")
    return parser


def validate_args(args):
    for name in ("steps", "save_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.max_layers < 0 or args.max_layers == 1:
        raise ValueError("--max-layers must be 0 or >= 2")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not math.isfinite(args.downscale) or args.downscale < 1:
        raise ValueError("--downscale must be finite and >= 1")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        raise ValueError("--grad-clip must be finite and nonnegative")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not (args.checkpoint.parent / "config.yaml").is_file():
        raise FileNotFoundError(args.checkpoint.parent / "config.yaml")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise FileExistsError(f"Output is not a directory: {args.output_dir}")
    if args.resume:
        for filename in ("resume_state.pt", "config.yaml", "layered_config.json"):
            if not (args.output_dir / filename).is_file():
                raise FileNotFoundError(f"--resume requires {args.output_dir / filename}")
    elif args.output_dir.exists() and any(path.name != ".run.lock" for path in args.output_dir.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.output_dir}; use --resume")
    return args


def variant_config(variant, grouping="instance", max_layers=0):
    if variant not in VARIANTS or grouping not in ("instance", "dynamic"):
        raise ValueError("Unknown variant or grouping")
    if isinstance(max_layers, bool) or not isinstance(max_layers, int) or max_layers < 0 or max_layers == 1:
        raise ValueError("max_layers must be 0 or >= 2")
    if variant == "B0":
        return None
    return {"grouping": "single" if variant == "B1" else grouping,
            "color_space": "linear" if variant in ("B1", "L1") else "srgb",
            "max_layers": 0 if variant == "B1" else max_layers,
            "ablation": "none", "diagnostics": False}


def checked_train_frames(runtime):
    train = sorted(int(frame) for frame in runtime.train_frames)
    if not train or len(set(train)) != len(train):
        raise ValueError("Training frames must be a nonempty unique sequence")
    if set(train) & (set(runtime.val_frames) | set(runtime.test_frames)):
        raise ValueError("Training and held-out frames overlap")
    return train


def _hash_value(digest, value):
    """Stream exact loaded reference values without retaining frame copies."""
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(memoryview(value).cast("B"))
    elif isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode())
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def training_input_digest(runtime, frames):
    """Fingerprint actual train RGB, teachers, masks, depth, timestamps and poses.

    Ray grids are deterministic from hashed cameras and configuration and are
    omitted. Only the saved training split is inspected, never held-out targets.
    """
    digest = hashlib.sha256()
    for index, frame in enumerate(frames):
        infos, cameras = runtime.camera.get_image(frame, "raw")
        missing = {"pixels", "gbuf_basecolor", "gbuf_metallic", "gbuf_roughness", "gbuf_normal"} - set(infos)
        if missing:
            raise ValueError(f"Training frame {frame} misses saved targets: {sorted(missing)}")
        _hash_value(digest, {"frame_id": frame, "camera": cameras,
                            "reference": {key: value for key, value in infos.items() if key not in
                                          {"origins", "viewdirs", "direction_norm", "pixel_coords"}}})
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(frames):
            print(json.dumps({"stage": "fingerprint_training_inputs", "frames": index + 1,
                              "total": len(frames)}), flush=True)
        del infos, cameras
    return digest.hexdigest()


def run_identity(args, input_digest, frames):
    options = {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items() if key not in {"output_dir", "resume", "save_every"}}
    sources = {Path(__file__).resolve(), ROOT / "tools/ego_motion_runtime.py",
               ROOT / "tools/finetune_ego_material.py", ROOT / "tools/render_temporal_material.py"}
    for directory in ("models", "datasets", "utils"):
        sources.update((ROOT / directory).rglob("*.py"))
    return {"schema": 1, "options": options,
            "checkpoint_sha256": file_digest(args.checkpoint),
            "config_sha256": file_digest(args.checkpoint.parent / "config.yaml"),
            "training_inputs_sha256": input_digest, "train_frames": frames,
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "code_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in sorted(sources)}}


def train_step(runtime, frame, optimizer, selected, grad_clip=1.):
    """One saved-loss step, exposed for CPU contract tests with a fake runtime."""
    if frame not in runtime.train_frames or frame in set(runtime.val_frames) | set(runtime.test_frames):
        raise ValueError("A held-out/nontraining frame cannot enter optimization")
    trainer = runtime.trainer
    source_step = int(trainer.step)
    assert_frozen_parameters(trainer, selected)
    optimizer.zero_grad(set_to_none=True)
    packet = runtime.render(frame, grad=True, return_outputs=True)
    losses = trainer.compute_losses(packet["training_outputs"], packet["training_image_infos"],
                                    packet["training_camera_infos"], weather="raw")
    if not isinstance(losses, dict) or not losses:
        raise ValueError("Saved trainer returned no losses")
    for name, value in losses.items():
        if not torch.is_tensor(value) or value.numel() != 1 or not torch.isfinite(value).all():
            raise FloatingPointError(f"Invalid original loss term: {name}")
    total = sum(losses.values())
    if not total.requires_grad or not torch.isfinite(total).all():
        raise RuntimeError("Original loss has no finite path to selected material decoders")
    total.backward()
    assert_frozen_parameters(trainer, selected)
    if not any(parameter.grad is not None for _, parameter in selected):
        raise RuntimeError("No selected material decoder received a gradient")
    if grad_clip:
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in selected], grad_clip,
                                      error_if_nonfinite=True)
    elif any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
             for _, parameter in selected):
        raise FloatingPointError("Nonfinite selected material gradient")
    optimizer.step()
    if int(trainer.step) != source_step:
        raise RuntimeError("The source trainer step must stay fixed during matched adaptation")
    return {"frame_id": int(frame), "loss": float(total.detach()),
            "losses": {name: float(value.detach()) for name, value in losses.items()}}


def _plain_config(value):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def restore_model_for_resume(runtime, model_state):
    """Restore registered weights and plain instance tensors to the GPU."""
    from tools.ego_motion_runtime import packet_to
    runtime.trainer.load_state_dict(packet_to(model_state, runtime.device), strict=True)


def _run(args):
    from tools.ego_motion_runtime import SceneRuntime
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runtime = SceneRuntime(args.checkpoint, args.downscale, antialiased=args.antialiased,
                           device="cuda", require_holdout=not args.allow_nonblock_diagnostic,
                           load_gbuffer=True)
    trainer = runtime.trainer
    frames = checked_train_frames(runtime)
    identity = run_identity(args, training_input_digest(runtime, frames), frames)
    renderer = variant_config(args.variant, args.grouping, args.max_layers)
    layer_metadata = {"schema": 1, "trained_variant": args.variant, "renderer": renderer,
                      "downscale": args.downscale, "antialiased": args.antialiased,
                      "reload_instruction": "Call trainer.configure_layered_gbuffer(renderer) after loading checkpoint; B0 uses None"}
    resumed = None
    if args.resume:
        resumed = torch.load(args.output_dir / "resume_state.pt", map_location="cpu", weights_only=False)
        if resumed["identity"] != identity:
            raise ValueError("Resume identity mismatch: source/code/training inputs/options changed")
        if file_digest(args.output_dir / "config.yaml") != identity["config_sha256"]:
            raise ValueError("Saved output config changed")
        if json.loads((args.output_dir / "layered_config.json").read_text()) != layer_metadata:
            raise ValueError("Saved layered renderer configuration changed")
        restore_model_for_resume(runtime, resumed["model"])
    else:
        (args.output_dir / "config.yaml").write_bytes((args.checkpoint.parent / "config.yaml").read_bytes())
        atomic_json(args.output_dir / "layered_config.json", layer_metadata)
    trainer.configure_layered_gbuffer(renderer)
    selected = runtime.freeze_material_parameters(update_features=False)
    assert_frozen_parameters(trainer, selected)
    selected_names = [name for name, _ in selected]
    optimizer = torch.optim.Adam([parameter for _, parameter in selected], lr=args.lr)
    sampler = random.Random(args.seed)
    step = 0
    if resumed is not None:
        if resumed["selected_parameter_names"] != selected_names:
            raise ValueError("Resume material parameter selection changed")
        optimizer.load_state_dict(resumed["optimizer"])
        sampler.setstate(resumed["sampler_rng"])
        restore_rng(resumed["rng"])
        step = int(resumed["completed_steps"])
        if not 0 <= step <= args.steps:
            raise ValueError("Invalid completed step count in resume state")
    del resumed
    source_step = int(trainer.step)
    metadata = {"identity": identity, **layer_metadata,
                "train_frames": frames, "validation_frames": list(map(int, runtime.val_frames)),
                "test_frames": list(map(int, runtime.test_frames)),
                "split_policy": "diagnostic_saved_split" if args.allow_nonblock_diagnostic else "saved_block_holdout",
                "loss_policy": "Original trainer.compute_losses; identical weights and fixed source trainer step; no new losses",
                "original_loss_config": _plain_config(trainer.losses_dict),
                "source_trainer_step": source_step,
                "material_parameters": runtime.material_parameter_summary,
                "feature_updates": False, "inference_history": False,
                "validation_or_test_used_for_optimization": False,
                "sampler": "random.Random(seed).choice(sorted(saved_training_frames))"}
    atomic_json(args.output_dir / "training_manifest.json", metadata)
    atomic_json(args.output_dir / "manifest.json", {**metadata, "status": "running"})

    def save_resume():
        assert_frozen_parameters(trainer, selected)
        if int(trainer.step) != source_step:
            raise RuntimeError("Source trainer step changed")
        atomic_torch_save(args.output_dir / "resume_state.pt", {
            "identity": identity, "model": trainer.state_dict(), "optimizer": optimizer.state_dict(),
            "selected_parameter_names": selected_names, "completed_steps": step,
            "sampler_rng": sampler.getstate(), "rng": capture_rng(),
            "layered_config": layer_metadata, "source_trainer_step": source_step,
        })

    if not args.resume:
        save_resume()
    log_path = args.output_dir / "train.jsonl"
    last_record = None
    if args.resume and log_path.exists():
        retained = []
        lines = log_path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                break  # A process interruption may leave an incomplete final line.
            if int(record["step"]) <= step:
                retained.append(line)
                last_record = record
        log_path.write_text("\n".join(retained) + ("\n" if retained else ""))
    started = time.monotonic()
    try:
        with log_path.open("a") as log:
            while step < args.steps:
                frame = sampler.choice(frames)
                last_record = {"step": step + 1, "variant": args.variant,
                               **train_step(runtime, frame, optimizer, selected, args.grad_clip)}
                step += 1
                log.write(json.dumps(last_record, allow_nan=False) + "\n")
                if step == 1 or step % 25 == 0:
                    log.flush()
                    print(json.dumps(last_record, allow_nan=False), flush=True)
                if step % args.save_every == 0 or step == args.steps:
                    log.flush()
                    save_resume()
        atomic_torch_save(args.output_dir / "checkpoint_final.pth", trainer.state_dict())
        atomic_json(args.output_dir / "training_summary.json", {
            "completed_steps": step, "trained_variant": args.variant, "seed": args.seed,
            "wall_seconds_this_run": time.monotonic() - started,
            "last_training_record": last_record, "checkpoint": str(args.output_dir / "checkpoint_final.pth"),
            "source_trainer_step": source_step, "validation_or_test_used_for_optimization": False,
            "inference_history": False, "feature_updates": False,
        })
        atomic_json(args.output_dir / "manifest.json", {**metadata, "status": "completed", "completed_steps": step})
    except BaseException as error:
        atomic_json(args.output_dir / "manifest.json", {
            **metadata, "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "completed_steps_in_process": step, "error": f"{type(error).__name__}: {error}",
            "resume_policy": "Resume from the last complete atomic resume_state.pt; newer log records are discarded",
        })
        raise


def main(argv=None):
    args = validate_args(make_parser().parse_args(argv))
    if not torch.cuda.is_available():
        raise RuntimeError("Layered material adaptation requires CUDA and the project's gsplat renderer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another adaptation process holds this output directory") from error
        validate_args(args)
        _run(args)


if __name__ == "__main__":
    main()
