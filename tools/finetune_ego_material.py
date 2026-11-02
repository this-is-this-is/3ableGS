#!/usr/bin/env python3
"""Matched-budget, material-only fine tuning for ego-motion experiments.

No validation or test image participates in optimization. Camera, geometry,
lighting, response, normals and affine parameters stay fixed; the trainer stays
in eval mode to disable topology updates. Temporal targets are detached renders
of training views in the same contiguous training block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


MODES = ("continue", "balanced", "legacy", "albedo", "albedo_roughness")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--downscale", type=float, default=2.)
    parser.add_argument("--temporal-weight", type=float, default=.05)
    parser.add_argument("--roughness-weight", type=float, default=.25)
    parser.add_argument("--anchor-weight", type=float, default=.1)
    parser.add_argument("--anchor-samples", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--motion-manifest", type=Path)
    parser.add_argument("--antialiased", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_args(args):
    for name in ("steps", "anchor_samples", "save_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    for name in ("temporal_weight", "roughness_weight", "anchor_weight", "grad_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not math.isfinite(args.downscale) or args.downscale < 1:
        raise ValueError("--downscale must be finite and >= 1")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not (args.checkpoint.parent / "config.yaml").is_file():
        raise FileNotFoundError(args.checkpoint.parent / "config.yaml")
    if args.motion_manifest is not None:
        args.motion_manifest = args.motion_manifest.expanduser().resolve()
        if not args.motion_manifest.is_file():
            raise FileNotFoundError(args.motion_manifest)
    if args.resume:
        if not (args.output_dir / "resume_state.pt").is_file():
            raise FileNotFoundError("--resume requires output-dir/resume_state.pt")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.output_dir}; use --resume")
    return args


def training_pairs(train_frames, gaps=(1, 2, 4)):
    """Return allowed predecessors without jumping across even one holdout."""
    frames = sorted(set(int(frame) for frame in train_frames))
    allowed = set(frames)
    if not gaps or any(int(gap) != gap or gap < 1 for gap in gaps):
        raise ValueError("History gaps must be positive integers")
    return {
        frame: tuple(frame - gap for gap in gaps
                     if all(value in allowed for value in range(frame - gap, frame + 1)))
        for frame in frames
    }


def motion_bins(train_frames, records):
    """Prefer frozen manifest labels; otherwise derive terciles on train only."""
    frames = sorted(set(int(frame) for frame in train_frames))
    by_frame = {}
    for record in records:
        frame = int(record["frame_id"])
        if frame in by_frame:
            raise ValueError(f"Duplicate motion frame: {frame}")
        by_frame[frame] = record
    missing = set(frames) - set(by_frame)
    if missing:
        raise ValueError(f"Motion manifest misses training frames: {sorted(missing)}")
    labels = {frame: by_frame[frame].get("motion_bin") for frame in frames}
    labels = {frame: "low" if label == "unknown" and by_frame[frame].get("displacement_p50") is None else label
              for frame, label in labels.items()}
    if all(value in {"low", "medium", "high"} for value in labels.values()):
        return labels
    values = {}
    for frame in frames:
        value = by_frame[frame].get("displacement_p50")
        if value is None:
            value = 0.
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Invalid motion score at frame {frame}")
        values[frame] = float(value)
    lower, upper = np.quantile(list(values.values()), [1 / 3, 2 / 3])
    # Equal scores retain the same class; no arbitrary high-motion assignment.
    return {frame: ("low" if value <= lower else "medium" if value <= upper else "high")
            for frame, value in values.items()}


def pose_motion_records(runtime, train_frames):
    """Fallback: dimensionless pose score, explicitly not pixel displacement.

    A translation is normalized by median train-frame translation, and rotation
    by median nonzero train-frame rotation. Raw poses are method-independent;
    use --motion-manifest for depth/pose projected pixel motion instead.
    """
    poses = runtime.dataset.pixel_source.camera_data[0].cam_to_worlds.detach().cpu()
    frames = sorted(set(int(frame) for frame in train_frames))
    pairs = training_pairs(frames, (1,))
    translation, rotation = {}, {}
    for frame in frames:
        if not pairs[frame]:
            translation[frame], rotation[frame] = 0., 0.
            continue
        current, previous = poses[frame], poses[frame - 1]
        translation[frame] = float((current[:3, 3] - previous[:3, 3]).norm())
        relative = previous[:3, :3].T @ current[:3, :3]
        rotation[frame] = math.acos(float(((relative.trace() - 1) / 2).clamp(-1, 1)))
    t_scale = np.median([value for value in translation.values() if value > 1e-8] or [1.])
    r_scale = np.median([value for value in rotation.values() if value > 1e-8] or [1.])
    return [{"frame_id": frame,
             "displacement_p50": translation[frame] / t_scale + rotation[frame] / r_scale,
             "translation_m": translation[frame], "rotation_rad": rotation[frame]}
            for frame in frames]


class TrainFrameSampler:
    def __init__(self, train_frames, bins, *, balanced, seed, gaps=(1, 2, 4)):
        self.frames = tuple(sorted(set(int(frame) for frame in train_frames)))
        if not self.frames:
            raise ValueError("No training frames")
        self.allowed = training_pairs(self.frames, gaps)
        self.balanced = bool(balanced)
        self.groups = tuple(tuple(frame for frame in self.frames if bins[frame] == label)
                            for label in ("low", "medium", "high"))
        self.groups = tuple(group for group in self.groups if group)
        self.rng = random.Random(seed)
        self.pair_rng = random.Random(seed + 1763)

    def sample(self):
        group = self.rng.choice(self.groups) if self.balanced else self.frames
        current = self.rng.choice(group)
        choices = self.allowed[current]
        previous = self.pair_rng.choice(choices) if choices else None
        return current, previous

    def state_dict(self):
        return {"rng": self.rng.getstate(), "pair_rng": self.pair_rng.getstate(), "frames": self.frames,
                "groups": self.groups, "balanced": self.balanced, "allowed": self.allowed}

    def load_state_dict(self, state):
        for key in ("frames", "groups", "balanced", "allowed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Resume sampler mismatch: {key}")
        self.rng.setstate(state["rng"])
        self.pair_rng.setstate(state["pair_rng"])


def all_named_parameters(trainer):
    """The trainer's model dictionary is not an nn.ModuleDict."""
    seen = set()
    for prefix, module in [("trainer", trainer), *sorted(trainer.models.items())]:
        for name, parameter in module.named_parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield f"{prefix}.{name}", parameter


def assert_frozen_parameters(trainer, selected):
    allowed = {id(parameter) for _, parameter in selected}
    actual = {id(parameter) for _, parameter in all_named_parameters(trainer)
              if parameter.requires_grad}
    if actual != allowed:
        raise RuntimeError("Only the explicitly selected material parameters may require gradients")
    if trainer.training or any(module.training for module in trainer.models.values()):
        raise RuntimeError("Material fine tuning must keep the scene in eval mode")
    for name, parameter in all_named_parameters(trainer):
        if id(parameter) not in allowed and parameter.grad is not None:
            raise RuntimeError(f"Unexpected gradient in frozen parameter: {name}")


def material_inputs(node, indices):
    value = node._features[indices]
    if getattr(node, "raw_spatial_xyz_enabled", False):
        origin = torch.as_tensor(node.scene_origin, dtype=value.dtype, device=value.device)
        xyz = ((node._means[indices].detach() - origin) / max(float(node.scene_scale), 1e-6)).clamp(-1, 1)
        value = torch.cat((value, xyz), -1)
    return value


def decoded_material(node, indices):
    logits = node.mlp_material["raw"](material_inputs(node, indices))
    return torch.cat((torch.sigmoid(logits[..., :3]),
                      torch.sigmoid(logits[..., 3:4]),
                      torch.sigmoid(logits[..., 4:5]) * .99 + .01), -1)


@torch.no_grad()
def build_material_anchors(trainer, chunk_size=131072):
    """Fixed initial per-Gaussian material targets, including frozen R/M."""
    anchors = {}
    for name, node in sorted(trainer.models.items()):
        if not getattr(node, "pbr_enabled", False):
            continue
        chunks = []
        for start in range(0, len(node._features), chunk_size):
            chunks.append(decoded_material(node, slice(start, start + chunk_size)).cpu())
        if chunks:
            anchors[name] = torch.cat(chunks)
    if not anchors:
        raise ValueError("No raw PBR Gaussian material heads found")
    return anchors


def linear_albedo(value):
    value = value.clamp(0, 1)
    return torch.where(value <= .04045, value / 12.92,
                       ((value + .055) / 1.055).pow(2.4))


def anchor_loss(trainer, anchors, samples, generator):
    terms = []
    for name, target in anchors.items():
        node = trainer.models[name]
        if len(target) != len(node._features):
            raise RuntimeError(f"Gaussian topology changed for {name}")
        indices = torch.randint(len(target), (min(samples, len(target)),), generator=generator)
        prediction = decoded_material(node, indices.to(node._features.device))
        reference = target[indices].to(prediction)
        terms.append(((linear_albedo(prediction[:, :3]) - linear_albedo(reference[:, :3])).abs().mean()
                      + (prediction[:, 3] - reference[:, 3]).abs().mean()
                      + (prediction[:, 4] - reference[:, 4]).abs().mean()) / 3)
    return sum(terms) / len(terms)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_identity(args):
    identity = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in {"output_dir", "resume", "save_every"}}
    identity["checkpoint_sha256"] = file_digest(args.checkpoint)
    identity["config_sha256"] = file_digest(args.checkpoint.parent / "config.yaml")
    identity["motion_sha256"] = file_digest(args.motion_manifest) if args.motion_manifest else None
    identity["code_sha256"] = {str(path.relative_to(ROOT)): file_digest(path) for path in (
        Path(__file__), ROOT / "tools/ego_motion_runtime.py", ROOT / "models/ego_temporal.py")}
    return identity


def temporal_loss(mode, current, previous, roughness_weight):
    if previous is None or mode in {"continue", "balanced"}:
        return current["albedo"].sum() * 0.
    if mode == "legacy":
        from models.temporal_training import material_temporal_loss
        target = {"gbuffer": {name: previous[name] for name in
                              ("albedo", "roughness", "metallic", "normal")},
                  **{name: previous[name] for name in
                     ("depth", "opacity", "static_mask", "camera_to_world", "intrinsics")}}
        return material_temporal_loss(current, current["static_mask"],
                                      current["camera_to_world"], current["intrinsics"], target)
    from models.ego_temporal import material_consistency
    result = material_consistency(current, previous,
                                  roughness_weight=roughness_weight if mode == "albedo_roughness" else 0.)
    return result["loss"]


def main(argv=None):
    args = validate_args(make_parser().parse_args(argv))
    if not torch.cuda.is_available():
        raise RuntimeError("Material fine tuning requires a CUDA gsplat renderer")
    from tools.ego_motion_runtime import SceneRuntime
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    identity = run_identity(args)
    resume = (torch.load(args.output_dir / "resume_state.pt", map_location="cuda", weights_only=False)
              if args.resume else None)
    if resume is not None and resume["identity"] != identity:
        raise ValueError("Resume identity mismatch: checkpoint, config, code or experiment options changed")
    runtime = SceneRuntime(args.checkpoint, args.downscale, antialiased=args.antialiased, device="cuda")
    trainer = runtime.trainer
    train_frames = sorted(int(frame) for frame in runtime.train_frames)
    heldout = set(runtime.val_frames) | set(runtime.test_frames)
    if set(train_frames) & heldout:
        raise RuntimeError("Training and held-out frame sets overlap")
    if args.motion_manifest:
        manifest = json.loads(args.motion_manifest.read_text())
        records = manifest["frames"]
        motion_source = "frozen_depth_pose_projection_manifest"
    else:
        records = pose_motion_records(runtime, train_frames)
        motion_source = "raw_pose_translation_rotation_normalized_proxy"
    bins = motion_bins(train_frames, records)
    sampler = TrainFrameSampler(train_frames, bins, balanced=args.mode in {"balanced", "albedo", "albedo_roughness"},
                                seed=args.seed, gaps=(1,) if args.mode == "legacy" else (1, 2, 4))
    anchor_generator = torch.Generator(device="cpu").manual_seed(args.seed + 931)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchors_path = args.output_dir / "material_anchors.pt"
    if resume is None:
        anchors = build_material_anchors(trainer)
        atomic_torch_save(anchors_path, anchors)
        # Keep the ORIGINAL resolution; a later --downscale must apply once.
        (args.output_dir / "config.yaml").write_bytes((args.checkpoint.parent / "config.yaml").read_bytes())
    else:
        if not anchors_path.is_file() or file_digest(anchors_path) != resume["anchors_sha256"]:
            raise ValueError("Resume material anchors are missing or changed")
        anchors = torch.load(anchors_path, map_location="cpu", weights_only=False)
        trainer.load_state_dict(resume["model"], strict=True)
    selected = runtime.freeze_material_parameters(update_features=True)
    assert_frozen_parameters(trainer, selected)
    optimizer = torch.optim.Adam([parameter for _, parameter in selected], lr=args.lr)
    step = 0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        sampler.load_state_dict(resume["sampler"])
        anchor_generator.set_state(resume["anchor_rng"].cpu())
        restore_rng(resume["rng"])
        step = int(resume["completed_steps"])
    del resume
    anchors_sha256 = file_digest(anchors_path)
    metadata = {"identity": identity, "train_frames": train_frames,
                "validation_frames": list(runtime.val_frames), "test_frames": list(runtime.test_frames),
                "motion_source": motion_source, "motion_bins": bins,
                "material_parameters": runtime.material_parameter_summary,
                "loss": "RGB L1 + fixed initial Gaussian A/R/M anchor + selected temporal loss",
                "normal_policy": "Features frozen where learned normal heads share them; raw decoder remains trainable",
                "history_policy": "Same contiguous training block; previous render detached; no inference history",
                "source_trainer_step": int(trainer.step)}
    atomic_json(args.output_dir / "training_manifest.json", metadata)
    atomic_json(args.output_dir / "manifest.json", {**metadata, "status": "running"})

    def save_resume():
        assert_frozen_parameters(trainer, selected)
        atomic_torch_save(args.output_dir / "resume_state.pt", {
            "identity": identity, "anchors_sha256": anchors_sha256,
            "model": trainer.state_dict(), "optimizer": optimizer.state_dict(),
            "completed_steps": step, "sampler": sampler.state_dict(),
            "anchor_rng": anchor_generator.get_state(), "rng": capture_rng(),
        })

    # Save a restart point even if interrupted before the first scheduled save.
    if not args.resume:
        save_resume()
    started = time.monotonic()
    last_record = None
    log_path = args.output_dir / "train.jsonl"
    if args.resume and log_path.exists():
        # Logs newer than the last atomic checkpoint represent discarded work.
        retained = [line for line in log_path.read_text().splitlines()
                    if int(json.loads(line)["step"]) <= step]
        log_path.write_text("\n".join(retained) + ("\n" if retained else ""))
        last_record = json.loads(retained[-1]) if retained else None
    with log_path.open("a") as log:
        while step < args.steps:
            current_frame, previous_frame = sampler.sample()
            if current_frame not in train_frames or (previous_frame is not None and previous_frame not in train_frames):
                raise RuntimeError("Sampler selected a held-out image")
            optimizer.zero_grad(set_to_none=True)
            previous = (runtime.render(previous_frame, grad=False)
                        if previous_frame is not None and args.mode not in {"continue", "balanced"} else None)
            current = runtime.render(current_frame, grad=True)
            rgb = (current["rgb"] - current["target_rgb"].detach()).abs().mean()
            anchor = anchor_loss(trainer, anchors, args.anchor_samples, anchor_generator)
            temporal = temporal_loss(args.mode, current, previous, args.roughness_weight)
            loss = rgb + args.anchor_weight * anchor + args.temporal_weight * temporal
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step + 1}")
            loss.backward()
            assert_frozen_parameters(trainer, selected)
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_([parameter for _, parameter in selected], args.grad_clip,
                                              error_if_nonfinite=True)
            elif any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                     for _, parameter in selected):
                raise FloatingPointError(f"Nonfinite material gradient at step {step + 1}")
            optimizer.step()
            step += 1
            last_record = {"step": step, "frame_id": current_frame, "previous_frame": previous_frame,
                           "motion_bin": bins[current_frame], "rgb_l1": float(rgb.detach()),
                           "anchor_loss": float(anchor.detach()), "temporal_loss": float(temporal.detach()),
                           "loss": float(loss.detach())}
            log.write(json.dumps(last_record, allow_nan=False) + "\n")
            if step == 1 or step % 25 == 0:
                log.flush()
                print(json.dumps(last_record), flush=True)
            # Drop image graphs before serializing the large scene checkpoint.
            del loss, rgb, anchor, temporal, current, previous
            if step % args.save_every == 0 or step == args.steps:
                log.flush()
                save_resume()
    atomic_torch_save(args.output_dir / "checkpoint_final.pth", trainer.state_dict())
    atomic_json(args.output_dir / "training_summary.json", {
        "completed_steps": step, "mode": args.mode, "seed": args.seed,
        "wall_seconds_this_run": time.monotonic() - started,
        "last_training_record": last_record, "checkpoint": str(args.output_dir / "checkpoint_final.pth"),
        "inference_history": False, "validation_or_test_used_for_optimization": False,
        "material_parameters": runtime.material_parameter_summary,
    })
    atomic_json(args.output_dir / "manifest.json", {
        **metadata, "status": "completed", "completed_steps": step,
    })


if __name__ == "__main__":
    main()
