#!/usr/bin/env python3
"""Frozen Base v4 material-attention pilot with original supervised losses.

The scene checkpoint, original renderer and train/test split remain fixed.
Only the temporal adapter is optimized. Ordinary tools/eval.py evaluates the
resulting checkpoint/config pair without a separate image-metric implementation.
"""
from __future__ import annotations

import argparse
import copy
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
VARIANTS = ("B0", "H0", "C0", "C1", "A0", "A1", "A2")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path, value):
    import torch
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


def code_identity():
    files = ("tools/train_gbuffer_attention.py", "models/gbuffer_attention.py",
             "models/gbuffer_attention_runtime.py", "models/ego_temporal.py",
             "models/rdg_fusion.py", "models/rdg_fusion_runtime.py",
             "models/trainers/scene_graph_pbr.py", "models/trainers/base.py",
             "models/boundary_losses.py", "datasets/base/pixel_source.py",
             "datasets/base/split_wrapper.py")
    return {name: file_digest(ROOT / name) for name in files if (ROOT / name).is_file()}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_args(args):
    if args.seed < 0 or args.steps < 0 or args.save_every < 1:
        raise ValueError("seed/steps must be nonnegative and save-every must be positive")
    if args.variant in ("B0", "H0") and args.steps != 0:
        raise ValueError("B0/H0 have no trainable adapter; use --steps 0")
    if args.variant not in ("B0", "H0") and args.steps == 0:
        raise ValueError("Learned pilot variants require positive --steps")
    for name in ("lr", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (args.checkpoint, args.checkpoint.parent / "config.yaml"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.resume:
        if not (args.output_dir / "resume_state.pt").is_file():
            raise FileNotFoundError("--resume requires output-dir/resume_state.pt")
    elif args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f"Output is not empty: {args.output_dir}; choose a new directory or --resume")
    return args


def pilot_config(variant, lr=1e-4):
    from models.gbuffer_attention import validate_attention_config
    return validate_attention_config({
        "variant": variant, "lr": float(lr), "start_step": 0,
        "ramp_steps": 200, "temporal_start": 0, "temporal_ramp": 200,
    })


def load_scene(checkpoint, *, load_gbuffer=False):
    """Load a raw scene with its saved rendering policy; never force mean affine.

    CPU preloading changes storage only. The saved image dimensions, calibration,
    masks, split, AA and render settings are left intact. Teachers are loaded only
    for supervised training, not for inference features or temporal evaluation.
    """
    import torch
    from omegaconf import OmegaConf
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    checkpoint = Path(checkpoint).resolve()
    saved = OmegaConf.load(checkpoint.parent / "config.yaml")
    if str(saved.trainer.type) not in (
        "models.trainers.MultiTrainer_pbr", "models.trainers.scene_graph_pbr.MultiTrainer_pbr",
    ):
        raise ValueError("G-buffer attention requires a raw Base v4 MultiTrainer_pbr checkpoint")
    if list(saved.data.pixel_source.cameras) != [0]:
        raise ValueError("This experiment currently requires exactly camera 0")
    if not torch.cuda.is_available():
        raise RuntimeError("Scene rendering requires CUDA and the project's gsplat environment")
    cfg = copy.deepcopy(saved)
    cfg.data.preload_device = "cpu"
    cfg.data.pixel_source.load_weather = []
    cfg.data.pixel_source.load_gbuffer = bool(load_gbuffer)
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **copy.deepcopy(cfg.trainer), dataset=dataset, weathers=["raw"],
        vgg_config=cfg.get("vgg"), model_config=copy.deepcopy(cfg.model),
        device=torch.device("cuda"),
    )
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    trainer.load_state_dict(state, load_only_model=True, strict=True)
    trainer.set_eval()
    return saved, dataset, trainer


def all_named_parameters(trainer):
    """Scene modules are in a plain dictionary, outside Module.parameters()."""
    seen = set()
    for prefix, module in [("trainer", trainer), *sorted(trainer.models.items())]:
        for name, parameter in module.named_parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield f"{prefix}.{name}", parameter


def freeze_except_attention(trainer):
    for _, parameter in all_named_parameters(trainer):
        parameter.requires_grad_(False)
        parameter.grad = None
    head = getattr(trainer, "gbuffer_attention_head", None)
    if head is None:
        return []
    head.requires_grad_(True)
    return list(head.parameters())


def set_pilot_training(trainer):
    # The trainer flag enables train-only context/schedules. Scene modules stay
    # in eval and no topology, visibility or refinement hooks are called.
    trainer.train()
    for module in trainer.models.values():
        module.eval()
    if hasattr(trainer, "lpips"):
        trainer.lpips.eval()


def assert_only_attention_gradients(trainer, parameters):
    allowed = {id(parameter) for parameter in parameters}
    for name, parameter in all_named_parameters(trainer):
        if id(parameter) not in allowed and (parameter.requires_grad or parameter.grad is not None):
            raise RuntimeError(f"Frozen scene parameter received gradients: {name}")


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def seed_rng(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_identity(args, attention):
    return {
        "schema": 1, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_digest(args.checkpoint),
        "config_sha256": file_digest(args.checkpoint.parent / "config.yaml"),
        "variant": args.variant, "steps": args.steps, "seed": args.seed,
        "lr": args.lr, "grad_clip": args.grad_clip, "attention": attention,
        "code": code_identity(),
    }


def run(args):
    import torch
    from omegaconf import OmegaConf
    from tools.render_temporal_material import render_inputs, to_device

    args = validate_args(args)
    os.chdir(ROOT)
    config = pilot_config(args.variant, args.lr)
    identity = training_identity(args, config)
    resumed = None
    if args.resume:
        resumed = torch.load(args.output_dir / "resume_state.pt", map_location="cpu", weights_only=False)
        if resumed["identity"] != identity:
            raise ValueError("Resume source/code/config/options changed; use a new output directory")
    seed_rng(args.seed)
    saved, dataset, trainer = load_scene(args.checkpoint, load_gbuffer=args.steps > 0)
    source_attention = saved.trainer.get("gbuffer_attention")
    if source_attention is not None and source_attention.get("variant", "A2") != "B0":
        raise ValueError("Frozen pilot source must be the original Base v4 checkpoint, without an adapter")
    source_step = int(trainer.step)
    trainer.configure_gbuffer_attention(config)
    if resumed is not None:
        trainer.load_state_dict(resumed["model"], load_only_model=True, strict=True)
    parameters = freeze_except_attention(trainer)
    if args.steps and not parameters:
        raise RuntimeError("Learned variant has no trainable adapter parameters")
    optimizer = torch.optim.Adam(parameters, lr=args.lr) if args.steps else None
    train_ids = {int(value) for value in dataset.train_indices}
    if not train_ids:
        raise ValueError("Saved dataset has no training images")
    # Equal data/crop RNG sequence across head variants, independent of their
    # parameter count and construction. The original dataset sampler is used.
    seed_rng(args.seed)
    completed = 0
    if resumed is not None:
        completed = int(resumed["completed_updates"])
        if optimizer is not None:
            optimizer.load_state_dict(resumed["optimizer"])
        restore_rng(resumed["rng"])
        assert_only_attention_gradients(trainer, parameters)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_cfg = copy.deepcopy(saved)
    output_cfg.trainer.gbuffer_attention = config
    output_cfg.seed = args.seed
    output_cfg.log_dir = str(args.output_dir)
    OmegaConf.save(output_cfg, args.output_dir / "config.yaml")
    manifest = {
        "identity": identity, "status": "training", "source_scene_step": source_step,
        "train_image_ids": sorted(train_ids), "test_image_ids": list(map(int, dataset.test_indices)),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "scene_frozen": True, "sampling": "original train_image_set.next; identical RNG seed",
        "image_evaluation": "tools/eval.py, saved test split, native saved resolution and affine",
        "context": "generated past G-buffers only; train IDs only during training",
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    log_path = args.output_dir / "losses.jsonl"
    if resumed is not None and log_path.exists():
        retained = [line for line in log_path.read_text().splitlines()
                    if int(json.loads(line)["update"]) <= completed]
        log_path.write_text("".join(line + "\n" for line in retained))

    def save_resume():
        atomic_torch_save(args.output_dir / "resume_state.pt", {
            "identity": identity, "completed_updates": completed,
            "model": trainer.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
            "rng": rng_state(),
        })

    if resumed is None:
        save_resume()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with log_path.open("a") as log:
        for update in range(completed + 1, args.steps + 1):
            set_pilot_training(trainer)
            trainer.attention_step = update
            images, cameras = dataset.train_image_set.next(1., ["raw"])
            image_infos = to_device(images["raw"], trainer.device, torch)
            camera_infos = to_device(cameras["raw"], trainer.device, torch)
            image_id = int(image_infos["img_idx"].flatten()[0])
            if image_id not in train_ids:
                raise RuntimeError("Training sampler returned a held-out image")
            optimizer.zero_grad(set_to_none=True)
            outputs = trainer(render_inputs(image_infos, trainer.device, torch), camera_infos, "raw")
            losses = trainer.compute_losses(outputs, image_infos, camera_infos, "raw")
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {update}")
            if not loss.requires_grad:
                raise RuntimeError("Pilot loss is disconnected from the attention head")
            loss.backward()
            assert_only_attention_gradients(trainer, parameters)
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if int(trainer.step) != source_step:
                raise RuntimeError("Frozen pilot must retain the source scene loss/density step")
            completed = update
            record = {"update": update, "image_id": image_id,
                      "loss": float(loss.detach()), "gradient_norm": float(norm),
                      "losses": {key: float(value.detach()) for key, value in losses.items()},
                      "context_ids": list(outputs.get("attention_context_ids", []))}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            if update == 1 or update % 20 == 0 or update == args.steps:
                log.flush()
                print(json.dumps(record, allow_nan=False), flush=True)
            if update % args.save_every == 0 or update == args.steps:
                save_resume()
    trainer.set_eval()
    atomic_torch_save(args.output_dir / "checkpoint_final.pth", trainer.state_dict())
    manifest.update(status="complete", completed_updates=completed,
                    elapsed_seconds=time.perf_counter() - started,
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                    checkpoint_sha256=file_digest(args.output_dir / "checkpoint_final.pth"))
    atomic_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main(argv=None):
    args = make_parser().parse_args(argv)
    result = run(args)
    print(json.dumps({"status": result["status"], "completed_updates": result["completed_updates"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
