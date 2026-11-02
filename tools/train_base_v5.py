#!/usr/bin/env python3
"""Base v5 scratch training, with total-update accounting.

Default 30,001 camera updates matches tools/train.py's inclusive 0..30000
Base-v4 budget inherited by Base v5. No pretrained initializer is accepted. Existing runs resume
their optimizer, density accumulators, camera clock and random streams.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_training_config
from utils.rgb_joint_training import (validate_config, validate_inputs, training_signature,
    read_training_state, save_training_state, load_model_state, restore_training_state)
from utils.weather_training import exclusive_run_lock, atomic_json_save


def prepare(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    saved_path = run_dir / "config.yaml"
    if saved_path.exists():
        saved = load_training_config(saved_path)
        cfg = OmegaConf.merge(saved, OmegaConf.from_dotlist(args.opts or []))
        if training_signature(cfg) != training_signature(saved):
            raise ValueError("Existing run uses different fitting settings; choose a new RUN_TAG")
    else:
        if run_dir.exists() and any(p.name != ".training.lock" for p in run_dir.iterdir()):
            raise ValueError("Output has files without config.yaml; choose a new RUN_TAG")
        cfg = load_training_config(args.config_file, args.opts)
    validate_config(cfg)
    inputs = validate_inputs(cfg)
    total = args.iterations if args.iterations is not None else int(
        OmegaConf.select(cfg, "rgb_joint_run.total_iterations", default=int(cfg.trainer.optim.num_iters) + 1))
    if type(total) is not int or total < 1:
        raise ValueError("--iterations must be a positive TOTAL update target")
    checkpoint = None
    state = None
    for filename in ("checkpoint_training_latest.pth", "checkpoint_final.pth"):
        candidate = run_dir / filename
        if candidate.is_file():
            loaded = read_training_state(candidate, cfg)
            if state is None or loaded["step"] > state["step"]:
                checkpoint, state = candidate, loaded
    if state is not None and not saved_path.exists():
        raise ValueError("Resume requires config.yaml")
    next_step = 0 if state is None else state["step"] + 1
    cfg.log_dir = str(run_dir)
    cfg.rgb_joint_run = {"initialization": "scratch_dataset_seed_points", "pretrained_checkpoint": None,
                         "total_iterations": total, "camera_only": True,
                         "native_schedule_final_step": int(cfg.trainer.optim.num_iters)}
    return cfg, run_dir, total, checkpoint, next_step, inputs


def to_device(values, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in values.items()}


def fit_step(trainer, dataset, step, *, collect_diagnostics=True):
    trainer.set_train()
    trainer.preprocess_per_train_step(step)
    trainer.optimizer_zero_grad()
    images, cameras = dataset.train_image_set.next(trainer._get_downscale_factor(), ["raw"])
    image, camera = to_device(images["raw"], trainer.device), to_device(cameras["raw"], trainer.device)
    outputs = trainer(image, camera, "raw")
    trainer.update_visibility_filter()
    losses = trainer.compute_losses(outputs, image, camera, "raw")
    for name, value in losses.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {name} at step {step}")
    trainer.backward(losses)
    trainer.optimizer_step()
    trainer.postprocess_per_train_step(step)
    with torch.no_grad():
        values = {name: float(value.detach()) for name, value in losses.items()}
        values.update({"metric/" + name: float(value) for name, value in trainer.compute_metrics(outputs, image).items()})
        values["camera_updates"] = int(trainer.rgb_joint_camera_updates)
        values["residual_strength"] = float(outputs["rgb_residual_strength"])
        values["gaussian_counts"] = trainer.get_gaussian_count()
        if collect_diagnostics:
            stats = getattr(trainer, "_last_dynamic_gbuffer_stats", {})
            tensor_keys = [name for name, value in stats.items() if isinstance(value, torch.Tensor)]
            if tensor_keys:
                # One device transfer at logging updates, rather than one CUDA
                # synchronization per diagnostic on every camera update.
                tensor_values = torch.stack([stats[name].detach().float() for name in tensor_keys]).cpu().tolist()
                values.update({"dynamic_gbuffer/" + name: value for name, value in zip(tensor_keys, tensor_values)})
            values.update({"dynamic_gbuffer/" + name: float(value)
                           for name, value in stats.items() if not isinstance(value, torch.Tensor)})
    return values


def run(args):
    os.chdir(ROOT)
    cfg, run_dir, total, checkpoint, next_step, inputs = prepare(args)
    print(json.dumps({"run_dir": str(run_dir), "initialization": "scratch_dataset_seed_points",
        "resume_checkpoint": str(checkpoint) if checkpoint else None, "next_step": next_step,
        "total_camera_updates": total, "remaining_updates": max(0, total - next_step),
        "joint": OmegaConf.to_container(cfg.trainer.rgb_joint, resolve=True), "inputs": inputs}, indent=2), flush=True)
    if args.prepare_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires the 3ableGS CUDA environment")
    with exclusive_run_lock(run_dir / ".training.lock"):
        cfg, run_dir, total, checkpoint, next_step, _ = prepare(args)
        if next_step < total:
            torch.cuda.reset_peak_memory_stats()
            from datasets.driving_dataset import DrivingDataset
            from utils.misc import import_str
            from utils.logging import setup_logging
            from utils.backup import backup_project
            setup_logging(output=str(run_dir), level="INFO", time_string=time.strftime("%Y-%m-%d_%H-%M-%S"))
            random.seed(int(cfg.seed)); np.random.seed(int(cfg.seed)); torch.manual_seed(int(cfg.seed))
            torch.cuda.manual_seed_all(int(cfg.seed))
            # Save before trainer construction mutates class optimizer configs.
            OmegaConf.save(cfg, run_dir / "config.yaml")
            if not (run_dir / "backup").exists():
                backup_project(str(run_dir / "backup"), str(ROOT), ["configs", "datasets", "models", "utils", "tools"],
                               [".py", ".yaml", ".sh"])
            dataset = DrivingDataset(data_cfg=deepcopy(cfg.data))
            trainer = import_str(cfg.trainer.type)(**deepcopy(cfg.trainer), dataset=dataset,
                weathers=["raw"], model_config=deepcopy(cfg.model), vgg_config=cfg.get("vgg"), device=torch.device("cuda"))
            state = read_training_state(checkpoint, cfg) if checkpoint else None
            if state is None:
                trainer.init_gaussians_from_dataset(dataset)
            else:
                load_model_state(trainer, state)
            trainer.initialize_optimizer()
            if state is not None:
                next_step = restore_training_state(trainer, dataset, state, cfg)
                del state
            save_every = max(1, int(cfg.logging.saveckpt_freq))
            log_every = max(1, int(cfg.logging.print_freq))
            started = time.monotonic()
            with (run_dir / "training_metrics.jsonl").open("a") as log:
                for step in range(next_step, total):
                    values = fit_step(trainer, dataset, step,
                                      collect_diagnostics=(step % log_every == 0 or step + 1 == total))
                    if step % log_every == 0 or step + 1 == total:
                        row = {"step": step, "elapsed_seconds": time.monotonic() - started, **values}
                        log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
                        print(json.dumps(row, allow_nan=False), flush=True)
                    if (step + 1) % save_every == 0 or step + 1 == total:
                        save_training_state(trainer, dataset, cfg, run_dir / "checkpoint_training_latest.pth")
                        atomic_json_save({"completed_updates": step + 1, "total_updates": total,
                            "camera_updates": int(trainer.rgb_joint_camera_updates),
                            "process_start_update": next_step,
                            "loop_elapsed_seconds_this_process": time.monotonic() - started,
                            "peak_allocated_vram_bytes_this_process": torch.cuda.max_memory_allocated(),
                            }, run_dir / "training_status.json")
                save_training_state(trainer, dataset, cfg, run_dir / "checkpoint_final.pth")
            checkpoint = run_dir / "checkpoint_final.pth"
            del trainer, dataset
            torch.cuda.empty_cache()
        else:
            print("Saved checkpoint meets the total update target; skipping training.", flush=True)
        if not args.skip_eval:
            subprocess.run([sys.executable, "-u", str(ROOT / "tools/eval_base_v5.py"),
                            "--checkpoint", str(checkpoint), "--save-images"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=ROOT / "configs/base_v5/scene788_100f.yaml", type=Path)
    parser.add_argument("--run-dir", default=ROOT / "output/pbr/rebuild/scene_788_100f/base_v5", type=Path)
    parser.add_argument("--iterations", type=int, help="Total camera updates; Base v5 defaults to 30,001")
    parser.add_argument("--prepare-only", "--dry-run", dest="prepare_only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("opts", nargs="*", help="Fresh-run key=value overrides; resume requires unchanged fitting settings")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
