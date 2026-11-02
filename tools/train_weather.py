#!/usr/bin/env python3
"""Train one frozen-base Rain/Snow stage, or evaluate its saved checkpoint."""
import argparse
import copy
from datetime import datetime
import logging
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from omegaconf import OmegaConf
import torch

from utils.weather_training import (
    atomic_json_save, atomic_torch_save, exclusive_run_lock, load_weather_config,
    model_state, restore_training_state, save_training_state, training_signature,
    validate_weather_inputs,
)

logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(values, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in values.items()}


def create_trainer(cfg, dataset, device):
    from utils.misc import import_str
    return import_str(cfg.trainer.type)(
        **copy.deepcopy(cfg.trainer), dataset=dataset,
        weathers=["raw", cfg.trainer.weather_training.weather],
        model_config=copy.deepcopy(cfg.model), vgg_config=cfg.vgg, device=device,
    )


def fit_step(trainer, dataset, step):
    """One update using matching camera/frame indices for both targets."""
    trainer.set_train()
    trainer.preprocess_per_train_step(step)
    trainer.optimizer_zero_grad()
    weather = trainer.weather_name
    requested = ["raw", weather] if trainer.requires_paired_raw else [weather]
    images, cameras = dataset.train_image_set.next(trainer._get_downscale_factor(), requested)
    weather_info = to_device(images[weather], trainer.device)
    camera = to_device(cameras[weather], trainer.device)
    raw_info, raw_output = {}, None
    if trainer.requires_paired_raw:
        raw_info = to_device(images["raw"], trainer.device)
        raw_camera = to_device(cameras["raw"], trainer.device)
        if not torch.equal(raw_info["img_idx"], weather_info["img_idx"]):
            raise RuntimeError("Raw and weather samples must refer to the same frame")
        raw_output = trainer.render_raw_reference(raw_info, raw_camera)
    output = trainer(weather_info, camera, weather)
    losses = trainer.compute_weather_losses(output, raw_output, weather_info, raw_info, camera, weather)
    for name, value in losses.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {name} at step {step}")
    trainer.backward(losses)
    # Small residual parameter set only; catches numerical failures before Adam.
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradient in {group['name']} at step {step}")
    trainer.optimizer_step()
    trainer.postprocess_per_train_step(step)
    with torch.no_grad():
        values = {name: float(value.detach()) for name, value in losses.items()}
        values.update({name: float(value) for name, value in trainer.compute_metrics(output, weather_info).items()})
    return values


def evaluate(trainer, dataset, cfg):
    from tools.eval import do_evaluation
    if not cfg.render.render_test or dataset.test_image_set is None or len(dataset.test_image_set) == 0:
        raise ValueError("Weather stage evaluation requires render.render_test=true and a nonempty test split")
    args = argparse.Namespace(enable_wandb=False, render_video_postfix=None)
    do_evaluation(
        step=trainer.step, cfg=cfg, trainer=trainer, dataset=dataset,
        args=args, weather=trainer.weather_name,
        render_keys=["gt_rgbs", "rgbs", "Background_rgbs", "Dynamic_rgbs",
                     "RigidNodes_rgbs", "DeformableNodes_rgbs", "SMPLNodes_rgbs"],
    )


def run(args):
    source = Path(args.init_from_base or args.resume_from).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    cfg = load_weather_config(args.config_file, source, args.weather, args.weather_stage,
                              args.opts, resume=bool(args.resume_from))
    input_report = validate_weather_inputs(cfg)
    run_dir = Path(args.run_dir).resolve() if args.run_dir else source.parent
    if args.init_from_base and (not args.run_dir or run_dir == source.parent):
        raise ValueError("A new weather stage needs a separate --run_dir")
    if args.resume_from and run_dir != source.parent:
        raise ValueError("Resume/evaluation must use the checkpoint's existing run directory")
    if args.evaluate and not args.resume_from:
        raise ValueError("--evaluate requires --resume_from")
    cfg.log_dir = str(run_dir)
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        print(input_report)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Weather PBR fitting requires CUDA")
    with exclusive_run_lock(run_dir / ".weather_train.lock"):
        for folder in ("metrics", "videos", "images"):
            (run_dir / folder / cfg.trainer.weather_training.weather).mkdir(parents=True, exist_ok=True)
        saved_config = run_dir / "config.yaml"
        if saved_config.exists():
            prior = OmegaConf.load(saved_config)
            if training_signature(prior) != training_signature(cfg):
                raise ValueError("Existing run config differs; choose a new run directory")
        if args.init_from_base and any(run_dir.glob("checkpoint*.pth")):
            raise FileExistsError("Run already has checkpoints; resume it or choose a new directory")
        if not args.evaluate:
            OmegaConf.save(cfg, saved_config)
        handlers = [logging.StreamHandler(), logging.FileHandler(run_dir / "train_weather.log")]
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=handlers, force=True)
        logger.info("Paired inputs: %s; stage=%s", input_report, cfg.trainer.weather_training.stage)
        seed_everything(int(cfg.seed))
        from datasets.driving_dataset import DrivingDataset
        dataset = DrivingDataset(data_cfg=cfg.data)
        trainer = create_trainer(cfg, dataset, torch.device("cuda"))
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if args.init_from_base:
            trainer.init_from_base_state(model_state(checkpoint, device=trainer.device))
        else:
            trainer.load_state_dict(model_state(checkpoint, device=trainer.device), load_only_model=True, strict=True)
        if args.evaluate:
            del checkpoint
            evaluate(trainer, dataset, cfg)
            atomic_json_save({"status": "completed", "weather": trainer.weather_name,
                              "stage": trainer.weather_stage, "step": trainer.step,
                              "evaluation_done": True, "finished_at": datetime.now().astimezone().isoformat()},
                             run_dir / "stage_status.json")
            return
        trainer.initialize_optimizer()
        start = restore_training_state(trainer, checkpoint, cfg) if args.resume_from else 1
        del checkpoint
        total = int(cfg.trainer.optim.num_iters)
        if start > total + 1:
            raise ValueError("Checkpoint step exceeds this stage's planned iterations")
        logger.info("%s/%s: steps %s..%s, frozen base step=%s, optimizer=%s",
                    trainer.weather_name, trainer.weather_stage, start, total, trainer.base_model_step,
                    [(g["name"], g["lr"]) for g in trainer.optimizer.param_groups])
        final_path = run_dir / "checkpoint_final.pth"
        latest_path = run_dir / "checkpoint_training_latest.pth"
        status_path = run_dir / "stage_status.json"
        status = {"status": "running", "weather": trainer.weather_name, "stage": trainer.weather_stage,
                  "step": start - 1, "num_iters": total, "evaluation_done": False}
        atomic_json_save(status, status_path)
        started, last_print = time.monotonic(), time.monotonic()
        print_freq, save_freq = int(cfg.logging.print_freq), int(cfg.logging.saveckpt_freq)
        if print_freq < 1 or save_freq < 1:
            raise ValueError("print_freq and saveckpt_freq must be positive")
        with (run_dir / "weather_metrics.jsonl").open("a") as metrics:
            import json
            for step in range(start, total + 1):
                values = fit_step(trainer, dataset, step)
                if step == start or step % print_freq == 0 or step == total:
                    elapsed = time.monotonic() - started
                    speed = elapsed / (step - start + 1)
                    record = {"step": step, "elapsed_seconds": elapsed,
                              "seconds_per_iteration": speed, **values}
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    logger.info("[%d/%d] %.3f s/it, ETA %.1f min, losses=%s", step, total, speed,
                                (total - step) * speed / 60, values)
                    last_print = time.monotonic()
                elif time.monotonic() - last_print > 60:
                    logger.info("[%d/%d] training continues", step, total)
                    last_print = time.monotonic()
                if step % save_freq == 0 or step == total:
                    save_training_state(trainer, cfg, latest_path)
                    status["step"] = step
                    atomic_json_save(status, status_path)
                    logger.info("Saved optimizer/RNG continuation: %s", latest_path)
        # A final-step continuation can reach here without any repeated update.
        atomic_torch_save(trainer.state_dict(only_model=True), final_path)
        status.update(status="trained", step=trainer.step,
                      training_seconds_this_invocation=time.monotonic() - started)
        atomic_json_save(status, status_path)
        if not args.skip_eval:
            evaluate(trainer, dataset, cfg)
            status["evaluation_done"] = True
        status.update(status="completed", finished_at=datetime.now().astimezone().isoformat())
        atomic_json_save(status, status_path)
        logger.info("Stage finished: %s", run_dir)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", default=str(ROOT / "configs/base_v4/weather.yaml"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--init_from_base")
    source.add_argument("--resume_from")
    parser.add_argument("--run_dir")
    parser.add_argument("--weather", choices=("rain", "snow"))
    parser.add_argument("--weather_stage", choices=("material", "environment", "joint"))
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Load all saved heads and evaluate without fitting")
    parser.add_argument("--dry_run", action="store_true", help="Resolve config and check paired file paths only")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


if __name__ == "__main__":
    run(make_parser().parse_args())
