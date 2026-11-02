#!/usr/bin/env python3
"""Validate scene 788 initialization/render/backward without fitting a model.

Uses real raw frames 000..029 and SMPL/LiDAR at 1/4 image resolution, with
10,000 Background seed points. No optimizer update or pseudo-label generation
is performed. G-buffer loss is covered separately by the synthetic CUDA test.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer
from models.trainers.scene_graph_pbr import MultiTrainer_pbr
from utils.config import load_training_config


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("This scene validation requires CUDA")
    torch.manual_seed(0)
    cfg = load_training_config(ROOT / "configs/base_v3/scene788_30f.yaml", [
        "data.pixel_source.load_gbuffer=false",
        "data.pixel_source.downscale_when_loading=[4]",
        "model.Background.init.from_lidar.num_samples=10000",
        "model.Background.init.near_randoms=0",
        "model.Background.init.far_randoms=0",
        "trainer.res_schedule.downscale_times=0",
    ])
    dataset = DrivingDataset(cfg.data)
    trainer = MultiTrainer_pbr(
        **cfg.trainer, dataset=dataset, weathers=["raw"], model_config=cfg.model,
        vgg_config=cfg.vgg, device=torch.device("cuda"),
    )
    trainer.init_gaussians_from_dataset(dataset)
    trainer.initialize_optimizer()
    trainer.set_train()
    # Exercise deformation once enabled; this does not run earlier iterations.
    trainer.preprocess_per_train_step(3001)
    image, camera = dataset.pixel_source.camera_data[0].get_image(15)
    image = {k: v.cuda() if torch.is_tensor(v) else v for k, v in image.items()}
    camera = {k: v.cuda() if torch.is_tensor(v) else v for k, v in camera.items()}
    outputs = trainer(image, camera, "raw")
    trainer.update_visibility_filter()
    # No generated substitutes: only RGB/depth/mask and Gaussian regularizers.
    losses = BasicTrainer.compute_losses(trainer, outputs, image, camera, "raw")
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite real-scene loss")
    loss.backward()
    summary = {
        "kind": "real scene initialization/render/backward; no optimizer step",
        "scene": 788, "frames": [0, 29], "render_frame": 15,
        "load_downscale": 4, "background_lidar_samples": 10000,
        "gbuffer_targets": False, "gbuffer_loss_tested": False,
        "gaussians": trainer.get_gaussian_count(),
        "rgb_shape": list(outputs["rgb"].shape),
        "losses": {k: float(v.detach()) for k, v in losses.items()},
        "gradient_max": {},
    }
    for name, model in trainer.models.items():
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError(f"Missing/non-finite gradient: {name}")
        summary["gradient_max"][name] = max(float(g.abs().max()) for g in gradients)
    if trainer.info["means2d"].absgrad is None:
        raise AssertionError("Missing densification gradient")
    summary["absgrad_max"] = float(trainer.info["means2d"].absgrad.max())
    summary["status"] = "passed"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "logs/reconstruction/phase3_real_scene_smoke.json")
    args = parser.parse_args()
    summary = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
