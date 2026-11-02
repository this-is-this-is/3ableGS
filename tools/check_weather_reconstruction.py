#!/usr/bin/env python3
"""Validate the real raw checkpoint and paired weather inputs without a long fit.

Each weather runs three in-memory updates (one per stage), including a full
resolution joint update. No measured checkpoint or training run is modified.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from datasets.driving_dataset import DrivingDataset
from tools.train_weather import create_trainer, fit_step, seed_everything, to_device
from utils.weather_training import load_weather_config, model_state, validate_weather_inputs


def frozen_digest(trainer):
    digest = hashlib.sha256()
    for name, model in trainer.models.items():
        for key, value in model.state_dict().items():
            if key.startswith(("weather_material_decoders.", "weather_log_residuals.")):
                continue
            digest.update(f"{name}:{key}".encode())
            if torch.is_tensor(value):
                digest.update(value.detach().contiguous().cpu().numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def get_frame(dataset, index, weather, downscale=4):
    source = dataset.pixel_source
    source.update_downscale_factor(source.downscale_factor / downscale)
    try:
        image, camera = source.get_image(index, weather)
    finally:
        source.reset_downscale_factor()
    return to_device(image, "cuda"), to_device(camera, "cuda")


def check_one(args, checkpoint, weather):
    seed_everything(0)
    cfg = load_weather_config(ROOT / "configs/base_v4/weather.yaml", args.base_ckpt, weather, "material")
    validate_weather_inputs(cfg)
    dataset = DrivingDataset(cfg.data)
    trainer = create_trainer(cfg, dataset, torch.device("cuda"))
    trainer.init_from_base_state(model_state(checkpoint, device=trainer.device))
    before = frozen_digest(trainer)
    counts = trainer.get_gaussian_count()
    coverage = []
    # Confidence filters should be meaningful for actual generated pseudo maps.
    for frame in (0, 15, 29):
        raw, camera = get_frame(dataset, frame, "raw")
        target, _ = get_frame(dataset, frame, weather)
        raw_output = trainer.render_raw_reference(raw, camera)
        mask = trainer._weather_material_mask(raw_output, raw, target, raw_output["albedo"].shape[:2])
        valid = int(mask.sum())
        if valid < cfg.trainer.weather_training.mask.min_valid_pixels:
            raise AssertionError(f"{weather} frame{frame}: insufficient confident pixels {valid}")
        coverage.append({"frame": frame, "valid_pixels": valid, "fraction": float(mask.mean())})
        del raw, camera, target, raw_output, mask
    steps = []
    for stage, step in (("material", 1), ("environment", 1), ("joint", 500)):
        trainer.configure_weather_stage(stage)
        trainer.initialize_optimizer()
        stats = fit_step(trainer, dataset, step)
        if trainer.get_gaussian_count() != counts:
            raise AssertionError("Weather fitting changed Gaussian topology")
        steps.append({"stage": stage, "step": step, "losses": stats})
    after = frozen_digest(trainer)
    if before != after:
        raise AssertionError("Weather fitting modified frozen base model state")
    trainer.set_eval()
    trainer.set_render_weather_strengths(material=0, environment=0)
    heldout = []
    for frame in dataset.test_timesteps:
        raw, camera = get_frame(dataset, int(frame), "raw")
        with torch.no_grad():
            baseline = trainer(raw, camera, "raw")["rgb"]
            weather_rgb = trainer(raw, camera, weather)["rgb"]
        error = float((baseline - weather_rgb).abs().max())
        torch.testing.assert_close(baseline, weather_rgb, atol=0, rtol=0)
        heldout.append({"frame": int(frame), "zero_strength_rgb_max_error": error})
    report = {"weather": weather, "status": "passed", "base_model_step": trainer.base_model_step,
              "gaussians": counts, "frozen_base_sha256_before": before,
              "frozen_base_sha256_after": after, "mask_coverage_quarter_resolution": coverage,
              "updates": steps, "heldout_quarter_resolution": heldout,
              "joint_update_full_resolution": True, "saved_training_checkpoint": False}
    del trainer, dataset, raw, camera, baseline, weather_rgb
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_ckpt", type=Path, default=ROOT / "output/pbr/rebuild/scene_788_30f/base_v4_raw/checkpoint_final.pth")
    parser.add_argument("--out", type=Path, default=ROOT / "logs/reconstruction/phase5_real_scene.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    report = {"base_checkpoint": str(args.base_ckpt), "results": []}
    for weather in ("rain", "snow"):
        result = check_one(args, checkpoint, weather)
        report["results"].append(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    report["status"] = "passed"
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
