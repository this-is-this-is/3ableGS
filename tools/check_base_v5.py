#!/usr/bin/env python3
"""Tiny real-CUDA check of Base v5; no scene training."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tools.check_base_v3_reconstruction import (
    make_trainer, initialize_background, synthetic_inputs, group_gradients,
)
from utils.config import load_training_config


def run(config_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RGB joint integration check")
    cfg = load_training_config(config_path)
    # Shorten only this synthetic warmup. Native geometry/environment clocks
    # are exercised at representative global steps without a long training run.
    cfg.trainer.rgb_joint.warmup_camera_updates = 1
    cfg.trainer.rgb_joint.ramp_camera_updates = 1
    cfg.trainer.res_schedule.downscale_times = 0
    torch.manual_seed(int(cfg.seed))
    trainer = make_trainer(cfg)
    background = initialize_background(trainer)
    trainer.initialize_optimizer()
    images, cameras = synthetic_inputs()
    report = {"status": "passed", "synthetic_only": True, "config": str(config_path),
              "device": torch.cuda.get_device_name(), "updates": []}
    original_count = background.num_points
    for update, step in enumerate((0, 1, 2, 3, 600, 3001)):
        trainer.set_train()
        trainer.preprocess_per_train_step(step)
        trainer.optimizer_zero_grad()
        outputs = trainer(images, cameras, "raw")
        trainer.update_visibility_filter()
        active = update >= 2
        if bool(outputs["rgb_residual_active"]) != active:
            raise AssertionError("Residual warmup did not use the completed camera counter")
        if update <= 2:
            torch.testing.assert_close(outputs["rgb"], outputs["rgb_pure_pbr"], atol=0, rtol=0)
        losses = trainer.compute_losses(outputs, images, cameras, "raw")
        if ("rgb_joint_pure_pbr_loss" in losses) != active:
            raise AssertionError("Pure PBR auxiliary does not follow residual activation")
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise AssertionError("Nonfinite joint loss")
        trainer.backward(losses)
        stats = group_gradients(trainer)
        head = stats["RGBResidual#all"]["gradient_l1"]
        feature = stats["Background#correction_feature"]["gradient_l1"]
        if active and head <= 0:
            raise AssertionError("RGB residual head received no gradient")
        if update > 2 and feature <= 0:
            raise AssertionError("Gaussian correction features received no gradient")
        if update < 2 and (head != 0 or feature != 0):
            raise AssertionError("Warmup must preserve the original PBR objective")
        trainer.optimizer_step()
        if step == 600:
            # Force native refinement to split, verifying the same learned
            # feature rows and their already-initialized Adam moments survive.
            background.xys_grad_norm = torch.full((background.num_points,), 100., device="cuda")
            background.vis_counts = torch.ones(background.num_points, device="cuda")
            background.max_2Dsize = torch.zeros(background.num_points, device="cuda")
        trainer.postprocess_per_train_step(step)
        trainer.complete_camera_update(step)  # idempotent public hook
        if int(trainer.rgb_joint_camera_updates) != update + 1:
            raise AssertionError("Completed camera counter advanced incorrectly")
        feature_parameter = background._correction_features
        if len(feature_parameter) != background.num_points:
            raise AssertionError("Correction rows lost density alignment")
        group = next(g for g in trainer.optimizer.param_groups if g["name"] == "Background#correction_feature")
        if group["params"][0] is not feature_parameter:
            raise AssertionError("Density changed the feature parameter without its optimizer")
        moments = trainer.optimizer.state[feature_parameter]
        for key in ("exp_avg", "exp_avg_sq"):
            if moments[key].shape != feature_parameter.shape:
                raise AssertionError("Density changed Gaussian rows without Adam moments")
        report["updates"].append({"global_step": step, "camera_updates": update + 1,
            "residual_strength": float(outputs["rgb_residual_strength"]),
            "gaussians": background.num_points,
            "losses": {name: float(value.detach()) for name, value in losses.items()},
            "gradients": stats})
    if background.num_points <= original_count:
        raise AssertionError("Synthetic density check did not grow the population")
    trainer.set_eval()
    with torch.no_grad():
        expected = trainer(images, cameras, "raw")
        if torch.equal(expected["rgb_hybrid"], expected["rgb_pure_pbr"]):
            raise AssertionError("Trained RGB residual did not alter the image")
    with tempfile.TemporaryDirectory(prefix="base_v5_check_") as temp:
        trainer.save_checkpoint(temp, save_only_model=True, is_final=True)
        state = torch.load(Path(temp) / "checkpoint_final.pth", map_location="cuda", weights_only=False)
        clone = make_trainer(cfg)
        clone.load_state_dict(state, load_only_model=True, strict=True)
        clone.set_eval()
        with torch.no_grad():
            actual = clone(images, cameras, "raw")
        for key in ("rgb_hybrid", "rgb_pure_pbr", "albedo", "normal"):
            torch.testing.assert_close(actual[key], expected[key], atol=2e-7, rtol=2e-6)
        if int(clone.rgb_joint_camera_updates) != 6:
            raise AssertionError("Checkpoint lost completed camera count")
        report["checkpoint"] = {"strict_reload": True, "camera_updates": 6,
            "max_rgb_difference": float((actual["rgb"] - expected["rgb"]).abs().max())}
        with torch.no_grad():
            clone.env.external = torch.full_like(clone.env._log_env, .7)
            relit = clone(images, cameras, "raw")
        if bool(relit["rgb_residual_active"]) or not torch.equal(relit["rgb"], relit["rgb_pure_pbr"]):
            raise AssertionError("External HDR did not exactly bypass display RGB residual")
        report["external_hdr_exact_bypass"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base_v5/scene788_100f.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.config)
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
