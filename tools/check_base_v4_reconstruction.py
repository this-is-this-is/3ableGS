#!/usr/bin/env python3
"""Check Base v4's real LPIPS objective on a tiny synthetic CUDA scene.

This performs no long training and never opens or modifies the measured scene
checkpoint. The production PBR renderer, material heads, HDR64 lighting,
camera response, pretrained AlexNet LPIPS and checkpoint code are exercised.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from tools.check_base_v3_reconstruction import (
    group_gradients, initialize_background, make_trainer, synthetic_inputs,
)
from utils.config import load_training_config


def assert_same_state(first, second, path="state"):
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError(f"Different state keys at {path}")
        for name in first:
            assert_same_state(first[name], second[name], f"{path}.{name}")
    elif isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, atol=0, rtol=0, msg=path)
    elif first != second:
        raise AssertionError(f"Different state values at {path}")


def metric_state(metric):
    return {
        "update_count": metric.update_count,
        "values": copy.deepcopy({name: getattr(metric, name) for name in metric._defaults}),
    }


def make_inputs(size=160):
    images, cameras = synthetic_inputs(size)
    # Keep the original fixture's camera field of view as the image grows.
    cameras["intrinsics"][:2, :2] *= size / 32
    ys, xs = torch.meshgrid(
        torch.arange(size, device="cuda", dtype=torch.float32),
        torch.arange(size, device="cuda", dtype=torch.float32), indexing="ij",
    )
    images["viewdirs"] = torch.nn.functional.normalize(torch.stack((
        (xs + .5 - size / 2) / size,
        (ys + .5 - size / 2) / size,
        torch.ones_like(xs),
    ), -1), dim=-1)
    images["sky_masks"] = ((xs < .15 * size) | (xs > .85 * size)
                            | (ys < .15 * size) | (ys > .85 * size)).float()
    images["lidar_depth_map"] = (1 - images["sky_masks"]) * 3.7
    # A visible target discontinuity makes the boundary-biased sampling useful.
    images["pixels"][size // 2:, :, 0] += .25
    return images, cameras


def run(config_path, baseline_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Base v4 integration check")
    cfg = load_training_config(config_path)
    baseline_cfg = load_training_config(baseline_path)
    torch.manual_seed(int(baseline_cfg.seed))
    baseline = make_trainer(baseline_cfg)
    initialize_background(baseline)
    torch.manual_seed(int(cfg.seed))
    trainer = make_trainer(cfg)
    background = initialize_background(trainer)
    assert_same_state(baseline.state_dict(), trainer.state_dict())
    baseline_parameters = {name: p.requires_grad for name, p in baseline.named_parameters()}
    if baseline_parameters != {name: p.requires_grad for name, p in trainer.named_parameters()}:
        raise AssertionError("Base v4 introduced or changed registered trainable parameters")
    del baseline
    trainer.initialize_optimizer()
    trainer.set_train()
    trainer.preprocess_per_train_step(6000)
    trainer.optimizer_zero_grad()
    images, cameras = make_inputs()
    outputs = trainer(images, cameras, "raw")
    trainer.update_visibility_filter()
    outputs["rgb"].retain_grad()
    if trainer.lpips.net.training or any(p.requires_grad for p in trainer.lpips.net.parameters()):
        raise AssertionError("LPIPS must stay frozen and in evaluation mode during training")
    initial_metric_state = metric_state(trainer.lpips)
    calls = []

    def record_forward(module, args, output):
        calls.append({
            "prediction_shape": list(args[0].shape),
            "target_shape": list(args[1].shape),
            "raw_lpips": float(output.detach().mean()),
            "evaluation_mode": not module.training,
        })

    hook = trainer.lpips.net.register_forward_hook(record_forward)
    schedule = []
    raw_value = None
    perceptual_loss = None
    for step, expected_weight in ((4999, None), (5000, 0.), (5500, .01), (6000, .02)):
        trainer.step = step
        before = len(calls)
        # Use identical crop locations to isolate the schedule multiplier.
        torch.manual_seed(31415)
        losses = trainer.compute_losses(outputs, images, cameras, "raw")
        if expected_weight is None:
            if "boundary_perceptual_loss" in losses or len(calls) != before:
                raise AssertionError("LPIPS was executed before step 5000")
            schedule.append({"step": step, "weight": 0., "network_calls": 0})
            continue
        perceptual_loss = losses["boundary_perceptual_loss"]
        if len(calls) != before + 1:
            raise AssertionError("Expected one batched LPIPS forward per active step")
        call = calls[-1]
        if call["prediction_shape"] != [4, 3, 128, 128] or call["target_shape"] != [4, 3, 128, 128]:
            raise AssertionError("LPIPS must receive four 128 x 128 RGB patches")
        if raw_value is None:
            raw_value = call["raw_lpips"]
        elif abs(call["raw_lpips"] - raw_value) > 1e-7:
            raise AssertionError("Fixed crop RNG did not reproduce the LPIPS score")
        torch.testing.assert_close(
            perceptual_loss.detach(), perceptual_loss.new_tensor(expected_weight * raw_value),
            atol=1e-8, rtol=1e-6,
        )
        if not bool(torch.isfinite(perceptual_loss)):
            raise AssertionError("Non-finite perceptual loss")
        schedule.append({"step": step, "weight": expected_weight,
                         "network_calls": 1, "loss": float(perceptual_loss.detach()), **call})
    hook.remove()
    assert_same_state(initial_metric_state, metric_state(trainer.lpips), "LPIPS_metric")
    if perceptual_loss is None or float(perceptual_loss.detach()) <= 0:
        raise AssertionError("Fixture did not produce a positive LPIPS loss")
    # Crucially, only the perceptual term participates in this backward pass.
    perceptual_loss.backward()
    gradients = group_gradients(trainer)
    rgb_gradient = outputs["rgb"].grad
    if not bool(torch.isfinite(rgb_gradient).all()) or float(rgb_gradient.abs().sum()) <= 0:
        raise AssertionError("LPIPS did not backpropagate into rendered RGB")
    if any(parameter.grad is not None for parameter in trainer.lpips.net.parameters()):
        raise AssertionError("Frozen LPIPS parameters accumulated gradients")
    absgrad = trainer.info["means2d"].absgrad
    if not bool(torch.isfinite(absgrad).all()) or float(absgrad.abs().sum()) <= 0:
        raise AssertionError("LPIPS lost the main rasterization's densification gradient")
    trainer.set_eval()
    with torch.no_grad():
        expected_rgb = trainer(images, cameras, "raw")["rgb"]
    with tempfile.TemporaryDirectory(prefix="base_v4_smoke_") as tmp:
        trainer.save_checkpoint(tmp, save_only_model=True, is_final=True)
        path = Path(tmp) / "checkpoint_final.pth"
        saved = torch.load(path, weights_only=False, map_location="cuda")
        clone = make_trainer(cfg)
        clone.load_state_dict(saved, load_only_model=True, strict=True)
        clone.set_eval()
        with torch.no_grad():
            actual_rgb = clone(images, cameras, "raw")["rgb"]
        torch.testing.assert_close(actual_rgb, expected_rgb, atol=2e-7, rtol=2e-6)
        checkpoint_report = {
            "model_only_strict_reload": True,
            "max_rgb_difference": float((actual_rgb - expected_rgb).abs().max()),
            "file_bytes": path.stat().st_size,
            "step": clone.step,
        }
    return {
        "status": "passed", "synthetic_only": True,
        "config": str(Path(config_path).resolve()),
        "baseline_config": str(Path(baseline_path).resolve()),
        "device": torch.cuda.get_device_name(), "image_size": [160, 160],
        "gaussian_count": background.num_points,
        "hdr": OmegaConf.to_container(cfg.model.EnvLight.params),
        "fresh_v3_v4_initialization_exact_match": True,
        "same_registered_parameter_names_and_requires_grad": True,
        "same_state_keys": True,
        "lpips_network": type(trainer.lpips.net).__name__,
        "lpips_parameters_frozen": True,
        "lpips_evaluation_metric_state_unchanged": True,
        "schedule": schedule,
        "backward_objective": "boundary_perceptual_loss only",
        "rgb_gradient_l1": float(rgb_gradient.abs().sum()),
        "means2d_absgrad_l1": float(absgrad.abs().sum()),
        "gradients": gradients,
        "checkpoint": checkpoint_report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base_v4/scene788_30f.yaml")
    parser.add_argument("--baseline", type=Path, default=ROOT / "configs/base_v3/scene788_30f.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "logs/reconstruction/phase4_integration.json")
    args = parser.parse_args()
    report = run(args.config, args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
