#!/usr/bin/env python3
"""Validate local DiffReg through real CUDA PBR and benchmark its native loss.

The scene is freshly initialized and synthetic. No training checkpoint is read,
and no long experiment is performed. Native loss timings exclude rendering and
must not be interpreted as an end-to-end training overhead percentage.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from models.diffreg_material import material_relation_loss, validate_diffreg_config
from tools.check_base_v3_reconstruction import initialize_background, make_trainer
from tools.check_base_v4_reconstruction import assert_same_state, make_inputs
from utils.config import load_training_config


def initialize(cfg):
    torch.manual_seed(int(cfg.seed))
    trainer = make_trainer(cfg)
    background = initialize_background(trainer)
    # Overlapping, opaque surfaces give the actual .95-opacity gate support.
    with torch.no_grad():
        background._scales[:, :2].fill_(torch.tensor(.65).log().item())
        background._opacities.fill_(torch.logit(torch.tensor(.99)).item())
        generator = torch.Generator(device="cuda").manual_seed(517)
        background._features.copy_(torch.rand(background._features.shape, device="cuda", generator=generator))
    return trainer


def gradients(trainer):
    result = {}
    for group in trainer.optimizer.param_groups:
        values = [p.grad for p in group["params"] if p.grad is not None]
        if not all(bool(torch.isfinite(g).all()) for g in values):
            raise AssertionError(f"Nonfinite gradient: {group['name']}")
        result[group["name"]] = sum(float(g.abs().sum()) for g in values)
    return result


def native_loss_benchmark(repeats):
    device = "cuda"
    h, w = 1280, 1920
    generator = torch.Generator(device=device).manual_seed(241)
    prediction = {name: (.2 + .6 * torch.rand(h, w, channels, device=device, generator=generator)).requires_grad_()
                  for name, channels in (("albedo", 3), ("roughness", 1), ("metallic", 1))}
    teacher = {name: torch.full_like(value, .5) for name, value in prediction.items()}
    valid = torch.ones(h, w, device=device, dtype=torch.bool)
    depth = torch.full((h, w, 1), 5., device=device)
    normal = torch.tensor([0., 0., 1.], device=device).expand(h, w, 3)
    cfg = validate_diffreg_config({"enabled": True})
    timings = []
    torch.cuda.reset_peak_memory_stats()
    memory_start = torch.cuda.memory_allocated()
    for iteration in range(repeats + 5):
        for value in prediction.values():
            value.grad = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss, stats = material_relation_loss(prediction, teacher, valid, step=4000 + iteration,
                                             config=cfg, depth=depth, normal=normal)
        loss.backward()
        torch.cuda.synchronize()
        if iteration >= 5:
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "scope": "synthetic native material-map loss forward+backward only, excludes alignment and rendering",
        "resolution": [h, w], "repeats": repeats,
        "median_ms": statistics.median(timings), "mean_ms": statistics.mean(timings),
        "incremental_peak_bytes": torch.cuda.max_memory_allocated() - memory_start,
        "last_diagnostics": {name: float(value) for name, value in stats.items()},
    }


def run(config_path, repeats=30):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this integration check")
    cfg = load_training_config(config_path)
    baseline = initialize(cfg)
    variant_cfg = copy.deepcopy(cfg)
    variant_cfg.trainer.losses.diffreg_material = validate_diffreg_config({"enabled": True})
    variant = initialize(variant_cfg)
    assert_same_state(baseline.state_dict(), variant.state_dict())
    images, cameras = make_inputs(160)
    images["dynamic_masks"] = torch.zeros_like(images["sky_masks"])
    baseline.initialize_optimizer()
    variant.initialize_optimizer()
    for trainer in (baseline, variant):
        trainer.set_train()
        trainer.preprocess_per_train_step(4000)
        trainer.optimizer_zero_grad()
    baseline_output = baseline(images, cameras, "raw")
    variant_output = variant(images, cameras, "raw")
    torch.testing.assert_close(baseline_output["rgb"], variant_output["rgb"], rtol=2e-6, atol=2e-7)
    render_error = float((baseline_output["rgb"] - variant_output["rgb"]).detach().abs().max())
    # Reuse one render for exact loss-hook comparison; independent CUDA
    # rasterization/shading launches can differ by float32 roundoff.
    variant._diffreg_config["enabled"] = False
    original_losses = variant.compute_losses(variant_output, images, cameras, "raw")
    variant._diffreg_config["enabled"] = True
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    losses = variant.compute_losses(variant_output, images, cameras, "raw")
    for name in original_losses:
        torch.testing.assert_close(original_losses[name], losses[name], rtol=0, atol=0)
    torch.testing.assert_close(cpu_rng, torch.get_rng_state(), rtol=0, atol=0)
    torch.testing.assert_close(cuda_rng, torch.cuda.get_rng_state(), rtol=0, atol=0)
    loss = losses["diffreg_material_loss"]
    if not bool(torch.isfinite(loss)) or float(loss.detach()) <= 0:
        raise AssertionError("Fixture did not produce a finite, nonzero material regularizer")
    variant.update_visibility_filter()
    # Isolate gradients from this new objective, rather than the RGB teacher.
    loss.backward()
    grad = gradients(variant)
    for name in ("Background#feature", "Background#mlp.raw", "Background#xyz"):
        if grad[name] <= 0:
            raise AssertionError(f"No regularization gradient into {name}")
    absgrad = variant.info["means2d"].absgrad
    if not bool(torch.isfinite(absgrad).all()) or float(absgrad.abs().sum()) <= 0:
        raise AssertionError("Material regularization lost rasterization/densification gradients")
    stats = {name: float(value) for name, value in variant._last_diffreg_stats.items()}
    variant.optimizer_step()
    variant.set_eval()
    with torch.no_grad():
        expected = variant(images, cameras, "raw")["rgb"]
    with tempfile.TemporaryDirectory(prefix="diffreg_cuda_") as directory:
        variant.save_checkpoint(directory, save_only_model=True, is_final=True)
        clone = make_trainer(variant_cfg)
        clone.resume_from_checkpoint(str(Path(directory) / "checkpoint_final.pth"))
        clone.set_eval()
        with torch.no_grad():
            actual = clone(images, cameras, "raw")["rgb"]
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        checkpoint_error = float((actual - expected).abs().max())
    report = {
        "status": "passed", "synthetic_only": True, "device": torch.cuda.get_device_name(),
        "fresh_initialization_identical": True, "original_losses_identical": True,
        "global_rng_unchanged": True, "render_before_update_max_rgb_error": render_error,
        "isolated_regularizer_loss": float(loss.detach()), "diagnostics": stats,
        "gradients": grad, "means2d_absgrad_l1": float(absgrad.abs().sum()),
        "strict_checkpoint_reload_max_rgb_error": checkpoint_error,
    }
    del baseline, variant, clone, baseline_output, variant_output, original_losses, losses, loss
    torch.cuda.empty_cache()
    report["native_loss_benchmark"] = native_loss_benchmark(repeats)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base_v4/scene788_30f.yaml")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=Path, default=ROOT / "logs/diffreg_validation/gpu_check.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    result = run(args.config, args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
