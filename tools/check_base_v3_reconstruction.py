#!/usr/bin/env python3
"""Run two synthetic CUDA updates through the restored Base v3 trainer.

No scene images, G-buffer generation, or long training are started. The actual
configured material decoder, HDR64 lighting, camera response, rasterizer,
losses, optimizer scheduling and model-only checkpoint path are exercised.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from utils.config import load_training_config
from utils.misc import import_str


class SyntheticDataset:
    num_img_timesteps = 2
    train_image_set = range(2)
    full_image_set = range(2)
    test_timesteps = []
    pixel_source = SimpleNamespace(gbuffer_crop_frac=0.0)

    def get_aabb(self):
        return torch.tensor([[-2., -2., 0.], [2., 2., 5.]])


def make_trainer(cfg):
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    for name in ("RigidNodes", "DeformableNodes", "SMPLNodes"):
        model_cfg.pop(name, None)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        dataset=SyntheticDataset(),
        model_config=model_cfg,
        weathers=["raw"],
        vgg_config=cfg.vgg,
        device=torch.device("cuda"),
    )
    trainer.render_dynamic_mask = True  # Exercise auxiliary render/info isolation.
    return trainer


def synthetic_inputs(size=32):
    device = "cuda"
    ys, xs = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32), indexing="ij",
    )
    fx = fy = 32.
    center = size / 2
    rays = F.normalize(torch.stack(((xs + .5 - center) / fx,
                                   (ys + .5 - center) / fy,
                                   torch.ones_like(xs)), -1), dim=-1)
    sky = ((xs < 5) | (xs > 26) | (ys < 5) | (ys > 26)).float()
    pixels = torch.stack((.2 + .3 * xs / size,
                          .25 + .25 * ys / size,
                          .4 + .1 * (xs + ys) / (2 * size)), -1)
    normal = F.normalize(torch.tensor([.12, .2, .97], device=device), dim=0)
    images = {
        "pixels": pixels,
        "viewdirs": rays,
        "normed_time": torch.zeros(size, size, device=device),
        "img_idx": torch.zeros(size, size, device=device, dtype=torch.long),
        "sky_masks": sky,
        "egocar_masks": torch.zeros_like(sky),
        "lidar_depth_map": (1 - sky) * 3.7,
        "gbuf_basecolor": torch.full_like(pixels, .55),
        "gbuf_metallic": torch.full((size, size, 1), .1, device=device),
        "gbuf_roughness": torch.full((size, size, 1), .65, device=device),
        "gbuf_normal": normal.expand(size, size, 3).clone(),
    }
    cameras = {
        "camera_to_world": torch.eye(4, device=device),
        "intrinsics": torch.tensor([[fx, 0., center], [0., fy, center], [0., 0., 1.]], device=device),
        "height": size,
        "width": size,
    }
    return images, cameras


def initialize_background(trainer):
    grid = torch.linspace(-.7, .7, 4, device="cuda")
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack((x.flatten(), y.flatten(), torch.full_like(x.flatten(), 3.5)), -1)
    background = trainer.models["Background"]
    background.create_from_pcd(points, torch.full_like(points, .5))
    with torch.no_grad():
        background._quats.zero_()
        background._quats[:, 0] = 1.
        background._scales.copy_(torch.log(torch.tensor([.18, .16, .035], device="cuda")).expand_as(background._scales))
        background._opacities.fill_(torch.logit(torch.tensor(.55)).item())
    return background


def group_gradients(trainer):
    result = {}
    for group in trainer.optimizer.param_groups:
        tensors = [p.grad for p in group["params"] if p.grad is not None]
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise AssertionError(f"Non-finite gradient: {group['name']}")
        result[group["name"]] = {
            "lr": float(group["lr"]),
            "gradient_l1": sum(float(value.detach().abs().sum()) for value in tensors),
            "gradient_tensors": len(tensors),
        }
    required = {
        "Background#xyz", "Background#feature", "Background#opacity",
        "Background#scaling", "Background#rotation", "Background#mlp.raw",
        "Background#mlp.raw.normal_residual", "EnvLight#all", "Affine#all",
        "CameraResponse#all",
    }
    for name in required:
        if result[name]["gradient_l1"] <= 0:
            raise AssertionError(f"Missing gradient through {name}")
    if result["CamPose#all"]["lr"] != 0:
        raise AssertionError("Base v3 CamPose must remain frozen")
    return result


def run(config_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this synthetic rasterizer smoke check")
    cfg = load_training_config(config_path)
    if OmegaConf.to_container(cfg.model.EnvLight.params) != {
        "resolution": 64, "n_mips": 5, "brdf_lut_size": 64, "ggx_tail_resolution": 16,
    }:
        raise AssertionError("Use the restored Base v3 HDR configuration")
    torch.manual_seed(int(cfg.seed))
    trainer = make_trainer(cfg)
    background = initialize_background(trainer)
    trainer.initialize_optimizer()
    images, cameras = synthetic_inputs()
    report = {
        "status": "passed",
        "config": str(Path(config_path).resolve()),
        "device": torch.cuda.get_device_name(),
        "synthetic_only": True,
        "image_size": [32, 32],
        "gaussian_count": background.num_points,
        "models": sorted(trainer.models),
        "hdr": OmegaConf.to_container(cfg.model.EnvLight.params),
        "updates": [],
    }
    for step in (0, 3001):
        trainer.set_train()
        trainer.preprocess_per_train_step(step)
        trainer.optimizer_zero_grad()
        env_before = trainer.env._log_env.detach().clone()
        outputs = trainer(images, cameras, "raw")
        trainer.update_visibility_filter()
        if outputs["rgb"].shape != (32, 32, 3):
            raise AssertionError("Unexpected render size")
        if not all(bool(torch.isfinite(value).all()) for value in outputs.values()):
            raise AssertionError("Non-finite rendered layer")
        if outputs["rgb_diffuse"].requires_grad or outputs["rgb_specular"].requires_grad:
            raise AssertionError("Diagnostic layers must not retain camera-response graphs")
        losses = trainer.compute_losses(outputs, images, cameras, "raw")
        required_losses = {"rgb_loss", "ssim_loss", "sky_loss_opacity", "depth_loss",
                           "albedo_loss", "metallic_loss", "roughness_loss", "normal_loss",
                           "affine_loss", "envmap_tv", "camera_response_identity",
                           "camera_response_smoothness", "Background_flatten", "Background_normal_residual"}
        if not required_losses.issubset(losses):
            raise AssertionError(f"Missing losses: {required_losses - set(losses)}")
        if not all(bool(torch.isfinite(value).all()) for value in losses.values()):
            raise AssertionError("Non-finite loss")
        trainer.backward(losses)
        gradient_stats = group_gradients(trainer)
        absgrad = trainer.info["means2d"].absgrad
        if absgrad.shape != (1, background.num_points, 2) or not bool(torch.isfinite(absgrad).all()):
            raise AssertionError("Densification lost the main rasterization's gradient statistics")
        if float(absgrad.abs().sum()) <= 0:
            raise AssertionError("No projected Gaussian gradient")
        trainer.optimizer_step()
        trainer.postprocess_per_train_step(step)
        env_update = float((trainer.env._log_env.detach() - env_before).abs().max())
        if step == 0 and (env_update != 0. or gradient_stats["EnvLight#all"]["lr"] != 0.):
            raise AssertionError("HDR should not update before opt_after=3000")
        if step > 3000 and (env_update <= 0. or gradient_stats["EnvLight#all"]["lr"] <= 0.):
            raise AssertionError("HDR did not update after opt_after=3000")
        if background.xys_grad_norm is None or not bool(torch.isfinite(background.xys_grad_norm).all()):
            raise AssertionError("Densification statistics were not accumulated")
        report["updates"].append({
            "step": step,
            "losses": {name: float(value.detach()) for name, value in losses.items()},
            "gradients": gradient_stats,
            "env_max_parameter_update": env_update,
            "means2d_absgrad_l1": float(absgrad.abs().sum()),
            "densification_accumulated_gradient_l1": float(background.xys_grad_norm.abs().sum()),
            "visible_gaussians": int((trainer.info["radii"] > 0).sum()),
        })
    missing = dict(images)
    missing.pop("gbuf_normal")
    try:
        trainer.compute_losses(outputs, missing, cameras, "raw")
    except ValueError:
        report["missing_teacher_rejected"] = True
    else:
        raise AssertionError("Training silently accepted missing teacher supervision")
    trainer.set_eval()
    with torch.no_grad():
        expected = trainer(images, cameras, "raw")["rgb"]
    with tempfile.TemporaryDirectory(prefix="base_v3_smoke_") as tmp:
        trainer.save_checkpoint(tmp, save_only_model=True, is_final=True)
        ckpt = Path(tmp) / "checkpoint_final.pth"
        saved = torch.load(ckpt, map_location="cuda", weights_only=False)
        clone = make_trainer(cfg)
        clone.load_state_dict(saved, load_only_model=True, strict=True)
        clone.set_eval()
        with torch.no_grad():
            actual = clone(images, cameras, "raw")["rgb"]
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        report["checkpoint"] = {
            "model_only_strict_reload": True,
            "max_rgb_difference": float((actual - expected).abs().max()),
            "file_bytes": ckpt.stat().st_size,
            "step": clone.step,
            "gaussian_count": clone.models["Background"].num_points,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base_v3/base.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "logs/reconstruction/phase3_integration.json")
    args = parser.parse_args()
    report = run(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
