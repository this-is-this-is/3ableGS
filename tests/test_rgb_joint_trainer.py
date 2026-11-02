"""Joint RGB residual through native PBR forward, loss and checkpoint hooks.

Scene construction and rasterization are synthetic; the native trainer's
material unpack, shading/affine ordering and mixin are exercised directly.
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import unittest
import torch
from torch import nn
from omegaconf import OmegaConf
from pytorch_msssim import SSIM

from models.rgb_joint_runtime import validate_rgb_joint_config
from models.trainers.scene_graph_pbr import MultiTrainer_pbr
from tests.test_temporal_material_integration import SyntheticTrainer


class JointTrainer(SyntheticTrainer):
    def __init__(self, *, enabled=True):
        super().__init__(size=32)
        self.scene_value = nn.Parameter(torch.tensor(.4))
        self.correction_image = nn.Parameter(torch.zeros(32, 32, 8))
        self.models["Background"].rgb_residual_enabled = True
        self.models["Background"].correction_feature_dim = 8
        self.models["EnvLight"].compute_reg_loss = lambda cfg: {}
        self.crop_frac = 0.
        self._opencv_to_opengl = torch.tensor([1., -1., -1.])
        self.losses_dict = OmegaConf.create({"rgb": {"w": .7}, "ssim": {"w": .3},
            "albedo": {"w": .5}, "metallic": {"w": .2}, "roughness": {"w": .2},
            "normal": {"w": .15}, "boundary": {"enabled": False}})
        self.ssim = SSIM(data_range=1., size_average=True, channel=3)
        self.configure_rgb_joint({"warmup_camera_updates": 2, "ramp_camera_updates": 2} if enabled else None)

    def render_gaussians(self, gs, cam, **kwargs):
        self.albedo = self.scene_value.expand(self.size, self.size, 3)
        outputs, render = super().render_gaussians(gs, cam, **kwargs)
        if self.rgb_joint_config is not None:
            outputs["_rgb_correction_features"] = self.correction_image
        return outputs, render

    def inputs(self, frame=0):
        images, cameras = super().inputs(frame)
        images.update(pixels=torch.full((32, 32, 3), .65), sky_masks=torch.zeros(32, 32),
                      egocar_masks=torch.zeros(32, 32), gbuf_basecolor=torch.full((32, 32, 3), .4),
                      gbuf_metallic=torch.full((32, 32, 1), .1), gbuf_roughness=self.roughness,
                      gbuf_normal=self.normal)
        return images, cameras


def test_forward_zero_init_and_post_affine_correction_preserve_materials():
    base, joint = JointTrainer(enabled=False), JointTrainer()
    base.eval(); joint.eval()
    joint.rgb_joint_camera_updates.fill_(4)
    images, cameras = joint.inputs(1)
    original = base(images, cameras)
    output = joint(images, cameras)
    for key in ("rgb", "albedo", "metallic", "roughness", "normal", "depth", "opacity"):
        torch.testing.assert_close(output[key], original[key], rtol=0, atol=0)
    assert output["rgb_residual_active"] and torch.equal(output["rgb_pure_pbr"], output["rgb_hybrid"])
    with torch.no_grad():
        joint.rgb_residual_head.net[-1].bias[3:].fill_(.5)
    corrected = joint(images, cameras)
    torch.testing.assert_close(corrected["rgb"] - corrected["rgb_pure_pbr"],
                               torch.full_like(corrected["rgb"], .1 * torch.tensor(.5).tanh()))
    for key in ("albedo", "roughness", "normal"):
        torch.testing.assert_close(corrected[key], output[key], rtol=0, atol=0)


def test_joint_gradients_clock_ramp_and_base_losses():
    trainer = JointTrainer().train()
    images, cameras = trainer.inputs()
    optimizer = torch.optim.Adam(trainer.parameters(), lr=.001)
    with patch("models.trainers.scene_graph_weather.MultiTrainer_weather.compute_losses",
               side_effect=lambda self_outputs, *args: {"kept_base_loss": .4 * self_outputs["rgb"].square().mean()}):
        for step in range(6):
            trainer.step = step
            optimizer.zero_grad()
            outputs = trainer(images, cameras)
            losses = trainer.compute_losses(outputs, images, cameras)
            assert "kept_base_loss" in losses and "albedo_loss" in losses
            assert int(trainer.rgb_joint_camera_updates) == step
            if step <= 2:
                assert not outputs["rgb_residual_active"]
                assert "rgb_joint_pure_pbr_loss" not in losses
            else:
                assert outputs["rgb_residual_active"]
                pure = outputs["rgb_pure_pbr"]
                expected = .7 * (pure - images["pixels"]).abs().mean() + .3 * (
                    1 - trainer.ssim(pure.permute(2, 0, 1)[None], images["pixels"].permute(2, 0, 1)[None]))
                torch.testing.assert_close(losses["rgb_joint_pure_pbr_loss"],
                                           .3 * trainer.rgb_residual_strength() * expected)
            sum(losses.values()).backward()
            assert trainer.scene_value.grad is not None and trainer.scene_value.grad.abs() > 0
            if step <= 2:
                assert trainer.rgb_residual_head.net[-1].weight.grad is None
            else:
                assert trainer.rgb_residual_head.net[-1].weight.grad.abs().sum() > 0
                if step > 3:
                    assert trainer.correction_image.grad.abs().sum() > 0
            optimizer.step()
            trainer.complete_camera_update(step)
            trainer.complete_camera_update(step)
            assert int(trainer.rgb_joint_camera_updates) == step + 1
    trainer.eval()
    with torch.no_grad():
        trainer(*trainer.inputs())
    trainer.complete_camera_update(6)
    assert int(trainer.rgb_joint_camera_updates) == 6


def test_external_hdr_bypasses_cnn_with_current_pbr():
    trainer = JointTrainer().eval()
    trainer.rgb_joint_camera_updates.fill_(4)
    trainer.env.external = torch.ones(1)
    with patch.object(trainer.rgb_residual_head, "forward", side_effect=AssertionError("CNN called")):
        outputs = trainer(*trainer.inputs())
    assert not outputs["rgb_residual_active"]
    assert torch.equal(outputs["rgb"], outputs["rgb_pure_pbr"])
    assert torch.count_nonzero(outputs["rgb_residual"]) == 0


def test_checkpoint_roundtrip_counter_head_and_config_guard():
    trainer = JointTrainer()
    trainer.rgb_joint_camera_updates.fill_(3)
    with torch.no_grad():
        trainer.rgb_residual_head.net[-1].bias.fill_(.3)
    state = deepcopy(trainer.state_dict())
    assert state["rgb_joint"]["camera_updates"] == 3
    assert "rgb_residual_head.net.2.weight" in state and "rgb_joint_camera_updates" in state
    resumed = JointTrainer()
    resumed.load_state_dict(state, strict=True)
    assert resumed.rgb_residual_strength() == .5
    torch.testing.assert_close(resumed.rgb_residual_head.net[-1].bias, trainer.rgb_residual_head.net[-1].bias)
    torch.testing.assert_close(resumed(*resumed.inputs())["rgb"], trainer(*trainer.inputs())["rgb"])
    assert "models" in state and "step" in state  # native loader must not mutate caller envelope
    invalid = deepcopy(state)
    invalid["rgb_joint_camera_updates"] = torch.tensor(2)
    with unittest.TestCase().assertRaisesRegex(ValueError, "counter"):
        resumed.load_state_dict(invalid)
    with unittest.TestCase().assertRaisesRegex(ValueError, "unchanged"):
        JointTrainer(enabled=False).load_state_dict(state)
    with unittest.TestCase().assertRaisesRegex(ValueError, "pretrained"):
        JointTrainer().load_state_dict(JointTrainer(enabled=False).state_dict())


def test_native_raster_keeps_features_in_main_projection_during_warmup():
    trainer = JointTrainer().train()
    trainer.render_cfg = OmegaConf.create({"batch_size": 1, "packed": False, "absgrad": False,
                                           "sparse_grad": False, "antialiased": False})
    colors = torch.randn(2, 8, requires_grad=True)
    features = torch.zeros(2, 8, requires_grad=True)
    gs = SimpleNamespace(rgbs=colors, extras={"correction_features": features},
                         means=torch.zeros(2, 3), quats=torch.ones(2, 4),
                         scales=torch.ones(2, 3), opacities=torch.ones(2, 1))
    cam = SimpleNamespace(camtoworlds=torch.eye(4), Ks=torch.eye(3), W=4, H=4)
    calls = []
    def raster(**kwargs):
        calls.append(kwargs["colors"].shape[-1])
        image = kwargs["colors"].sum(0).expand(1, 4, 4, -1)
        image = torch.cat((image, torch.ones(1, 4, 4, 1)), -1)
        return image, torch.ones(1, 4, 4, 1), {"means2d": torch.zeros(1, 2, 2, requires_grad=True)}
    with patch("models.trainers.scene_graph_pbr.rasterization", raster):
        outputs, _ = MultiTrainer_pbr.render_gaussians(trainer, gs, cam)
    assert calls == [16]
    assert outputs["rgb_gaussians"].shape[-1] == 8
    assert outputs["_rgb_correction_features"].shape[-1] == 8
    outputs["rgb_gaussians"].sum().backward()
    assert features.grad is not None and features.grad.count_nonzero() == 0
    assert colors.grad.abs().sum() > 0


def test_invalid_schedule_and_bounds_rejected():
    for option, value in (("warmup_camera_updates", -1), ("ramp_camera_updates", 0),
                         ("gain_lambda", float("nan")), ("max_additive", -1), ("enabled", "yes")):
        with unittest.TestCase().assertRaises(ValueError):
            validate_rgb_joint_config({option: value})


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
                              for key, value in globals().items() if key.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
