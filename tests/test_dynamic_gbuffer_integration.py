"""D1 production forward/loss wiring; synthetic images never start training."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from models.trainers.scene_graph_pbr import MultiTrainer_pbr
from tests.test_rgb_joint_trainer import JointTrainer
from utils.config import load_training_config
from utils.rgb_joint_training import validate_config


class ActorTrainer(JointTrainer):
    def __init__(self, mode="purity"):
        super().__init__()
        self.gaussian_classes["RigidNodes"] = 1
        node = torch.nn.Identity()
        node.point_ids = torch.tensor([[0], [1]])
        self.models["RigidNodes"] = node
        self.pts_labels = torch.tensor([0, 1, 1])
        self.mass_source = torch.nn.Parameter(torch.tensor([.1, .6, .3]))
        self.configure_dynamic_gbuffer({"enabled": True, "mode": mode, "start_step": 2,
            "ramp_steps": 2, "chunk_size": 1, "min_pixels": 4, "visibility_downscale": 2})
        self.rgb_joint_camera_updates.fill_(4)
        self.step = 4
        self.feature_calls = []

    def collect_gaussians(self, *args):
        return SimpleNamespace(rgbs=torch.zeros(3, 8))

    def render_gaussians(self, *args, **kwargs):
        outputs, _ = super().render_gaussians(*args, **kwargs)

        def render_fn(opacity_mask=None, *, features=None, feature_size=None):
            assert opacity_mask is None, "D1 must preserve all occluders"
            assert not torch.is_grad_enabled(), "Visibility cannot contribute gradients"
            self.feature_calls.append((features.detach().clone(), feature_size))
            h, w = feature_size
            weights = self.mass_source.expand(h, w, 3).clone()
            # Different actor dominates each half; their image areas differ.
            weights[:, :w // 4, 1] = self.mass_source[2]
            weights[:, :w // 4, 2] = self.mass_source[1]
            return weights @ features

        return outputs, render_fn

    def inputs(self, frame=0):
        images, cameras = super().inputs(frame)
        images["gbuf_basecolor"] = torch.full((32, 32, 3), .7, requires_grad=True)
        return images, cameras


class DynamicGBufferIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_existing_losses_exact_and_auxiliary_gradients_detach_visibility_teacher(self):
        trainer = ActorTrainer().train()
        image, camera = trainer.inputs()
        rng = torch.get_rng_state().clone()
        output = trainer(image, camera)
        with patch("models.trainers.scene_graph_weather.MultiTrainer_weather.compute_losses",
                   side_effect=lambda *args, **kwargs: {}):
            actual = trainer.compute_losses(output, image, camera)
            trainer.dynamic_gbuffer_config["enabled"] = False
            baseline = trainer.compute_losses(output, image, camera)
        for name, value in baseline.items():
            torch.testing.assert_close(value, actual[name], atol=0, rtol=0)
        auxiliary = {name: value for name, value in actual.items() if name not in baseline}
        self.assertEqual(set(auxiliary), {f"dynamic_gbuffer_{name}_loss" for name in
                                          ("albedo", "metallic", "roughness", "normal")})
        sum(auxiliary.values()).backward()
        self.assertGreater(float(trainer.scene_value.grad.abs()), 0.)
        self.assertIsNone(trainer.mass_source.grad)
        self.assertIsNone(image["gbuf_basecolor"].grad)
        self.assertEqual(len(trainer.feature_calls), 2)
        for features, size in trainer.feature_calls:
            self.assertEqual(size, (16, 16))
            torch.testing.assert_close(features[:, -1], torch.ones(3))
            self.assertEqual(float(features[0, 0]), 0.)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)

    def test_balanced_and_purity_use_same_support_but_different_loss(self):
        trainer = ActorTrainer().train()
        image, camera = trainer.inputs()
        output = trainer(image, camera)
        purity = trainer.dynamic_gbuffer_losses(output, image, camera)["dynamic_gbuffer_albedo_loss"]
        trainer.dynamic_gbuffer_config["mode"] = "balanced"
        balanced = trainer.dynamic_gbuffer_losses(output, image, camera)["dynamic_gbuffer_albedo_loss"]
        self.assertGreater(float(balanced.detach()), float(purity.detach()))
        self.assertGreater(float(purity.detach()), 0.)
        self.assertEqual(int(trainer._last_dynamic_gbuffer_stats["albedo/eligible_instances"]), 2)

    def test_eval_disabled_warmup_and_no_grad_have_no_auxiliary_render(self):
        trainer = ActorTrainer().eval()
        image, camera = trainer.inputs()
        for name in tuple(image):
            if name.startswith("gbuf_"):
                del image[name]
        with torch.no_grad():
            active_config = trainer(image, camera)
            trainer.dynamic_gbuffer_config["enabled"] = False
            inactive_config = trainer(image, camera)
        self.assertEqual(set(active_config), set(inactive_config))
        for key in active_config:
            torch.testing.assert_close(active_config[key], inactive_config[key], atol=0, rtol=0)
        trainer.train()
        trainer.dynamic_gbuffer_config["enabled"] = True
        trainer.rgb_joint_camera_updates.fill_(2)
        self.assertNotIn("_dynamic_gbuffer_ownership", trainer(image, camera))
        trainer.rgb_joint_camera_updates.fill_(4)
        with torch.no_grad():
            self.assertNotIn("_dynamic_gbuffer_ownership", trainer(image, camera))
        self.assertEqual(trainer.feature_calls, [])

    def test_current_crop_masks_and_zero_actor_frames(self):
        trainer = ActorTrainer().train()
        trainer.crop_frac = .25
        image, camera = trainer.inputs()
        for name in tuple(image):
            if name.startswith("gbuf_"):
                image[name] = image[name][8:24]
        image["egocar_masks"][8:16] = 1
        output = trainer(image, camera)
        self.assertEqual(output["_dynamic_gbuffer_ownership"]["owner"].shape, (16, 32))
        losses = trainer.dynamic_gbuffer_losses(output, image, camera)
        self.assertEqual(int(trainer._last_dynamic_gbuffer_stats["albedo/eligible_pixels"]), 8 * 32)
        sum(losses.values()).backward()
        # No actor IDs: no auxiliary raster, graph-connected zero losses.
        trainer._layered_group_ids = lambda _: torch.zeros(3, dtype=torch.long)
        trainer.feature_calls.clear()
        output = trainer(image, camera)
        losses = trainer.dynamic_gbuffer_losses(output, image, camera)
        self.assertEqual(trainer.feature_calls, [])
        self.assertEqual(float(sum(losses.values()).detach()), 0.)

    def test_missing_or_stale_visibility_fails_instead_of_silent_zero(self):
        trainer = ActorTrainer().train()
        image, camera = trainer.inputs()
        with self.assertRaisesRegex(RuntimeError, "visibility"):
            trainer.dynamic_gbuffer_losses({}, image, camera)
        output = trainer(image, camera)
        output["_dynamic_gbuffer_weight"] = .01
        with self.assertRaisesRegex(RuntimeError, "clock"):
            trainer.dynamic_gbuffer_losses(output, image, camera)

    def test_config_accepts_d1_and_rejects_typos_or_mixed_experiments(self):
        from pathlib import Path
        config = load_training_config(Path(__file__).resolve().parents[1] /
                                      "configs/base_v5/scene788_100f.yaml")
        config.trainer.dynamic_gbuffer = {"enabled": True, "mode": "purity"}
        validate_config(config)
        config.trainer.dynamic_gbuffer.typo = 1
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_config(config)
        del config.trainer.dynamic_gbuffer.typo
        config.trainer.teacher_confidence = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "cannot also enable"):
            validate_config(config)

    def test_auxiliary_raster_keeps_opacities_primary_state_and_scales_intrinsics(self):
        trainer = ActorTrainer().train()
        trainer.render_cfg = OmegaConf.create({"batch_size": 1, "packed": False,
            "absgrad": True, "sparse_grad": False, "antialiased": False})
        gs = SimpleNamespace(rgbs=torch.randn(3, 8), extras={"correction_features": torch.zeros(3, 8)},
            means=torch.nn.Parameter(torch.randn(3, 3)), quats=torch.zeros(3, 4),
            scales=torch.ones(3, 3), opacities=torch.tensor([.5, .75, .4]))
        camera = SimpleNamespace(W=20, H=10, Ks=torch.tensor([[8., 0., 10.], [0., 8., 5.], [0., 0., 1.]]),
                                 camtoworlds=torch.eye(4))
        calls = []

        def raster(**kwargs):
            calls.append(kwargs)
            alpha = kwargs["opacities"]
            weights = alpha * torch.cat((alpha.new_ones(1), (1 - alpha).cumprod(0)[:-1]))
            pixel = weights @ kwargs["colors"]
            if kwargs["render_mode"] == "RGB+ED":
                pixel = torch.cat((pixel, pixel.new_tensor([3.])))
            render = pixel.expand(1, kwargs["height"], kwargs["width"], -1)
            coverage = weights.sum().expand(1, kwargs["height"], kwargs["width"], 1)
            return render, coverage, {"means2d": kwargs["means"][:, :2][None]}

        with patch("models.trainers.scene_graph_pbr.rasterization", side_effect=raster):
            outputs, render = MultiTrainer_pbr.render_gaussians(trainer, gs, camera, render_mode="RGB+ED")
            primary_info = trainer.info
            intrinsics = camera.Ks.clone()
            with torch.no_grad():
                features = torch.tensor([[0., 1.], [1., 1.], [0., 1.]])
                accumulated = render(features=features, feature_size=(5, 10))
        self.assertIs(trainer.info, primary_info)
        self.assertEqual(outputs["_rgb_correction_features"].shape, (10, 20, 8))
        torch.testing.assert_close(camera.Ks, intrinsics, atol=0, rtol=0)
        torch.testing.assert_close(calls[1]["opacities"], gs.opacities, atol=0, rtol=0)
        torch.testing.assert_close(calls[1]["Ks"][0, :2], intrinsics[:2] / 2)
        self.assertEqual(calls[1]["render_mode"], "RGB")
        torch.testing.assert_close(accumulated[0, 0], torch.tensor([.375, .925]))


if __name__ == "__main__":
    unittest.main()
