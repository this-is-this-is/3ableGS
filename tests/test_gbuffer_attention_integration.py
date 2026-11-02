"""CPU integration through the real PBR forward, state and loss hooks.

Only rasterization/scene construction are synthetic. These checks exercise
split-safe history, affine, current gradients and checkpoint serialization.
"""
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

from models.gbuffer_attention_runtime import history_image_ids, preserve_render_state
from tests.test_temporal_material_integration import SyntheticTrainer


class Sequence:
    def __init__(self, trainer):
        self.trainer = trainer
        self.reads = []

    def __len__(self):
        return 8

    def get_image(self, index, camera_downscale, weathers):
        self.reads.append(index)
        images, cameras = self.trainer.inputs(index)
        # Reference payloads deliberately disagree with generated materials.
        images.update(pixels=torch.full((12, 12, 3), 99.),
                      gbuf_basecolor=torch.full((12, 12, 3), -99.))
        return {"raw": images}, {"raw": cameras}


class AttentionTrainer(SyntheticTrainer):
    def __init__(self):
        super().__init__(size=12)
        self.scene_value = nn.Parameter(torch.tensor(.3))
        self.normalized_timestamps = torch.linspace(0, 1, 8)
        self.test_set_indices = [1, 4]
        self.res_schedule = OmegaConf.create({"downscale_times": 0, "double_steps": 250})
        self.dataset = SimpleNamespace(
            train_indices=[0, 2, 3, 5, 6, 7], pixel_source=SimpleNamespace(num_cams=1))
        self.dataset.full_image_set = Sequence(self)
        self.current_render_ids = []
        self.context_keys = []

    def inputs(self, frame):
        images, cameras = super().inputs(frame)
        images["normed_time"].fill_(frame / 7)
        return images, cameras

    def render_gaussians(self, gs, cam, update_info=True, **kwargs):
        if getattr(self, "_gbuffer_attention_context", False):
            # Simulate a stochastic scene branch; its RNG must be isolated.
            torch.rand(1)
            random.random()
            np.random.rand()
        self.albedo = self.scene_value.expand(self.size, self.size, 3) + .04 * int(self.cur_frame)
        self.models["Background"]._raw_normal_residual_penalty = self.scene_value * (1 + self.cur_frame)
        self.models["Background"]._gs_cache = {"frame": int(self.cur_frame)}
        if update_info:
            self.info = {"current_frame": int(self.cur_frame)}
            self.current_render_ids.append(int(self.cur_frame))
        return super().render_gaussians(gs, cam, **kwargs)

    def forward(self, image_infos, camera_infos, *args, **kwargs):
        if getattr(self, "_gbuffer_attention_context", False):
            self.context_keys.append(set(image_infos))
        return super().forward(image_infos, camera_infos, *args, **kwargs)

    def configure(self, variant="A2", **kwargs):
        self.configure_gbuffer_attention({"variant": variant, "width": 8, "chunk_size": 31,
                                         "start_step": 0, "ramp_steps": 1,
                                         "temporal_start": 0, "temporal_ramp": 1, **kwargs})
        self.attention_step = 1


class AttentionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_history_indices_use_timestamps_camera_and_saved_train_split(self):
        self.assertEqual(history_image_ids(16, 2, 40, set(), False), [14, 12, 8])
        self.assertEqual(history_image_ids(16, 2, 40, {16, 12, 8}, True), [12, 8])
        with self.assertRaisesRegex(ValueError, "train split"):
            history_image_ids(16, 2, 40, {12}, True)
        self.assertEqual(history_image_ids(1, 2, 40, set(), False), [])

    def test_configuration_preserves_scene_rng_and_warmup_is_exact_bypass(self):
        trainer = AttentionTrainer()
        trainer.train()
        state = torch.get_rng_state().clone()
        base = trainer(*trainer.inputs(3))
        trainer.configure()
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
        trainer.attention_step = 0
        result = trainer(*trainer.inputs(3))
        torch.testing.assert_close(result["rgb"], base["rgb"], rtol=0, atol=0)
        self.assertEqual(trainer.dataset.full_image_set.reads, [])
        self.assertNotIn("attention_albedo", result)

    def test_original_material_teacher_payload_current_affine_and_shading(self):
        trainer = AttentionTrainer()
        trainer.configure("H0")
        trainer.eval()
        with torch.no_grad():
            result = trainer(*trainer.inputs(4))
        torch.testing.assert_close(result["albedo"], torch.full((12, 12, 3), .46))
        self.assertLess(result["attention_albedo"][5, 5, 0], result["albedo"][5, 5, 0])
        expected = result["attention_albedo"] * .5 + .5 * .2 + trainer.normal.abs() * .05 + .1
        torch.testing.assert_close(result["rgb"], expected)
        self.assertEqual(result["attention_context_ids"], [3, 2, 0])
        self.assertTrue(trainer.models["Affine"].in_test_set)
        self.assertEqual(trainer.models["Affine"].calls[-1], (True, True))

    def test_joint_gradients_and_train_only_context_preserve_current_raster_state(self):
        trainer = AttentionTrainer()
        trainer.configure()
        trainer.train()
        result = trainer(*trainer.inputs(3))
        losses = trainer.attention_losses(result)
        (result["rgb"].square().mean() + sum(losses.values())).backward()
        self.assertEqual(result["attention_context_ids"], [2])
        self.assertTrue(set(trainer.dataset.full_image_set.reads) <= set(trainer.dataset.train_indices))
        self.assertEqual(trainer.current_render_ids, [3])
        self.assertEqual(trainer.info["current_frame"], 3)
        self.assertEqual(trainer.models["Background"]._gs_cache["frame"], 3)
        self.assertTrue(trainer.models["Background"]._raw_normal_residual_penalty.requires_grad)
        self.assertGreater(trainer.scene_value.grad.abs(), 0)
        query = trainer.gbuffer_attention_head.attention[-1].query.weight
        self.assertIsNotNone(query.grad)
        self.assertGreater(query.grad.abs().sum(), 0)
        self.assertIn("attention_temporal_loss", losses)
        for keys in trainer.context_keys:
            self.assertNotIn("pixels", keys)
            self.assertNotIn("gbuf_basecolor", keys)
            self.assertNotIn("dynamic_masks", keys)

    def test_reference_inputs_never_change_inference_and_dynamic_is_unchanged(self):
        trainer = AttentionTrainer()
        trainer.configure("H0")
        trainer.dynamic_opacity[5, 5] = 1
        images, cameras = trainer.inputs(3)
        with torch.no_grad():
            first = trainer(images, cameras)
            images.update(pixels=torch.randn(12, 12, 3) * 100,
                          gbuf_basecolor=torch.ones(12, 12, 3) * 100)
            images["dynamic_masks"].zero_()
            second = trainer(images, cameras)
        torch.testing.assert_close(first["rgb"], second["rgb"], atol=0, rtol=0)
        torch.testing.assert_close(second["albedo"][5, 5], second["attention_albedo"][5, 5], atol=0, rtol=0)
        self.assertEqual(second["attention_history_weight"][5, 5], 0)

    def test_real_loss_hook_supervises_unfused_teacher_and_corrected_rgb(self):
        from models.trainers.scene_graph_weather import MultiTrainer_weather
        trainer = AttentionTrainer()
        trainer.configure()
        trainer.train()
        trainer.crop_frac = 0.
        trainer.losses_dict = OmegaConf.create({"albedo": {"w": .5},
                                               "metallic": {"w": .2}, "roughness": {"w": .2}})
        trainer.env.compute_reg_loss = lambda _config: {}
        images, cameras = trainer.inputs(3)
        output = trainer(images, cameras)
        images.update(pixels=torch.zeros(12, 12, 3), sky_masks=torch.zeros(12, 12),
                      gbuf_basecolor=output["albedo"].detach().clone(),
                      gbuf_metallic=output["metallic"].detach().clone(),
                      gbuf_roughness=output["roughness"].detach().clone(),
                      gbuf_normal=output["normal"].detach().clone())

        def rgb_loss(_trainer, values, references, _cameras, _weather):
            return {"rgb_loss": .7 * (values["rgb"] - references["pixels"]).abs().mean()}

        with patch.object(MultiTrainer_weather, "compute_losses", rgb_loss):
            losses = trainer.compute_losses(output, images, cameras)
        self.assertGreater((output["attention_albedo"] - output["albedo"]).abs().max(), 0)
        self.assertEqual(float(losses["albedo_loss"]), 0.)
        torch.testing.assert_close(losses["rgb_loss"], .7 * output["rgb"].abs().mean())
        self.assertIn("attention_anchor_loss", losses)
        self.assertIn("attention_temporal_loss", losses)

    def test_random_access_and_cpu_stream_cache_are_order_independent(self):
        trainer = AttentionTrainer()
        trainer.configure()
        with torch.no_grad():
            expected = {frame: trainer(*trainer.inputs(frame))["rgb"] for frame in (1, 4, 7)}
            trainer.set_gbuffer_attention_cache(True)
            trainer.context_keys.clear()
            for frame in range(8):
                result = trainer(*trainer.inputs(frame))
                if frame in expected:
                    torch.testing.assert_close(result["rgb"], expected[frame], atol=0, rtol=0)
            self.assertEqual(trainer.context_keys, [], "Streaming context must reuse each base render")
            for frame in (7, 1, 4):
                torch.testing.assert_close(trainer(*trainer.inputs(frame))["rgb"], expected[frame], atol=0, rtol=0)
        self.assertLessEqual(len(trainer._attention_history.cache), 4)

    def test_checkpoint_requires_saved_config_and_preserves_head_object(self):
        trainer = AttentionTrainer()
        trainer.configure()
        trainer.set_gbuffer_attention_cache(True)
        with torch.no_grad():
            before = trainer(*trainer.inputs(4))["rgb"]
        saved = copy.deepcopy(trainer.state_dict())
        self.assertIn("_gbuffer_attention", saved)
        clone = AttentionTrainer()
        clone.configure()
        head = clone.gbuffer_attention_head
        clone.load_state_dict(saved, strict=True)
        self.assertIs(clone.gbuffer_attention_head, head)
        with torch.no_grad():
            torch.testing.assert_close(clone(*clone.inputs(4))["rgb"], before, atol=0, rtol=0)
        trainer.load_state_dict(saved)
        self.assertFalse(trainer._attention_history.cache)
        fresh = AttentionTrainer()
        with self.assertRaisesRegex(ValueError, "saved trainer"):
            fresh.load_state_dict(saved)
        fresh.configure("A0")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            fresh.load_state_dict(saved)
        self.assertIn("models", saved)
        self.assertIn("_gbuffer_attention", saved)

    def test_context_exception_restores_render_and_random_state(self):
        trainer = AttentionTrainer()
        marker = object()
        trainer.info = marker
        pixel = trainer.dataset.pixel_source
        pixel._downscale_factor, pixel._old_downscale_factor = 1., []
        camera = SimpleNamespace(downscale_factor=1.)
        pixel.camera_data = {0: camera}
        state = torch.get_rng_state().clone()
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with preserve_render_state(trainer):
                trainer.info = None
                trainer._gbuffer_attention_context = True
                trainer.models["Background"].filter_mask = torch.ones(3)
                pixel._old_downscale_factor.append(1.)
                pixel._downscale_factor = camera.downscale_factor = .25
                torch.rand(10)
                random.random()
                np.random.rand()
                raise RuntimeError("deliberate")
        self.assertIs(trainer.info, marker)
        self.assertFalse(hasattr(trainer, "_gbuffer_attention_context"))
        self.assertFalse(hasattr(trainer.models["Background"], "filter_mask"))
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
        self.assertEqual(random.getstate(), random_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        self.assertEqual(pixel._old_downscale_factor, [])
        self.assertEqual(pixel._downscale_factor, 1.)
        self.assertEqual(camera.downscale_factor, 1.)

    def test_joint_optimizer_registers_head_and_updates_both_scene_and_adapter(self):
        from models.trainers.scene_graph_weather import MultiTrainer_weather
        trainer = AttentionTrainer()
        trainer.configure()
        trainer.train()

        def initialize_scene_optimizer(value):
            value.param_groups = {"Background#value": [value.scene_value]}
            value.lr_schedulers = {}
            value.optimizer = torch.optim.Adam([
                {"params": [value.scene_value], "name": "Background#value", "lr": .001}])

        with patch.object(MultiTrainer_weather, "initialize_optimizer", initialize_scene_optimizer):
            trainer.initialize_optimizer()
        self.assertEqual([group["name"] for group in trainer.optimizer.param_groups],
                         ["Background#value", "GBufferAttention#all"])
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], .001)
        scene_before = trainer.scene_value.detach().clone()
        head_before = trainer.gbuffer_attention_head.residual_head.weight.detach().clone()
        output = trainer(*trainer.inputs(3))
        output["rgb"].square().mean().backward()
        trainer.optimizer_step()
        self.assertFalse(torch.equal(scene_before, trainer.scene_value))
        self.assertFalse(torch.equal(head_before, trainer.gbuffer_attention_head.residual_head.weight))


if __name__ == "__main__":
    unittest.main()
