"""CPU integration of real PBR hooks with a size-aware synthetic rasterizer."""
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from tests.test_temporal_material_integration import SyntheticTrainer


class Sequence:
    def __init__(self, trainer):
        self.trainer, self.reads = trainer, []

    def __len__(self):
        return 8

    def get_image(self, index, camera_downscale, weathers):
        self.reads.append((index, camera_downscale))
        image, camera = self.trainer.inputs(index, int(self.trainer.size / camera_downscale))
        image.update(pixels=torch.tensor(float("nan")), gbuf_basecolor=torch.tensor(99.))
        return {"raw": image}, {"raw": camera}


class RDGTrainer(SyntheticTrainer):
    def __init__(self):
        super().__init__(size=48)
        self.scene_value = nn.Parameter(torch.tensor(.3))
        self.normalized_timestamps = torch.linspace(0, 1, 8)
        self.test_set_indices = [1, 4]
        self.res_schedule = OmegaConf.create({"downscale_times": 0, "double_steps": 250})
        self.dataset = SimpleNamespace(train_indices=[0, 2, 3, 5, 6, 7],
                                       pixel_source=SimpleNamespace(num_cams=1))
        self.dataset.full_image_set = Sequence(self)
        self.current_render_ids, self.context_keys = [], []
        self.configure_gbuffer_attention(None)

    def inputs(self, frame, size=None):
        size = self.size if size is None else size
        image = {"normed_time": torch.full((size, size), frame / 7),
                 "img_idx": torch.full((size, size), frame, dtype=torch.long),
                 "frame_idx": torch.full((size, size), frame, dtype=torch.long),
                 "viewdirs": torch.tensor([0., 0., 1.]).expand(size, size, 3).clone(),
                 "dynamic_masks": torch.ones(size, size)}
        camera = {"camera_to_world": torch.eye(4),
                  "intrinsics": torch.tensor([[float(size), 0., size / 2],
                                               [0., float(size), size / 2], [0., 0., 1.]]),
                  "cam_id": torch.zeros(size, size, dtype=torch.long),
                  "height": size, "width": size}
        return image, camera

    def process_camera(self, cameras, image_ids, novel_view=False):
        return SimpleNamespace(camtoworlds=cameras["camera_to_world"], Ks=cameras["intrinsics"],
                               H=cameras["height"], W=cameras["width"])

    def render_gaussians(self, gs, cam, update_info=True, **kwargs):
        h, w = cam.H, cam.W
        if getattr(self, "_rdg_context", False):
            torch.rand(1)
            random.random()
            np.random.rand()
        node = self.models["Background"]
        node._raw_normal_residual_penalty = self.scene_value * (1 + self.cur_frame)
        node._gs_cache = {"frame": int(self.cur_frame)}
        if update_info:
            self.info = {"current_frame": int(self.cur_frame)}
            self.current_render_ids.append(int(self.cur_frame))

        def resize(value):
            return F.interpolate(value.permute(2, 0, 1)[None], (h, w), mode="nearest")[0].permute(1, 2, 0)

        opacity, depth = resize(self.opacity), resize(self.depth)
        albedo = self.scene_value.expand(h, w, 3) + .04 * int(self.cur_frame)
        feat = torch.cat((albedo, torch.full((h, w, 1), .1),
                          resize(self.roughness), resize(self.normal)), -1) * opacity

        def render_fn(dynamic):
            self.auxiliary_renders += 1
            return feat, depth, resize(self.dynamic_opacity)

        return {"rgb_gaussians": feat, "depth": depth, "opacity": opacity}, render_fn

    def forward(self, images, cameras, *args, **kwargs):
        if getattr(self, "_rdg_context", False):
            self.context_keys.append(set(images))
        return super().forward(images, cameras, *args, **kwargs)

    def configure(self, **kwargs):
        self._configure_rdg_fusion({"enabled": True, "width": 8,
                                    "start_step": 0, "ramp_steps": 1, "interval": 1, **kwargs})
        self.step = 1


class RDGIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_disabled_and_warmup_are_exact_bypass_with_no_additional_renders(self):
        trainer = RDGTrainer()
        trainer.train()
        base = trainer(*trainer.inputs(3))
        state = torch.get_rng_state().clone()
        trainer.configure(start_step=5000)
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
        result = trainer(*trainer.inputs(3))
        torch.testing.assert_close(result["rgb"], base["rgb"], atol=0, rtol=0)
        self.assertEqual(trainer.dataset.full_image_set.reads, [])
        self.assertEqual(trainer.auxiliary_renders, 0)
        self.assertNotIn("rdg_albedo", result)

    def test_quarter_history_current_shading_and_original_teacher_payload(self):
        trainer = RDGTrainer()
        trainer.configure()
        with torch.no_grad():
            output = trainer(*trainer.inputs(4))
        self.assertEqual(trainer.dataset.full_image_set.reads, [(3, 4)])
        self.assertEqual(output["rdg_context_ids"], [3])
        torch.testing.assert_close(output["albedo"], torch.full((48, 48, 3), .46))
        self.assertLess(output["rdg_albedo"][24, 24, 0], output["albedo"][24, 24, 0])
        expected = output["rdg_albedo"] * .5 + .5 * .2 + trainer.normal.abs() * .05 + .1
        torch.testing.assert_close(output["rgb"], expected)
        self.assertEqual(len(trainer.env.shade_inputs), 1, "No guide-RGB or history shading")
        self.assertEqual(trainer.models["Affine"].calls, [(True, True)])

    def test_training_history_split_rng_current_state_and_joint_gradients(self):
        trainer = RDGTrainer()
        trainer.configure()
        trainer.train()
        torch_state, python_state, numpy_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
        output = trainer(*trainer.inputs(5))
        losses = trainer.rdg_losses(output)
        (output["rgb"].square().mean() + sum(losses.values())).backward()
        self.assertEqual(output["rdg_context_ids"], [3], "Held-out t-1 must be skipped while training")
        self.assertEqual(trainer.dataset.full_image_set.reads, [(3, 4)])
        self.assertEqual(trainer.current_render_ids, [5])
        self.assertEqual(trainer.info["current_frame"], 5)
        self.assertEqual(trainer.models["Background"]._gs_cache["frame"], 5)
        self.assertTrue(trainer.models["Background"]._raw_normal_residual_penalty.requires_grad)
        self.assertGreater(trainer.scene_value.grad.abs().sum(), 0)
        self.assertGreater(sum(p.grad.abs().sum() for p in trainer.rdg_fusion_head.parameters()
                               if p.grad is not None), 0)
        self.assertEqual(set(losses), {"rdg_anchor", "rdg_temporal"})
        torch.testing.assert_close(torch.get_rng_state(), torch_state, atol=0, rtol=0)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        self.assertTrue(all(not ({"pixels", "gbuf_basecolor", "dynamic_masks"} & keys)
                            for keys in trainer.context_keys))

    def test_generated_static_guard_and_reference_independence(self):
        trainer = RDGTrainer()
        trainer.configure()
        trainer.dynamic_opacity[24, 24] = 1.
        images, cameras = trainer.inputs(3)
        with torch.no_grad():
            first = trainer(images, cameras)
            images.update(pixels=torch.full((48, 48, 3), 99.),
                          gbuf_basecolor=torch.full((48, 48, 3), -99.))
            images["dynamic_masks"].zero_()
            second = trainer(images, cameras)
        torch.testing.assert_close(first["rgb"], second["rgb"], atol=0, rtol=0)
        torch.testing.assert_close(second["rdg_albedo"][24, 24], second["albedo"][24, 24], atol=0, rtol=0)
        self.assertEqual(second["rdg_history_weight"][24, 24], 0)

    def test_real_loss_hook_supervises_unfused_material(self):
        from models.trainers.scene_graph_weather import MultiTrainer_weather
        trainer = RDGTrainer()
        trainer.configure()
        trainer.train()
        trainer.crop_frac = 0.
        trainer.losses_dict = OmegaConf.create({"albedo": {"w": .5},
                                               "metallic": {"w": .2}, "roughness": {"w": .2}})
        trainer.env.compute_reg_loss = lambda config: {}
        images, cameras = trainer.inputs(3)
        output = trainer(images, cameras)
        images.update(pixels=torch.zeros(48, 48, 3), sky_masks=torch.zeros(48, 48),
                      gbuf_basecolor=output["albedo"].detach().clone(),
                      gbuf_metallic=output["metallic"].detach().clone(),
                      gbuf_roughness=output["roughness"].detach().clone(),
                      gbuf_normal=output["normal"].detach().clone())
        with patch.object(MultiTrainer_weather, "compute_losses", lambda self, out, *args:
                          {"rgb_loss": .7 * out["rgb"].abs().mean()}):
            losses = trainer.compute_losses(output, images, cameras)
        self.assertGreater((output["rdg_albedo"] - output["albedo"]).abs().max(), 0)
        self.assertEqual(float(losses["albedo_loss"].detach()), 0.)
        self.assertIn("rdg_anchor", losses)
        self.assertIn("rdg_temporal", losses)
        torch.testing.assert_close(losses["rgb_loss"], .7 * output["rgb"].abs().mean())

    def test_eval_uses_actual_previous_frame_independent_of_access_order(self):
        trainer = RDGTrainer()
        trainer.configure(interval=8)
        trainer.step = 7  # Evaluation does not skip off-interval steps.
        with torch.no_grad():
            before = trainer(*trainer.inputs(5))
            trainer(*trainer.inputs(1))
            after = trainer(*trainer.inputs(5))
        self.assertEqual(after["rdg_context_ids"], [4])
        torch.testing.assert_close(before["rgb"], after["rgb"], atol=0, rtol=0)

    def test_checkpoint_roundtrip_preserves_head_and_saved_strength(self):
        trainer = RDGTrainer()
        trainer.configure()
        with torch.no_grad():
            expected = trainer(*trainer.inputs(4))["rgb"]
        state = copy.deepcopy(trainer.state_dict())
        self.assertIn("_rdg_fusion", state)
        clone = RDGTrainer()
        clone.configure()
        head = clone.rdg_fusion_head
        clone.load_state_dict(state, strict=True)
        self.assertIs(clone.rdg_fusion_head, head)
        with torch.no_grad():
            torch.testing.assert_close(clone(*clone.inputs(4))["rgb"], expected, atol=0, rtol=0)
        self.assertIn("_rdg_fusion", state, "Checkpoint caller must not be mutated")

    def test_later_temporal_configuration_cannot_silently_compose_renderers(self):
        trainer = RDGTrainer()
        trainer.configure()
        trainer.configure_temporal_material({"mode": "normal", "erode_radius": 0,
                                             "history_weight": .3})
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "without other experiments"):
            trainer(*trainer.inputs(4))
        self.assertEqual(trainer.dataset.full_image_set.reads, [])


if __name__ == "__main__":
    unittest.main()
