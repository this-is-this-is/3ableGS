"""Train-only sparse consistency wiring; synthetic renders, no optimizer runs."""
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from models.gbuffer_attention_runtime import RENDER_IMAGE_KEYS
from models.temporal_training import (
    TemporalMaterialTraining, temporal_training_weight, validate_temporal_training_config,
)
from models.trainers.scene_graph_weather import MultiTrainer_weather
from tests.test_temporal_material_integration import SyntheticTrainer


CONFIG = {"w": .05, "start_step": 3000, "ramp_steps": 1000,
          "interval": 8, "max_gap": 2, "num_samples": 4096, "seed": 20261101}


class SparseTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        trainer = SyntheticTrainer(size=12)
        trainer.train()
        trainer.step = 4000
        trainer.crop_frac = 0.
        trainer.losses_dict = OmegaConf.create({
            "albedo": {"w": .5}, "roughness": {"w": .2}, "metallic": {"w": .2},
        })
        trainer.env.compute_reg_loss = lambda _: {}
        trainer.albedo = torch.nn.Parameter(torch.full((12, 12, 3), .4))
        trainer.roughness = torch.nn.Parameter(torch.full((12, 12, 1), .5))
        current, cameras = trainer.inputs(2)
        previous, previous_camera = trainer.inputs(0)
        for image in (current, previous):
            image.update(
                pixels=torch.zeros(12, 12, 3), sky_masks=torch.zeros(12, 12),
                dynamic_masks=torch.zeros(12, 12), egocar_masks=torch.zeros(12, 12),
                gbuf_basecolor=torch.full((12, 12, 3), .4),
                gbuf_roughness=torch.full((12, 12, 1), .5),
                gbuf_metallic=torch.full((12, 12, 1), .1),
                gbuf_normal=trainer.normal.clone(),
            )
        loads = []

        def get_image(position, downscale, weathers):
            loads.append((position, downscale, weathers))
            self.assertEqual(position, 0)  # Frame 1 is held out: pair 2 -> 0.
            return {"raw": previous}, {"raw": previous_camera}

        pixel_source = SimpleNamespace(
            num_cams=1, _downscale_factor=1., _old_downscale_factor=[.5],
            camera_data={0: SimpleNamespace(downscale_factor=1.)},
        )
        dataset = SimpleNamespace(train_indices=[0, 2, 3], pixel_source=pixel_source,
                                  train_image_set=SimpleNamespace(get_image=get_image))
        trainer.dataset = dataset
        hook = TemporalMaterialTraining(dataset, CONFIG)
        return trainer, hook, current, cameras, previous, loads

    def test_schedule_is_sparse_ramped_and_not_interval_rescaled(self):
        cfg = validate_temporal_training_config(CONFIG)
        for step in (0, 2999, 3000, 3001, 3007, 3999, 4001):
            self.assertEqual(temporal_training_weight(step, cfg), 0.)
        self.assertAlmostEqual(temporal_training_weight(3008, cfg), .0004)
        self.assertAlmostEqual(temporal_training_weight(3992, cfg), .0496)
        self.assertEqual(temporal_training_weight(4000, cfg), .05)
        self.assertEqual(temporal_training_weight(30000, cfg), .05)
        for invalid in (-1, True, 3.5):
            with self.assertRaises(ValueError):
                temporal_training_weight(invalid, cfg)
        for invalid in ({"interval": 0}, {"interval": True}, {"ramp_steps": -1},
                        {"num_samples": -.5}, {"num_samples": True}, {"seed": -1}):
            with self.assertRaises(ValueError):
                validate_temporal_training_config(invalid)

    def test_disabled_warmup_skipped_weather_and_eval_do_no_work(self):
        trainer, hook, current, _, _, loads = self.fixture()
        for step, weather in ((2999, "raw"), (3000, "raw"), (4001, "raw"), (4000, "snow")):
            self.assertIsNone(hook.prepare_previous(trainer, {}, 1, weather, step))
            self.assertEqual(hook.compute(trainer, {}, {}, None, weather, step), {})
        trainer.eval()
        self.assertIsNone(hook.prepare_previous(trainer, {}, 1, "raw", 4000))
        self.assertEqual(hook.compute(trainer, {}, {}, None, "raw", 4000), {})
        trainer.train()
        hook.config["w"] = 0.
        self.assertIsNone(hook.prepare_previous(trainer, current, 1, "raw", 4000))
        self.assertFalse(loads)
        self.assertEqual(hook.counters["neighbor_renders"], 0)

    def test_history_uses_train_pair_and_renderer_receives_no_reference_images(self):
        trainer, hook, current, _, previous, loads = self.fixture()
        previous["sky_masks"][3, 3] = 1
        seen = []

        def capture(_module, args):
            seen.append((set(args[0]), torch.is_grad_enabled()))

        handle = trainer.register_forward_pre_hook(capture)
        try:
            packet = hook.prepare_previous(trainer, current, 1, "raw", 4000)
        finally:
            handle.remove()
        self.assertEqual(loads, [(0, 1, ["raw"])])
        self.assertEqual(len(seen), 1)
        self.assertLessEqual(seen[0][0], RENDER_IMAGE_KEYS)
        self.assertFalse(seen[0][1])
        self.assertEqual(packet["_image_idx"], 0)
        self.assertEqual(packet["_current_image_idx"], 2)
        self.assertFalse(packet["static_mask"][3, 3])
        self.assertTrue(all(not value.requires_grad for value in packet["gbuffer"].values()))
        self.assertEqual(hook.counters, {"scheduled_updates": 1, "neighbor_renders": 1, "missing_pairs": 0})
        heldout = {"img_idx": torch.tensor([1])}
        with self.assertRaises(ValueError):
            hook.prepare_previous(trainer, heldout, 1, "raw", 4000)
        self.assertEqual(len(loads), 1)

    def test_history_restores_rng_render_caches_and_downscale_after_success_or_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                trainer, hook, current, cameras, _, _ = self.fixture()
                trainer(current, cameras)
                cache = object()
                penalty = trainer.albedo.sum()
                node = trainer.models["Background"]
                node._gs_cache, node._raw_normal_residual_penalty = cache, penalty
                original_frame = trainer.cur_frame
                original_camera = trainer._render_c2w
                original_intrinsics = trainer._render_intrinsics
                source = trainer.dataset.pixel_source
                loader = trainer.dataset.train_image_set.get_image
                random.seed(7)
                np.random.seed(7)
                torch.manual_seed(7)
                py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()

                def mutate(*args):
                    random.random()
                    np.random.rand()
                    torch.rand(3)
                    node._gs_cache = "history"
                    node._raw_normal_residual_penalty = torch.tensor(0.)
                    trainer.info = "history"
                    source._downscale_factor = 4.
                    source._old_downscale_factor.append(4.)
                    source.camera_data[0].downscale_factor = 4.
                    if fail:
                        raise RuntimeError("synthetic loader failure")
                    return loader(*args)

                trainer.dataset.train_image_set.get_image = mutate
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "synthetic loader failure"):
                        hook.prepare_previous(trainer, current, 1, "raw", 4000)
                else:
                    hook.prepare_previous(trainer, current, 1, "raw", 4000)
                self.assertIs(node._gs_cache, cache)
                self.assertIs(node._raw_normal_residual_penalty, penalty)
                self.assertIs(trainer.cur_frame, original_frame)
                self.assertIs(trainer._render_c2w, original_camera)
                self.assertIs(trainer._render_intrinsics, original_intrinsics)
                self.assertFalse(hasattr(trainer, "info"))
                self.assertEqual(source._downscale_factor, 1.)
                self.assertEqual(source._old_downscale_factor, [.5])
                self.assertEqual(source.camera_data[0].downscale_factor, 1.)
                self.assertEqual(random.getstate(), py_state)
                np.testing.assert_equal(np.random.get_state(), np_state)
                torch.testing.assert_close(torch.get_rng_state(), torch_state, atol=0, rtol=0)

    def test_original_losses_and_current_gradient_survive_history_render(self):
        trainer, hook, current, cameras, _, _ = self.fixture()
        packet = hook.prepare_previous(trainer, current, 1, "raw", 4000)
        # Simulate view-dependent rendered material disagreement, without an
        # optimization step or any real scene data.
        with torch.no_grad():
            trainer.albedo.fill_(.6)
            trainer.roughness.fill_(.8)
        outputs = trainer(current, cameras)
        main_camera = trainer._render_c2w
        with patch.object(MultiTrainer_weather, "compute_losses", side_effect=lambda *a, **k: {}):
            original = trainer.compute_losses(outputs, current, cameras)
        combined = dict(original)
        combined.update(hook.compute(trainer, outputs, current, packet, "raw", 4000))
        self.assertEqual(set(combined) - set(original), {"temporal_material_loss"})
        for key, value in original.items():
            self.assertIs(combined[key], value)
        loss = combined["temporal_material_loss"]
        self.assertGreater(float(loss.detach()), 0.)
        loss.backward()
        self.assertGreater(float(trainer.albedo.grad.abs().sum()), 0.)
        self.assertGreater(float(trainer.roughness.grad.abs().sum()), 0.)
        self.assertIs(trainer._render_c2w, main_camera)
        self.assertGreater(int(hook.last_stats["valid_sample_count"]), 0)
        self.assertLessEqual(int(hook.last_stats["sample_count"]), 4096)
        self.assertTrue(all(not value.requires_grad for value in hook.last_stats.values()))
        self.assertEqual(trainer.auxiliary_renders, 0)

    def test_stale_step_wrong_pair_and_heldout_current_are_rejected(self):
        trainer, hook, current, cameras, _, _ = self.fixture()
        packet = hook.prepare_previous(trainer, current, 1, "raw", 4000)
        outputs = trainer(current, cameras)
        for field, value in (("_step", 3992), ("_current_image_idx", 3), ("_image_idx", 1)):
            bad_packet = {**packet, field: value}
            with self.assertRaisesRegex(ValueError, "different step or training pair"):
                hook.compute(trainer, outputs, current, bad_packet, "raw", 4000)
        with self.assertRaisesRegex(ValueError, "held-out"):
            hook.compute(trainer, outputs, {"img_idx": torch.tensor([1])}, None, "raw", 4000)

    def test_missing_pair_is_zero_and_history_is_rendered_fresh_on_next_active_step(self):
        trainer, hook, current, cameras, _, loads = self.fixture()
        first = {**current, "img_idx": torch.zeros_like(current["img_idx"])}
        self.assertIsNone(hook.prepare_previous(trainer, first, 1, "raw", 4000))
        outputs = trainer(first, cameras)
        loss = hook.compute(trainer, outputs, first, None, "raw", 4000)["temporal_material_loss"]
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertEqual(float(trainer.albedo.grad.abs().sum()), 0.)
        self.assertEqual(hook.counters["missing_pairs"], 1)
        first_packet = hook.prepare_previous(trainer, current, 1, "raw", 4000)
        next_packet = hook.prepare_previous(trainer, current, 1, "raw", 4008)
        self.assertEqual(len(loads), 2)
        self.assertIsNot(first_packet, next_packet)
        self.assertEqual(next_packet["_step"], 4008)


if __name__ == "__main__":
    unittest.main()
