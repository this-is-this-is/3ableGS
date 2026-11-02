"""Production teacher-loss wiring with synthetic scene rendering."""
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from models.diffreg_material import validate_diffreg_config
from models.trainers.scene_graph_weather import MultiTrainer_weather
from tests.test_temporal_material_integration import SyntheticTrainer


class DiffRegIntegrationTests(unittest.TestCase):
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
        trainer._diffreg_config = validate_diffreg_config({"enabled": True})
        generator = torch.Generator().manual_seed(712)
        trainer.albedo = torch.nn.Parameter(.2 + .5 * torch.rand(12, 12, 3, generator=generator))
        images, cameras = trainer.inputs(0)
        images.update(
            pixels=torch.zeros(12, 12, 3), sky_masks=torch.zeros(12, 12),
            dynamic_masks=torch.zeros(12, 12), egocar_masks=torch.zeros(12, 12),
            gbuf_basecolor=torch.full((12, 12, 3), .4, requires_grad=True),
            gbuf_roughness=torch.full((12, 12, 1), .5),
            gbuf_metallic=torch.full((12, 12, 1), .1),
            gbuf_normal=trainer.normal.clone(),
        )
        return trainer, images, cameras

    def test_original_losses_preserved_and_added_loss_backpropagates_without_render(self):
        trainer, images, cameras = self.fixture()
        outputs = trainer(images, cameras)
        rng = torch.get_rng_state().clone()
        with patch.object(MultiTrainer_weather, "compute_losses", side_effect=lambda *a, **k: {}):
            active = trainer.compute_losses(outputs, images, cameras)
            trainer._diffreg_config["enabled"] = False
            baseline = trainer.compute_losses(outputs, images, cameras)
        self.assertEqual(set(active) - set(baseline), {"diffreg_material_loss"})
        for key in baseline:
            torch.testing.assert_close(active[key], baseline[key], atol=0, rtol=0)
        self.assertGreater(float(active["diffreg_material_loss"].detach()), 0)
        active["diffreg_material_loss"].backward()
        self.assertGreater(float(trainer.albedo.grad.abs().sum()), 0)
        self.assertIsNone(images["gbuf_basecolor"].grad)
        self.assertEqual(trainer.auxiliary_renders, 0)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)

    def test_static_masks_and_low_opacity_exclude_both_centers_and_neighbors(self):
        trainer, images, cameras = self.fixture()
        images["sky_masks"][:2] = 1
        images["dynamic_masks"][2:4] = 1
        images["egocar_masks"][4:6] = 1
        trainer.opacity[6:8] = .5
        outputs = trainer(images, cameras)
        loss = trainer.diffreg_losses(outputs, images)["diffreg_material_loss"]
        loss.backward()
        self.assertEqual(float(trainer.albedo.grad[:8].abs().sum()), 0.)
        self.assertGreater(float(trainer.albedo.grad[8:].abs().sum()), 0.)

    def test_disabled_warmup_eval_do_not_require_reference_maps(self):
        trainer, images, cameras = self.fixture()
        with patch("models.diffreg_material.material_relation_loss", side_effect=AssertionError("unused")):
            trainer.step = 3000
            self.assertEqual(trainer.diffreg_losses({}, {}), {})
            trainer.step = 4000
            trainer.eval()
            self.assertEqual(trainer.diffreg_losses({}, {}), {})
            trainer.train()
            trainer._diffreg_config["enabled"] = False
            self.assertEqual(trainer.diffreg_losses({}, {}), {})

    def test_enabled_training_requires_explicit_dynamic_mask(self):
        trainer, images, cameras = self.fixture()
        del images["dynamic_masks"]
        with self.assertRaisesRegex(ValueError, "dynamic_masks"):
            trainer.diffreg_losses(trainer(images, cameras), images)

    def test_inference_ignores_teacher_and_regularizer_setting(self):
        trainer, images, cameras = self.fixture()
        trainer.eval()
        with torch.no_grad():
            first = trainer(images, cameras)
            trainer._diffreg_config["enabled"] = False
            for key in list(images):
                if key.startswith("gbuf_"):
                    images[key] = torch.full_like(images[key], 99.)
            second = trainer(images, cameras)
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        self.assertEqual(trainer.auxiliary_renders, 0)

    def test_teacher_crop_is_identical_to_existing_supervision(self):
        trainer, images, cameras = self.fixture()
        trainer.crop_frac = .25
        images["gbuf_basecolor"] = torch.full((6, 12, 3), .4)
        images["gbuf_roughness"] = torch.full((6, 12, 1), .5)
        images["gbuf_metallic"] = torch.full((6, 12, 1), .1)
        outputs = trainer(images, cameras)
        with patch("models.diffreg_material.material_relation_loss", return_value=(outputs["albedo"].sum() * 0, {})) as call:
            trainer.diffreg_losses(outputs, images)
        prediction, teacher, valid = call.call_args.args
        torch.testing.assert_close(prediction["albedo"], outputs["albedo"][3:9], rtol=0, atol=0)
        torch.testing.assert_close(teacher["albedo"], images["gbuf_basecolor"], rtol=0, atol=0)
        self.assertEqual(valid.shape, (6, 12))


if __name__ == "__main__":
    unittest.main()
