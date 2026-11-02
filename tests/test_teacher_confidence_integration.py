"""Real PBR loss/forward wiring with synthetic geometry; no scene jobs."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from models.trainers.scene_graph_pbr import OPENCV_TO_OPENGL
from models.trainers.scene_graph_weather import MultiTrainer_weather
from tests.test_temporal_material_integration import SyntheticTrainer


class TeacherConfidenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, enabled=True):
        trainer = SyntheticTrainer(size=12)
        trainer.train()
        trainer.step = 2999
        trainer.crop_frac = 0.
        trainer._opencv_to_opengl = OPENCV_TO_OPENGL
        trainer.losses_dict = OmegaConf.create({
            "albedo": {"w": .5}, "roughness": {"w": .2},
            "metallic": {"w": .2}, "normal": {"w": .15},
        })
        trainer.env.compute_reg_loss = lambda _: {}
        trainer.albedo = torch.nn.Parameter(torch.full((12, 12, 3), .6))
        trainer.roughness = torch.nn.Parameter(torch.full((12, 12, 1), .7))
        train_ids = [0, 2, 3]
        images, cameras = {}, {}
        loads = []
        for image_id in train_ids:
            image, camera = trainer.inputs(image_id)
            albedo = torch.full((12, 12, 3), .2)
            if image_id == 2:
                albedo[:, :6] = .8
            image.update(
                pixels=torch.zeros(12, 12, 3), sky_masks=torch.zeros(12, 12),
                dynamic_masks=torch.zeros(12, 12), egocar_masks=torch.zeros(12, 12),
                gbuf_basecolor=albedo, gbuf_roughness=torch.full((12, 12, 1), .5),
                gbuf_metallic=torch.full((12, 12, 1), .1), gbuf_normal=-trainer.normal.clone(),
            )
            images[image_id], cameras[image_id] = image, camera

        def get_image(position, downscale, weathers):
            image_id = train_ids[position]
            loads.append((image_id, downscale, weathers))
            return {"raw": images[image_id]}, {"raw": cameras[image_id]}

        trainer.dataset = SimpleNamespace(
            train_indices=train_ids,
            pixel_source=SimpleNamespace(num_cams=1, gbuffer_crop_frac=0.),
            train_image_set=SimpleNamespace(get_image=get_image),
        )
        trainer.configure_teacher_confidence({"enabled": enabled})
        return trainer, images, cameras, loads

    @staticmethod
    def build(trainer):
        trainer.step = 3000
        trainer.prepare_teacher_confidence(3000)
        trainer.step = 4000

    def test_disabled_and_prebuild_warmup_are_exact_original_losses(self):
        trainer, images, cameras, loads = self.fixture(enabled=False)
        outputs = trainer(images[2], cameras[2])
        baseline = trainer.gbuffer_losses(outputs, images[2], cameras[2])
        trainer.configure_teacher_confidence({"enabled": True})
        trainer.prepare_teacher_confidence(2999)
        warmup = trainer.gbuffer_losses(outputs, images[2], cameras[2])
        self.assertEqual(baseline.keys(), warmup.keys())
        for name in baseline:
            torch.testing.assert_close(baseline[name], warmup[name], atol=0, rtol=0)
        self.assertFalse(loads)

    def test_cache_uses_train_only_once_and_main_render_owns_current_state(self):
        trainer, images, cameras, loads = self.fixture()
        before = trainer(images[2], cameras[2])
        old_camera = trainer._render_c2w
        old_frame = trainer.cur_frame
        rng = torch.get_rng_state().clone()
        self.build(trainer)
        self.assertEqual([value[0] for value in loads], [0, 2, 3])
        self.assertTrue(all(value[1:] == (4, ["raw"]) for value in loads))
        self.assertIs(trainer._render_c2w, old_camera)
        self.assertIs(trainer.cur_frame, old_frame)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        trainer.prepare_teacher_confidence(4000)
        after = trainer(images[2], cameras[2])
        self.assertEqual(len(loads), 3)
        torch.testing.assert_close(before["rgb"], after["rgb"], atol=0, rtol=0)
        saved = trainer.state_dict()["_teacher_confidence"]
        self.assertEqual(set(saved["frames"]), {0, 2, 3})
        confidence = saved["frames"][2]
        self.assertLess(float(confidence[4, 4, 0]), float(confidence[4, 8, 0]))
        self.assertFalse(confidence.requires_grad)
        self.assertEqual(confidence.device.type, "cpu")
        self.assertEqual(confidence.dtype, torch.float16)

    def test_only_material_weighting_changes_rgb_normal_and_teacher_values_do_not(self):
        trainer, images, cameras, _ = self.fixture()
        image = images[2]
        originals = {key: value.clone() for key, value in image.items() if torch.is_tensor(value)}
        outputs = trainer(image, cameras[2])

        def inherited_losses(*args, **kwargs):
            return {"rgb_loss": outputs["rgb"].abs().mean(), "opacity_loss": outputs["opacity"].mean()}

        with patch.object(MultiTrainer_weather, "compute_losses", side_effect=inherited_losses):
            baseline = trainer.compute_losses(outputs, image, cameras[2])
            self.build(trainer)
            current = trainer(image, cameras[2])
            active = trainer.compute_losses(current, image, cameras[2])
        self.assertEqual(baseline.keys(), active.keys())
        for key in ("rgb_loss", "normal_loss", "opacity_loss"):
            torch.testing.assert_close(baseline[key], active[key], atol=0, rtol=0)
        self.assertNotEqual(float(baseline["albedo_loss"].detach()), float(active["albedo_loss"].detach()))
        for key in originals:
            torch.testing.assert_close(image[key], originals[key], atol=0, rtol=0)
        self.assertEqual(trainer.auxiliary_renders, 0)

    def test_dynamic_gradient_is_original_and_static_gradient_mass_is_preserved(self):
        trainer, images, cameras, _ = self.fixture()
        self.build(trainer)
        image = images[2]
        image["dynamic_masks"][6:] = 1
        image["sky_masks"][0] = 1
        image["egocar_masks"][1] = 1
        outputs = trainer(image, cameras[2])
        weights = trainer.teacher_confidence_weights(image, (12, 12),
                                                     (1 - image["sky_masks"] - image["egocar_masks"])[..., None])
        torch.testing.assert_close(weights[6:], torch.ones_like(weights[6:]), atol=0, rtol=0)
        self.assertAlmostEqual(float(weights[2:6].mean()), 1., places=6)
        losses = trainer.gbuffer_losses(outputs, image, cameras[2])
        losses["albedo_loss"].backward()
        expected_dynamic_grad = .5 / (10 * 12 * 3)
        torch.testing.assert_close(trainer.albedo.grad[6:].abs(),
                                   torch.full_like(trainer.albedo.grad[6:], expected_dynamic_grad))
        self.assertEqual(float(trainer.albedo.grad[:2].abs().sum()), 0.)
        self.assertGreater(float(trainer.albedo.grad[3, 8].abs().mean()),
                           float(trainer.albedo.grad[3, 4].abs().mean()))
        self.assertAlmostEqual(float(trainer.albedo.grad[2:6].abs().sum()), .5 * 4 / 10, places=6)

    def test_inference_never_requires_teacher_or_rebuilds_cache_and_reload_matches(self):
        trainer, images, cameras, loads = self.fixture()
        self.build(trainer)
        saved = copy.deepcopy(trainer.state_dict())
        clone, _, _, clone_loads = self.fixture()
        clone.load_state_dict(saved)
        self.assertFalse(clone_loads)
        trainer.eval()
        clone.eval()
        bare = {key: value for key, value in images[2].items() if not key.startswith("gbuf_")}
        with torch.no_grad():
            before = trainer(bare, cameras[2])
            after = clone(bare, cameras[2])
        for key in before:
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        clone.prepare_teacher_confidence(4000)
        self.assertFalse(clone_loads)
        self.assertEqual(len(loads), 3)
        self.assertIsNone(clone.teacher_confidence_weights({}, (12, 12), torch.ones(12, 12, 1)))

    def test_baseline_checkpoint_is_unchanged_and_missing_active_cache_fails(self):
        baseline, _, _, _ = self.fixture(enabled=False)
        saved = copy.deepcopy(baseline.state_dict())
        self.assertNotIn("_teacher_confidence", saved)
        baseline.load_state_dict(saved)
        trainer, images, cameras, _ = self.fixture()
        trainer.step = 4000
        with self.assertRaises(ValueError):
            trainer.prepare_teacher_confidence(4000)
        with self.assertRaises(ValueError):
            trainer.gbuffer_losses(trainer(images[2], cameras[2]), images[2], cameras[2])


if __name__ == "__main__":
    unittest.main()
