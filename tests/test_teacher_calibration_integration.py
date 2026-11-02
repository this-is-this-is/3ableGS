"""Synthetic CPU checks of teacher calibration, training and checkpoint wiring."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from models.trainers.scene_graph_pbr import OPENCV_TO_OPENGL
from models.trainers.scene_graph_weather import MultiTrainer_weather
from tests.test_temporal_material_integration import SyntheticTrainer


class TeacherCalibrationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, enabled=True, train_indices=(0, 2, 3), **options):
        trainer = SyntheticTrainer(size=12)
        trainer.train()
        trainer.dataset = SimpleNamespace(
            train_indices=list(train_indices), pixel_source=SimpleNamespace(num_cams=1))
        trainer.step = 4000
        trainer.crop_frac = 0.
        trainer._opencv_to_opengl = OPENCV_TO_OPENGL
        trainer.losses_dict = OmegaConf.create({
            "albedo": {"w": .5}, "roughness": {"w": .2},
            "metallic": {"w": .2}, "normal": {"w": .15},
        })
        trainer.env.compute_reg_loss = lambda _: {}
        trainer.albedo = torch.nn.Parameter(torch.full((12, 12, 3), .6))
        trainer.roughness = torch.nn.Parameter(torch.full((12, 12, 1), .7))
        trainer.configure_teacher_calibration({"enabled": enabled, **options})
        images, cameras = trainer.inputs(2)
        images.update(
            pixels=torch.zeros(12, 12, 3), sky_masks=torch.zeros(12, 12),
            dynamic_masks=torch.zeros(12, 12), egocar_masks=torch.zeros(12, 12),
            gbuf_basecolor=torch.full((12, 12, 3), .4, requires_grad=True),
            gbuf_roughness=torch.full((12, 12, 1), .5, requires_grad=True),
            gbuf_metallic=torch.full((12, 12, 1), .1, requires_grad=True),
            gbuf_normal=-trainer.normal.clone(),
        )
        return trainer, images, cameras

    @staticmethod
    def change_head(trainer):
        with torch.no_grad():
            trainer.teacher_calibration_head.raw[0].fill_(-.3)
            trainer.teacher_calibration_head.raw[1].fill_(.3)

    def test_disabled_and_warmup_preserve_all_original_losses_exactly(self):
        trainer, images, cameras = self.fixture(enabled=False)
        outputs = trainer(images, cameras)
        original = trainer.gbuffer_losses(outputs, images, cameras)
        trainer.configure_teacher_calibration({"enabled": True})
        self.change_head(trainer)
        trainer.step = 3000
        warmup = trainer.gbuffer_losses(outputs, images, cameras)
        self.assertEqual(original.keys(), warmup.keys())
        for name in original:
            torch.testing.assert_close(warmup[name], original[name], atol=0, rtol=0)
        sum(warmup.values()).backward()
        self.assertIsNone(trainer.teacher_calibration_head.raw.grad)
        self.assertFalse(trainer._last_teacher_calibration_stats)

    def test_zero_initialization_preserves_original_losses_and_random_stream(self):
        trainer, images, cameras = self.fixture(enabled=False)
        outputs = trainer(images, cameras)
        original = trainer.gbuffer_losses(outputs, images, cameras)
        rng = torch.get_rng_state().clone()
        trainer.configure_teacher_calibration({"enabled": True})
        active = trainer.gbuffer_losses(outputs, images, cameras)
        for name in original:
            torch.testing.assert_close(original[name], active[name], atol=0, rtol=0)
        self.assertEqual(float(active["teacher_calibration_identity_loss"].detach()), 0.)
        self.assertEqual(float(active["teacher_calibration_smooth_loss"].detach()), 0.)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)

    def test_teacher_and_render_inputs_unchanged_normal_rgb_and_other_losses_preserved(self):
        trainer, images, cameras = self.fixture(enabled=False)
        outputs = trainer(images, cameras)
        originals = {key: value.clone() for key, value in images.items() if torch.is_tensor(value)}
        rendered = {key: value.clone() for key, value in outputs.items() if torch.is_tensor(value)}

        def original_rgb(*args, **kwargs):
            return {"rgb_loss": outputs["rgb"].abs().mean(), "mask_loss": outputs["opacity"].mean()}

        with patch.object(MultiTrainer_weather, "compute_losses", side_effect=original_rgb):
            before = trainer.compute_losses(outputs, images, cameras)
            trainer.configure_teacher_calibration({"enabled": True})
            self.change_head(trainer)
            after = trainer.compute_losses(outputs, images, cameras)
        self.assertEqual(set(after) - set(before), {
            "teacher_calibration_identity_loss", "teacher_calibration_smooth_loss"})
        for name in ("normal_loss", "rgb_loss", "mask_loss"):
            torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)
        self.assertNotEqual(float(before["albedo_loss"].detach()), float(after["albedo_loss"].detach()))
        for key, value in originals.items():
            torch.testing.assert_close(images[key], value, atol=0, rtol=0)
        for key, value in rendered.items():
            torch.testing.assert_close(outputs[key], value, atol=0, rtol=0)
        sum(after.values()).backward()
        self.assertGreater(float(trainer.albedo.grad.abs().sum()), 0.)
        self.assertGreater(float(trainer.teacher_calibration_head.raw.grad.abs().sum()), 0.)
        for key in ("gbuf_basecolor", "gbuf_roughness", "gbuf_metallic"):
            self.assertIsNone(images[key].grad)
        self.assertEqual(trainer.auxiliary_renders, 0)

    def test_dynamic_and_low_opacity_keep_original_supervision_and_denominator(self):
        trainer, images, cameras = self.fixture()
        self.change_head(trainer)
        images["sky_masks"][0] = 1
        images["egocar_masks"][1] = 1
        images["dynamic_masks"][6:] = 1
        trainer.opacity[2] = .5
        outputs = trainer(images, cameras)
        losses = trainer.gbuffer_losses(outputs, images, cameras)
        gain, bias = trainer.teacher_calibration_head.coefficients(4000)
        adjusted = images["gbuf_basecolor"].detach().clone()
        adjusted[3:6] = (adjusted[3:6] * gain[1, :3] + bias[1, :3]).clamp(0, 1)
        expected = .5 * (outputs["albedo"][2:] - adjusted[2:]).abs().mean()
        torch.testing.assert_close(losses["albedo_loss"], expected)
        losses["albedo_loss"].backward()
        self.assertGreater(float(trainer.albedo.grad[6:].abs().sum()), 0.)
        self.assertEqual(float(trainer.albedo.grad[:2].abs().sum()), 0.)

    def test_crop_and_resize_match_existing_teacher_alignment(self):
        trainer, images, cameras = self.fixture()
        trainer.crop_frac = .25
        for key in ("gbuf_basecolor", "gbuf_roughness", "gbuf_metallic", "gbuf_normal"):
            images[key] = images[key][3:9].detach()
        outputs = trainer(images, cameras)
        head = trainer.teacher_calibration_head
        with patch.object(head, "correct", wraps=head.correct) as corrected:
            trainer.gbuffer_losses(outputs, images, cameras)
        maps, valid, index, step = corrected.call_args.args
        self.assertEqual(valid.shape, (6, 12))
        self.assertEqual(index, 2)
        self.assertEqual(step, 4000)
        torch.testing.assert_close(maps["albedo"], images["gbuf_basecolor"], atol=0, rtol=0)

    def test_heldout_and_missing_static_annotation_fail_before_learning(self):
        trainer, images, cameras = self.fixture()
        outputs = trainer(images, cameras)
        images["img_idx"].fill_(1)
        with self.assertRaises(ValueError):
            trainer.gbuffer_losses(outputs, images, cameras)
        images["img_idx"].fill_(2)
        del images["dynamic_masks"]
        with self.assertRaisesRegex(ValueError, "dynamic_masks"):
            trainer.gbuffer_losses(outputs, images, cameras)

    def test_evaluation_never_uses_head_or_requires_teacher(self):
        trainer, images, cameras = self.fixture()
        trainer.eval()
        with torch.no_grad():
            before = trainer(images, cameras)
            self.change_head(trainer)
            bare = {key: value for key, value in images.items() if not key.startswith("gbuf_")}
            after = trainer(bare, cameras)
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        # Even optional evaluation of teacher losses uses unmodified teachers.
        with patch.object(trainer.teacher_calibration_head, "correct", side_effect=AssertionError("unused")):
            trainer.gbuffer_losses(before, images, cameras)

    def test_optimizer_adds_only_tiny_head_and_keeps_scene_group(self):
        trainer, images, cameras = self.fixture()

        def scene_optimizer(value):
            value.param_groups = {"Background#albedo": [value.albedo]}
            value.lr_schedulers = {}
            value.optimizer = torch.optim.Adam([
                {"params": [value.albedo], "name": "Background#albedo", "lr": .002}])

        with patch.object(MultiTrainer_weather, "initialize_optimizer", scene_optimizer):
            trainer.initialize_optimizer()
        self.assertEqual([group["name"] for group in trainer.optimizer.param_groups],
                         ["Background#albedo", "TeacherCalibration#all"])
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], .002)
        self.assertEqual(trainer.optimizer.param_groups[1]["lr"], .001)
        self.assertIs(trainer.optimizer.param_groups[1]["params"][0], trainer.teacher_calibration_head.raw)
        self.assertEqual(sum(p.numel() for p in trainer.teacher_calibration_head.parameters()), 30)

    def test_checkpoint_roundtrip_keeps_same_head_and_accepts_original_base(self):
        trainer, images, cameras = self.fixture()
        self.change_head(trainer)
        saved = copy.deepcopy(trainer.state_dict())
        clone, clone_images, clone_cameras = self.fixture()
        head = clone.teacher_calibration_head
        clone.load_state_dict(saved)
        self.assertIs(clone.teacher_calibration_head, head)
        torch.testing.assert_close(head.raw, trainer.teacher_calibration_head.raw, atol=0, rtol=0)
        self.assertIn("_teacher_calibration", saved)
        with torch.no_grad():
            torch.testing.assert_close(clone(clone_images, clone_cameras)["rgb"],
                                       trainer(images, cameras)["rgb"], atol=0, rtol=0)
        base, _, _ = self.fixture(enabled=False)
        base_state = copy.deepcopy(base.state_dict())
        self.assertFalse(any("teacher_calibration" in key for key in base_state))
        base.load_state_dict(base_state)

    def test_checkpoint_rejects_config_split_missing_metadata_and_corrupt_id_buffer(self):
        trainer, _, _ = self.fixture()
        saved = copy.deepcopy(trainer.state_dict())
        for options in ({"enabled": False}, {"max_bias": .01}, {"train_indices": (0, 2, 4)}):
            with self.subTest(options=options):
                incompatible, _, _ = self.fixture(**options)
                with self.assertRaises(ValueError):
                    incompatible.load_state_dict(saved)
        clone, _, _ = self.fixture()
        without_metadata = dict(saved)
        without_metadata.pop("_teacher_calibration")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            clone.load_state_dict(without_metadata)
        wrong_buffer = copy.deepcopy(saved)
        wrong_buffer["teacher_calibration_head.train_indices"][0] = 1
        with self.assertRaisesRegex(ValueError, "buffer mismatch"):
            clone.load_state_dict(wrong_buffer)
        wrong_parameters = copy.deepcopy(saved)
        wrong_parameters["teacher_calibration_head.raw"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid parameters"):
            clone.load_state_dict(wrong_parameters)


if __name__ == "__main__":
    unittest.main()
