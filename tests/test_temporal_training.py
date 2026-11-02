"""CPU checks for train-only temporal pairing, holdouts and detached targets."""

import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from models.temporal_training import (
    TemporalMaterialTraining,
    material_temporal_loss,
    static_training_mask,
    validate_temporal_training_config,
)
from tests import test_temporal_material
from utils.temporal_split import (
    previous_train_image_index,
    resolve_test_frames,
    training_bracket,
)


class TemporalSplitTests(unittest.TestCase):
    def test_explicit_ranges_override_stride_and_use_exclusive_ends(self):
        self.assertEqual(resolve_test_frames(100, [[20, 30], [60, 70]], 8),
                         list(range(20, 30)) + list(range(60, 70)))
        self.assertEqual(resolve_test_frames(10, [], 2), [])
        config = OmegaConf.create({"ranges": [[2, 4], [6, 8]]})
        self.assertEqual(resolve_test_frames(10, config.ranges), [2, 3, 6, 7])

    def test_legacy_stride_and_no_holdout_are_unchanged(self):
        self.assertEqual(resolve_test_frames(20, test_image_stride=8), [8, 16])
        self.assertEqual(resolve_test_frames(20), [])
        self.assertEqual(resolve_test_frames(1, test_image_stride=8), [])

    def test_invalid_ranges_fail(self):
        for value in ([[2, 4], [3, 5]], [[6, 8], [2, 4]], [[3, 3]], [[-1, 2]],
                      [[1, 11]], [[0, 10]], [[1., 3]], [[True, 3]], [[1, 2, 3]], "2:4"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_test_frames(10, value)

    def test_interpolation_brackets_skip_whole_holdout(self):
        train = [frame for frame in range(100) if not 20 <= frame < 30 and not 60 <= frame < 70]
        self.assertEqual(training_bracket(20, train), (19, 30, 1 / 11))
        self.assertEqual(training_bracket(25, train), (19, 30, 6 / 11))
        self.assertEqual(training_bracket(69, train), (59, 70, 10 / 11))
        self.assertEqual(training_bracket(30, train), (30, 30, 0.))
        self.assertEqual(training_bracket(0, [3, 4]), (3, 3, 0.))
        self.assertEqual(training_bracket(8, [3, 4]), (4, 4, 0.))
        with self.assertRaises(ValueError):
            training_bracket(0, [])

    def test_pair_selection_uses_same_camera_and_only_training_frames(self):
        # Two cameras, held-out frames 2 and 3.
        allowed = {frame * 2 + camera for frame in (0, 1, 4, 5) for camera in (0, 1)}
        self.assertEqual(previous_train_image_index(11, allowed, 2), 9)
        self.assertEqual(previous_train_image_index(2, allowed, 2), 0)
        self.assertIsNone(previous_train_image_index(8, allowed, 2))
        self.assertIsNone(previous_train_image_index(1, allowed, 2))
        self.assertEqual(previous_train_image_index(9, allowed, 2, max_gap=3), 3)
        with self.assertRaises(ValueError):
            previous_train_image_index(5, allowed, 2)


class TemporalTrainingTests(unittest.TestCase):
    @staticmethod
    def frame(roughness=.4):
        return test_temporal_material.TemporalMaterialTests.frame(roughness=roughness)

    @staticmethod
    def output(frame):
        return {**frame["gbuffer"], "depth": frame["depth"], "opacity": frame["opacity"]}

    @staticmethod
    def previous(frame):
        return {key: value for key, value in frame.items()
                if key in ("gbuffer", "depth", "opacity", "static_mask", "camera_to_world", "intrinsics")}

    def test_temporal_gradient_enters_only_current_material(self):
        previous = self.frame(.2)
        current = self.frame(.8)
        for frame in (previous, current):
            for value in frame["gbuffer"].values():
                value.requires_grad_()
            frame["camera_to_world"].requires_grad_()
            frame["depth"].requires_grad_()
        loss = material_temporal_loss(
            self.output(current), current["static_mask"], current["camera_to_world"],
            current["intrinsics"], self.previous(previous),
        )
        self.assertAlmostEqual(loss.item(), .2, places=6)
        loss.backward()
        self.assertAlmostEqual(current["gbuffer"]["roughness"].grad.sum().item(), 1 / 3, places=6)
        self.assertIsNone(previous["gbuffer"]["roughness"].grad)
        self.assertIsNone(current["camera_to_world"].grad)
        self.assertIsNone(current["depth"].grad)
        self.assertIsNone(current["gbuffer"]["normal"].grad)
        self.assertIsNone(previous["camera_to_world"].grad)

    def test_known_subpixel_correspondence_matches_analytic_loss(self):
        previous = self.frame(.2)
        previous["gbuffer"]["roughness"][:] = torch.arange(7)[None, :, None] / 10
        current = self.frame(.8)
        current["camera_to_world"][0, 3] = .5
        # Static erosion leaves x=1..4 with targets .15,.25,.35,.45.
        loss = material_temporal_loss(
            self.output(current), current["static_mask"], current["camera_to_world"],
            current["intrinsics"], self.previous(previous),
        )
        self.assertAlmostEqual(loss.item(), 1 / 6, places=6)

    def test_empty_overlap_and_missing_pair_have_zero_current_gradient(self):
        for missing in (True, False):
            current = self.frame(.8)
            current["gbuffer"]["roughness"].requires_grad_()
            previous = self.frame(.2)
            previous["static_mask"].zero_()
            loss = material_temporal_loss(
                self.output(current), current["static_mask"], current["camera_to_world"],
                current["intrinsics"], None if missing else self.previous(previous), .05,
            )
            self.assertEqual(loss.item(), 0.)
            loss.backward()
            self.assertTrue((current["gbuffer"]["roughness"].grad == 0).all())

    def test_static_mask_excludes_sky_dynamic_and_ego_including_nan(self):
        frame = self.frame()
        infos = {name: torch.zeros(5, 7) for name in ("sky_masks", "dynamic_masks", "egocar_masks")}
        infos["sky_masks"][1, 1] = 1.
        infos["dynamic_masks"][2, 2] = 1.
        infos["egocar_masks"][3, 3] = 1.
        infos["dynamic_masks"][3, 4] = float("nan")
        mask = static_training_mask(infos, frame["depth"])
        self.assertEqual(mask.sum().item(), 31)
        for y, x in ((1, 1), (2, 2), (3, 3), (3, 4)):
            self.assertFalse(mask[y, x])
        with self.assertRaises(ValueError):
            static_training_mask({"sky_masks": torch.zeros(5, 7)}, frame["depth"])

    def test_config_validation_and_omegaconf(self):
        self.assertEqual(validate_temporal_training_config()["w"], 0.)
        config = validate_temporal_training_config(OmegaConf.create({"w": .05, "start_step": 0}))
        self.assertEqual(config, {"w": .05, "start_step": 0, "max_gap": 1,
                                  "interval": 1, "ramp_steps": 0, "num_samples": 0,
                                  "seed": 20261101})
        for invalid in ({"w": -1}, {"w": float("nan")}, {"w": True}, {"max_gap": 0},
                        {"start_step": -.1}, {"start_step": True}, {"unexpected": 2}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_temporal_training_config(invalid)

    def test_training_hook_never_loads_test_image_and_renders_before_main(self):
        calls = []
        frame = self.frame(.2)

        class FakeSplit:
            def get_image(self, split_position, camera_downscale, weathers):
                calls.append(("load", split_position))
                image = {
                    "pixels": torch.zeros(5, 7, 3),
                    "sky_masks": torch.zeros(5, 7),
                    "dynamic_masks": torch.zeros(5, 7),
                }
                return {"raw": image}, {"raw": {}}

        class FakeTrainer:
            def __call__(self, image, camera, weather):
                calls.append(("render", torch.is_grad_enabled()))
                self.info = "previous"
                self._render_c2w = frame["camera_to_world"]
                self._render_intrinsics = frame["intrinsics"]
                return TemporalTrainingTests.output(frame)

        dataset = SimpleNamespace(
            train_indices=[0, 1, 4, 5], pixel_source=SimpleNamespace(num_cams=1),
            train_image_set=FakeSplit(),
        )
        hook = TemporalMaterialTraining(dataset, {"w": .05, "start_step": 3})
        trainer = FakeTrainer()
        image = {"img_idx": torch.tensor([5]), "pixels": torch.zeros(5, 7, 3)}
        self.assertIsNone(hook.prepare_previous(trainer, image, 2, "raw", 2))
        self.assertFalse(calls)
        target = hook.prepare_previous(trainer, image, 2, "raw", 3)
        self.assertEqual(calls, [("load", 2), ("render", False)])
        self.assertTrue(all(not value.requires_grad for value in target["gbuffer"].values()))
        # The main training render executes after acquisition and owns info.
        trainer.info = "main differentiable render"
        current = self.frame(.8)
        infos = {"sky_masks": torch.zeros(5, 7), "dynamic_masks": torch.zeros(5, 7)}
        loss = hook.compute(trainer, self.output(current), infos, target, "raw", 3)
        self.assertIn("temporal_material_loss", loss)
        self.assertEqual(trainer.info, "main differentiable render")
        image["img_idx"] = torch.tensor([4])
        self.assertIsNone(hook.prepare_previous(trainer, image, 2, "raw", 3))
        self.assertEqual(len(calls), 2)
        image["img_idx"] = torch.tensor([3])
        with self.assertRaises(ValueError):
            hook.prepare_previous(trainer, image, 2, "raw", 3)


if __name__ == "__main__":
    unittest.main()
