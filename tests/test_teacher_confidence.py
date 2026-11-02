"""CPU geometry, moments and loss-mass tests; no scene training or evaluation."""
import copy
import math
import random
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from models.teacher_confidence import (
    TeacherConfidenceMoments, normalized_teacher_confidence,
    teacher_confidence_strength, validate_teacher_confidence_config,
)
from tests import test_temporal_material as temporal_fixture


class TeacherConfidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def frame(**kwargs):
        kwargs.setdefault("dtype", torch.float64)
        return temporal_fixture.TemporalMaterialTests.frame(**kwargs)

    @staticmethod
    def moments(**kwargs):
        return TeacherConfidenceMoments({"enabled": True, "erode_radius": 0,
                                         "depth_edge_rel": 0., **kwargs})

    def pair(self, previous=None, current=None, **kwargs):
        processor = self.moments(**kwargs)
        processor.update(previous or self.frame(roughness=.2), 0, 0)
        return processor.update(current or self.frame(roughness=.8), 1, 0)

    @staticmethod
    def confidence_for_variance(variance, sigma=.1):
        # Only one of three equally weighted material groups differs.
        return .25 + .75 * math.exp(-.5 * variance / (3 * sigma ** 2))

    def test_first_view_and_identical_history_are_neutral(self):
        processor = self.moments()
        first, stats = processor.update(self.frame(), 0, 0)
        self.assertEqual(first.shape, (5, 7, 1))
        self.assertTrue((first == 1).all())
        self.assertEqual(float(stats["support_fraction"]), 0.)
        self.assertEqual(float(stats["history_length_mean"]), 1.)
        self.assertEqual(float(stats["reset"]), 1.)
        second, stats = processor.update(self.frame(), 1, 0)
        torch.testing.assert_close(second, torch.ones_like(second), atol=1e-12, rtol=0)
        self.assertEqual(float(stats["history_length_mean"]), 2.)
        self.assertEqual(float(stats["support_fraction"]), 1.)
        self.assertEqual(float(stats["reset"]), 0.)

    def test_two_view_variance_is_analytic_and_not_rejected_by_material_difference(self):
        confidence, stats = self.pair()
        expected = self.confidence_for_variance((.8 - .2) ** 2 / 4)
        torch.testing.assert_close(confidence, torch.full_like(confidence, expected), atol=1e-12, rtol=0)
        self.assertEqual(float(stats["support_fraction"]), 1.)
        self.assertAlmostEqual(float(stats["variance_score_mean"]), 3.)

    def test_accumulated_moments_and_capped_effective_history(self):
        processor = self.moments(max_history=4)
        values = [.2, .8, .5, .5, .5]
        mean, second = values[0], values[0] ** 2
        for frame_id, value in enumerate(values):
            confidence, stats = processor.update(self.frame(roughness=value), frame_id, 0)
            if frame_id:
                weight = min(frame_id / (frame_id + 1), .75)
                mean = (1 - weight) * value + weight * mean
                second = (1 - weight) * value ** 2 + weight * second
                expected = self.confidence_for_variance(second - mean ** 2)
                self.assertAlmostEqual(float(confidence[2, 3, 0]), expected, places=12)
            self.assertEqual(float(stats["history_length_mean"]), min(frame_id + 1, 4))

    def test_translation_uses_true_previous_surface_material(self):
        previous, current = self.frame(), self.frame(roughness=.8)
        previous["gbuffer"]["roughness"][:] = torch.arange(7, dtype=torch.float64)[None, :, None] / 10
        current["camera_to_world"][0, 3] = 1.
        confidence, stats = self.pair(previous, current)
        expected = self.confidence_for_variance((.8 - .3) ** 2 / 4)
        self.assertAlmostEqual(float(confidence[2, 2, 0]), expected, places=12)
        self.assertEqual(float(confidence[2, -1, 0]), 1.)
        self.assertAlmostEqual(float(stats["support_fraction"]), 6 / 7)
        self.assertLess(float(stats["cycle_error_px"]), 1e-12)

    def test_subpixel_second_moment_transport_matches_analytic_variance(self):
        previous, current = self.frame(), self.frame(roughness=.8)
        previous["gbuffer"]["roughness"][:] = torch.arange(7, dtype=torch.float64)[None, :, None] / 10
        current["camera_to_world"][0, 3] = .5
        confidence, _ = self.pair(previous, current)
        warped_mean, warped_second = (.2 + .3) / 2, (.2 ** 2 + .3 ** 2) / 2
        variance = (.8 ** 2 + warped_second) / 2 - ((.8 + warped_mean) / 2) ** 2
        self.assertAlmostEqual(float(confidence[2, 2, 0]), self.confidence_for_variance(variance), places=12)
        wrong_variance = (.8 ** 2 + warped_mean ** 2) / 2 - ((.8 + warped_mean) / 2) ** 2
        self.assertGreater(abs(float(confidence[2, 2, 0]) - self.confidence_for_variance(wrong_variance)), .001)

    def test_albedo_linearization_precedes_spatial_interpolation(self):
        previous, current = self.frame(albedo=0.), self.frame(albedo=0.)
        previous["gbuffer"]["albedo"][:, 3:] = 1.
        current["camera_to_world"][0, 3] = .5
        confidence, _ = self.pair(previous, current)
        # Previous RGB mean=.5, second=.5; current is zero. Variance=.1875.
        self.assertAlmostEqual(float(confidence[2, 2, 0]), self.confidence_for_variance(.1875), places=12)

    def test_camera_rotation_and_camera_z_translation_keep_correspondence(self):
        previous, current = self.frame(roughness=.2), self.frame(roughness=.8)
        current["camera_to_world"][:3, :3] = torch.tensor(
            [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
        confidence, stats = self.pair(previous, current)
        self.assertLess(float(confidence[2, 3, 0]), 1.)
        self.assertAlmostEqual(float(stats["support_fraction"]), 25 / 35)
        previous["depth"].fill_(3.)
        current["camera_to_world"] = torch.eye(4, dtype=torch.float64)
        current["camera_to_world"][2, 3] = 1.
        confidence, stats = self.pair(previous, current)
        self.assertLess(float(confidence[2, 3, 0]), 1.)
        self.assertLess(float(stats["cycle_error_px"]), 1e-12)

    def test_occlusion_normal_disagreement_and_opacity_fall_back_to_one(self):
        for change in ("previous_depth", "previous_opacity", "current_opacity", "normal"):
            with self.subTest(change=change):
                previous, current = self.frame(roughness=.2), self.frame(roughness=.8)
                if change == "previous_depth":
                    previous["depth"].fill_(4.)
                elif change == "previous_opacity":
                    previous["opacity"].fill_(.5)
                elif change == "current_opacity":
                    current["opacity"].fill_(.5)
                else:
                    current["gbuffer"]["normal"][:] = torch.tensor([1., 0., 0.])
                confidence, stats = self.pair(previous, current)
                self.assertTrue((confidence == 1).all())
                self.assertEqual(float(stats["support_fraction"]), 0.)
        current = self.frame(roughness=.8)
        current["gbuffer"]["normal"] *= -1
        confidence, _ = self.pair(current=current)
        self.assertLess(float(confidence[2, 3, 0]), 1.)

    def test_independent_cycle_rejects_when_forward_depth_test_still_passes(self):
        previous, current = self.frame(roughness=.2), self.frame(roughness=.8)
        previous["depth"].fill_(2.05)  # Forward error .05 < tolerance .09.
        current["camera_to_world"][0, 3] = 1.
        permissive, _ = self.pair(previous, current, cycle_px=1.)
        strict, stats = self.pair(previous, current, cycle_px=.01)
        self.assertLess(float(permissive[2, 3, 0]), 1.)
        self.assertEqual(float(strict[2, 3, 0]), 1.)
        self.assertEqual(float(stats["support_fraction"]), 0.)

    def test_returned_depth_rejects_even_with_zero_pixel_cycle_error(self):
        previous, current = self.frame(roughness=.2), self.frame(roughness=.8)
        current["camera_to_world"][2, 3] = .5
        previous["depth"].fill_(2.595)
        # Forward error .095 is below .05+.02*2.5=.1, but returned current
        # error .095 exceeds .05+.02*2=.09, including the central optical ray.
        confidence, stats = self.pair(previous, current)
        self.assertEqual(float(confidence[2, 3, 0]), 1.)
        self.assertEqual(float(stats["support_fraction"]), 0.)

    def test_complete_bilinear_support_static_erosion_and_depth_edges(self):
        previous, current = self.frame(roughness=.2), self.frame(roughness=.8)
        previous["static_mask"][2, 3] = False
        current["camera_to_world"][0, 3] = .5
        current["static_mask"][1, 1] = False
        confidence, _ = self.pair(previous, current)
        for y, x in ((2, 2), (2, 3), (1, 1)):
            self.assertEqual(float(confidence[y, x, 0]), 1.)
        self.assertLess(float(confidence[3, 3, 0]), 1.)
        previous, current = self.frame(roughness=.2), self.frame(roughness=.8)
        current["depth"][2, 3, 0] = 3.
        confidence, _ = self.pair(previous, current, depth_edge_rel=.1)
        self.assertTrue((confidence[1:4, 2:5] == 1).all())
        confidence, _ = self.pair(erode_radius=1)
        self.assertTrue((confidence[[0, -1]] == 1).all())
        self.assertTrue((confidence[:, [0, -1]] == 1).all())
        self.assertLess(float(confidence[2, 3, 0]), 1.)

    def test_invalid_materials_do_not_poison_other_confidence_or_moments(self):
        previous, current = self.frame(roughness=.2), self.frame(roughness=.8)
        previous["gbuffer"]["albedo"][1, 1] = float("nan")
        current["gbuffer"]["roughness"][2, 2] = float("inf")
        current["gbuffer"]["metallic"][3, 3] = -1.
        confidence, stats = self.pair(previous, current)
        self.assertTrue(torch.isfinite(confidence).all())
        self.assertTrue(all(torch.isfinite(value) for value in stats.values()))
        for y, x in ((1, 1), (2, 2), (3, 3)):
            self.assertEqual(float(confidence[y, x, 0]), 1.)
        self.assertLess(float(confidence[2, 4, 0]), 1.)

    def test_actual_frame_gap_and_camera_or_layout_reset(self):
        processor = self.moments()
        processor.update(self.frame(roughness=.2), 7, 0)
        confidence, stats = processor.update(self.frame(roughness=.8), 9, 0)
        self.assertLess(float(confidence[2, 3, 0]), 1.)
        self.assertEqual(float(stats["reset"]), 0.)
        confidence, stats = processor.update(self.frame(roughness=.2), 12, 0)
        self.assertTrue((confidence == 1).all())
        self.assertEqual(float(stats["reset"]), 1.)
        confidence, stats = processor.update(self.frame(roughness=.8), 0, 1)
        self.assertTrue((confidence == 1).all())
        self.assertEqual(float(stats["reset"]), 1.)
        confidence, stats = processor.update(self.frame(h=6, roughness=.2), 1, 1)
        self.assertEqual(confidence.shape, (6, 7, 1))
        self.assertTrue((confidence == 1).all())
        self.assertEqual(float(stats["reset"]), 1.)
        with self.assertRaisesRegex(ValueError, "increase"):
            processor.update(self.frame(h=6), 1, 1)
        processor.reset()
        confidence, _ = processor.update(self.frame(), 0, 0)
        self.assertTrue((confidence == 1).all())

    def test_disabled_single_sample_and_cpu_half_are_supported(self):
        for option in ({"enabled": False}, {"max_history": 1}):
            confidence, _ = self.pair(**option)
            self.assertTrue((confidence == 1).all())
        confidence, stats = self.pair(self.frame(dtype=torch.float16, roughness=.2),
                                      self.frame(dtype=torch.float16, roughness=.8))
        self.assertEqual(confidence.dtype, torch.float32)
        self.assertTrue(torch.isfinite(confidence).all())
        self.assertGreater(float(stats["support_fraction"]), 0.)

    def test_state_is_detached_inputs_immutable_and_global_rng_unchanged(self):
        previous, current = self.frame(roughness=.2), self.frame(roughness=.8)
        original = copy.deepcopy(current)
        for packet in (previous, current):
            for value in packet["gbuffer"].values():
                value.requires_grad_()
            for key in ("depth", "opacity", "camera_to_world", "intrinsics"):
                packet[key].requires_grad_()
        random_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        confidence, stats = self.pair(previous, current)
        self.assertFalse(confidence.requires_grad)
        self.assertTrue(all(not value.requires_grad for value in stats.values()))
        for key, value in current.items():
            if isinstance(value, dict):
                for name in value:
                    torch.testing.assert_close(value[name], original[key][name], atol=0, rtol=0)
                    self.assertIsNone(value[name].grad)
            else:
                torch.testing.assert_close(value, original[key], atol=0, rtol=0)
        self.assertEqual(random.getstate(), random_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, atol=0, rtol=0)
        processor = self.moments()
        first = self.frame(roughness=.2)
        processor.update(first, 0, 0)
        first["gbuffer"]["roughness"].fill_(.8)
        confidence, _ = processor.update(self.frame(roughness=.8), 1, 0)
        self.assertLess(float(confidence[2, 3, 0]), 1.)

    def test_configuration_schedule_and_bad_packet_validation(self):
        cfg = validate_teacher_confidence_config(OmegaConf.create({"enabled": True}))
        self.assertEqual(teacher_confidence_strength(3000, cfg), 0.)
        self.assertEqual(teacher_confidence_strength(3500, cfg), .5)
        self.assertEqual(teacher_confidence_strength(4000, cfg), 1.)
        self.assertEqual(teacher_confidence_strength(10000), 0.)
        self.assertEqual(teacher_confidence_strength(3000, {"enabled": True, "ramp_steps": 0}), 1.)
        for config in ({"unknown": 1}, {"enabled": 1}, {"build_step": -1}, {"downscale": 1.5},
                       {"max_gap": 0}, {"max_history": True}, {"confidence_floor": 0}, {"confidence_floor": 1e-8},
                       {"confidence_floor": 1.01}, {"albedo_sigma": 0}, {"roughness_sigma": float("nan")},
                       {"depth_abs": True}, {"normal_cos": 0}, {"min_opacity": 1.1}, {"max_cache_mb": 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_teacher_confidence_config(config)
        for step in (-1, True, 3.5):
            with self.assertRaises(ValueError):
                teacher_confidence_strength(step, cfg)
        for frame_id, camera_id in ((-1, 0), (0, True), (1.5, 0)):
            with self.assertRaises(ValueError):
                self.moments().update(self.frame(), frame_id, camera_id)
        for change in ("missing", "shape", "mask", "singular", "camera_nan"):
            packet = self.frame()
            if change == "missing":
                packet.pop("opacity")
            elif change == "shape":
                packet["gbuffer"]["roughness"] = packet["gbuffer"]["roughness"][..., 0]
            elif change == "mask":
                packet["static_mask"] = packet["static_mask"].float()
            elif change == "singular":
                packet["intrinsics"].zero_()
            else:
                packet["camera_to_world"][0, 0] = float("nan")
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.moments().update(packet, 0, 0)


class ConfidenceNormalizationTests(unittest.TestCase):
    def fixture(self):
        confidence = torch.tensor([[[.25], [1.], [.5]], [[.75], [.25], [.9]]], requires_grad=True)
        static = torch.tensor([[True, True, False], [True, True, False]])
        surface = torch.tensor([[[1.], [1.], [1.]], [[1.], [0.], [0.]]])
        return confidence, static, surface

    def test_static_loss_mass_and_dynamic_supervision_are_preserved(self):
        confidence, static, surface = self.fixture()
        weights = normalized_teacher_confidence(confidence, static, surface, 1.)
        eligible = static & surface[..., 0].bool()
        self.assertAlmostEqual(float(weights[..., 0][eligible].mean()), 1., places=7)
        self.assertTrue((weights[..., 0][~eligible] == 1).all())
        self.assertLess(float(weights[0, 0, 0]), float(weights[0, 1, 0]))
        self.assertFalse(weights.requires_grad)
        # A constant material residual keeps its original objective magnitude.
        residual = torch.full((2, 3, 3), .3, requires_grad=True)
        weighted = (surface * weights * residual).sum() / (surface.sum() * 3)
        ordinary = (surface * residual).sum() / (surface.sum() * 3)
        torch.testing.assert_close(weighted, ordinary)
        weighted.backward()
        self.assertIsNone(confidence.grad)
        torch.testing.assert_close(residual.grad[0, 2], torch.full((3,), 1 / 12))

    def test_warmup_empty_uniform_and_half_normalization(self):
        confidence, static, surface = self.fixture()
        for mask, strength in ((static, 0.), (torch.zeros_like(static), 1.)):
            weights = normalized_teacher_confidence(confidence, mask, surface, strength)
            self.assertTrue((weights == 1).all())
        constant = torch.full((2, 3, 1), .25)
        self.assertTrue((normalized_teacher_confidence(constant, static, surface, 1.) == 1).all())
        # Large half-precision reductions must not overflow 65,504.
        large = torch.full((256, 320, 1), .5, dtype=torch.float16)
        weights = normalized_teacher_confidence(large, torch.ones(256, 320, dtype=torch.bool),
                                                 torch.ones_like(large), 1.)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertTrue((weights == 1).all())

    def test_ramp_is_applied_before_normalization_and_inputs_are_unchanged(self):
        confidence, static, surface = self.fixture()
        before = confidence.detach().clone()
        weights = normalized_teacher_confidence(confidence, static, surface, .5)
        # Eligible raw ramped values .625,1,.875 sum to 2.5.
        self.assertAlmostEqual(float(weights[0, 0, 0]), .625 * 3 / 2.5, places=7)
        self.assertAlmostEqual(float(weights[0, 1, 0]), 3 / 2.5, places=7)
        torch.testing.assert_close(confidence, before, atol=0, rtol=0)

    def test_invalid_normalization_inputs_fail(self):
        confidence, static, surface = self.fixture()
        for strength in (-1., 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                normalized_teacher_confidence(confidence, static, surface, strength)
        for invalid in (torch.zeros_like(confidence), torch.full_like(confidence, float("nan")),
                        torch.full_like(confidence, 1.1)):
            with self.assertRaises(ValueError):
                normalized_teacher_confidence(invalid, static, surface, 1.)
        with self.assertRaises(ValueError):
            normalized_teacher_confidence(confidence, static.float(), surface, 1.)
        with self.assertRaises(ValueError):
            normalized_teacher_confidence(confidence, static, surface[..., 0], 1.)


if __name__ == "__main__":
    unittest.main()
