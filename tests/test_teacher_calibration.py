"""CPU invariants for the bounded train-view teacher calibration module."""
import copy
import random
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from models.teacher_calibration import (
    TeacherCalibration, teacher_calibration_strength, validate_teacher_calibration_config,
)


ACTIVE = {"enabled": True, "start_step": 0, "ramp_steps": 0}


def maps(grad=False):
    return {name: torch.full((3, 4, channels), .5, dtype=torch.float64, requires_grad=grad)
            for name, channels in (("albedo", 3), ("roughness", 1), ("metallic", 1))}


class TeacherCalibrationTests(unittest.TestCase):
    def head(self, ids=(0, 1, 2), cameras=1, config=None):
        return TeacherCalibration(ids, cameras, ACTIVE if config is None else config).double()

    def apply(self, head, teacher=None, valid=None, image_idx=0, step=0):
        teacher = maps() if teacher is None else teacher
        valid = torch.ones(3, 4, dtype=torch.bool) if valid is None else valid
        return head.correct(teacher, valid, image_idx, step)

    def test_config_validation_and_copy(self):
        defaults = validate_teacher_calibration_config()
        self.assertFalse(defaults["enabled"])
        self.assertEqual(defaults["max_gain_delta"], .15)
        self.assertEqual(defaults["min_opacity"], .95)
        cfg = OmegaConf.create({"enabled": True, "max_bias": .03})
        result = validate_teacher_calibration_config(cfg)
        result["max_bias"] = .08
        self.assertEqual(cfg.max_bias, .03)
        for invalid in (False, [], "enabled", {"enabled": 1}, {"start_step": True},
                        {"start_step": -1}, {"ramp_steps": -.1}, {"max_gap": 0},
                        {"max_gap": 1.2}, {"lr": 0}, {"lr": float("nan")},
                        {"identity_w": -1}, {"smooth_w": True}, {"max_bias": 1.1},
                        {"max_gain_delta": 1}, {"min_opacity": -1}, {"min_opacity": 1.1},
                        {"max_bias": float("inf")}, {"unknown": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_teacher_calibration_config(invalid)

    def test_start_ramp_and_disabled_strength(self):
        for step, value in ((0, 0.), (2999, 0.), (3000, 0.), (3500, .5), (4000, 1.), (30000, 1.)):
            self.assertEqual(teacher_calibration_strength(step, {"enabled": True}), value)
        self.assertEqual(teacher_calibration_strength(30000), 0.)
        self.assertEqual(teacher_calibration_strength(3000, {"enabled": True, "ramp_steps": 0}), 1.)
        for step in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                teacher_calibration_strength(step, ACTIVE)

    def test_split_is_sorted_unique_and_metadata_is_a_copy(self):
        head = self.head(ids=(3, 0, 1))
        self.assertEqual(head.train_indices.tolist(), [0, 1, 3])
        self.assertEqual(tuple(head.raw.shape), (3, 2, 5))
        metadata = head.metadata()
        self.assertEqual(metadata["train_indices"], [0, 1, 3])
        self.assertEqual(metadata["num_cams"], 1)
        metadata["train_indices"].append(10)
        metadata["config"]["enabled"] = False
        self.assertTrue(head.config["enabled"])
        self.assertEqual(head.metadata()["train_indices"], [0, 1, 3])
        for ids in ([], [1, 1], [-1, 1], [0, True], [0., 1.], "0,1", {0: 1},
                    torch.tensor([[0, 1]]), torch.tensor([0., 1.])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.head(ids=ids)
        for cameras in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.head(cameras=cameras)

    def test_construction_and_correction_preserve_all_rng_streams(self):
        random.seed(902)
        np.random.seed(902)
        torch.manual_seed(902)
        py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        head = self.head(ids=(0, 1, 2, 3), cameras=2)
        self.apply(head)
        self.assertEqual(random.getstate(), py)
        np.testing.assert_equal(np.random.get_state(), numpy)
        torch.testing.assert_close(torch.get_rng_state(), cpu, atol=0, rtol=0)

    def test_initialization_is_exact_identity_without_teacher_gradient(self):
        teacher = maps(grad=True)
        teacher["albedo"] = torch.linspace(0, 1, 36, dtype=torch.float64).reshape(3, 4, 3).requires_grad_()
        before = {key: value.detach().clone() for key, value in teacher.items()}
        corrected, losses, stats = self.apply(self.head(), teacher)
        for key in teacher:
            torch.testing.assert_close(corrected[key], before[key], atol=0, rtol=0)
            torch.testing.assert_close(teacher[key], before[key], atol=0, rtol=0)
        self.assertEqual(set(losses), {"teacher_calibration_identity_loss", "teacher_calibration_smooth_loss"})
        self.assertTrue(all(float(value.detach()) == 0. for value in losses.values()))
        self.assertEqual(float(stats["correction_abs_mean"]), 0.)
        self.assertTrue(all(not value.requires_grad for value in stats.values()))
        sum(value.sum() for value in corrected.values()).backward()
        self.assertTrue(all(value.grad is None for value in teacher.values()))

    def test_disabled_warmup_and_eval_leave_maps_detached_and_parameters_unconnected(self):
        for config, step, training in (({}, 4000, True), ({"enabled": True}, 3000, True),
                                       (ACTIVE, 0, False)):
            head = self.head(config=config)
            head.train(training)
            with torch.no_grad():
                head.raw[0].fill_(1.)
            teacher = maps(grad=True)
            corrected, losses, stats = self.apply(head, teacher, step=step)
            self.assertFalse(losses)
            self.assertFalse(stats)
            for key in teacher:
                torch.testing.assert_close(corrected[key], teacher[key], atol=0, rtol=0)
                self.assertFalse(corrected[key].requires_grad)
            self.assertIsNone(head.raw.grad)

    def test_camera_centered_bounds_and_single_view_camera(self):
        head = self.head(ids=(0, 1, 2, 4, 5, 6, 9), cameras=3)
        with torch.no_grad():
            head.raw.copy_(torch.arange(head.raw.numel(), dtype=torch.float64).reshape_as(head.raw) - 30.)
        gain, bias = head.coefficients(0)
        self.assertTrue(bool(((gain >= .85) & (gain <= 1.15)).all()))
        self.assertTrue(bool((bias.abs() <= .05).all()))
        for camera in (0, 1, 2):
            selected = head.train_indices.remainder(3) == camera
            torch.testing.assert_close(gain[selected].mean(0), torch.ones(5, dtype=torch.float64), atol=1e-15, rtol=0)
            torch.testing.assert_close(bias[selected].mean(0), torch.zeros(5, dtype=torch.float64), atol=1e-15, rtol=0)
        # Camera 2 has frames 0 and 1 in this fixture; a genuinely lone view
        # cannot introduce camera-wide material scale drift.
        single = self.head(ids=(0, 1, 2), cameras=3)
        with torch.no_grad():
            single.raw.fill_(3.)
        gain, bias = single.coefficients(0)
        self.assertTrue(torch.equal(gain, torch.ones_like(gain)))
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))

    def test_analytic_affine_transform_and_ramp(self):
        head = self.head(ids=(0, 1), config={"enabled": True, "start_step": 0, "ramp_steps": 10})
        bounded = torch.tensor([[.4, -.2, .6, .8, -.6], [.2, .4, -.2, -.4, .6]], dtype=torch.float64)
        with torch.no_grad():
            head.raw[0] = bounded.atanh()
            head.raw[1] = -bounded.atanh()
        corrected, _, stats = self.apply(head, step=5)
        expected = .5 + .5 * .15 * (.5 * bounded[0]) * .5 + .5 * .05 * (.5 * bounded[1])
        packed = torch.cat([corrected[key][0, 0] for key in ("albedo", "roughness", "metallic")])
        torch.testing.assert_close(packed, expected, atol=1e-15, rtol=0)
        self.assertEqual(float(stats["strength"]), .5)

    def test_parameter_edges_use_same_camera_train_neighbors_and_frame_gaps(self):
        head = self.head(ids=(0, 1, 2, 3, 6, 9, 16), cameras=2)
        self.assertEqual(head.edge_left.tolist(), [0, 2, 1])
        self.assertEqual(head.edge_right.tolist(), [2, 4, 3])
        self.assertEqual(head.edge_gaps.tolist(), [1, 2, 1])

    def test_smooth_penalty_divides_by_elapsed_frames_and_does_not_cross_camera(self):
        head = self.head(ids=(0, 1, 4, 5), cameras=2)
        with torch.no_grad():
            head.raw[0].fill_(math_atanh(.4))
            head.raw[2].fill_(-math_atanh(.4))
            head.raw[1].fill_(math_atanh(.8))
            head.raw[3].fill_(math_atanh(.8))
        _, losses, stats = self.apply(head)
        # Camera 0: normalized coefficients +/- .2, frame gap 2, squared .04.
        # Camera 1: common coefficient shift gauges to zero. Average edges=.02.
        self.assertAlmostEqual(float(stats["smooth_unweighted"]), .02, places=12)
        self.assertAlmostEqual(float(losses["teacher_calibration_smooth_loss"].detach()), .0002, places=12)
        # Identity prior still penalizes a common raw offset removed by gauge.
        self.assertGreater(float(stats["identity_unweighted"]), 0.)

    def test_empty_edge_set_has_graph_safe_zero_smoothness(self):
        head = self.head(ids=(0, 10))
        _, losses, _ = self.apply(head)
        self.assertEqual(head.edge_left.numel(), 0)
        loss = losses["teacher_calibration_smooth_loss"]
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(head.raw.grad, torch.zeros_like(head.raw)))

    def test_only_static_pixels_affect_data_gradients_and_teacher_is_detached(self):
        head = self.head(ids=(0, 1))
        teacher = maps(grad=True)
        valid = torch.zeros(3, 4, dtype=torch.bool)
        valid[1, 2] = True
        corrected, _, _ = self.apply(head, teacher, valid)
        ignored = sum(value[~valid].sum() for value in corrected.values())
        ignored.backward(retain_graph=True)
        self.assertTrue(torch.equal(head.raw.grad, torch.zeros_like(head.raw)))
        head.raw.grad = None
        sum(value[valid].sum() for value in corrected.values()).backward()
        self.assertGreater(float(head.raw.grad.abs().sum()), 0.)
        self.assertTrue(all(value.grad is None for value in teacher.values()))

    def test_invalid_excluded_pixels_remain_unchanged_with_finite_gradients(self):
        head = self.head(ids=(0, 1))
        teacher = maps()
        teacher["albedo"][0, 0] = float("nan")
        teacher["roughness"][0, 1] = float("inf")
        valid = torch.ones(3, 4, dtype=torch.bool)
        valid[0, :2] = False
        corrected, _, stats = self.apply(head, teacher, valid)
        self.assertTrue(bool(torch.isnan(corrected["albedo"][0, 0]).all()))
        self.assertTrue(bool(torch.isinf(corrected["roughness"][0, 1]).all()))
        sum(torch.nan_to_num(value, nan=0., posinf=0., neginf=0.).sum() for value in corrected.values()).backward()
        self.assertTrue(bool(torch.isfinite(head.raw.grad).all()))
        self.assertTrue(all(bool(torch.isfinite(value)) for value in stats.values()))

    def test_empty_support_is_identity_with_zero_data_gradient(self):
        head = self.head(ids=(0, 1))
        with torch.no_grad():
            head.raw[0].fill_(1.)
        teacher = maps()
        corrected, losses, stats = self.apply(head, teacher, torch.zeros(3, 4, dtype=torch.bool))
        for key in teacher:
            torch.testing.assert_close(corrected[key], teacher[key], atol=0, rtol=0)
        self.assertEqual(float(stats["valid_fraction"]), 0.)
        self.assertEqual(float(stats["clamp_fraction"]), 0.)
        self.assertEqual(float(stats["correction_abs_mean"]), 0.)
        sum(value.sum() for value in corrected.values()).backward()
        self.assertTrue(torch.equal(head.raw.grad, torch.zeros_like(head.raw)))
        self.assertGreater(float(losses["teacher_calibration_identity_loss"].detach()), 0.)

    def test_clipping_is_reported_and_output_stays_in_bounds(self):
        head = self.head(ids=(0, 1))
        with torch.no_grad():
            head.raw[0].fill_(10.)
            head.raw[1].fill_(-10.)
        teacher = {name: value.fill_(1.) for name, value in maps().items()}
        corrected, _, stats = self.apply(head, teacher)
        self.assertEqual(float(stats["clamp_fraction"]), 1.)
        self.assertTrue(all(bool((value == 1.).all()) for value in corrected.values()))

    def test_shared_material_error_gives_descent_direction_for_opposite_view_biases(self):
        head = self.head(ids=(0, 1), config={**ACTIVE, "max_gain_delta": 0., "max_bias": .1,
                                           "identity_w": 0., "smooth_w": 0.})
        low = {key: value.fill_(.49) for key, value in maps().items()}
        high = {key: value.fill_(.51) for key, value in maps().items()}

        def objective():
            first, _, _ = self.apply(head, low, image_idx=0)
            second, _, _ = self.apply(head, high, image_idx=1)
            return sum(((first[key] - .5).square().mean()
                        + (second[key] - .5).square().mean()) / 6. for key in first)

        before = objective()
        before.backward()
        # One analytic descent probe on tiny synthetic maps, not scene training.
        with torch.no_grad():
            head.raw.add_(head.raw.grad, alpha=-1000.)
        self.assertLess(float(objective().detach()), float(before.detach()))
        gain, bias = head.coefficients(0)
        self.assertTrue(bool((bias[0] > 0.).all()))
        self.assertTrue(bool((bias[1] < 0.).all()))
        torch.testing.assert_close(bias.mean(0), torch.zeros(5, dtype=torch.float64), atol=1e-15, rtol=0)

    def test_heldout_and_invalid_inputs_are_rejected(self):
        head = self.head(ids=(0, 2))
        for image_idx in (1, True, 0., -1):
            with self.assertRaisesRegex(ValueError, "training split"):
                self.apply(head, image_idx=image_idx)
        for invalid in ({}, None, {**maps(), "roughness": torch.zeros(3, 4, dtype=torch.float64)},
                        {**maps(), "metallic": torch.zeros(3, 4, 1, dtype=torch.float32)}):
            with self.assertRaises(ValueError):
                head.correct(invalid, torch.ones(3, 4, dtype=torch.bool), 0, 0)
        for valid in (torch.ones(3, 4), torch.ones(3, 4, 1, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                self.apply(head, valid=valid)
        for invalid in (-.1, 1.1, float("nan"), float("inf")):
            teacher = maps()
            teacher["roughness"][1, 1] = invalid
            with self.assertRaisesRegex(ValueError, "finite and in"):
                self.apply(head, teacher)

    def test_state_round_trip_preserves_coefficients_and_edge_structure(self):
        head = self.head(ids=(0, 2, 4, 8), cameras=2)
        with torch.no_grad():
            head.raw[0, 0, 0] = .7
        saved = copy.deepcopy(head.state_dict())
        self.assertEqual(set(saved), {"raw", "train_indices"})
        restored = self.head(ids=(8, 4, 0, 2), cameras=2)
        restored.load_state_dict(saved, strict=True)
        for current, loaded in zip(head.coefficients(0), restored.coefficients(0)):
            torch.testing.assert_close(current, loaded, atol=0, rtol=0)
        self.assertEqual(restored.metadata(), head.metadata())
        self.assertEqual(restored.edge_gaps.tolist(), head.edge_gaps.tolist())


def math_atanh(value):
    return float(torch.tensor(value, dtype=torch.float64).atanh())


if __name__ == "__main__":
    unittest.main()
