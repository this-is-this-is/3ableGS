"""Meaningful invariants of sampled teacher-guided material regularization."""
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.diffreg_material import (
    diffreg_weight, material_relation_loss, validate_diffreg_config,
)


ACTIVE = {"enabled": True, "w": 1., "start_step": 0, "ramp_steps": 0,
          "num_samples": 4096, "radius": 1}


def maps(height=4, width=5, *, grad=False, dtype=torch.float64):
    return {key: torch.full((height, width, channels), .5, dtype=dtype, requires_grad=grad)
            for key, channels in (("albedo", 3), ("roughness", 1), ("metallic", 1))}


def evaluate(prediction, teacher=None, valid=None, **kwargs):
    height, width = prediction["albedo"].shape[:2]
    if teacher is None:
        teacher = maps(height, width)
    if valid is None:
        valid = torch.ones(height, width, dtype=torch.bool)
    return material_relation_loss(prediction, teacher, valid,
                                  step=kwargs.pop("step", 6000),
                                  config=kwargs.pop("config", ACTIVE), **kwargs)


class DiffRegMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_config_rejects_invalid_or_silent_settings(self):
        bad = [{"enabled": 1}, {"num_samples": 0}, {"radius": 0}, {"radius": True},
               {"start_step": -1}, {"ramp_steps": 1.2}, {"seed": -1},
               {"w": float("nan")}, {"w": -1}, {"w": True}, {"sigma_g": 0},
               {"sigma_xy": float("inf")}, {"normal_cos": 1.01}, {"eps": 1},
               {"eps": 0}, {"depth_abs": -1}, {"typo": 1}]
        for cfg in bad:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_diffreg_config(cfg)
        for cfg in (False, [], 3):
            with self.assertRaises(ValueError):
                validate_diffreg_config(cfg)
        original = {"enabled": True}
        result = validate_diffreg_config(original)
        result["enabled"] = False
        self.assertTrue(original["enabled"])

    def test_ramp_and_disabled_paths_have_no_required_maps(self):
        cfg = {"enabled": True}
        for step, expected in ((2999, 0), (3000, 0), (3500, .025), (4000, .05)):
            self.assertEqual(diffreg_weight(step, cfg), expected)
        self.assertEqual(diffreg_weight(30000, {}), 0)
        with patch("models.diffreg_material._validate_maps", side_effect=AssertionError("unused")):
            for config, step in (({}, 30000), (cfg, 3000), ({**ACTIVE, "w": 0}, 30000)):
                loss, diagnostic = material_relation_loss(None, None, None, step=step, config=config)
                self.assertEqual(loss.item(), 0)
                self.assertEqual(diagnostic, {})
        self.assertEqual(diffreg_weight(3000, {**cfg, "ramp_steps": 0}), .05)
        for step in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                diffreg_weight(step, cfg)

    def test_constant_material_is_exact_zero_with_nonconstant_teacher(self):
        prediction = maps(7, 9, grad=True)
        teacher = maps(7, 9)
        teacher["roughness"][:, 4:] = .57
        loss, diagnostic = evaluate(prediction, teacher)
        self.assertEqual(loss.item(), 0)
        self.assertGreater(diagnostic["valid_sample_count"].item(), 0)
        loss.backward()
        for value in prediction.values():
            self.assertTrue(torch.equal(value.grad, torch.zeros_like(value)))

    def test_varying_material_has_finite_gradients_only_to_predictions(self):
        prediction = maps(4, 5, grad=True)
        teacher = maps(4, 5, grad=True)
        with torch.no_grad():
            for value in prediction.values():
                value[:, 2:] = .8
        depth = torch.full((4, 5, 1), 5., dtype=torch.float64, requires_grad=True)
        normal = torch.zeros(4, 5, 3, dtype=torch.float64)
        normal[..., 2] = 1.
        normal.requires_grad_()
        valid = torch.ones(4, 5, dtype=torch.float64, requires_grad=True)
        loss, diagnostic = evaluate(prediction, teacher, valid, depth=depth, normal=normal)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        for value in prediction.values():
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for value in list(teacher.values()) + [depth, normal, valid]:
            self.assertIsNone(value.grad)
        self.assertTrue(all(not value.requires_grad for value in diagnostic.values()))

    def test_both_center_and_neighbor_receive_gradients(self):
        prediction = maps(1, 3, grad=True)
        with torch.no_grad():
            prediction["roughness"][0, 1] = .9
        with patch("models.diffreg_material._sample_indices", return_value=torch.tensor([0])):
            loss, _ = evaluate(prediction)
        loss.backward()
        self.assertGreater(prediction["roughness"].grad[0, 0].abs().item(), 0)
        self.assertGreater(prediction["roughness"].grad[0, 1].abs().item(), 0)
        self.assertEqual(prediction["roughness"].grad[0, 2].item(), 0)

    def test_joint_guide_preserves_material_boundary_in_other_channel(self):
        prediction = maps(1, 2, grad=True)
        with torch.no_grad():
            prediction["albedo"][0, 0] = .2
            prediction["albedo"][0, 1] = .8
        guide = maps(1, 2)
        unguarded, _ = evaluate(prediction, guide)
        guide["roughness"][0, 0] = 0
        guide["roughness"][0, 1] = 1
        guarded, diagnostics = evaluate(prediction, guide, config={**ACTIVE, "sigma_g": .01})
        self.assertGreater(unguarded.item(), .01)
        self.assertEqual(guarded.item(), 0)
        self.assertEqual(diagnostics["valid_sample_count"].item(), 0)

    def test_zero_support_is_graph_connected_and_finite(self):
        for shape, valid_value in (((1, 1), True), ((5, 6), False)):
            prediction = maps(*shape, grad=True)
            loss, diagnostics = evaluate(prediction, valid=torch.full(shape, valid_value))
            self.assertEqual(loss.item(), 0)
            self.assertTrue(loss.requires_grad)
            self.assertEqual(diagnostics["valid_sample_count"].item(), 0)
            loss.backward()
            for value in prediction.values():
                self.assertEqual(value.grad.abs().sum().item(), 0)

    def test_invalid_pixels_and_nans_do_not_leak_or_get_gradients(self):
        prediction = maps(3, 5, grad=True)
        guide = maps(3, 5)
        valid = torch.ones(3, 5, dtype=torch.bool)
        valid[:, 2] = False
        with torch.no_grad():
            for value in prediction.values():
                value[:, 2] = float("nan")
            guide["albedo"][:, 2] = float("nan")
        loss, _ = evaluate(prediction, guide, valid)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        for value in prediction.values():
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertEqual(value.grad.abs().sum().item(), 0)

    def test_stencils_do_not_wrap_across_image_edges(self):
        prediction = maps(2, 4, grad=True)
        valid = torch.zeros(2, 4, dtype=torch.bool)
        valid[0, 3] = valid[1, 0] = True
        with torch.no_grad():
            for value in prediction.values():
                value[1, 0] = .9
        loss, diagnostics = evaluate(prediction, valid=valid)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(diagnostics["valid_sample_count"].item(), 0)

    def test_depth_and_normal_boundaries_reject_false_relations(self):
        prediction = maps(1, 2)
        prediction["roughness"][0, 1] = .9
        baseline, _ = evaluate(prediction)
        self.assertGreater(baseline.item(), 0)
        depth = torch.tensor([[[1.], [2.]]])
        loss, _ = evaluate(prediction, depth=depth)
        self.assertEqual(loss.item(), 0)
        orthogonal = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        loss, _ = evaluate(prediction, normal=orthogonal)
        self.assertEqual(loss.item(), 0)
        opposite = torch.tensor([[[1., 0., 0.], [-1., 0., 0.]]])
        loss, _ = evaluate(prediction, normal=opposite)
        torch.testing.assert_close(loss, baseline)

    def test_log_albedo_does_not_collapse_equal_a_log_a_values(self):
        # .25*log(.25) == .5*log(.5): applying A*log(A) before JBF would
        # erase this real material difference. Official code weights AFTER JBF.
        prediction = maps(1, 2, grad=True)
        with torch.no_grad():
            prediction["albedo"][0, 0] = .25
        values = torch.tensor([.25, .5], dtype=torch.float64)
        transformed = values * values.log()
        self.assertEqual(transformed[0].item(), transformed[1].item())
        loss, diagnostics = evaluate(prediction)
        self.assertGreater(diagnostics["albedo"].item(), .01)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(prediction["albedo"].grad.abs().sum().item(), 0)

    def test_albedo_compensation_is_detached_and_has_expected_gradient(self):
        prediction = maps(1, 2, grad=True)
        with torch.no_grad():
            prediction["albedo"][0, 0] = .25
        with patch("models.diffreg_material._sample_indices", return_value=torch.tensor([0])):
            loss, _ = evaluate(prediction)
        loss.backward()
        # For one anchor, w1=exp(-1/8)/(1+exp(-1/8)); channel and material
        # averaging divides each albedo channel by 9. Detached .25 cancels
        # the derivative 1/.25 at the center (no log-dependent extra term).
        neighbor_weight = np.exp(-1 / 8) / (1 + np.exp(-1 / 8))
        self.assertAlmostEqual(prediction["albedo"].grad[0, 0, 0].item(), -neighbor_weight / 9)
        self.assertAlmostEqual(prediction["albedo"].grad[0, 1, 0].item(), neighbor_weight / 18)

    def test_albedo_gradient_has_no_uniform_scale_shrink_direction(self):
        prediction = maps(4, 5, grad=True)
        with torch.no_grad():
            prediction["albedo"][:, 2:] = .8
        loss, _ = evaluate(prediction)
        loss.backward()
        # A multiplicative perturbation A -> (1+d)A adds a constant in log
        # space. The detached confidence must not create a shrinkage gradient.
        directional = (prediction["albedo"].detach() * prediction["albedo"].grad).sum()
        self.assertAlmostEqual(directional.item(), 0., places=13)

    def test_sampling_is_repeatable_without_changing_global_rng(self):
        prediction = maps(25, 27)
        for value in prediction.values():
            value[:, 7:11] = .9
        cfg = {**ACTIVE, "num_samples": 19}
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        one, d1 = evaluate(prediction, config=cfg, step=6000, sample_key=17)
        two, d2 = evaluate(prediction, config=cfg, step=6000, sample_key=17)
        torch.testing.assert_close(one, two, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        actual_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], actual_numpy[0])
        np.testing.assert_array_equal(numpy_state[1], actual_numpy[1])
        self.assertEqual(numpy_state[2:], actual_numpy[2:])
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(d1["sample_count"].item(), 19)
        for key in d1:
            torch.testing.assert_close(d1[key], d2[key], rtol=0, atol=0)

    def test_invalid_shapes_and_keys_are_rejected(self):
        prediction = maps()
        teacher = maps()
        with self.assertRaises(ValueError):
            evaluate(prediction, {"albedo": teacher["albedo"]})
        with self.assertRaises(ValueError):
            evaluate(prediction, valid=torch.ones(2, 3))
        with self.assertRaises(ValueError):
            evaluate(prediction, normal=torch.ones(4, 5))
        with self.assertRaises(ValueError):
            evaluate(prediction, sample_key=True)
        with self.assertRaises(ValueError):
            evaluate(prediction, {**teacher, "roughness": torch.ones(4, 5)})

    def test_half_precision_uses_finite_float_computation(self):
        prediction = maps(3, 4, grad=True, dtype=torch.float16)
        teacher = maps(3, 4, dtype=torch.float16)
        with torch.no_grad():
            prediction["albedo"][:, 1:] = .001
            prediction["roughness"][:, 1:] = .8
        loss, _ = evaluate(prediction, teacher)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in prediction.values()))


if __name__ == "__main__":
    unittest.main()
