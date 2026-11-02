"""Geometry, gradient and compute-budget contracts of sparse material pairs."""
import copy
import unittest

import torch

from models.sparse_material_consistency import sparse_material_temporal_loss
from models.temporal_training import material_temporal_loss
from tests import test_temporal_material as temporal_fixture


class SparseMaterialConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def frame(**kwargs):
        return temporal_fixture.TemporalMaterialTests.frame(**kwargs)

    @staticmethod
    def output(frame):
        return {**frame["gbuffer"], "depth": frame["depth"], "opacity": frame["opacity"]}

    def loss(self, current, previous, **kwargs):
        options = dict(weight=1., erode_radius=0, depth_edge_rel=0.)
        options.update(kwargs)
        return sparse_material_temporal_loss(
            self.output(current), current["static_mask"], current["camera_to_world"],
            current["intrinsics"], previous, **options,
        )

    def test_identical_material_has_exact_zero_and_all_surface_points(self):
        # The sRGB endpoint is exactly representable; arbitrary pow inputs can
        # differ by one ULP across CPU SIMD kernels for Nx3 and Nx4x3 tensors.
        current = self.frame(albedo=1.)
        current["gbuffer"]["albedo"].requires_grad_()
        loss, stats = self.loss(current, self.frame(albedo=1.))
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(int(stats["sample_count"]), 35)
        self.assertEqual(int(stats["valid_sample_count"]), 35)
        loss.backward()
        self.assertEqual(float(current["gbuffer"]["albedo"].grad.abs().sum()), 0.)
        self.assertTrue(all(not value.requires_grad for value in stats.values()))

    def test_material_difference_is_not_used_to_reject_correspondence(self):
        current, previous = self.frame(roughness=.9), self.frame(roughness=.1)
        loss, stats = self.loss(current, previous)
        self.assertAlmostEqual(float(loss), .8 / 3, places=6)
        self.assertEqual(int(stats["valid_sample_count"]), 35)

    def test_subpixel_projection_matches_analytic_material_loss(self):
        previous = self.frame(roughness=.2)
        previous["gbuffer"]["roughness"][:] = torch.arange(7)[None, :, None] / 10
        current = self.frame(roughness=.8)
        current["camera_to_world"][0, 3] = .5
        loss, stats = self.loss(current, previous)
        # Previous samples .05,.15,...,.55 average to .3 at 30 valid pixels.
        self.assertAlmostEqual(float(loss), (.8 - .3) / 3, places=6)
        self.assertEqual(int(stats["valid_sample_count"]), 30)
        self.assertLess(float(stats["cycle_error_px"]), 1e-6)

    def test_albedo_is_linearized_before_previous_interpolation(self):
        previous = self.frame()
        previous["gbuffer"]["albedo"][:] = (torch.arange(7)[None, :, None] / 6).expand(5, 7, 3)
        current = self.frame(albedo=.95)
        current["camera_to_world"][0, 3] = .5
        loss, stats = self.loss(current, previous)
        values = torch.arange(7) / 6
        linear = lambda x: torch.where(x <= .04045, x / 12.92, ((x + .055) / 1.055).pow(2.4))
        correct_target = .5 * (linear(values[:-1]) + linear(values[1:]))
        expected = (linear(torch.tensor(.95)) - correct_target).abs().mean() / 3
        incorrect = (linear(torch.tensor(.95)) - linear(.5 * (values[:-1] + values[1:]))).abs().mean() / 3
        torch.testing.assert_close(loss, expected)
        self.assertGreater(float((loss - incorrect).abs()), .001)

    def test_only_current_material_receives_gradient(self):
        current, previous = self.frame(roughness=.8), self.frame(roughness=.2)
        for frame in (current, previous):
            for value in frame["gbuffer"].values():
                value.requires_grad_()
            for key in ("depth", "opacity", "camera_to_world", "intrinsics"):
                frame[key].requires_grad_()
        loss, _ = self.loss(current, previous, weight=.05)
        loss.backward()
        self.assertAlmostEqual(float(current["gbuffer"]["roughness"].grad.sum()), .05 / 3, places=7)
        for value in previous["gbuffer"].values():
            self.assertIsNone(value.grad)
        for frame in (current, previous):
            self.assertIsNone(frame["gbuffer"]["normal"].grad)
            for key in ("depth", "opacity", "camera_to_world", "intrinsics"):
                self.assertIsNone(frame[key].grad)

    def test_sparse_matches_legacy_dense_on_clean_correspondence(self):
        previous = self.frame(roughness=.2)
        previous["gbuffer"]["roughness"][:] = torch.arange(7)[None, :, None] / 10
        current = self.frame(roughness=.8)
        current["camera_to_world"][0, 3] = .5
        sparse, _ = self.loss(current, previous, erode_radius=1, depth_edge_rel=.05)
        dense = material_temporal_loss(self.output(current), current["static_mask"],
                                       current["camera_to_world"], current["intrinsics"], previous)
        torch.testing.assert_close(sparse, dense)

    def test_camera_z_translation_compares_projected_depth(self):
        current, previous = self.frame(roughness=.8), self.frame(roughness=.2)
        current["camera_to_world"][2, 3] = 1.
        previous["depth"].fill_(3.)
        loss, stats = self.loss(current, previous)
        self.assertGreater(int(stats["valid_sample_count"]), 0)
        self.assertAlmostEqual(float(loss), .2, places=6)

    def test_camera_rotation_preserves_world_normal_and_analytic_correspondence(self):
        current, previous = self.frame(roughness=.8), self.frame(roughness=.2)
        previous["gbuffer"]["roughness"][:] = torch.arange(7)[None, :, None] / 10
        current["camera_to_world"][:3, :3] = torch.tensor([[0., -1., 0.],
                                                        [1., 0., 0.],
                                                        [0., 0., 1.]])
        current["gbuffer"]["roughness"].requires_grad_()
        loss, stats = self.loss(current, previous)
        # A quarter-turn maps rows onto columns 1..5, with mean roughness .3.
        self.assertAlmostEqual(float(loss.detach()), (.8 - .3) / 3, places=6)
        self.assertEqual(int(stats["valid_sample_count"]), 25)
        self.assertLess(float(stats["cycle_error_px"]), 1e-6)
        loss.backward()
        gradient = current["gbuffer"]["roughness"].grad[..., 0]
        self.assertTrue(bool((gradient[:, 1:-1] > 0).all()))
        self.assertEqual(float(gradient[:, [0, -1]].abs().sum()), 0.)

    def test_depth_occlusion_and_opacity_reject_pair(self):
        for field, value in (("depth", 4.), ("opacity", .5)):
            current, previous = self.frame(roughness=.8), self.frame(roughness=.2)
            current["gbuffer"]["roughness"].requires_grad_()
            previous[field].fill_(value)
            loss, stats = self.loss(current, previous)
            self.assertEqual(float(loss.detach()), 0.)
            self.assertEqual(int(stats["valid_sample_count"]), 0)
            loss.backward()
            self.assertEqual(float(current["gbuffer"]["roughness"].grad.abs().sum()), 0.)

    def test_world_normal_orientation_is_sign_invariant(self):
        for normal, expected in (([0., 0., -1.], 35), ([1., 0., 0.], 0)):
            current, previous = self.frame(roughness=.8), self.frame(roughness=.2)
            previous["gbuffer"]["normal"][:] = torch.tensor(normal)
            _, stats = self.loss(current, previous)
            self.assertEqual(int(stats["valid_sample_count"]), expected)

    def test_roundtrip_rejects_incorrect_depth_even_with_loose_depth_gate(self):
        current, previous = self.frame(roughness=.8), self.frame(roughness=.2)
        current["camera_to_world"][0, 3] = 1.
        previous["depth"].fill_(4.)
        loss, stats = self.loss(current, previous, depth_abs=10., cycle_px=.1)
        self.assertGreater(int(stats["normal_count"]), 0)
        self.assertEqual(int(stats["cycle_count"]), 0)
        self.assertEqual(float(loss), 0.)
        relaxed, _ = self.loss(current, previous, depth_abs=10., cycle_px=1.)
        self.assertGreater(float(relaxed), 0.)

    def test_complete_bilinear_footprint_required_and_no_row_wrap(self):
        current, previous = self.frame(roughness=.8), self.frame(roughness=.2)
        current["gbuffer"]["roughness"].requires_grad_()
        current["camera_to_world"][0, 3] = .5
        previous["static_mask"][2, 3] = False
        loss, stats = self.loss(current, previous)
        loss.backward()
        gradient = current["gbuffer"]["roughness"].grad[..., 0]
        self.assertEqual(int(stats["valid_sample_count"]), 28)
        self.assertEqual(float(gradient[2, 2:4].abs().sum()), 0.)
        self.assertEqual(float(gradient[:, -1].abs().sum()), 0.)
        self.assertGreater(float(gradient[1, 1]), 0.)

    def test_static_erode_excludes_dynamic_edges_and_image_border(self):
        current, previous = self.frame(roughness=.8), self.frame(roughness=.2)
        current["static_mask"][2, 3] = False
        current["gbuffer"]["roughness"].requires_grad_()
        loss, stats = self.loss(current, previous, erode_radius=1)
        loss.backward()
        gradient = current["gbuffer"]["roughness"].grad[..., 0]
        self.assertEqual(int(stats["available_count"]), 6)
        self.assertEqual(float(gradient[1:4, 2:5].abs().sum()), 0.)
        self.assertEqual(float(gradient[[0, -1]].abs().sum()), 0.)

    def test_depth_edge_mask_does_not_join_two_layers(self):
        current, previous = self.frame(roughness=.8), self.frame(roughness=.2)
        for frame in (current, previous):
            frame["depth"][:, 4:] = 5.
        _, stats = self.loss(current, previous, depth_edge_rel=.1)
        self.assertEqual(int(stats["available_count"]), 25)

    def test_sampling_is_bounded_deterministic_and_global_rng_unchanged(self):
        current, previous = self.frame(h=31, w=43, roughness=.8), self.frame(h=31, w=43, roughness=.2)
        previous["gbuffer"]["roughness"][:] = torch.linspace(.1, .6, 43)[None, :, None]
        rng = torch.get_rng_state().clone()
        first, stats = self.loss(current, previous, num_samples=31, seed=91, step=88, sample_key=7)
        second, other = self.loss(current, previous, num_samples=31, seed=91, step=88, sample_key=7)
        changed, _ = self.loss(current, previous, num_samples=31, seed=91, step=89, sample_key=7)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertNotEqual(float(first), float(changed))
        self.assertEqual(int(stats["sample_count"]), 31)
        self.assertEqual(int(stats["valid_sample_count"]), 31)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)

    def test_nan_inf_invalid_materials_do_not_poison_loss_or_gradient(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            current, previous = self.frame(roughness=.8), self.frame(roughness=.2)
            current["gbuffer"]["albedo"][1, 1] = invalid
            previous["gbuffer"]["roughness"][2, 2] = invalid
            for value in current["gbuffer"].values():
                value.requires_grad_()
            loss, _ = self.loss(current, previous)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            for key in ("albedo", "roughness", "metallic"):
                self.assertTrue(bool(torch.isfinite(current["gbuffer"][key].grad).all()))
            self.assertEqual(float(current["gbuffer"]["albedo"].grad[1, 1].abs().sum()), 0.)
            self.assertEqual(float(current["gbuffer"]["roughness"].grad[2, 2].abs().sum()), 0.)

    def test_missing_pair_and_empty_current_support_are_graph_safe(self):
        for missing, no_support in ((True, False), (False, True)):
            current = self.frame(roughness=.8)
            current["gbuffer"]["roughness"].requires_grad_()
            if no_support:
                current["static_mask"].zero_()
            loss, stats = self.loss(current, None if missing else self.frame(roughness=.2))
            self.assertEqual(float(loss.detach()), 0.)
            loss.backward()
            self.assertEqual(float(current["gbuffer"]["roughness"].grad.abs().sum()), 0.)
            self.assertEqual(int(stats["sample_count"]), 0)

    def test_disabled_loss_does_not_require_geometry(self):
        current = {"albedo": torch.ones(2, 2, 3, requires_grad=True)}
        loss, _ = sparse_material_temporal_loss(current, None, None, None, {}, weight=0.)
        loss.backward()
        self.assertEqual(float(current["albedo"].grad.abs().sum()), 0.)

    def test_bad_options_shapes_masks_and_singular_cameras_fail(self):
        current, previous = self.frame(), self.frame()
        for invalid in ({"weight": float("nan")}, {"num_samples": 0}, {"step": True},
                        {"num_samples": 1.2}, {"cycle_px": -1}, {"min_opacity": 1.1},
                        {"normal_cos": 0}, {"erode_radius": -1}, {"sample_key": .2}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.loss(current, previous, **invalid)
        malformed = copy.deepcopy(previous)
        malformed["gbuffer"]["roughness"] = torch.zeros(5, 7)
        with self.assertRaisesRegex(ValueError, "roughness"):
            self.loss(current, malformed)
        malformed = copy.deepcopy(previous)
        malformed["static_mask"] = malformed["static_mask"].float()
        with self.assertRaisesRegex(ValueError, "static mask"):
            self.loss(current, malformed)
        for location in ("camera_to_world", "intrinsics"):
            malformed = copy.deepcopy(previous)
            malformed[location].zero_()
            with self.assertRaisesRegex(ValueError, "invertible"):
                self.loss(current, malformed)


if __name__ == "__main__":
    unittest.main()
