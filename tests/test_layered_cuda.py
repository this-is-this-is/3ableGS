"""Tiny actual-gsplat checks; skipped when CUDA or gsplat is unavailable.

These exercise feature-only rasterization with overlapping Gaussians, including
the trainer's material8+expected-depth baseline and RGB-only auxiliary passes.
They do not load a scene, checkpoint, PBR environment, or teacher model.
"""

from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch

from models.layered_gbuffer import render_layered_gbuffer, srgb_to_linear, unpack_features


def analytic_shader(gbuffer):
    a, m, r, n = (gbuffer[key] for key in ("albedo", "metallic", "roughness", "normal"))
    diffuse = srgb_to_linear(a) * (1 - m) * (.2 + n[..., 2:3].square())
    specular = ((.1 + m) * (1 - r).square() * (.2 + n[..., :1].square())).expand_as(diffuse)
    return {"rgb": diffuse + specular, "diffuse": diffuse,
            "specular": specular, "normal_oriented": n}


class LayeredCUDAIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable; run in the project's GPU environment")
        try:
            from gsplat.rendering import rasterization
        except ImportError as exc:
            raise unittest.SkipTest(f"gsplat is unavailable: {exc}") from exc
        cls.rasterization = staticmethod(rasterization)

    def setUp(self):
        self.device = torch.device("cuda")
        self.features = torch.tensor([
            [.2, .3, .5, .0, .8, 0., 0., 1.],
            [.9, .1, .2, .8, .1, .6, 0., .8],
            [.3, .7, .2, .2, .5, 0., .8, .6],
        ], device=self.device)
        self.ids = torch.arange(3, device=self.device)
        self.geometry = dict(
            means=torch.tensor([[-.08, 0., 3.], [.1, .02, 3.4], [0., -.04, 4.]], device=self.device),
            quats=torch.tensor([[1., 0., 0., 0.]] * 3, device=self.device),
            scales=torch.tensor([[.45, .30, .06]] * 3, device=self.device),
            opacities=torch.tensor([.75, .65, .85], device=self.device),
            viewmats=torch.eye(4, device=self.device)[None],
            Ks=torch.tensor([[[40., 0., 24.], [0., 40., 16.], [0., 0., 1.]]], device=self.device),
            width=48, height=32, packed=True, absgrad=False, sparse_grad=False,
            rasterize_mode="classic", near_plane=.01, far_plane=100., radius_clip=0.,
        )

    def raster_features(self, colors):
        self.assertEqual(colors.shape[0], self.features.shape[0])
        renders, _, _ = self.rasterization(**self.geometry, colors=colors, render_mode="RGB")
        self.assertEqual(renders.shape, (1, 32, 48, colors.shape[-1]))
        return renders[0]

    def baseline(self, features=None):
        renders, alphas, _ = self.rasterization(
            **self.geometry, colors=self.features if features is None else features, render_mode="RGB+ED")
        self.assertEqual(renders.shape[-1], 9)
        return renders[0, ..., :8], alphas[0]

    def test_actual_global_visibility_conserves_material_and_alpha(self):
        raw, alpha = self.baseline()
        self.assertGreater(float(alpha.max()), .7)
        self.assertGreater(int((alpha > .01).sum()), 20)
        for space in ("srgb", "linear"):
            for slots in (0, 2):
                with self.subTest(space=space, slots=slots):
                    result = render_layered_gbuffer(
                        self.features, self.ids, self.raster_features, analytic_shader,
                        color_space=space, max_layers=slots,
                        merged_feature_srgb=raw, merged_opacity=alpha)
                    self.assertLess(result["diagnostics"]["feature_reconstruction_max_abs"], 2e-5)
                    self.assertLess(result["diagnostics"]["mass_reconstruction_max_abs"], 2e-5)
                    self.assertLess(result["diagnostics"]["raster_mass_max_abs"], 2e-5)
                    torch.testing.assert_close(result["opacity"], alpha, rtol=0, atol=0)
                    torch.testing.assert_close(result["merged_feature_srgb"], raw, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(result["linear_surface"]).all())

    def test_actual_single_group_matches_material8_expected_depth_baseline(self):
        raw, alpha = self.baseline()
        result = render_layered_gbuffer(
            self.features, torch.zeros_like(self.ids), self.raster_features, analytic_shader,
            merged_feature_srgb=raw, merged_opacity=alpha)
        expected = analytic_shader(unpack_features(raw, alpha))["rgb"] * alpha
        torch.testing.assert_close(result["linear_surface"], expected, atol=2e-6, rtol=2e-5)

    def test_actual_grouped_shading_matches_rasterized_surface_radiance_without_double_alpha(self):
        raw, alpha = self.baseline()
        result = render_layered_gbuffer(
            self.features, self.ids, self.raster_features, analytic_shader,
            merged_feature_srgb=raw, merged_opacity=alpha)
        point_radiance = analytic_shader(unpack_features(
            self.features, self.features.new_ones(3, 1)))["rgb"]
        expected = self.raster_features(point_radiance)
        torch.testing.assert_close(result["linear_surface"], expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(result["linear_surface"],
                                   result["diffuse_surface"] + result["specular_surface"],
                                   atol=2e-6, rtol=2e-5)

    def test_actual_material_gradients_are_finite(self):
        for space in ("srgb", "linear"):
            with self.subTest(space=space):
                features = self.features.detach().clone().requires_grad_(True)
                raw, alpha = self.baseline(features)
                result = render_layered_gbuffer(
                    features, self.ids, self.raster_features, analytic_shader,
                    color_space=space, max_layers=2,
                    merged_feature_srgb=raw, merged_opacity=alpha)
                result["linear_surface"].square().mean().backward()
                self.assertIsNotNone(features.grad)
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertGreater(float(features.grad.abs().sum()), 0)

    def test_actual_joint_geometry_opacity_and_absgrad_with_shader_checkpoint(self):
        from models.trainers.scene_graph_pbr import _LayeredScreenGradients

        outcomes = []
        for checkpointed in (False, True):
            features = self.features.detach().clone().requires_grad_(True)
            geometry = dict(self.geometry)
            geometry.update(packed=False, absgrad=True)
            for key in ("means", "opacities", "scales", "quats"):
                geometry[key] = geometry[key].detach().clone().requires_grad_(True)
            raw, alpha, primary_info = self.rasterization(
                **geometry, colors=features, render_mode="RGB+ED")
            primary_info["means2d"].retain_grad()
            collector = _LayeredScreenGradients(primary_info, True)
            raster_count = 0

            def raster(colors):
                nonlocal raster_count
                raster_count += 1
                image, _, info = self.rasterization(
                    **geometry, colors=colors, render_mode="RGB")
                collector.track(info)
                return image[0]

            def eager(function, *args, **kwargs):
                return function(*args)

            context = nullcontext() if checkpointed else patch(
                "models.layered_gbuffer.checkpoint", side_effect=eager)
            with context:
                output = render_layered_gbuffer(
                    features, self.ids, raster, analytic_shader,
                    merged_feature_srgb=raw[0, ..., :8], merged_opacity=alpha[0],
                    checkpoint_raster=False)
                output["linear_surface"].square().mean().backward()
            collector.finalize(primary_info)
            self.assertEqual(raster_count, 3)
            stats = primary_info["layered_gradient_stats"]
            self.assertEqual(stats["auxiliary_backward_passes"], 3)
            self.assertGreater(stats["auxiliary_grad_norm"], 0)
            self.assertTrue(stats["absgrad"])
            gradients = [features.grad] + [geometry[key].grad for key in
                                          ("means", "opacities", "scales", "quats")]
            for gradient in gradients:
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)
            outcomes.append([output["linear_surface"].detach(),
                             primary_info["means2d"].absgrad.detach(), *gradients])
        for actual, expected in zip(outcomes[1], outcomes[0]):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
