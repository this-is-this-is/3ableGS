"""CPU checks for visibility conservation and grouped deferred shading."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from models.layered_gbuffer import (
    linear_to_srgb, render_layered_gbuffer, srgb_to_linear, unpack_features,
)


def materials(count, *, dtype=torch.float64):
    row = torch.tensor([.4, .5, .6, .2, .35, 0., 0., 1.], dtype=dtype)
    return row.repeat(count, 1)


def analytic_shader(gbuffer):
    """A nonlinear local shader; no geometry, rasterizer, or GPU dependency."""
    a, m, r, n = (gbuffer[key] for key in ("albedo", "metallic", "roughness", "normal"))
    diffuse = srgb_to_linear(a) * (1 - m) * (.2 + n[..., 2:3].square())
    specular = ((.1 + m) * (1 - r).square() * (.2 + n[..., :1].square())).expand_as(diffuse)
    return {"rgb": diffuse + specular, "diffuse": diffuse,
            "specular": specular, "normal_oriented": n}


class FixedVisibilityRasterizer:
    """A premultiplied feature accumulator with immutable global visibility."""

    def __init__(self, weights):
        self.weights = weights
        self.calls = []

    def __call__(self, colors):
        if colors.shape[0] != self.weights.shape[-1]:
            raise AssertionError("A group pass dropped occluding Gaussians")
        self.calls.append(colors.detach().clone())
        return torch.einsum("...n,nc->...c", self.weights, colors)


class LayeredGBufferTests(unittest.TestCase):
    def render(self, feat, weights, ids=None, **kwargs):
        raster = FixedVisibilityRasterizer(weights)
        ids = torch.arange(len(feat), device=feat.device) if ids is None else ids
        result = render_layered_gbuffer(feat, ids, raster, analytic_shader, **kwargs)
        return result, raster

    def test_single_group_is_the_legacy_merged_shader(self):
        feat = materials(3)
        feat[1, :3] = .9
        feat[2, 3:5] = torch.tensor([.8, .15])
        weights = torch.tensor([[[.2, .3, .4], [0., 0., 0.]],
                                [[1e-7, 2e-7, 3e-7], [.1, .2, .1]]], dtype=feat.dtype)
        result, raster = self.render(feat, weights, torch.zeros(3, dtype=torch.long))
        raw = torch.einsum("...n,nc->...c", weights, feat)
        mass = weights.sum(-1, keepdim=True)
        expected = mass * analytic_shader(unpack_features(raw, mass))["rgb"]
        torch.testing.assert_close(result["linear_surface"], expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(len(raster.calls), 1)

    def test_feature_and_mass_conservation_with_occluding_foreground(self):
        feat = materials(3)
        feat[1] = torch.tensor([.8, .2, .1, .7, .12, 1., 0., 0.])
        alpha = torch.tensor([.8, .6, .9], dtype=feat.dtype)
        transmittance = torch.cat((torch.ones(1, dtype=feat.dtype), (1 - alpha).cumprod(0)[:-1]))
        weights = (alpha * transmittance)[None, None]
        result, raster = self.render(feat, weights, torch.tensor([10, 20, 20]))
        self.assertEqual(len(raster.calls), 3)
        # Background group mass is .2 * (.6 + .4*.9), not .6+.4*.9.
        torch.testing.assert_close(raster(weights.new_ones(3, 1)), weights.sum(-1, keepdim=True))
        group_background = raster.calls[2]
        torch.testing.assert_close(group_background[:, -1], weights.new_tensor([0, 1, 1]))
        torch.testing.assert_close(result["opacity"], weights.sum(-1, keepdim=True))
        self.assertLess(result["diagnostics"]["feature_reconstruction_max_abs"], 1e-12)
        self.assertLess(result["diagnostics"]["mass_reconstruction_max_abs"], 1e-12)

    def test_identical_materials_invariant_including_fading_tiny_groups(self):
        feat = materials(4)
        weights = torch.tensor([[[.6, .1, .04, 1e-12], [1e-9, 2e-9, 3e-9, 1e-12]],
                                [[0., 0., 0., 0.], [.3, 1e-7, .1, .01]]], dtype=feat.dtype)
        for space in ("srgb", "linear"):
            with self.subTest(space=space):
                baseline, _ = self.render(feat, weights, torch.zeros(4, dtype=torch.long), color_space=space)
                for cap in (0, 1, 2, 4):
                    grouped, _ = self.render(feat, weights, color_space=space, max_layers=cap)
                    torch.testing.assert_close(grouped["linear_surface"], baseline["linear_surface"],
                                               atol=1e-12, rtol=1e-10)

    def test_conservation_diagnostics_detect_illegal_opacity_masking(self):
        feat = materials(2)
        alpha = feat.new_tensor([.8, .6])

        def wrong_group_raster(colors):
            # This intentionally incorrect callback removes non-group opacity.
            # Group renders then see through objects that should still occlude.
            changed_alpha = alpha * colors[:, -1]
            trans = torch.cat((torch.ones_like(changed_alpha[:1]),
                               (1 - changed_alpha).cumprod(0)[:-1]))
            return ((changed_alpha * trans)[:, None] * colors).sum(0)[None, None]

        result = render_layered_gbuffer(feat, torch.arange(2), wrong_group_raster, analytic_shader)
        self.assertGreater(result["diagnostics"]["mass_reconstruction_max_abs"], .4)
        self.assertGreater(result["diagnostics"]["feature_reconstruction_max_abs"], .4)

    def test_separate_shading_equals_analytic_weighted_surfaces_without_double_alpha(self):
        feat = materials(2)
        feat[1, 5:8] = feat.new_tensor([1, 0, 0])
        feat[1, 3:5] = feat.new_tensor([.9, .1])
        weights = feat.new_tensor([[[.4, .3]]])
        grouped, _ = self.render(feat, weights)
        expected_rgb = sum(weights[0, 0, i] * analytic_shader(
            unpack_features(feat[i:i + 1], feat.new_ones(1, 1)))["rgb"][0] for i in range(2))
        torch.testing.assert_close(grouped["linear_surface"][0, 0], expected_rgb)
        merged, _ = self.render(feat, weights, torch.zeros(2, dtype=torch.long))
        self.assertGreater(float((grouped["linear_surface"] - merged["linear_surface"]).abs().max()), .01)
        torch.testing.assert_close(grouped["linear_surface"],
                                   grouped["diffuse_surface"] + grouped["specular_surface"])

    def test_linear_albedo_control_linearizes_before_averaging(self):
        feat = materials(2)
        feat[0, :3] = .05
        feat[1, :3] = .95
        weights = feat.new_tensor([[[.4, .4]]])
        srgb, _ = self.render(feat, weights, torch.zeros(2, dtype=torch.long), color_space="srgb")
        linear, _ = self.render(feat, weights, torch.zeros(2, dtype=torch.long), color_space="linear")
        restored = unpack_features(linear["merged_feature"], linear["opacity"], "linear")
        expected = srgb_to_linear(feat[:, :3]).mean(0)
        torch.testing.assert_close(srgb_to_linear(restored["albedo"])[0, 0], expected)
        torch.testing.assert_close(linear["merged_feature_srgb"], srgb["merged_feature_srgb"])
        self.assertGreater(float((linear["linear_surface"] - srgb["linear_surface"]).abs().max()), .05)

    def test_top_contribution_plus_overflow_preserves_all_mass(self):
        feat = materials(4)
        feat[:, :3] = feat.new_tensor([[.1, .2, .3], [.4, .8, .6], [.9, .2, .1], [.5, .6, .2]])
        feat[2, 5:8] = feat.new_tensor([0, 1, 0])
        weights = feat.new_tensor([[[.5, .2, .1, .05], [.1, .1, .2, .5]]])
        result, _ = self.render(feat, weights, max_layers=2)
        for pixel in range(2):
            w = weights[0, pixel]
            top = int(w.argmax())
            other = torch.arange(4) != top
            expected = w[top] * analytic_shader(unpack_features(feat[top:top + 1], feat.new_ones(1, 1)))["rgb"][0]
            other_mass = w[other].sum().reshape(1, 1)
            other_feat = (w[other, None] * feat[other]).sum(0, keepdim=True)
            expected += other_mass[0] * analytic_shader(unpack_features(other_feat, other_mass))["rgb"][0]
            torch.testing.assert_close(result["linear_surface"][0, pixel], expected)
            torch.testing.assert_close(result["diagnostic_maps"]["overflow_mass"][0, pixel, 0], other_mass[0, 0])
        self.assertEqual(result["diagnostics"]["shaded_layer_count"], 2)
        self.assertGreater(result["diagnostics"]["overflow_mass_fraction"], 0)
        torch.testing.assert_close(result["opacity"], weights.sum(-1, keepdim=True))

    def test_one_slot_is_merged_control_and_enough_slots_equal_full_groups(self):
        feat = materials(3)
        feat[1, 4] = .05
        feat[2, 5:8] = feat.new_tensor([1, 0, 0])
        weights = feat.new_tensor([[[.2, .3, .4]]])
        merged, _ = self.render(feat, weights, torch.zeros(3, dtype=torch.long))
        one_slot, _ = self.render(feat, weights, max_layers=1)
        full, _ = self.render(feat, weights)
        three_slots, _ = self.render(feat, weights, max_layers=3)
        torch.testing.assert_close(merged["linear_surface"], one_slot["linear_surface"])
        torch.testing.assert_close(full["linear_surface"], three_slots["linear_surface"])

    def test_component_ablations_replace_only_requested_material_fields(self):
        feat = materials(2)
        feat[1] = feat.new_tensor([.8, .1, .3, .8, .1, 1., 0., 0.])
        weights = feat.new_tensor([[[.4, .5]]])
        merged = unpack_features(torch.einsum("...n,nc->...c", weights, feat), weights.sum(-1, keepdim=True))
        for ablation, replaced in (("normal", ("normal",)), ("brdf", ("metallic", "roughness")),
                                   ("albedo", ("albedo",))):
            captured = []

            def record(gbuffer):
                captured.append(gbuffer)
                return analytic_shader(gbuffer)

            render_layered_gbuffer(feat, torch.arange(2), FixedVisibilityRasterizer(weights), record,
                                   ablation=ablation)
            for index, observed in enumerate(captured):
                own = unpack_features(feat[index:index + 1].reshape(1, 1, 8), feat.new_ones(1, 1, 1))
                for key in own:
                    torch.testing.assert_close(observed[key], merged[key] if key in replaced else own[key])

    def test_optional_merged_inputs_reused_and_legacy_features_preserved(self):
        feat = materials(2)
        weights = feat.new_tensor([[[.4, .3]]])
        raw = torch.einsum("...n,nc->...c", weights, feat)
        mass = weights.sum(-1, keepdim=True)
        for space, expected_calls in (("srgb", 2), ("linear", 3)):
            result, raster = self.render(feat, weights, color_space=space,
                                         merged_feature_srgb=raw, merged_opacity=mass)
            self.assertEqual(len(raster.calls), expected_calls)
            self.assertIs(result["merged_feature_srgb"], raw)
            self.assertIs(result["opacity"], mass)

    def test_zero_and_tiny_visibility_have_finite_material_and_visibility_gradients(self):
        for space in ("srgb", "linear"):
            for cap in (0, 1, 2):
                with self.subTest(space=space, cap=cap):
                    feat = materials(3, dtype=torch.float32).requires_grad_(True)
                    weights = torch.tensor([[[.4, .3, 1e-12], [0., 0., 0.]],
                                            [[1e-9, 2e-9, 0.], [.1, .2, .3]]], requires_grad=True)
                    result, _ = self.render(feat, weights, color_space=space, max_layers=cap)
                    result["linear_surface"].square().sum().backward()
                    self.assertTrue(torch.isfinite(result["linear_surface"]).all())
                    self.assertTrue(torch.isfinite(feat.grad).all())
                    self.assertTrue(torch.isfinite(weights.grad).all())
                    self.assertGreater(float(feat.grad.abs().sum()), 0)

    def test_checkpoint_recomputation_matches_eager_outputs_and_gradients(self):
        base_features = materials(3)
        base_features[1] = base_features.new_tensor([.8, .2, .1, .7, .12, .8, 0., .6])
        base_features[2] = base_features.new_tensor([.1, .7, .4, .1, .6, 0., .6, .8])
        base_weights = base_features.new_tensor([[[.4, .3, .1], [.1, .2, .6]],
                                                 [[1e-9, 2e-9, 0.], [0., 0., 0.]]])

        def eager(function, *args, **kwargs):
            self.assertFalse(kwargs["use_reentrant"])
            return function(*args)

        for space in ("srgb", "linear"):
            for cap in (0, 2):
                for ablation in ("none", "normal", "brdf", "albedo"):
                    with self.subTest(space=space, cap=cap, ablation=ablation):
                        outcomes = []
                        for checkpointed in (False, True):
                            feat = base_features.clone().requires_grad_(True)
                            weights = base_weights.clone().requires_grad_(True)
                            context = nullcontext() if checkpointed else patch(
                                "models.layered_gbuffer.checkpoint", side_effect=eager)
                            with context:
                                result, raster = self.render(feat, weights, color_space=space,
                                                             max_layers=cap, ablation=ablation)
                                forward_passes = len(raster.calls)
                                loss = (result["linear_surface"].square().sum()
                                        + .3 * result["diffuse_surface"].sum()
                                        + .7 * result["specular_surface"].sum()
                                        + .1 * result["normal_oriented"].sum())
                                loss.backward()
                            outcomes.append((result["linear_surface"].detach(),
                                             feat.grad, weights.grad))
                            if checkpointed:
                                self.assertGreater(len(raster.calls), forward_passes)
                            else:
                                self.assertEqual(len(raster.calls), forward_passes)
                        for actual, expected in zip(outcomes[1], outcomes[0]):
                            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)

    def test_inference_does_not_checkpoint_or_recompute(self):
        feat = materials(3).requires_grad_(True)
        weights = feat.new_tensor([[[.4, .3, .1]]])
        with patch("models.layered_gbuffer.checkpoint", side_effect=AssertionError("inference checkpoint")):
            with torch.no_grad():
                result, raster = self.render(feat, weights)
        self.assertEqual(len(raster.calls), 4)
        self.assertFalse(result["linear_surface"].requires_grad)

    def test_zero_gaussians_returns_empty_surface_and_valid_diagnostics(self):
        feat = materials(0)
        result, _ = self.render(feat, torch.zeros(2, 3, 0, dtype=feat.dtype))
        self.assertEqual(result["diagnostics"]["group_count"], 0)
        self.assertEqual(result["diagnostics"]["feature_reconstruction_max_abs"], 0)
        self.assertEqual(float(result["linear_surface"].abs().sum()), 0)

    def test_invalid_configuration_fails_before_rasterization(self):
        feat = materials(2)
        weights = feat.new_ones(1, 1, 2) / 3
        for kwargs in ({"color_space": "display"}, {"max_layers": -1},
                       {"max_layers": 1.5}, {"ablation": "random"}):
            with self.assertRaises(ValueError):
                self.render(feat, weights, **kwargs)
        with self.assertRaises(ValueError):
            self.render(feat, weights, torch.tensor([0., 1.]))

    def test_color_transfer_roundtrip_and_gradients_at_zero(self):
        x = torch.tensor([0., .01, .04, .1, .5, 1.], requires_grad=True)
        restored = linear_to_srgb(srgb_to_linear(x))
        torch.testing.assert_close(restored, x, rtol=1e-6, atol=1e-7)
        restored.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
