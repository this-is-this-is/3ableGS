"""CPU geometry, gradient, and bounded-material contracts for experiment 5."""

import copy
import unittest
from unittest.mock import patch

import torch

from models.ego_temporal import linear_to_srgb, project_history, srgb_to_linear
from models.rdg_fusion import (
    DEFAULT_CONFIG, RDGFusion, downsample_packet, rdg_strength, validate_rdg_config,
)
from tests.test_ego_temporal import packet


def packets(h=64, w=80):
    current = packet(h, w)
    current["intrinsics"][0, 0] = 32
    current["intrinsics"][1, 1] = 32
    previous = downsample_packet(copy.deepcopy(current), 4)
    previous["frame_id"] = 0
    previous["albedo"].fill_(.7)
    previous["rgb"] = previous["albedo"]
    return current, previous


class RDGFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, **overrides):
        return RDGFusion({"enabled": True, "width": 8, **overrides})

    def fixed_attention(self, history_logit=-2., **overrides):
        model = self.model(**overrides)
        with torch.no_grad():
            for value in model.parameters():
                value.zero_()
            model.logit_prior[1] = history_logit
        return model

    def test_config_is_opt_in_independent_and_rejects_invalid_settings(self):
        config = validate_rdg_config()
        self.assertFalse(config["enabled"])
        config["width"] = 100
        self.assertEqual(DEFAULT_CONFIG["width"], 16)
        bad = [[], {"unknown": 1}, {"enabled": 1}, {"width": 3}, {"width": 8.},
               {"feature_downscale": 0}, {"interval": True}, {"start_step": -1},
               {"ramp_steps": -1}, {"max_gap": 0}, {"max_blend": 1.01},
               {"max_delta_linear": 2}, {"temporal_weight": -1}, {"anchor_weight": True},
               {"lr": 0}, {"lr": float("nan")}, {"grad_clip": 0},
               {"weight_decay": float("inf")}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_rdg_config(value)

    def test_warmup_interval_and_evaluation_saved_step_strength(self):
        cfg = {"enabled": True}
        self.assertEqual(rdg_strength(cfg, 4999), 0)
        self.assertEqual(rdg_strength(cfg, 5000), 0)
        self.assertEqual(rdg_strength(cfg, 5001), 0)
        self.assertAlmostEqual(rdg_strength(cfg, 5008), .008)
        self.assertAlmostEqual(rdg_strength(cfg, 5001, training=False), .001)
        self.assertEqual(rdg_strength(cfg, 6000), 1)
        self.assertEqual(rdg_strength(cfg, 6001), 0)
        self.assertEqual(rdg_strength(cfg, 6001, training=False), 1)
        self.assertEqual(rdg_strength({}, 30000), 0)
        self.assertEqual(rdg_strength({**cfg, "ramp_steps": 0}, 5000), 1)
        for step in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                rdg_strength(cfg, step)

    def test_bypasses_are_bit_exact_and_keep_current_gradient(self):
        current, previous = packets()
        current["albedo"].requires_grad_()
        for model, history, strength in ((RDGFusion(), previous, 1),
                                         (self.model(), None, 1),
                                         (self.model(), previous, 0),
                                         (self.model(max_blend=0), previous, 1),
                                         (self.model(max_delta_linear=0), previous, 1)):
            with patch("models.rdg_fusion.project_history", side_effect=AssertionError("bypass projected")):
                output = model(current, history, strength)
            self.assertIs(output["albedo"], current["albedo"])
            self.assertEqual(output["loss_anchor"], 0)
            self.assertEqual(output["loss_temporal"], 0)
            self.assertFalse(output["valid"].any())
        output["albedo"].sum().backward()
        torch.testing.assert_close(current["albedo"].grad, torch.ones_like(current["albedo"]))

    def test_bad_strength_is_rejected(self):
        current, previous = packets()
        model = self.model()
        for strength in (True, -1, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                model(current, previous, strength)

    def test_downsampling_scales_actual_pixel_center_intrinsics(self):
        current, _ = packets(65, 79)
        original = current["intrinsics"].clone()
        reduced = downsample_packet(current, 4)
        self.assertEqual(reduced["albedo"].shape, (17, 20, 3))
        torch.testing.assert_close(reduced["intrinsics"][0], original[0] * 20 / 79)
        torch.testing.assert_close(reduced["intrinsics"][1], original[1] * 17 / 65)
        torch.testing.assert_close(current["intrinsics"], original, atol=0, rtol=0)
        torch.testing.assert_close(reduced["camera_to_world"], current["camera_to_world"])

    def test_odd_native_size_accepts_floor_sized_history_with_its_own_intrinsics(self):
        current, _ = packets(65, 79)
        previous = packet(16, 19, frame_id=0)
        previous["intrinsics"] = current["intrinsics"].clone()
        previous["intrinsics"][0] *= 19 / 79
        previous["intrinsics"][1] *= 16 / 65
        previous["albedo"].fill_(.7)
        output = self.model()(current, previous)
        self.assertTrue(output["valid"][32, 39, 0])
        self.assertGreater(output["albedo"][32, 39, 0], current["albedo"][32, 39, 0])
        self.assertTrue(torch.isfinite(output["albedo"]).all())

    def test_area_reduction_is_linear_light_and_keeps_only_albedo_gradient(self):
        current, _ = packets()
        current["albedo"][:, ::2] = 0
        current["albedo"][:, 1::2] = 1
        for key in ("albedo", "depth", "normal", "roughness", "intrinsics"):
            current[key].requires_grad_()
        low = downsample_packet(current, 4)
        torch.testing.assert_close(srgb_to_linear(low["albedo"]), torch.full((16, 20, 3), .5))
        low["albedo"].sum().backward()
        self.assertGreater(current["albedo"].grad.abs().sum(), 0)
        for key in ("depth", "normal", "roughness", "intrinsics"):
            self.assertIsNone(current[key].grad)

    def test_geometry_and_attention_run_at_reduced_resolution(self):
        current, previous = packets()
        model = self.model()
        shapes = []
        hook = model.material_encoder.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
        with patch("models.rdg_fusion.project_history", wraps=project_history) as projection:
            output = model(current, previous)
        hook.remove()
        self.assertEqual(projection.call_count, 1)
        self.assertEqual(projection.call_args.args[0]["albedo"].shape, (16, 20, 3))
        self.assertEqual(projection.call_args.args[1]["albedo"].shape, (16, 20, 3))
        self.assertEqual(shapes, [torch.Size([1, 8, 16, 20]), torch.Size([1, 8, 16, 20])])
        self.assertEqual(output["albedo"].shape, current["albedo"].shape)
        self.assertTrue(output["valid"][32, 40, 0])

    def test_pixel_center_camera_translation_aligns_one_low_resolution_pixel(self):
        current, previous = packets()
        previous["albedo"][:] = torch.linspace(.1, .8, 20)[None, :, None]
        previous["rgb"] = previous["albedo"]
        current["camera_to_world"][0, 3] = .25  # Native fx32 / depth2 ->4px, quarter->1px.
        projected = project_history(downsample_packet(current, 4), previous)
        self.assertTrue(projected["valid"][8, 10, 0])
        torch.testing.assert_close(projected["warped"]["albedo"][8, 10], previous["albedo"][8, 11], atol=1e-6, rtol=0)
        shifted = self.fixed_attention()(current, previous)
        current["camera_to_world"][0, 3] = 0
        unshifted = self.fixed_attention()(current, previous)
        self.assertGreater(shifted["albedo"][32, 40, 0], unshifted["albedo"][32, 40, 0])

    def test_linear_correction_and_history_weight_obey_bounds_and_ramp(self):
        current, previous = packets()
        previous["albedo"].fill_(1.)
        model = self.fixed_attention(history_logit=20, max_delta_linear=.02)
        for strength in (.25, 1.):
            output = model(current, previous, strength)
            self.assertLessEqual(float(output["residual_linear"].detach().abs().max()), .02 * strength + 1e-7)
            self.assertLessEqual(float(output["history_weight"].detach().max()), .35 * strength + 1e-7)
            self.assertGreater(output["albedo"][32, 40, 0], current["albedo"][32, 40, 0])
            torch.testing.assert_close(output["residual_linear"][32, 40], torch.full((3,), .02 * strength), atol=1e-7, rtol=0)

    def test_native_high_frequency_contrast_survives_additive_low_frequency_fusion(self):
        current, previous = packets()
        linear = torch.full_like(current["albedo"], .2)
        linear[:, 1::2] = .3
        current["albedo"] = linear_to_srgb(linear)
        current["rgb"] = current["albedo"]
        previous["albedo"].fill_(.9)
        output = self.fixed_attention(history_logit=10)(current, previous)
        result = srgb_to_linear(output["albedo"])
        torch.testing.assert_close(result[20:40, 21:41:2] - result[20:40, 20:40:2],
                                   torch.full((20, 10, 3), .1), atol=2e-7, rtol=0)
        self.assertGreater(output["residual_linear"][32, 40, 0], 0)

    def test_dynamic_thin_invalid_regions_and_depth_edges_are_unchanged(self):
        current, previous = packets()
        current["static_mask"][:, 39] = False
        current["depth"][40:, :, 0] = 4
        output = self.fixed_attention(history_logit=10)(current, previous)
        torch.testing.assert_close(output["albedo"][:, 38:41], current["albedo"][:, 38:41], atol=0, rtol=0)
        torch.testing.assert_close(output["albedo"][39:], current["albedo"][39:], atol=0, rtol=0)
        self.assertFalse(output["valid"][:, 39].any())
        self.assertTrue(output["valid"][20, 20, 0])
        self.assertGreater(output["residual_linear"][20, 20, 0], 0)

    def test_occlusion_and_incompatible_normals_reject_history_exactly(self):
        current, previous = packets()
        for key, value in (("depth", 100.), ("normal", 0.), ("opacity", .1)):
            changed = copy.deepcopy(previous)
            changed[key].fill_(value)
            output = self.model()(current, changed)
            self.assertIs(output["albedo"], current["albedo"])
            self.assertFalse(output["valid"].any())
            self.assertEqual(output["loss_temporal"], 0)

    def test_nonfinite_history_is_excluded_without_poisoning_other_pixels(self):
        current, previous = packets()
        previous["albedo"][8, 10] = float("nan")
        previous["normal"][9, 10] = float("inf")
        output = self.model()(current, previous)
        self.assertTrue(torch.isfinite(output["albedo"]).all())
        self.assertTrue(torch.isfinite(output["loss_temporal"]))
        torch.testing.assert_close(output["albedo"][32:36, 40:44], current["albedo"][32:36, 40:44], atol=0, rtol=0)
        self.assertTrue(output["valid"][20, 20, 0])

    def test_out_of_range_current_albedo_is_preserved_without_unbounded_clipping(self):
        current, previous = packets()
        current["albedo"][30:35, 38:43] = 1.5
        output = self.model()(current, previous)
        torch.testing.assert_close(output["albedo"][30:35, 38:43], current["albedo"][30:35, 38:43], atol=0, rtol=0)
        self.assertFalse(output["valid"][30:35, 38:43].any())
        self.assertLessEqual(float(output["residual_linear"].detach().abs().max()), .05 + 1e-7)

    def test_history_and_geometry_are_detached_but_head_and_current_receive_gradient(self):
        current, previous = packets()
        keys = ("albedo", "normal", "roughness", "metallic", "depth", "opacity",
                "viewdirs", "camera_to_world", "intrinsics")
        for p in (current, previous):
            for key in keys:
                p[key].requires_grad_()
        model = self.model()
        output = model(current, previous)
        loss = output["albedo"][20:40, 20:40].square().mean() + output["loss_anchor"] + output["loss_temporal"]
        loss.backward()
        self.assertGreater(current["albedo"].grad.abs().sum(), 0)
        for key in keys[1:]:
            self.assertIsNone(current[key].grad, key)
        for key in keys:
            self.assertIsNone(previous[key].grad, key)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0)
        self.assertGreater(model.logit_prior.grad.abs().sum(), 0)
        self.assertGreater(model.geometry_encoder[-1].weight.grad.abs().sum(), 0)

    def test_temporal_loss_uses_geometry_support_even_when_learned_history_gate_closes(self):
        current, previous = packets()
        current["albedo"].requires_grad_()
        model = self.fixed_attention(history_logit=-100)
        output = model(current, previous)
        self.assertLess(float(output["history_weight"].detach().max()), 1e-30)
        self.assertGreater(output["loss_temporal"], 0)
        output["loss_temporal"].backward()
        self.assertGreater(current["albedo"].grad.abs().sum(), 0)

    def test_losses_are_weighted_once_and_scale_with_config(self):
        current, previous = packets()
        one = self.fixed_attention(anchor_weight=.01, temporal_weight=.02)
        two = self.fixed_attention(anchor_weight=.03, temporal_weight=.08)
        a, b = one(current, previous), two(current, previous)
        torch.testing.assert_close(b["loss_anchor"], a["loss_anchor"] * 3)
        torch.testing.assert_close(b["loss_temporal"], a["loss_temporal"] * 4)
        torch.testing.assert_close(a["albedo"], b["albedo"], atol=0, rtol=0)

    def test_unrelated_packet_targets_cannot_change_routing_or_values(self):
        current, previous = packets()
        model = self.model()
        baseline = model(current, previous)
        for p in (current, previous):
            p["teacher_albedo"] = torch.rand_like(p["albedo"])
            p["pixels"] = torch.rand_like(p["albedo"])
            p["dynamic_masks"] = torch.ones_like(p["static_mask"])
        changed = model(current, previous)
        for key in baseline:
            torch.testing.assert_close(baseline[key], changed[key], atol=0, rtol=0)

    def test_input_packets_and_non_albedo_materials_are_not_mutated(self):
        current, previous = packets()
        copies = [{k: v.clone() if torch.is_tensor(v) else v for k, v in p.items()}
                  for p in (current, previous)]
        self.model()(current, previous)
        for p, before in zip((current, previous), copies):
            for key in before:
                if torch.is_tensor(before[key]):
                    torch.testing.assert_close(p[key], before[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
