"""CPU checks for native-coordinate material attention and independent losses."""

import copy
import unittest

import torch

from models.gbuffer_attention import (
    GBufferAttentionHead, attention_material_loss, attention_strength,
    temporal_weight, validate_attention_config,
)
from tests.test_ego_temporal import packet


class GBufferAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, variant="A2", **kwargs):
        return GBufferAttentionHead({"variant": variant, "width": 8,
                                     "chunk_size": 97, **kwargs})

    def test_bypass_is_bit_exact_and_keeps_current_gradient(self):
        current, previous = packet(), packet(frame_id=0)
        current["albedo"].requires_grad_()
        previous["albedo"].fill_(.8)
        for model, strength in ((self.model(), 0.), (self.model("B0"), 1.)):
            output = model(current, [previous], strength=strength)
            self.assertIs(output["albedo"], current["albedo"])
            self.assertEqual(output["history_weight"].sum(), 0)
        output["albedo"].sum().backward()
        torch.testing.assert_close(current["albedo"].grad, torch.ones_like(current["albedo"]))

    def test_invalid_history_and_dynamic_pixels_cannot_leak_values(self):
        current, previous = packet(), packet(frame_id=0)
        previous["depth"].fill_(100)
        previous["albedo"].fill_(1)
        current["static_mask"][5:8, 5:8] = False
        for variant in ("H0", "C0", "C1", "A0", "A1", "A2"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                output = model(current, [previous])
                self.assertFalse(output["valid"].any())
                self.assertEqual(output["history_weight"].sum(), 0)
                torch.testing.assert_close(output["albedo"], current["albedo"], atol=1e-7, rtol=0)
                if model.learned:
                    with torch.no_grad():
                        model.residual_head.bias.fill_(2)
                    output = model(current, [previous])
                    torch.testing.assert_close(output["albedo"][5:8, 5:8], current["albedo"][5:8, 5:8], atol=0, rtol=0)
                    self.assertGreater(output["albedo"][3, 3, 0], current["albedo"][3, 3, 0])

    def test_native_pixel_candidates_are_checked_against_central_query(self):
        current, previous = packet(), packet(frame_id=0)
        # A 1-pixel movement in the previous image is 2 pixels in the current
        # image. Each previous pixel is a valid surface by itself, but the
        # off-center candidates fail the query's <=1.5 pixel cycle test.
        previous["intrinsics"][0, 0] = 2
        previous["intrinsics"][1, 1] = 2
        model = self.model()
        geometry = model._prepare(current, [previous])
        start = 8 * 20 + 10
        pixels = torch.tensor([[10.5, 8.5, 1.]])
        uv, valid, _, _ = model._candidates(geometry, pixels, start, start + 1)[0]
        self.assertTrue(valid[0, 4])
        self.assertEqual(int(valid.sum()), 1)
        torch.testing.assert_close(uv[0, 5] - uv[0, 4], torch.tensor([1., 0.]))

    def test_fixed_transport_averages_material_in_linear_space(self):
        current, previous = packet(), packet(frame_id=0)
        current["albedo"].zero_()
        previous["albedo"].fill_(1)
        output = self.model("H0")(current, [previous])
        # sRGB(.3 linear) is approximately .584, not the .3 sRGB average.
        self.assertAlmostEqual(float(output["albedo"][5, 5, 0]), .5838315, places=5)
        self.assertAlmostEqual(float(output["history_weight"][5, 5, 0]), .3, places=6)

    def test_current_value_and_attention_receive_gradient_not_routing_or_history(self):
        current, previous = packet(), packet(frame_id=0)
        previous["albedo"][:] = torch.linspace(.2, .9, 20)[None, :, None]
        for p in (current, previous):
            for key in ("albedo", "depth", "normal", "roughness", "camera_to_world", "intrinsics"):
                p[key].requires_grad_()
        model = self.model()
        output = model(current, [previous])
        output["albedo"][3:-3, 3:-3].square().mean().backward()
        self.assertGreater(current["albedo"].grad.abs().sum(), 0)
        for key in ("depth", "normal", "roughness", "camera_to_world", "intrinsics"):
            self.assertIsNone(current[key].grad)
        for key in ("albedo", "depth", "normal", "roughness", "camera_to_world", "intrinsics"):
            self.assertIsNone(previous[key].grad)
        for parameter in (model.encoder.layers[0].weight, model.attention[0].query.weight,
                          model.attention[1].query.weight, model.gate_head.weight,
                          model.residual_head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum(), 0)

    def test_geometry_ablation_removes_material_slots_and_biases(self):
        current, previous = packet(), packet(frame_id=0)
        previous["albedo"].fill_(.7)
        model = self.model("A0")
        with torch.no_grad():
            model.gate_head.weight.normal_(0, .1)
            model.residual_head.weight.normal_(0, .1)
        first = model(current, [previous])
        for p in (current, previous):
            p["normal"][:] = torch.tensor([1., 0., 0.])
            p["roughness"].fill_(.98)
            p["metallic"].fill_(.8)
        second = model(current, [previous])
        for key in ("albedo", "history_weight", "valid", "attention_entropy"):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_full_and_geometry_use_the_same_hard_support(self):
        current, previous = packet(), packet(frame_id=0)
        previous["normal"][:] = torch.tensor([1., 0., 0.])
        geometry = self.model("A0")(current, [previous])
        full = self.model("A2")(current, [previous])
        torch.testing.assert_close(geometry["valid"], full["valid"])
        self.assertTrue(full["valid"][5, 5])
        strict = self.model("A2", normal_check=True)(current, [previous])
        self.assertFalse(strict["valid"].any())

    def test_chunking_and_checkpointing_preserve_forward_and_backward(self):
        current, previous = packet(), packet(frame_id=0)
        previous["albedo"][:] = torch.linspace(.1, .9, 20)[None, :, None]
        small = self.model(chunk_size=47, checkpoint_chunks=True)
        big = self.model(chunk_size=10000, checkpoint_chunks=False)
        big.load_state_dict(small.state_dict())
        a, b = small(current, [previous]), big(current, [previous])
        torch.testing.assert_close(a["albedo"], b["albedo"], rtol=1e-6, atol=1e-7)
        a["albedo"].square().mean().backward()
        b["albedo"].square().mean().backward()
        for (name, p), (other_name, q) in zip(small.named_parameters(), big.named_parameters()):
            self.assertEqual(name, other_name)
            if p.grad is not None or q.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=1e-7)

    def test_loss_uses_zero_offset_geometry_and_detached_previous_target(self):
        current, previous = packet(), packet(frame_id=0)
        current["albedo"].requires_grad_()
        previous["albedo"].fill_(.6).requires_grad_()
        previous["normal"][:] = torch.tensor([1., 0., 0.])
        current["learned_acceptance"] = torch.zeros(16, 20, 1)
        result = attention_material_loss(current, previous)
        self.assertTrue(result["valid"][5, 5])
        self.assertGreater(result["loss"], 0)
        result["loss"].backward()
        self.assertGreater(current["albedo"].grad.abs().sum(), 0)
        self.assertIsNone(previous["albedo"].grad)
        previous["depth"].fill_(100)
        current["albedo"].grad = None
        empty = attention_material_loss(current, previous)
        self.assertEqual(empty["loss"], 0)
        empty["loss"].backward()
        self.assertEqual(current["albedo"].grad.abs().sum(), 0)

    def test_history_order_supervision_keys_and_checkpoint_roundtrip(self):
        current = packet(frame_id=5)
        histories = [packet(frame_id=i) for i in (4, 3, 1)]
        for i, previous in enumerate(histories):
            previous["albedo"].fill_(.4 + i * .1)
        model = self.model()
        first = model(current, histories)["albedo"]
        current["pixels"] = torch.rand_like(current["albedo"])
        for previous in histories:
            previous["gbuf_basecolor"] = torch.rand_like(current["albedo"])
        restored = self.model()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        second = restored(current, list(reversed(histories)))["albedo"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_missing_history_retains_the_cnn_age_slot(self):
        current, previous = packet(frame_id=5), packet(frame_id=3)
        previous["albedo"].fill_(.8)
        model = self.model("C1")
        with torch.no_grad():
            model.cnn_logits.weight.zero_()
            model.cnn_logits.bias.fill_(-100)
            # Self + nine t-1 candidates + t-2's center candidate.
            model.cnn_logits.bias[1 + 9 + 4] = 100
        output = model(current, [previous])
        self.assertAlmostEqual(float(output["history_weight"][5, 5, 0].detach()), .25, places=6)

    def test_controls_match_capacity_and_schedule_is_explicit(self):
        from omegaconf import OmegaConf
        self.assertEqual(validate_attention_config(OmegaConf.create({"history_offsets": [1, 2, 4]}))["history_offsets"], [1, 2, 4])
        counts = {variant: sum(p.numel() for p in GBufferAttentionHead({"variant": variant}).parameters())
                  for variant in ("C0", "C1", "A0", "A1", "A2")}
        self.assertEqual(len({counts[v] for v in ("C0", "A0", "A1", "A2")}), 1)
        self.assertLess(abs(counts["C1"] / counts["A2"] - 1), .05)
        config = validate_attention_config()
        self.assertEqual(attention_strength(config, 4999), 0)
        self.assertEqual(attention_strength(config, 5500), .5)
        self.assertEqual(attention_strength(config, 6000), 1)
        self.assertEqual(attention_strength(config, 0, training=False), 1)
        self.assertEqual(temporal_weight(config, 6000), 0)
        self.assertEqual(temporal_weight(config, 6500), .025)
        self.assertEqual(temporal_weight({"variant": "A1"}, 30000), 0)
        config["history_offsets"].append(9)
        self.assertEqual(validate_attention_config()["history_offsets"], [1, 2, 4])

    def test_malformed_configs_future_duplicate_cross_stream_and_wrong_age_fail(self):
        for config in ({"width": 0}, {"chunk_size": True}, {"history_offsets": [1, 1]},
                       {"max_blend": 2}, {"typo": 1}, {"temporal_w": float("nan")}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.model(**config)
        model = self.model()
        current, previous = packet(frame_id=5), packet(frame_id=4)
        for histories in ([packet(frame_id=6)], [previous, previous], [packet(frame_id=2)]):
            with self.assertRaises(ValueError):
                model(current, histories)
        previous["stream_id"] = "different-camera"
        with self.assertRaisesRegex(ValueError, "same camera"):
            model(current, [previous])


if __name__ == "__main__":
    unittest.main()
