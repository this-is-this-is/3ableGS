"""CPU contracts for geometry visibility and differentiable temporal heads."""

import unittest

import torch

from models.ego_temporal import (EgoTemporalHead, linear_to_srgb, material_consistency,
                                 project_history, routing_guides, srgb_to_linear)


def packet(h=16, w=20, frame_id=1):
    normal = torch.zeros(h, w, 3)
    normal[..., 2] = 1
    return {"rgb": torch.full((h, w, 3), .35), "albedo": torch.full((h, w, 3), .3),
            "roughness": torch.full((h, w, 1), .4), "metallic": torch.full((h, w, 1), .1),
            "normal": normal, "depth": torch.full((h, w, 1), 2.),
            "opacity": torch.ones(h, w, 1), "static_mask": torch.ones(h, w, dtype=torch.bool),
            "viewdirs": -normal.clone(), "camera_to_world": torch.eye(4),
            "intrinsics": torch.tensor([[4., 0., w / 2], [0., 4., h / 2], [0., 0., 1.]]),
            "frame_id": frame_id, "stream_id": "scene0-camera0-train"}


class EgoTemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_identity_and_pixel_center_translation(self):
        current, previous = packet(), packet(frame_id=0)
        previous["albedo"][:] = torch.arange(20).float()[None, :, None] / 20
        p = project_history(current, previous)
        self.assertTrue(p["valid"][1:-1, 1:-1].all())
        torch.testing.assert_close(p["warped"]["albedo"], previous["albedo"], atol=1e-6, rtol=0)
        torch.testing.assert_close(p["displacement"], torch.zeros_like(p["displacement"]), atol=1e-6, rtol=0)
        current["camera_to_world"][0, 3] = .5  # fx * x / depth = 1 pixel
        p = project_history(current, previous)
        torch.testing.assert_close(p["warped"]["albedo"][4, 4], previous["albedo"][4, 5], atol=1e-6, rtol=0)
        self.assertTrue(p["valid"][4, 4])
        self.assertFalse(p["valid"][4, -1])

    def test_albedo_interpolates_in_linear_space(self):
        current, previous = packet(), packet(frame_id=0)
        previous["albedo"][:, ::2] = 0
        previous["albedo"][:, 1::2] = 1
        offset = torch.zeros(16, 20, 2)
        offset[..., 0] = .5
        p = project_history(current, previous, offset)
        torch.testing.assert_close(p["warped"]["albedo_linear"][4, 4], torch.full((3,), .5), atol=1e-6, rtol=0)
        self.assertGreater(linear_to_srgb(p["warped"]["albedo_linear"])[4, 4, 0], .7)

    def test_occlusion_normal_sign_and_edge_rejection(self):
        current, previous = packet(), packet(frame_id=0)
        previous["normal"].neg_()
        self.assertTrue(project_history(current, previous)["valid"][4, 4])
        previous["normal"][:] = torch.tensor([1., 0., 0.])
        self.assertFalse(project_history(current, previous)["valid"].any())
        self.assertTrue(project_history(current, previous, normal_check=False)["valid"][4, 4])
        previous["normal"][:] = torch.tensor([0., 0., 1.])
        previous["depth"][:, 10:] = 4
        p = project_history(current, previous)
        self.assertFalse(p["valid"][:, 9:].any())
        self.assertTrue(p["valid"][4, 4])

    def test_offset_rechecks_occlusion_and_cycle(self):
        current, previous = packet(), packet(frame_id=0)
        previous["opacity"][:, 8] = .1
        self.assertTrue(project_history(current, previous)["valid"][4, 7])
        offset = torch.zeros(16, 20, 2)
        offset[..., 0] = 1
        self.assertFalse(project_history(current, previous, offset)["valid"][4, 7])
        previous["opacity"].fill_(1)
        offset[..., 0] = 2
        self.assertFalse(project_history(current, previous, offset)["valid"].any())

    def test_entire_interpolation_footprint_must_be_supported(self):
        current, previous = packet(), packet(frame_id=0)
        previous["opacity"][:, 8] = .94
        offset = torch.zeros(16, 20, 2)
        offset[..., 0] = .25
        self.assertFalse(project_history(current, previous, offset)["valid"][4, 7])
        self.assertTrue(project_history(current, previous, offset)["valid"][4, 5])

    def test_projection_only_offset_has_gradient(self):
        current, previous = packet(), packet(frame_id=0)
        previous["albedo"][:] = torch.arange(20).float()[None, :, None] / 20
        for p in (current, previous):
            for key in ("albedo", "depth", "camera_to_world", "intrinsics"):
                p[key].requires_grad_()
        offset = torch.zeros(16, 20, 2, requires_grad=True)
        projection = project_history(current, previous, offset)
        projection["warped"]["albedo_linear"][3:-3, 3:-3].sum().backward()
        self.assertGreater(offset.grad.abs().sum(), 0)
        self.assertFalse(projection["valid"].requires_grad)
        for p in (current, previous):
            for key in ("albedo", "depth", "camera_to_world", "intrinsics"):
                self.assertIsNone(p[key].grad)

    def test_consistency_gradients_do_not_reach_geometry_or_teacher(self):
        current, previous = packet(), packet(frame_id=0)
        previous["albedo"].fill_(.5)
        for p in (current, previous):
            for key in ("albedo", "roughness", "depth", "camera_to_world", "intrinsics", "normal"):
                p[key].requires_grad_()
        result = material_consistency(current, previous)
        self.assertGreater(result["valid_count"], 0)
        result["loss"].backward()
        self.assertGreater(current["albedo"].grad.abs().sum(), 0)
        for key in ("roughness", "depth", "camera_to_world", "intrinsics", "normal"):
            self.assertIsNone(current[key].grad)
        for key in ("albedo", "roughness", "depth", "camera_to_world", "intrinsics", "normal"):
            self.assertIsNone(previous[key].grad)

    def test_empty_consistency_returns_graph_safe_zero(self):
        current, previous = packet(), packet(frame_id=0)
        current["albedo"].requires_grad_()
        previous["depth"].fill_(50)
        result = material_consistency(current, previous)
        self.assertEqual(result["loss"].item(), 0)
        result["loss"].backward()
        self.assertEqual(current["albedo"].grad.abs().sum(), 0)

    def test_routing_corruption_preserves_payload_and_rng(self):
        current, previous = packet(), packet(frame_id=0)
        original = {key: value.clone() for key, value in previous.items() if isinstance(value, torch.Tensor)}
        previous["normal"][..., 0] = .2
        original["normal"] = previous["normal"].clone()
        rng = torch.random.get_rng_state()
        for ablation in ("constant_normal", "shuffle_normal", "constant_roughness", "shuffle_roughness"):
            guides = routing_guides(previous, ablation)
            p = project_history(current, previous, routing_normals=(current["normal"], guides["normal"]))
            torch.testing.assert_close(p["warped"]["normal"], original["normal"], atol=1e-6, rtol=0)
            for key, value in original.items():
                torch.testing.assert_close(previous[key], value, atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_cnn_and_transformer_gradient_to_each_output_head(self):
        for kind in ("cnn", "transformer"):
            with self.subTest(kind=kind):
                torch.manual_seed(14)
                current, previous = packet(), packet(frame_id=0)
                ramp = torch.linspace(.1, .9, 20)[None, :, None]
                previous["albedo"][:] = ramp
                model = EgoTemporalHead(kind=kind, width=8)
                output = model(current, [previous])
                output["albedo"][4:-4, 4:-4].square().mean().backward()
                for name in ("offset_head", "gate_head", "residual_head"):
                    grad = getattr(model, name).weight.grad
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum(), 0)
                self.assertLessEqual(output["offsets"].abs().max(), 2.)
                self.assertTrue(torch.isfinite(output["albedo"]).all())

    def test_invalid_history_cannot_affect_output(self):
        current, previous = packet(), packet(frame_id=0)
        previous["depth"].fill_(100)
        previous["albedo"].fill_(1)
        for kind in ("cnn", "transformer"):
            model = EgoTemporalHead(kind=kind, width=8, max_residual=0)
            output = model(current, [previous])
            torch.testing.assert_close(output["albedo"], current["albedo"], atol=1e-6, rtol=0)
            self.assertFalse(output["valid"].any())
            self.assertEqual(output["history_weight"].sum(), 0)

    def test_no_history_has_current_fallback_and_can_learn_residual(self):
        current = packet()
        model = EgoTemporalHead(width=8, history=False)
        with torch.no_grad():
            model.residual_head.bias.fill_(1)
        output = model(current, [])
        self.assertEqual(output["history_weight"].sum(), 0)
        self.assertGreater(output["albedo"][4, 4, 0], current["albedo"][4, 4, 0])
        self.assertLessEqual(output["residual_linear"].abs().max(), .1)
        # Dynamic geometry is preserved exactly even if a residual is predicted.
        current["static_mask"][4, 4] = False
        output = model(current, [])
        torch.testing.assert_close(output["albedo"][4, 4], current["albedo"][4, 4], atol=0, rtol=0)

    def test_geometric_control_no_material_routing_inputs(self):
        current, previous = packet(), packet(frame_id=0)
        model = EgoTemporalHead(width=8, guides="geometry")
        with torch.no_grad():
            for head in (model.offset_head, model.gate_head, model.residual_head):
                head.weight.normal_(0, .05)
        first = model(current, [previous])
        current["roughness"].fill_(.9)
        current["normal"][:] = torch.tensor([1., 0., 0.])
        previous["metallic"].fill_(.9)
        second = model(current, [previous])
        for key in ("albedo", "offsets", "gate_weights", "valid"):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_control_parameter_counts_and_layout_validation(self):
        full, geometry = EgoTemporalHead(width=8), EgoTemporalHead(width=8, guides="geometry")
        self.assertEqual(sum(p.numel() for p in full.parameters()),
                         sum(p.numel() for p in geometry.parameters()))
        for width in (0, 3, 4.5):
            with self.assertRaises(ValueError):
                EgoTemporalHead(width=width)
        current, previous = packet(), packet(frame_id=0)
        previous["depth"] = previous["depth"][..., 0]
        with self.assertRaisesRegex(ValueError, "depth must have shape"):
            project_history(current, previous)
        previous = packet(frame_id=0)
        previous["intrinsics"].zero_()
        with self.assertRaisesRegex(ValueError, "invertible"):
            project_history(current, previous)

    def test_future_cross_stream_and_excess_history_are_rejected(self):
        model = EgoTemporalHead(width=8)
        current, previous = packet(), packet(frame_id=2)
        with self.assertRaisesRegex(ValueError, "strictly precede"):
            model(current, [previous])
        previous["frame_id"], previous["stream_id"] = 0, "test"
        with self.assertRaisesRegex(ValueError, "same camera"):
            model(current, [previous])
        with self.assertRaisesRegex(ValueError, "At most three"):
            model(current, [previous] * 4)

    def test_rgb_ground_truth_extra_keys_cannot_enter_features(self):
        current, previous = packet(), packet(frame_id=0)
        model = EgoTemporalHead(width=8)
        with torch.no_grad():
            for head in (model.offset_head, model.gate_head, model.residual_head):
                head.weight.normal_(0, .05)
        original = model(current, [previous])["albedo"]
        current["ground_truth"] = torch.rand(16, 20, 3)
        previous["pixels"] = torch.rand(16, 20, 3)
        torch.testing.assert_close(model(current, [previous])["albedo"], original, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
