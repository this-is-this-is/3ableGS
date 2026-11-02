"""CPU regression tests for static-surface temporal material reprojection."""

import unittest

import torch

from models.temporal_material import TemporalMaterialResolver


class TemporalMaterialTests(unittest.TestCase):
    @staticmethod
    def frame(h=5, w=7, dtype=torch.float32, albedo=0.3, roughness=0.4):
        gbuffer = {
            "albedo": torch.full((h, w, 3), albedo, dtype=dtype),
            "roughness": torch.full((h, w, 1), roughness, dtype=dtype),
            "metallic": torch.full((h, w, 1), 0.1, dtype=dtype),
            "normal": torch.zeros(h, w, 3, dtype=dtype),
        }
        gbuffer["normal"][..., 2] = 1
        return dict(
            gbuffer=gbuffer,
            depth=torch.full((h, w, 1), 2., dtype=dtype),
            opacity=torch.ones(h, w, 1, dtype=dtype),
            static_mask=torch.ones(h, w, dtype=torch.bool),
            camera_to_world=torch.eye(4, dtype=dtype),
            intrinsics=torch.tensor([[2., 0., w / 2], [0., 2., h / 2], [0., 0., 1.]], dtype=dtype),
        )

    @staticmethod
    def resolver(**config):
        return TemporalMaterialResolver(dict(mode="depth", erode_radius=0,
                                             depth_edge_rel_tolerance=0, **config))

    @staticmethod
    def run_frame(resolver, frame, index=0, stream="camera-0"):
        return resolver.resolve(**frame, frame_id=index, stream_id=stream)

    def test_first_frame_is_exact_and_diagnostics_have_channel_dimension(self):
        frame = self.frame()
        result, info = self.run_frame(self.resolver(), frame)
        for key in frame["gbuffer"]:
            torch.testing.assert_close(result[key], frame["gbuffer"][key], rtol=0, atol=0)
            self.assertNotEqual(result[key].data_ptr(), frame["gbuffer"][key].data_ptr())
        self.assertEqual(set(info), {"temporal_weight", "temporal_valid", "temporal_history_length",
                                     "temporal_grid"})
        for key in ("temporal_weight", "temporal_valid", "temporal_history_length"):
            self.assertEqual(info[key].shape, (5, 7, 1))
        self.assertEqual(info["temporal_grid"].shape, (5, 7, 2))
        self.assertTrue((info["temporal_grid"] == -2).all())
        self.assertFalse(info["temporal_valid"].any())
        self.assertEqual(info["temporal_weight"].sum(), 0)
        self.assertTrue((info["temporal_history_length"] == 1).all())

    def test_camera_translation_reprojects_material_at_true_surface_location(self):
        resolver = self.resolver()
        previous = self.frame()
        ramp = torch.arange(7).float() / 10
        previous["gbuffer"]["roughness"][:] = ramp[None, :, None]
        self.run_frame(resolver, previous)
        current = self.frame(roughness=0.2)
        current["camera_to_world"][0, 3] = 1.  # fx * delta_x / depth = one pixel.
        result, info = self.run_frame(resolver, current, 1)
        expected = 0.5 * (ramp[1:] + .2)
        torch.testing.assert_close(result["roughness"][2, :-1, 0], expected)
        self.assertTrue(info["temporal_valid"][2, :-1].all())
        self.assertFalse(info["temporal_valid"][2, -1].any())
        self.assertEqual(result["roughness"][2, -1, 0], current["gbuffer"]["roughness"][2, -1, 0])

    def test_fractional_translation_uses_bilinear_pixel_centers(self):
        resolver = self.resolver()
        previous = self.frame()
        ramp = torch.arange(7).float() / 10
        previous["gbuffer"]["roughness"][:] = ramp[None, :, None]
        self.run_frame(resolver, previous)
        current = self.frame(roughness=0.2)
        current["camera_to_world"][0, 3] = .5
        result, _ = self.run_frame(resolver, current, 1)
        expected = .5 * (.5 * (ramp[1:] + ramp[:-1]) + .2)
        torch.testing.assert_close(result["roughness"][2, :-1, 0], expected)

    def test_camera_depth_translation_compares_reprojected_z_not_current_depth(self):
        resolver = self.resolver()
        previous = self.frame(roughness=.2)
        previous["depth"].fill_(3.)
        self.run_frame(resolver, previous)
        current = self.frame(roughness=.8)
        current["camera_to_world"][2, 3] = 1.
        result, info = self.run_frame(resolver, current, 1)
        self.assertTrue(info["temporal_valid"][2, 3, 0])
        self.assertAlmostEqual(result["roughness"][2, 3, 0].item(), .5)

    def test_disocclusion_rejects_history_without_changing_current_material(self):
        resolver = self.resolver()
        previous = self.frame(roughness=.1)
        previous["depth"].fill_(4.)
        self.run_frame(resolver, previous)
        current = self.frame(roughness=.9)
        result, info = self.run_frame(resolver, current, 1)
        self.assertFalse(info["temporal_valid"].any())
        torch.testing.assert_close(result["roughness"], current["gbuffer"]["roughness"], rtol=0, atol=0)

    def test_normal_mode_accepts_sign_flip_and_rejects_orthogonal_normal(self):
        for normal, should_accept in (([0., 0., -1.], True), ([1., 0., 0.], False)):
            resolver = TemporalMaterialResolver({"mode": "normal", "erode_radius": 0})
            self.run_frame(resolver, self.frame(roughness=.2))
            current = self.frame(roughness=.8)
            current["gbuffer"]["normal"][:] = torch.tensor(normal)
            result, info = self.run_frame(resolver, current, 1)
            self.assertEqual(info["temporal_valid"][2, 3, 0].item(), should_accept)
            torch.testing.assert_close(result["normal"], current["gbuffer"]["normal"], rtol=0, atol=0)
            self.assertAlmostEqual(result["roughness"][2, 3, 0].item(), .5 if should_accept else .8)

    def test_material_confidence_suppresses_boundary_but_accepts_small_noise(self):
        resolver = TemporalMaterialResolver({"mode": "material", "erode_radius": 0})
        previous = self.frame(roughness=.2)
        previous["gbuffer"]["roughness"][:, :3] = 1.
        self.run_frame(resolver, previous)
        current = self.frame(roughness=.22)
        current["gbuffer"]["roughness"][:, :3] = 0.
        result, info = self.run_frame(resolver, current, 1)
        self.assertLess(info["temporal_weight"][2, 1, 0], 1e-4)
        self.assertGreater(info["temporal_weight"][2, 4, 0], .49)
        self.assertLess(result["roughness"][2, 1, 0], 1e-4)
        self.assertLess(result["roughness"][2, 4, 0], .211)

    def test_static_mask_and_complete_previous_interpolation_support_are_required(self):
        resolver = self.resolver()
        previous = self.frame(roughness=.2)
        previous["static_mask"][2, 3] = False
        self.run_frame(resolver, previous)
        current = self.frame(roughness=.8)
        current["camera_to_world"][0, 3] = .25
        current["static_mask"][1, 1] = False
        result, info = self.run_frame(resolver, current, 1)
        for y, x in ((2, 2), (2, 3), (1, 1)):
            self.assertFalse(info["temporal_valid"][y, x, 0])
            self.assertEqual(result["roughness"][y, x, 0], current["gbuffer"]["roughness"][y, x, 0])
        self.assertTrue(info["temporal_valid"][3, 3, 0])

    def test_opacity_rejects_transparent_current_and_previous_surface(self):
        resolver = self.resolver()
        previous = self.frame(roughness=.2)
        previous["opacity"][2, 2, 0] = .5
        self.run_frame(resolver, previous)
        current = self.frame(roughness=.8)
        current["opacity"][2, 4, 0] = .9
        result, info = self.run_frame(resolver, current, 1)
        self.assertFalse(info["temporal_valid"][2, 2, 0])
        self.assertFalse(info["temporal_valid"][2, 4, 0])
        self.assertTrue(info["temporal_valid"][2, 3, 0])
        for x in (2, 4):
            self.assertEqual(result["roughness"][2, x, 0], current["gbuffer"]["roughness"][2, x, 0])

    def test_previous_albedo_is_interpolated_in_linear_space(self):
        resolver = self.resolver()
        previous = self.frame(albedo=0.)
        previous["gbuffer"]["albedo"][:, 3:] = 1.
        self.run_frame(resolver, previous)
        current = self.frame(albedo=0.)
        current["camera_to_world"][0, 3] = .5
        result, _ = self.run_frame(resolver, current, 1)
        # Half black/half white footprint yields .5 linear history; the
        # first temporal blend then contributes another factor of .5.
        expected_srgb = 1.055 * (.25 ** (1 / 2.4)) - .055
        self.assertAlmostEqual(result["albedo"][2, 2, 0].item(), expected_srgb, places=6)

    def test_static_erosion_and_depth_boundary_rejection_are_conservative(self):
        resolver = TemporalMaterialResolver({"mode": "depth"})
        frame = self.frame(h=9, w=9)
        frame["static_mask"][4, 4] = False
        frame["depth"][:, 7:] = 5.
        self.run_frame(resolver, frame)
        _, info = self.run_frame(resolver, frame, 1)
        self.assertFalse(info["temporal_valid"][3:6, 3:6].any())
        self.assertFalse(info["temporal_valid"][:, 6:8].any())
        self.assertFalse(info["temporal_valid"][0].any())
        self.assertTrue(info["temporal_valid"][2, 2, 0])

    def test_albedo_fuses_in_linear_space(self):
        resolver = self.resolver()
        self.run_frame(resolver, self.frame(albedo=0.))
        result, info = self.run_frame(resolver, self.frame(albedo=1.), 1)
        expected_srgb = 1.055 * (.5 ** (1 / 2.4)) - .055
        self.assertAlmostEqual(result["albedo"][2, 3, 0].item(), expected_srgb, places=6)
        self.assertEqual(info["temporal_weight"][2, 3, 0], .5)

    def test_nonfinite_history_never_poison_neighbor_interpolation(self):
        resolver = self.resolver()
        previous = self.frame(roughness=.2)
        previous["gbuffer"]["albedo"][2, 3, 0] = float("nan")
        previous["gbuffer"]["roughness"][1, 2, 0] = float("inf")
        previous["depth"][3, 3, 0] = float("nan")
        self.run_frame(resolver, previous)
        current = self.frame(roughness=.8)
        current["camera_to_world"][0, 3] = .25
        result, info = self.run_frame(resolver, current, 1)
        for value in result.values():
            self.assertTrue(torch.isfinite(value).all())
        for y, x in ((2, 2), (2, 3), (1, 1), (1, 2), (3, 2), (3, 3)):
            self.assertFalse(info["temporal_valid"][y, x, 0])
        self.assertTrue(info["temporal_valid"][1, 4, 0])

    def test_invalid_current_values_are_preserved_without_corrupting_state(self):
        resolver = self.resolver()
        self.run_frame(resolver, self.frame())
        current = self.frame()
        current["gbuffer"]["albedo"][2, 3, 0] = float("nan")
        current["depth"][1, 1, 0] = 0
        current["depth"][1, 2, 0] = -1
        current["depth"][1, 3, 0] = float("inf")
        current["opacity"][3, 3, 0] = float("nan")
        result, info = self.run_frame(resolver, current, 1)
        self.assertTrue(torch.isnan(result["albedo"][2, 3, 0]))
        for y, x in ((2, 3), (1, 1), (1, 2), (1, 3), (3, 3)):
            self.assertFalse(info["temporal_valid"][y, x, 0])
            self.assertEqual(info["temporal_history_length"][y, x, 0], 0)
        for value in resolver._history.values():
            self.assertTrue(torch.isfinite(value).all())
        next_result, _ = self.run_frame(resolver, self.frame(), 2)
        self.assertTrue(torch.isfinite(next_result["albedo"]).all())

    def test_stream_changes_gaps_reverse_order_shape_and_dtype_reset_history(self):
        cases = [
            (self.frame(), 1, "camera-1"),
            (self.frame(), 2, "camera-0"),
            (self.frame(), -1, "camera-0"),
            (self.frame(), 0, "camera-0"),
            (self.frame(h=6), 1, "camera-0"),
            (self.frame(dtype=torch.float64), 1, "camera-0"),
        ]
        for current, frame_id, stream_id in cases:
            with self.subTest(frame_id=frame_id, stream_id=stream_id,
                              shape=current["depth"].shape, dtype=current["depth"].dtype):
                resolver = self.resolver()
                self.run_frame(resolver, self.frame())
                _, info = self.run_frame(resolver, current, frame_id, stream_id)
                self.assertFalse(info["temporal_valid"].any())
                self.assertTrue((info["temporal_history_length"] == 1).all())
        resolver.reset()
        self.assertIsNone(resolver._history)
        self.assertIsNone(resolver._frame_id)
        self.assertIsNone(resolver._stream_id)
        self.assertIsNone(resolver._layout)

    def test_history_count_and_weight_are_bounded(self):
        resolver = self.resolver(max_history=3, history_weight=1.)
        frame = self.frame()
        for index in range(20):
            _, info = self.run_frame(resolver, frame, index)
            self.assertTrue((info["temporal_history_length"] <= 3).all())
            self.assertTrue((info["temporal_weight"] <= 2 / 3 + 1e-7).all())
        self.assertEqual(info["temporal_history_length"][2, 3, 0], 3)
        self.assertAlmostEqual(info["temporal_weight"][2, 3, 0].item(), 2 / 3, places=6)

    def test_disabled_fusion_settings_produce_exact_current_values(self):
        for config in ({"max_history": 1}, {"history_weight": 0}):
            resolver = self.resolver(**config)
            self.run_frame(resolver, self.frame(albedo=0.))
            frame = self.frame(albedo=1.)
            result, info = self.run_frame(resolver, frame, 1)
            self.assertFalse(info["temporal_valid"].any())
            torch.testing.assert_close(result["albedo"], frame["gbuffer"]["albedo"], rtol=0, atol=0)

    def test_history_and_outputs_are_detached_and_not_aliased(self):
        resolver = self.resolver()
        previous = self.frame(roughness=.2)
        previous["gbuffer"]["roughness"].requires_grad_()
        first, _ = self.run_frame(resolver, previous)
        with torch.no_grad():
            previous["gbuffer"]["roughness"].fill_(1.)
        first["roughness"].fill_(0.)
        current = self.frame(roughness=.8)
        current["gbuffer"]["roughness"].requires_grad_()
        result, _ = self.run_frame(resolver, current, 1)
        self.assertAlmostEqual(result["roughness"][2, 3, 0].item(), .5, places=6)
        self.assertAlmostEqual(current["gbuffer"]["roughness"][2, 3, 0].item(), .8, places=6)
        self.assertTrue(all(not value.requires_grad for value in resolver._history.values()))
        self.assertTrue(all(not value.requires_grad for value in result.values()))

    def test_invalid_configuration_fails_early(self):
        for config in ({"unknown": 1}, {"mode": "rgb"}, {"max_history": 0},
                       {"max_history": 1.5}, {"max_history": True}, {"erode_radius": -1},
                       {"history_weight": 1.1}, {"min_opacity": -1}, {"normal_cos_min": 2},
                       {"depth_abs_tolerance": float("nan")}, {"depth_rel_tolerance": float("inf")},
                       {"albedo_sigma": 0}, {"roughness_sigma": -.1}, {"metallic_sigma": True},
                       {"depth_edge_rel_tolerance": -.1}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    TemporalMaterialResolver(config)

    def test_invalid_shapes_and_camera_matrices_fail_clearly(self):
        for mutation in ("depth", "normal", "mask", "camera", "intrinsics"):
            frame = self.frame()
            if mutation == "depth":
                frame["depth"] = frame["depth"][..., 0]
            elif mutation == "normal":
                frame["gbuffer"]["normal"] = frame["gbuffer"]["normal"][..., :2]
            elif mutation == "mask":
                frame["static_mask"] = frame["static_mask"].float()
            elif mutation == "camera":
                frame["camera_to_world"][0, 0] = float("nan")
            else:
                frame["intrinsics"].zero_()
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.run_frame(self.resolver(), frame)

    def test_current_history_avoids_recursive_subpixel_blur(self):
        results = {}
        for source in ("resolved", "current"):
            resolver = self.resolver(history_source=source, history_weight=1.)
            first = self.frame(roughness=0.)
            first["gbuffer"]["roughness"][:, 3, 0] = .9
            self.run_frame(resolver, first)
            for frame_id in (1, 2):
                frame = self.frame(roughness=0.)
                frame["camera_to_world"][0, 3] = .5 * frame_id
                result, info = self.run_frame(resolver, frame, frame_id)
            results[source] = result["roughness"][2, 2, 0].item()
            self.assertAlmostEqual(info["temporal_weight"][2, 2, 0].item(), 2 / 3, places=6)
        # First interpolation spreads .9 over two pixels and halves it;
        # recursive history carries .225 forward with weight 2/3.
        self.assertAlmostEqual(results["resolved"], .15, places=6)
        self.assertAlmostEqual(results["current"], 0., places=6)

    def test_neighborhood_clamp_rejects_stale_material_overshoot(self):
        resolver = self.resolver(clamp_history=True)
        previous = self.frame(albedo=1., roughness=.9)
        previous["gbuffer"]["metallic"].fill_(.9)
        self.run_frame(resolver, previous)
        current = self.frame(albedo=.25, roughness=.2)
        current["gbuffer"]["roughness"][:, 3, 0] = .8
        result, info = self.run_frame(resolver, current, 1)
        self.assertTrue(info["temporal_valid"].all())
        self.assertAlmostEqual(result["roughness"][2, 0, 0].item(), .2, places=6)
        self.assertAlmostEqual(result["roughness"][2, 2, 0].item(), .5, places=6)
        torch.testing.assert_close(result["albedo"], current["gbuffer"]["albedo"])
        torch.testing.assert_close(result["metallic"], current["gbuffer"]["metallic"])

    def test_neighborhood_clamp_ignores_nonfinite_neighbors(self):
        resolver = self.resolver(clamp_history=True)
        self.run_frame(resolver, self.frame(albedo=1., roughness=1.))
        current = self.frame(albedo=.3, roughness=.2)
        current["gbuffer"]["roughness"][2, 3, 0] = float("nan")
        result, info = self.run_frame(resolver, current, 1)
        self.assertFalse(info["temporal_valid"][2, 3, 0])
        self.assertTrue(torch.isnan(result["roughness"][2, 3, 0]))
        self.assertTrue(info["temporal_valid"][2, 2, 0])
        self.assertAlmostEqual(result["roughness"][2, 2, 0].item(), .2, places=6)
        for value in resolver._history.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_all_guide_ablations_preserve_payload_when_history_disabled(self):
        for ablation in ("none", "shuffle_normal", "shuffle_roughness",
                         "constant_normal", "constant_roughness"):
            resolver = TemporalMaterialResolver({"guide_ablation": ablation, "history_weight": 0})
            self.run_frame(resolver, self.frame())
            frame = self.frame(albedo=.7, roughness=.2)
            frame["gbuffer"]["roughness"][1:4, 1:4] = .8
            frame["gbuffer"]["normal"][:, :3] = torch.tensor([1., 0., 0.])
            originals = {key: value.clone() for key, value in frame["gbuffer"].items()}
            result, _ = self.run_frame(resolver, frame, 1)
            for key, expected in originals.items():
                torch.testing.assert_close(result[key], expected, rtol=0, atol=0)
                torch.testing.assert_close(frame["gbuffer"][key], expected, rtol=0, atol=0)

    def test_constant_roughness_changes_only_confidence_not_payload(self):
        resolver = TemporalMaterialResolver({
            "mode": "material", "guide_ablation": "constant_roughness", "erode_radius": 0,
        })
        self.run_frame(resolver, self.frame(roughness=.2))
        result, info = self.run_frame(resolver, self.frame(roughness=.6), 1)
        self.assertAlmostEqual(info["temporal_weight"][2, 3, 0].item(), .5)
        self.assertAlmostEqual(result["roughness"][2, 3, 0].item(), .4, places=6)

    def test_constant_normal_changes_acceptance_without_changing_current_normal(self):
        resolver = TemporalMaterialResolver({
            "mode": "normal", "guide_ablation": "constant_normal", "erode_radius": 0,
        })
        self.run_frame(resolver, self.frame(roughness=.2))
        frame = self.frame(roughness=.8)
        frame["gbuffer"]["normal"][:] = torch.tensor([1., 0., 0.])
        result, info = self.run_frame(resolver, frame, 1)
        self.assertTrue(info["temporal_valid"].all())
        self.assertAlmostEqual(result["roughness"][2, 3, 0].item(), .5)
        torch.testing.assert_close(result["normal"], frame["gbuffer"]["normal"], rtol=0, atol=0)

    def test_shuffle_is_deterministic_frame_dependent_and_does_not_touch_rng(self):
        resolver = TemporalMaterialResolver({"guide_ablation": "shuffle_roughness", "ablation_seed": 3})
        frame = self.frame()
        frame["gbuffer"]["roughness"][:] = torch.arange(35).reshape(5, 7, 1) / 35
        rng_before = torch.get_rng_state().clone()
        first = resolver._routing_guides(frame["gbuffer"], 0)
        second = resolver._routing_guides(frame["gbuffer"], 1)
        again = resolver._routing_guides(frame["gbuffer"], 0)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        torch.testing.assert_close(first["roughness"], again["roughness"], rtol=0, atol=0)
        self.assertFalse(torch.equal(first["roughness"], second["roughness"]))
        torch.testing.assert_close(first["roughness"].flatten().sort().values,
                                   frame["gbuffer"]["roughness"].flatten().sort().values)
        torch.testing.assert_close(first["normal"], frame["gbuffer"]["normal"], rtol=0, atol=0)
        for h, w in ((1, 1), (1, 7), (5, 1)):
            small = self.frame(h=h, w=w)
            self.assertEqual(resolver._routing_guides(small["gbuffer"], 2)["roughness"].shape,
                             (h, w, 1))

    def test_shuffled_normal_changes_routing_without_changing_shading_normals(self):
        outputs = {}
        for ablation in ("none", "shuffle_normal"):
            resolver = TemporalMaterialResolver({
                "mode": "normal", "guide_ablation": ablation, "erode_radius": 0,
            })
            previous = self.frame(roughness=.2)
            previous["gbuffer"]["normal"][:, :3] = torch.tensor([1., 0., 0.])
            self.run_frame(resolver, previous)
            current = self.frame(roughness=.8)
            current["gbuffer"]["normal"][:, :3] = torch.tensor([1., 0., 0.])
            result, info = self.run_frame(resolver, current, 1)
            torch.testing.assert_close(result["normal"], current["gbuffer"]["normal"], rtol=0, atol=0)
            outputs[ablation] = info["temporal_valid"]
        self.assertTrue(outputs["none"].all())
        self.assertLess(outputs["shuffle_normal"].sum(), outputs["none"].sum())

    def test_rgb_signal_uses_same_grid_and_weight_without_brightness_gating(self):
        resolver = self.resolver()
        previous = self.frame(roughness=.2)
        previous["signal"] = torch.arange(7).float()[None, :, None].expand(5, 7, 3) / 10
        self.run_frame(resolver, previous)
        current = self.frame(roughness=.8)
        current["camera_to_world"][0, 3] = .5
        current["signal"] = torch.ones(5, 7, 3)
        result, info = self.run_frame(resolver, current, 1)
        history = resolver._sample(previous["signal"], info["temporal_grid"])
        expected = ((1 - info["temporal_weight"]) * current["signal"]
                    + info["temporal_weight"] * history)
        torch.testing.assert_close(info["temporal_signal"], expected)
        self.assertEqual(info["temporal_weight"][2, 3, 0], .5)
        self.assertAlmostEqual(result["roughness"][2, 3, 0].item(), .5)
        torch.testing.assert_close(info["temporal_signal"][2, -1], current["signal"][2, -1],
                                   rtol=0, atol=0)

    def test_signal_history_source_clamp_and_resets(self):
        for source, expected in (("resolved", 2 / 3), ("current", 1.)):
            resolver = self.resolver(history_source=source, history_weight=1.)
            for frame_id, level in enumerate((0., 1., 1.)):
                frame = self.frame()
                frame["signal"] = torch.full((5, 7, 3), level)
                _, info = self.run_frame(resolver, frame, frame_id)
            self.assertAlmostEqual(info["temporal_signal"][2, 3, 0].item(), expected, places=6)
        resolver = self.resolver(clamp_history=True)
        first = self.frame()
        first["signal"] = torch.ones(5, 7, 3)
        self.run_frame(resolver, first)
        second = self.frame()
        second["signal"] = torch.full((5, 7, 3), .2)
        _, info = self.run_frame(resolver, second, 1)
        torch.testing.assert_close(info["temporal_signal"], second["signal"])
        _, info = self.run_frame(resolver, second, 5)
        self.assertFalse(info["temporal_valid"].any())
        torch.testing.assert_close(info["temporal_signal"], second["signal"], rtol=0, atol=0)
        _, info = self.run_frame(resolver, self.frame(), 6)
        self.assertFalse(info["temporal_valid"].any())
        self.assertNotIn("temporal_signal", info)
        self.assertNotIn("signal", resolver._history)
        _, info = self.run_frame(resolver, second, 7)
        self.assertFalse(info["temporal_valid"].any())

    def test_invalid_signals_reject_correspondences_and_preserve_current(self):
        resolver = self.resolver()
        previous = self.frame()
        previous["signal"] = torch.ones(5, 7, 3)
        previous["signal"][2, 3, 0] = float("nan")
        self.run_frame(resolver, previous)
        current = self.frame()
        current["signal"] = torch.full((5, 7, 3), .2)
        current["signal"][1, 1, 0] = float("inf")
        current["camera_to_world"][0, 3] = .25
        _, info = self.run_frame(resolver, current, 1)
        for y, x in ((2, 2), (2, 3), (1, 1)):
            self.assertFalse(info["temporal_valid"][y, x, 0])
        self.assertTrue(torch.isinf(info["temporal_signal"][1, 1, 0]))
        self.assertTrue(torch.isfinite(info["temporal_signal"][2]).all())
        self.assertTrue(all(torch.isfinite(value).all() for value in resolver._history.values()))

    def test_new_options_and_signal_validation(self):
        for config in ({"history_source": "raw"}, {"clamp_history": 1},
                       {"guide_ablation": "shuffle_albedo"}, {"ablation_seed": -1},
                       {"ablation_seed": True}, {"ablation_seed": .5}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    TemporalMaterialResolver(config)
        for signal in (torch.zeros(5, 7), torch.zeros(5, 7, 4),
                       torch.zeros(5, 7, 3, dtype=torch.float64), "invalid"):
            frame = self.frame()
            frame["signal"] = signal
            with self.subTest(signal=type(signal)):
                with self.assertRaises(ValueError):
                    self.run_frame(self.resolver(), frame)


if __name__ == "__main__":
    unittest.main()
