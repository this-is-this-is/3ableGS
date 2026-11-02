"""CPU tests for deterministic, frozen sparse surface edits."""

import unittest

import numpy as np
import torch

from models.surface_material_edit import SurfaceMaterialEditor


class SurfaceMaterialEditTests(unittest.TestCase):
    @staticmethod
    def frame(w=9):
        gbuffer = {
            "albedo": torch.full((1, w, 3), 0.3),
            "roughness": torch.full((1, w, 1), 0.6),
            "metallic": torch.zeros(1, w, 1),
            "normal": torch.zeros(1, w, 3),
        }
        gbuffer["normal"][..., 2] = 1
        return dict(
            gbuffer=gbuffer, depth=torch.ones(1, w, 1), opacity=torch.ones(1, w, 1),
            static_mask=torch.ones(1, w, dtype=torch.bool), camera_to_world=torch.eye(4),
            intrinsics=torch.tensor([[10., 0., 0.], [0., 10., 0.], [0., 0., 1.]]),
            raw_rgb=torch.full((1, w, 3), 0.4),
        )

    @staticmethod
    def editor(**kwargs):
        return SurfaceMaterialEditor(dict(voxel_size=.04, sample_stride=1,
                                          neighbor_radius=.151, knn=3, steps=30, **kwargs))

    def build(self, editor, frame, seed_x=0):
        seed = torch.zeros_like(frame["static_mask"])
        seed[0, seed_x] = True
        editor.add_keyframe(**frame, frame_id=0, seed_mask=seed)
        editor.finalize()
        return editor

    def test_gbuffer_blocks_same_geometry_albedo_boundary(self):
        frame = self.frame()
        frame["gbuffer"]["albedo"][:, 4:] = .95
        geometry = self.build(self.editor(affinity="geometry"), frame)
        material = self.build(self.editor(affinity="gbuffer", albedo_sigma=.05), frame)
        _, g = geometry.apply(**frame)
        _, m = material.apply(**frame)
        self.assertGreater(g["edit_weight"][0, 5, 0], .1)
        self.assertLess(m["edit_weight"][0, 5, 0], 1e-10)
        self.assertGreater(m["edit_weight"][0, 2, 0], .4)

    def test_rgb_affinity_uses_observed_color_and_gbuffer_ignores_it(self):
        frame = self.frame()
        frame["raw_rgb"][:, 4:] = .95
        rgb = self.build(self.editor(affinity="rgb", rgb_sigma=.03), frame)
        gbuf = self.build(self.editor(affinity="gbuffer"), frame)
        self.assertLess(rgb.apply(**frame)[1]["edit_weight"][0, 5, 0], 1e-10)
        self.assertGreater(gbuf.apply(**frame)[1]["edit_weight"][0, 5, 0], .1)

    def test_seeds_locked_and_current_normals_metallic_preserved(self):
        frame = self.frame()
        editor = self.build(self.editor(), frame)
        self.assertTrue((editor._strength[editor._seeds] == 1).all())
        result, info = editor.apply(**frame)
        for key in ("normal", "metallic"):
            torch.testing.assert_close(result[key], frame["gbuffer"][key], rtol=0, atol=0)
        self.assertAlmostEqual(info["edit_weight"][0, 0, 0].item(), 1, places=6)
        expected = torch.sigmoid(torch.logit(torch.tensor(.6)) - 1)
        torch.testing.assert_close(result["roughness"][0, 0, 0], expected)
        self.assertLess(result["albedo"][0, 0, 0], frame["gbuffer"]["albedo"][0, 0, 0])

    def test_unobserved_dynamic_and_nonfinite_queries_preserve_original(self):
        frame = self.frame()
        editor = self.build(self.editor(), frame)
        query = self.frame()
        query["static_mask"][0, 2] = False
        query["opacity"][0, 3, 0] = .5
        query["depth"][0, 4, 0] = float("nan")
        query["depth"][0, 5, 0] = 100.
        result, info = editor.apply(**query)
        for x in (2, 3, 4, 5):
            self.assertEqual(info["edit_weight"][0, x, 0], 0)
            self.assertFalse(info["edit_support"][0, x, 0])
            torch.testing.assert_close(result["albedo"][0, x], query["gbuffer"]["albedo"][0, x])

    def test_finalized_bank_is_order_independent_and_cannot_be_updated(self):
        first, second = self.frame(), self.frame()
        second["gbuffer"]["roughness"] += .01
        seed = torch.zeros_like(first["static_mask"])
        seed[0, 0] = True
        a, b = self.editor(), self.editor()
        for editor, observations in ((a, [(0, first), (1, second)]),
                                     (b, [(1, second), (0, first)])):
            for frame_id, frame in observations:
                editor.add_keyframe(**frame, frame_id=frame_id, seed_mask=seed)
            editor.finalize()
        np.testing.assert_array_equal(a._points, b._points)
        np.testing.assert_array_equal(a._strength, b._strength)
        before = a.apply(**first)[1]["edit_weight"]
        a.apply(**second)
        after = a.apply(**first)[1]["edit_weight"]
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            a.add_keyframe(**first, frame_id=2, seed_mask=seed)
        self.assertEqual(a.summary, a.finalize())

    def test_training_whitelist_duplicate_frames_and_empty_seed_are_rejected(self):
        frame = self.frame()
        seed = torch.zeros_like(frame["static_mask"])
        editor = self.editor(allowed_frame_ids=[0])
        with self.assertRaisesRegex(ValueError, "allowed training"):
            editor.add_keyframe(**frame, frame_id=1, seed_mask=seed)
        editor.add_keyframe(**frame, frame_id=0, seed_mask=seed)
        with self.assertRaisesRegex(ValueError, "already"):
            editor.add_keyframe(**frame, frame_id=0, seed_mask=seed)
        with self.assertRaisesRegex(ValueError, "valid seed"):
            editor.finalize()

    def test_normal_sign_flip_accepts_and_perpendicular_rejects(self):
        frame = self.frame()
        editor = self.build(self.editor(), frame)
        frame["gbuffer"]["normal"] *= -1
        self.assertGreater(editor.apply(**frame)[1]["edit_weight"].sum(), 0)
        frame["gbuffer"]["normal"].zero_()
        frame["gbuffer"]["normal"][..., 0] = 1
        self.assertEqual(editor.apply(**frame)[1]["edit_weight"].sum(), 0)

    def test_material_mismatch_is_not_normalized_away_at_query(self):
        frame = self.frame(w=1)
        editor = self.build(self.editor(albedo_sigma=.01), frame)
        frame["gbuffer"]["albedo"].fill_(.9)
        result, info = editor.apply(**frame)
        self.assertEqual(info["edit_weight"].sum(), 0)
        torch.testing.assert_close(result["albedo"], frame["gbuffer"]["albedo"])

    def test_cap_preserves_seeds_and_sparse_sampling_does_not_miss_seed(self):
        frame = self.frame()
        editor = SurfaceMaterialEditor(dict(max_anchors=2, sample_stride=4, voxel_size=.01))
        seed = torch.zeros_like(frame["static_mask"])
        seed[0, 3] = True
        editor.add_keyframe(**frame, frame_id=0, seed_mask=seed)
        summary = editor.finalize()
        self.assertEqual(summary["anchor_count"], 2)
        self.assertEqual(summary["seed_anchor_count"], 1)

    def test_chunk_size_does_not_change_result_and_outputs_are_detached(self):
        frame = self.frame()
        frame["gbuffer"]["albedo"].requires_grad_(True)
        small = self.build(self.editor(query_chunk_size=2), frame)
        large = self.build(self.editor(query_chunk_size=1024), frame)
        a, _ = small.apply(**frame)
        b, _ = large.apply(**frame)
        torch.testing.assert_close(a["albedo"], b["albedo"], atol=0, rtol=0)
        self.assertFalse(a["albedo"].requires_grad)

    def test_config_and_lifecycle_validation(self):
        for config in ({"affinity": "oops"}, {"steps": -1}, {"voxel_size": 0},
                       {"delta_albedo_logit": [1, 2]}, {"knn": 0}, {"unknown": 3}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                SurfaceMaterialEditor(config)
        with self.assertRaisesRegex(RuntimeError, "finalize"):
            self.editor().apply(**self.frame())
        with self.assertRaisesRegex(ValueError, "No training"):
            self.editor().finalize()
        frame = self.frame()
        frame.pop("raw_rgb")
        with self.assertRaisesRegex(ValueError, "requires raw_rgb"):
            self.build(self.editor(affinity="rgb"), frame)

    def test_zero_opacity_is_never_an_anchor_even_with_zero_threshold(self):
        frame = self.frame()
        frame["opacity"].zero_()
        seed = torch.ones_like(frame["static_mask"])
        editor = self.editor(min_opacity=0)
        editor.add_keyframe(**frame, frame_id=0, seed_mask=seed)
        with self.assertRaisesRegex(ValueError, "no valid static"):
            editor.finalize()


if __name__ == "__main__":
    unittest.main()
