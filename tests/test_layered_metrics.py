"""CPU checks for common-reference layered renderer evaluation."""

import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from utils.layered_metrics import LayeredBoundaryEvaluator, summarize_layered_records
from utils.temporal_metrics import TemporalEvaluator


class LayeredMetricTests(unittest.TestCase):
    @staticmethod
    def image(value=.2, height=24, width=32):
        return np.full((height, width, 3), value, np.float32)

    @staticmethod
    def mask(image):
        mask = np.zeros(image.shape[:2], bool)
        mask[:, image.shape[1] // 2:] = True
        return mask

    @staticmethod
    def flow(image, dx=0., dy=0.):
        flow = np.zeros((*image.shape[:2], 2), np.float32)
        flow[..., 0], flow[..., 1] = dx, dy
        return flow

    def test_real_temporal_change_is_compensated_on_dynamic_edges(self):
        previous, current = self.image(.2), self.image(.4)
        mask = self.mask(previous)
        evaluator = LayeredBoundaryEvaluator()
        evaluator.evaluate(previous, {"perfect": previous, "frozen": previous}, mask, 0)
        with mock.patch.object(evaluator, "_flow", return_value=self.flow(current)):
            result = evaluator.evaluate(current, {"perfect": current, "frozen": previous}, mask, 1)
        self.assertEqual(result["global_temporal_valid_pixels"], 24 * 32)
        self.assertEqual(result["object_boundary_r3_pixels"], 24 * 6)
        self.assertEqual(result["object_boundary_r3_temporal_valid_pixels"], 24 * 6)
        self.assertEqual(result["perfect_object_boundary_r3_temporal_residual_l1"], 0.)
        self.assertAlmostEqual(result["perfect_object_boundary_r3_raw_warp_l1"], .2, places=6)
        self.assertAlmostEqual(result["frozen_object_boundary_r3_temporal_residual_l1"], .2, places=6)
        self.assertGreater(result["dynamic_interior_temporal_valid_pixels"], 0)

    def test_dynamic_masks_only_select_current_regions_not_correspondence(self):
        reference = self.image()
        evaluator = LayeredBoundaryEvaluator(boundary_radii=(1,))
        previous_mask = np.zeros(reference.shape[:2], bool)
        evaluator.evaluate(reference, {"method": reference}, previous_mask, 0)
        with mock.patch.object(evaluator, "_flow", return_value=self.flow(reference)):
            result = evaluator.evaluate(reference, {"method": reference}, self.mask(reference), 1)
        self.assertEqual(result["global_temporal_valid_coverage"], 1.)
        self.assertEqual(result["object_boundary_r1_temporal_valid_coverage"], 1.)
        self.assertEqual(result["dynamic_interior_temporal_valid_coverage"], 1.)

    def test_correspondence_and_regions_do_not_depend_on_predictions(self):
        reference = self.image()
        mask = self.mask(reference)
        evaluator = LayeredBoundaryEvaluator()
        evaluator.evaluate(reference, {"perfect": reference, "bad": reference}, mask, 0)
        backward, forward = self.flow(reference, -1), self.flow(reference, 1)
        with mock.patch.object(evaluator, "_flow", side_effect=[backward, forward]):
            result = evaluator.evaluate(reference, {"perfect": reference, "bad": self.image(.8)}, mask, 1)
        for region in ("global", "object_boundary_r3", "dynamic_interior"):
            for metric in ("rgb_mae", "gradient_l1", "temporal_residual_l1"):
                self.assertEqual(result[f"perfect_{region}_{metric}_count"],
                                 result[f"bad_{region}_{metric}_count"])
        self.assertEqual(result["global_temporal_valid_pixels"], 24 * 31)
        self.assertAlmostEqual(result["bad_global_temporal_residual_l1"], .6, places=6)

    def test_flow_rejection_is_current_rgb_error_not_occlusion(self):
        reference = self.image()
        mask = self.mask(reference)
        evaluator = LayeredBoundaryEvaluator()
        evaluator.evaluate(reference, {"method": reference}, mask, 0)
        forward = self.flow(reference)
        forward[10:12, 15:17, 0] = 2
        with mock.patch.object(evaluator, "_flow", side_effect=[self.flow(reference), forward]):
            result = evaluator.evaluate(reference, {"method": self.image(.5)}, mask, 1)
        self.assertEqual(result["global_flow_rejected_pixels"], 4)
        self.assertEqual(result["object_boundary_r3_flow_rejected_pixels"], 4)
        self.assertAlmostEqual(result["method_object_boundary_r3_flow_rejected_l1"], .3, places=6)
        self.assertFalse(any("occlusion" in key for key in result))

    def test_invalid_ego_pixels_are_excluded_without_eroding_neighbours(self):
        reference = self.image()
        dynamic = self.mask(reference)
        previous_valid = np.ones(reference.shape[:2], bool)
        previous_valid[10, 15] = False
        current_valid = np.ones_like(previous_valid)
        current_valid[10, 16] = False
        evaluator = LayeredBoundaryEvaluator()
        evaluator.evaluate(reference, {"method": reference}, dynamic, 0, previous_valid)
        with mock.patch.object(evaluator, "_flow", return_value=self.flow(reference)):
            result = evaluator.evaluate(reference, {"method": reference}, dynamic, 1, current_valid)
        self.assertEqual(result["global_pixels"], 24 * 32 - 1)
        self.assertEqual(result["global_temporal_valid_pixels"], 24 * 32 - 2)
        self.assertEqual(result["global_flow_rejected_pixels"], 1)
        self.assertEqual(result["object_boundary_r3_temporal_valid_pixels"], 24 * 6 - 2)

    def test_gap_reset_and_resolution_change_have_no_temporal_pair(self):
        evaluator = LayeredBoundaryEvaluator()
        reference = self.image()
        mask = self.mask(reference)
        first = evaluator.evaluate(reference, {"method": reference}, mask, 0)
        gap = evaluator.evaluate(reference, {"method": reference}, mask, 3)
        self.assertFalse(first["temporal_has_pair"])
        self.assertFalse(gap["temporal_has_pair"])
        self.assertIsNone(gap["method_global_temporal_residual_l1"])
        with mock.patch.object(evaluator, "_flow", return_value=self.flow(reference)):
            contiguous = evaluator.evaluate(reference, {"method": reference}, mask, 4)
        self.assertTrue(contiguous["temporal_has_pair"])
        smaller = reference[:12]
        resized = evaluator.evaluate(smaller, {"method": smaller}, mask[:12], 5)
        self.assertFalse(resized["temporal_has_pair"])
        evaluator.reset()
        self.assertFalse(evaluator.evaluate(reference, {"new": reference}, mask, 6)["temporal_has_pair"])

    def test_sparse_depth_holes_are_not_depth_discontinuities(self):
        reference = self.image()
        depth = np.zeros(reference.shape[:2], np.float32)
        depth[5, 5], depth[10, 10] = 10, 100
        evaluator = LayeredBoundaryEvaluator(boundary_radii=(1,))
        result = evaluator.evaluate(reference, {"method": reference}, self.mask(reference), 0, depth=depth)
        self.assertEqual(result["sparse_reference_depth_boundary_r1_pixels"], 0)
        depth[5, 6] = 20
        result = evaluator.evaluate(reference, {"method": reference}, self.mask(reference), 2, depth=depth)
        self.assertEqual(result["sparse_reference_depth_boundary_r1_pixels"], 2)
        self.assertEqual(result["method_sparse_reference_depth_boundary_r1_rgb_mae_count"], 6)

    def test_cache_shared_by_methods_and_dynamic_regions_but_not_old_evaluator(self):
        reference = self.image()
        mask = self.mask(reference)
        with tempfile.TemporaryDirectory() as directory:
            old = TemporalEvaluator(directory, {"camera": 0})
            valid = np.ones(reference.shape[:2], bool)
            old.evaluate(reference, {"old": reference}, valid, 0)
            with mock.patch.object(old, "_flow", return_value=self.flow(reference)):
                old.evaluate(reference, {"old": reference}, valid, 1)
            first = LayeredBoundaryEvaluator(directory, {"camera": 0})
            first.evaluate(reference, {"a": reference}, mask, 0)
            with mock.patch.object(first, "_flow", return_value=self.flow(reference)) as flow:
                result = first.evaluate(reference, {"a": reference}, mask, 1)
                self.assertEqual(flow.call_count, 2)
                self.assertFalse(result["temporal_cache_hit"])
            second = LayeredBoundaryEvaluator(directory, {"camera": 0})
            second.evaluate(reference, {"b": self.image(.5)}, ~mask, 0)
            with mock.patch.object(second, "_flow", side_effect=AssertionError("must reuse reference flow")):
                result = second.evaluate(reference, {"b": self.image(.8)}, ~mask, 1)
            self.assertTrue(result["temporal_cache_hit"])
            self.assertEqual(result["global_temporal_valid_pixels"], 24 * 32)

    def test_pooled_statistics_and_psnr_are_channel_weighted(self):
        reference = self.image(0)
        evaluator = LayeredBoundaryEvaluator()
        mask = self.mask(reference)
        valid = np.zeros(reference.shape[:2], bool)
        valid[0, 0] = True
        first = evaluator.evaluate(reference, {"method": self.image(1)}, mask, 0, valid)
        valid[0, 1:4] = True
        second = evaluator.evaluate(reference, {"method": reference}, mask, 2, valid)
        summary = summarize_layered_records([first, second])
        self.assertEqual(summary["method_global_rgb_mse_count"], 15)
        self.assertAlmostEqual(summary["method_global_rgb_mse"], .2)
        self.assertAlmostEqual(summary["method_global_rgb_mae"], .2)
        self.assertAlmostEqual(summary["method_global_rgb_psnr"], -10 * np.log10(.2))
        self.assertIsNone(summary["global_temporal_valid_coverage"])
        self.assertEqual(summary["temporal_pairs"], 0)
        json.dumps(summary, allow_nan=False)
        json.dumps(first, allow_nan=False)
        self.assertEqual(summarize_layered_records([])["frames"], 0)

    def test_empty_regions_and_uniform_masks_are_safe(self):
        reference = self.image()
        evaluator = LayeredBoundaryEvaluator()
        mask = np.zeros(reference.shape[:2], bool)
        for dynamic in (mask, ~mask):
            result = evaluator.evaluate(reference, {"method": reference}, dynamic, 0)
            self.assertEqual(result["object_boundary_r3_pixels"], 0)
            self.assertEqual(result["method_global_rgb_psnr"], 120.)
            self.assertIsNone(result["method_object_boundary_r3_rgb_psnr"])
            json.dumps(result, allow_nan=False)

    def test_input_validation_and_history_copies(self):
        reference = self.image()
        evaluator = LayeredBoundaryEvaluator()
        mask = self.mask(reference)
        with self.assertRaises(ValueError):
            LayeredBoundaryEvaluator(boundary_radii=(0,))
        with self.assertRaises(ValueError):
            evaluator.evaluate(reference, {"method": reference}, mask.astype(np.uint8), 0)
        evaluator.evaluate(reference, {"method": reference}, mask, 0)
        with self.assertRaises(ValueError):
            evaluator.evaluate(reference, {"other": reference}, mask, 1)
        reference[:] = .4
        with mock.patch.object(evaluator, "_flow", return_value=self.flow(reference)):
            result = evaluator.evaluate(reference, {"method": reference}, mask, 1)
        self.assertAlmostEqual(result["method_global_raw_warp_l1"], .2, places=6)
        self.assertEqual(result["method_global_temporal_residual_l1"], 0.)


if __name__ == "__main__":
    unittest.main()
