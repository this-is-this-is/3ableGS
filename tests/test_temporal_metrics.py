"""CPU tests for common-reference, pixel-weighted temporal evaluation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from utils.temporal_metrics import TemporalEvaluator, summarize_metric_records


class TemporalMetricTests(unittest.TestCase):
    @staticmethod
    def image(height=16, width=20, value=0.):
        return np.full((height, width, 3), value, np.float32)

    @staticmethod
    def flow(image, dx=0., dy=0.):
        result = np.empty((*image.shape[:2], 2), np.float32)
        result[..., 0] = dx
        result[..., 1] = dy
        return result

    @staticmethod
    def mask(image):
        return np.ones(image.shape[:2], bool)

    def test_first_gap_and_reset_have_edges_without_temporal_metrics(self):
        reference = self.image()
        reference[:, 10:] = 1
        evaluator = TemporalEvaluator()
        for frame in (0, 7):
            result = evaluator.evaluate(reference, {"method": reference}, self.mask(reference), frame)
            self.assertFalse(result["temporal_has_pair"])
            self.assertIsNone(result["method_temporal_residual_l1"])
            self.assertGreater(result["edge_pixels"], 0)
            self.assertEqual(result["method_edge_gradient_l1"], 0.)
            self.assertEqual(result["method_edge_strength_ratio"], 1.)
        evaluator.reset()
        result = evaluator.evaluate(reference, {"different": reference}, self.mask(reference), 8)
        self.assertFalse(result["temporal_has_pair"])

    def test_known_translation_uses_correct_direction_and_common_masks(self):
        rng = np.random.RandomState(5)
        previous = rng.uniform(.1, .8, (16, 20, 3)).astype(np.float32)
        current = np.roll(previous, 1, axis=1)
        evaluator = TemporalEvaluator()
        evaluator.evaluate(previous, {"perfect": previous, "flicker": previous}, self.mask(previous), 0)
        with mock.patch.object(evaluator, "_flow", side_effect=[
            self.flow(current, -1), self.flow(current, 1),
        ]):
            result = evaluator.evaluate(
                current, {"perfect": current, "flicker": current + .1}, self.mask(current), 1,
            )
        # One-pixel mask erosion plus one-pixel horizontal translation.
        self.assertEqual(result["temporal_valid_pixels"], 14 * 17)
        self.assertEqual(result["perfect_temporal_residual_l1"], 0.)
        self.assertAlmostEqual(result["flicker_temporal_residual_l1"], .1, places=6)
        self.assertEqual(result["perfect_temporal_residual_l1_count"],
                         result["flicker_temporal_residual_l1_count"])
        self.assertEqual(result["flow_rejected_pixels"], 14)
        self.assertAlmostEqual(result["flicker_flow_rejected_l1"], .1, places=6)

    def test_real_reference_change_is_compensated_and_static_cheat_penalized(self):
        previous, current = self.image(value=.2), self.image(value=.4)
        evaluator = TemporalEvaluator()
        evaluator.evaluate(previous, {"faithful": previous, "frozen": previous}, self.mask(previous), 0)
        with mock.patch.object(evaluator, "_flow", return_value=self.flow(current)):
            result = evaluator.evaluate(
                current, {"faithful": current, "frozen": previous}, self.mask(current), 1,
            )
        self.assertEqual(result["faithful_temporal_residual_l1"], 0.)
        self.assertAlmostEqual(result["faithful_raw_warp_l1"], .2, places=6)
        self.assertAlmostEqual(result["frozen_temporal_residual_l1"], .2, places=6)
        self.assertEqual(result["frozen_raw_warp_l1"], 0.)

    def test_edge_metrics_penalize_blur_and_constant(self):
        reference = self.image()
        reference[:, 10:] = 1
        blurred = cv2.GaussianBlur(reference, (9, 9), 2)
        result = TemporalEvaluator().evaluate(
            reference, {"perfect": reference, "blurred": blurred, "constant": self.image(value=.5)},
            self.mask(reference), 0,
        )
        self.assertEqual(result["perfect_edge_gradient_l1"], 0.)
        self.assertEqual(result["perfect_edge_strength_ratio"], 1.)
        self.assertGreater(result["blurred_edge_gradient_l1"], 0.)
        self.assertGreater(result["constant_edge_gradient_l1"], result["blurred_edge_gradient_l1"])
        self.assertLess(result["blurred_edge_strength_ratio"], 1.)
        self.assertEqual(result["constant_edge_strength_ratio"], 0.)
        self.assertEqual(result["perfect_edge_gradient_l1_count"], result["blurred_edge_gradient_l1_count"])

    def test_previous_dynamic_mask_and_cycle_failures_are_rejected(self):
        reference = self.image(value=.3)
        previous_mask = self.mask(reference)
        previous_mask[8, 10] = False
        evaluator = TemporalEvaluator()
        evaluator.evaluate(reference, {"method": reference}, previous_mask, 0)
        forward = self.flow(reference)
        forward[3:5, 3:5, 0] = 2  # Fails 1.5px cycle threshold at four extra pixels.
        with mock.patch.object(evaluator, "_flow", side_effect=[self.flow(reference), forward]):
            result = evaluator.evaluate(reference, {"method": reference}, self.mask(reference), 1)
        self.assertEqual(result["flow_rejected_pixels"], 9 + 4)
        self.assertEqual(result["temporal_valid_pixels"], 14 * 18 - 13)

    def test_empty_regions_are_null_and_json_serializable(self):
        reference = self.image()
        evaluator = TemporalEvaluator()
        empty_mask = np.zeros(reference.shape[:2], bool)
        first = evaluator.evaluate(reference, {"method": reference}, empty_mask, 0)
        with mock.patch.object(evaluator, "_flow", return_value=self.flow(reference)):
            second = evaluator.evaluate(reference, {"method": reference}, empty_mask, 1)
        summary = summarize_metric_records([first, second])
        self.assertIsNone(summary["method_temporal_residual_l1"])
        self.assertIsNone(summary["method_edge_strength_ratio"])
        self.assertIsNone(summary["temporal_valid_coverage"])
        self.assertEqual(summary["method_temporal_residual_l1_count"], 0)
        json.dumps(summary, allow_nan=False)
        json.dumps(second, allow_nan=False)
        self.assertEqual(summarize_metric_records([])["temporal_pairs"], 0)

    def test_summary_weights_pixels_and_edge_reference_strength(self):
        records = [
            {"temporal_has_pair": True, "temporal_valid_pixels": 2, "temporal_static_pixels": 4,
             "method_temporal_residual_l1_sum": 6., "method_temporal_residual_l1_count": 6,
             "method_edge_strength_ratio_numerator": 2., "method_edge_strength_ratio_denominator": 4.},
            {"temporal_has_pair": True, "temporal_valid_pixels": 8, "temporal_static_pixels": 8,
             "method_temporal_residual_l1_sum": 0., "method_temporal_residual_l1_count": 24,
             "method_edge_strength_ratio_numerator": 8., "method_edge_strength_ratio_denominator": 8.},
        ]
        summary = summarize_metric_records(records)
        self.assertEqual(summary["method_temporal_residual_l1_count"], 30)
        self.assertAlmostEqual(summary["method_temporal_residual_l1"], .2)
        self.assertAlmostEqual(summary["temporal_valid_coverage"], 10 / 12)
        self.assertAlmostEqual(summary["method_edge_strength_ratio"], 10 / 12)

    def test_cache_is_shared_across_methods_and_keyed_by_reference_mask_metadata(self):
        reference = self.image(value=.2)
        mask = self.mask(reference)
        with tempfile.TemporaryDirectory() as temporary:
            def pair(metadata, first=reference, second=reference, current_mask=mask, fail=False):
                evaluator = TemporalEvaluator(temporary, metadata)
                evaluator.evaluate(first, {"other": first}, mask, 0)
                with mock.patch.object(evaluator, "_flow", return_value=self.flow(first)) as flow:
                    if fail:
                        flow.side_effect = AssertionError("Cache should have been reused")
                    result = evaluator.evaluate(second, {"other": second}, current_mask, 1)
                    return result, flow.call_count

            first, count = pair({"camera": 0})
            self.assertEqual(count, 2)
            self.assertFalse(first["temporal_cache_hit"])
            reused, count = pair({"camera": 0}, fail=True)
            self.assertTrue(reused["temporal_cache_hit"])
            self.assertEqual(count, 0)
            changed = reference.copy()
            changed[3, 3] += .01
            self.assertEqual(pair({"camera": 0}, second=changed)[1], 2)
            self.assertEqual(pair({"camera": 0}, first=changed)[1], 2)
            changed_mask = mask.copy()
            changed_mask[3, 3] = False
            self.assertEqual(pair({"camera": 0}, current_mask=changed_mask)[1], 2)
            self.assertEqual(pair({"camera": 1})[1], 2)
            self.assertEqual(len(list(Path(temporary).glob("*.npz"))), 5)
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_invalid_inputs_and_mutated_input_history(self):
        reference = self.image(value=.2)
        evaluator = TemporalEvaluator()
        for invalid in (np.nan, np.inf, -1., 2.):
            bad = reference.copy()
            bad[0, 0, 0] = invalid
            with self.assertRaises(ValueError):
                evaluator.evaluate(bad, {"method": reference}, self.mask(reference), 0)
        with self.assertRaises(ValueError):
            evaluator.evaluate(reference, {"method": reference}, self.mask(reference).astype(np.uint8), 0)
        with self.assertRaises(ValueError):
            evaluator.evaluate(reference, {"method": reference[:3]}, self.mask(reference), 0)
        evaluator.evaluate(reference, {"method": reference}, self.mask(reference), 0)
        with self.assertRaisesRegex(ValueError, "names"):
            evaluator.evaluate(reference, {"changed": reference}, self.mask(reference), 1)
        reference[:] = .4  # Must not mutate stored previous GT/prediction arrays.
        with mock.patch.object(evaluator, "_flow", return_value=self.flow(reference)):
            result = evaluator.evaluate(reference, {"method": reference}, self.mask(reference), 1)
        self.assertAlmostEqual(result["method_raw_warp_l1"], .2, places=6)
        self.assertEqual(result["method_temporal_residual_l1"], 0.)

    def test_actual_flow_on_identical_textured_frames(self):
        rng = np.random.RandomState(42)
        reference = rng.uniform(0, 1, (64, 80, 3)).astype(np.float32)
        evaluator = TemporalEvaluator()
        evaluator.evaluate(reference, {"method": reference}, self.mask(reference), 0)
        result = evaluator.evaluate(reference, {"method": reference}, self.mask(reference), 1)
        self.assertGreater(result["temporal_valid_coverage"], .95)
        self.assertEqual(result["method_temporal_residual_l1"], 0.)


if __name__ == "__main__":
    unittest.main()
