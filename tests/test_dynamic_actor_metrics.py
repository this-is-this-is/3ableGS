"""Reference-only support, honest aggregation and chronological dynamic diagnostics."""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from tools import eval_base_v5
from utils.dynamic_actor_metrics import (DynamicTemporalDiagnostic, component_crops,
    evaluate_dynamic_frame, reference_mask, summarize_dynamic_frames)
from utils.temporal_metrics import TemporalEvaluator


class LPIPSSpy:
    def __init__(self):
        self.inputs = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, prediction, target):
        self.inputs.append((prediction.clone(), target.clone()))
        return (prediction - target).abs().mean()


def masks(mask):
    return {key: torch.from_numpy(mask.copy()) for key in
            ("dynamic_masks", "human_masks", "vehicle_masks")}


class DynamicActorMetricTests(unittest.TestCase):
    def test_reference_support_excludes_sky_ego_and_requires_masks(self):
        mask = np.ones((9, 9), bool)
        images = masks(mask)
        images["sky_masks"] = np.eye(9)
        images["egocar_masks"] = np.flip(np.eye(9), 0)
        result = reference_mask(images, "dynamic_masks", mask.shape)
        self.assertEqual(int(result.sum()), 64)
        with self.assertRaisesRegex(ValueError, "requires reference"):
            reference_mask({}, "dynamic_masks", mask.shape)
        with self.assertRaisesRegex(ValueError, "finite"):
            reference_mask({"dynamic_masks": np.full(mask.shape, np.nan)}, "dynamic_masks", mask.shape)

    def test_masked_error_ignores_background_and_ssim_uses_entire_valid_window(self):
        mask = np.zeros((19, 19), bool)
        mask[5:14, 5:14] = True
        target = torch.full((19, 19, 3), .2)
        pred = torch.full_like(target, .9)
        pred[mask] = .3
        metric = LPIPSSpy()
        result = evaluate_dynamic_frame({"hybrid": pred}, target, masks(mask), metric)
        row = result["dynamic"]
        self.assertEqual(row["pixels"], 81)
        self.assertEqual(row["ssim_valid_pixels"], 9)
        self.assertAlmostEqual(row["branches"]["hybrid"]["psnr"], 20., places=5)
        # Unmasked crops include the background; black-mask LPIPS would conceal it.
        self.assertGreater(row["branches"]["hybrid"]["component_crop_lpips"][0], .1)
        self.assertTrue(all(min(prediction.shape[-2:]) >= 64 for prediction, _ in metric.inputs))
        self.assertEqual(metric.resets, 2 * len(metric.inputs))

    def test_empty_regions_are_null_not_perfect_and_tiny_objects_have_no_ssim_windows(self):
        mask = np.zeros((9, 9), bool)
        images = masks(mask)
        images["human_masks"][4, 4] = True
        target = torch.zeros(9, 9, 3)
        result = evaluate_dynamic_frame({"hybrid": target}, target, images, LPIPSSpy())
        self.assertIsNone(result["dynamic"]["branches"]["hybrid"]["psnr"])
        self.assertEqual(result["human"]["branches"]["hybrid"]["psnr"], 120.)
        self.assertIsNone(result["human"]["branches"]["hybrid"]["ssim_valid_window"])
        self.assertEqual(result["human"]["eligible_crop_count"], 0)
        summary = summarize_dynamic_frames([result])
        self.assertEqual(summary["dynamic"]["nonempty_frames"], 0)
        self.assertIsNone(summary["dynamic"]["branches"]["hybrid"]["component_mean_crop_lpips"])

    def test_summary_is_pixel_weighted_and_crops_are_shared_across_branches(self):
        target = torch.zeros(16, 16, 3)
        a = np.zeros((16, 16), bool); a[:4, :4] = True
        b = np.zeros((16, 16), bool); b[:8, :8] = True
        spy = LPIPSSpy()
        first = evaluate_dynamic_frame({"pure": target + .1, "hybrid": target + .2}, target, masks(a), spy)
        second = evaluate_dynamic_frame({"pure": target + .2, "hybrid": target + .3}, target, masks(b), spy)
        summary = summarize_dynamic_frames([first, second])["dynamic"]
        expected = -10 * np.log10((16 * .01 + 64 * .04) / 80)
        self.assertAlmostEqual(summary["branches"]["pure"]["pixel_weighted_psnr"], expected, places=5)
        self.assertEqual(summary["branches"]["pure"]["crop_count"], 2)
        # Same reference-defined bbox and interpolation for every branch.
        self.assertEqual(spy.inputs[0][0].shape, spy.inputs[3][0].shape)
        self.assertEqual(summary["pixels"], 80)

    def test_components_are_regions_and_small_components_are_reported_but_not_cropped(self):
        mask = np.zeros((40, 40), bool)
        mask[1:5, 1:5] = True
        mask[20:25, 20:25] = True
        mask[39, 39] = True
        crops, count = component_crops(mask)
        self.assertEqual(count, 3)
        self.assertEqual(len(crops), 2)
        self.assertEqual(crops[0]["box_xyxy"], [0, 0, 13, 13])

    def test_invalid_rgb_is_rejected(self):
        target = torch.zeros(9, 9, 3)
        images = masks(np.ones((9, 9), bool))
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_dynamic_frame({"hybrid": target + float("nan")}, target, images, LPIPSSpy())
        with self.assertRaisesRegex(ValueError, "Reference RGB"):
            evaluate_dynamic_frame({"hybrid": target}, target + 2, images, LPIPSSpy())

    def test_temporal_changes_are_reference_subtracted_and_splits_are_honest(self):
        diagnostic = DynamicTemporalDiagnostic([0, 2], [1])
        images = masks(np.ones((12, 12), bool))
        flow = lambda _self, source, target: np.zeros((*source.shape, 2), np.float32)
        with patch.object(TemporalEvaluator, "_flow", flow):
            for frame in range(3):
                reference = np.full((12, 12, 3), .1 + .2 * frame, np.float32)
                diagnostic.update(reference, {"pure": reference + .1,
                                               "hybrid": reference + (.05 if frame == 1 else 0)},
                                  images, frame, 0)
        result = diagnostic.summary()
        self.assertFalse(result["heldout_temporal_benchmark"])
        self.assertEqual(result["subsets"]["all_adjacent"]["temporal_pairs"], 2)
        self.assertEqual(result["subsets"]["train_to_test"]["temporal_pairs"], 1)
        self.assertEqual(result["subsets"]["test_to_train"]["temporal_pairs"], 1)
        self.assertEqual(result["subsets"]["test_to_test"]["temporal_pairs"], 0)
        all_pairs = result["subsets"]["all_adjacent"]
        self.assertIn("temporal_dynamic_pixels", all_pairs)
        self.assertNotIn("temporal_static_pixels", all_pairs)
        self.assertAlmostEqual(all_pairs["pure_temporal_residual_l1"], 0., places=6)
        self.assertAlmostEqual(all_pairs["hybrid_temporal_residual_l1"], .05, places=6)
        self.assertAlmostEqual(all_pairs["pure_raw_warp_l1"], .2, places=6)

    def test_isolated_test_frames_have_no_temporal_pairs(self):
        diagnostic = DynamicTemporalDiagnostic([], [8, 16])
        rgb = np.zeros((9, 9, 3), np.float32)
        images = masks(np.ones((9, 9), bool))
        diagnostic.update(rgb, {"hybrid": rgb}, images, 8, 0)
        diagnostic.update(rgb, {"hybrid": rgb}, images, 16, 0)
        self.assertEqual(diagnostic.summary()["subsets"]["all_adjacent"]["temporal_pairs"], 0)

    def test_temporal_evaluator_rejects_overlapping_or_unknown_split_ids(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            DynamicTemporalDiagnostic([0], [0])
        diagnostic = DynamicTemporalDiagnostic([0], [1])
        rgb = np.zeros((9, 9, 3), np.float32)
        with self.assertRaisesRegex(ValueError, "outside"):
            diagnostic.update(rgb, {"hybrid": rgb}, {}, 2, 0)

    def test_flags_are_opt_in(self):
        args = eval_base_v5.make_parser().parse_args(["--checkpoint", "x.pth"])
        self.assertFalse(args.dynamic_metrics)
        self.assertFalse(args.dynamic_temporal)

    def test_full_sequence_inference_never_receives_reference_masks_or_images(self):
        class Split:
            def __len__(self):
                return 3

            def get_image(self, index, downscale, weathers):
                images = {**masks(np.ones((9, 9), bool)), "pixels": torch.full((9, 9, 3), .2),
                          "frame_idx": torch.tensor([index]), "img_idx": torch.tensor([index]),
                          "gbuf_basecolor": torch.ones(9, 9, 3)}
                return {"raw": images}, {"raw": {"cam_id": torch.tensor([0])}}

        class Trainer:
            device = torch.device("cpu")

            def _get_downscale_factor(self):
                return 1.

            def __call__(self, inputs, cameras, weather):
                assert set(inputs) == {"frame_idx", "img_idx"}
                return {"rgb_pure_pbr": torch.full((9, 9, 3), .3), "rgb_hybrid": torch.full((9, 9, 3), .4)}

        dataset = SimpleNamespace(full_image_set=Split(), train_timesteps=[0, 2], test_timesteps=[1])
        config = SimpleNamespace(data=SimpleNamespace(scene_idx=788, pixel_source=SimpleNamespace(cameras=[0])))
        result = eval_base_v5.evaluate_dynamic_sequence(config, dataset, Trainer())
        self.assertEqual(result["subsets"]["all_adjacent"]["temporal_pairs"], 2)

    def test_optional_diagnostics_preserve_native_metrics_and_report_support(self):
        from tests.test_rgb_joint_evaluation import config, outputs

        class Split:
            def __len__(self):
                return 2

            def get_image(self, index, downscale, weathers):
                frame = [8, 16][index]
                images = {**masks(np.ones((9, 9), bool)), "pixels": torch.full((9, 9, 3), .5),
                          "frame_idx": torch.tensor([frame]), "img_idx": torch.tensor([frame])}
                return {"raw": images}, {"raw": {"cam_id": torch.tensor([0])}}

        class Trainer:
            device = torch.device("cpu")
            step = 30000
            rgb_joint_camera_updates = torch.tensor(30001)
            lpips = LPIPSSpy()

            def _get_downscale_factor(self):
                return 1.

            def __call__(self, inputs, cameras, weather):
                assert set(inputs) == {"frame_idx", "img_idx"}
                return outputs()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_final.pth"
            checkpoint.write_bytes(b"mock")
            OmegaConf.save(config(), Path(directory) / "config.yaml")
            dataset = SimpleNamespace(test_image_set=Split(), test_timesteps=[8, 16])
            with patch.object(eval_base_v5, "load_scene", return_value=(config(), dataset, Trainer())):
                baseline = eval_base_v5.run(eval_base_v5.make_parser().parse_args(["--checkpoint", str(checkpoint)]))
                supplemented = eval_base_v5.run(eval_base_v5.make_parser().parse_args(
                    ["--checkpoint", str(checkpoint), "--dynamic-metrics"]))
            self.assertNotIn("dynamic_object_metrics", baseline)
            for key in ("image_metrics", "pure_pbr_image_metrics", "delta_hybrid_minus_pure_pbr"):
                self.assertEqual(baseline[key], supplemented[key])
            self.assertEqual(supplemented["dynamic_object_metrics"]["summary"]["dynamic"]["nonempty_frames"], 2)
            report = (checkpoint.parent / "rgb_joint_eval/report.md").read_text()
            self.assertIn("actor/track 평균이 아닙니다", report)


if __name__ == "__main__":
    unittest.main()
