"""CPU checks for experiment scope, metric masks and CLI failure behavior."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from tools.render_temporal_material import (
    edit_keyframe_ids, evaluation_static_mask, frame_indices, inference_config, make_parser, quality_stats,
    render_inputs, summarize, validate_args,
)

ROOT = Path(__file__).resolve().parents[1]


class TemporalRunnerTests(unittest.TestCase):
    def test_edit_keyframes_are_train_only_deterministic_and_spread(self):
        train_frames = [0, 1, 2, 7, 8, 9]
        self.assertEqual(edit_keyframe_ids(train_frames, 3), [0, 2, 9])
        self.assertEqual(edit_keyframe_ids(train_frames, 1), [0])
        with self.assertRaises(ValueError):
            edit_keyframe_ids(train_frames, 7)

    def test_reference_mask_excludes_all_annotations_and_requires_them(self):
        inputs = {
            "pixels": torch.zeros(2, 2, 3),
            "sky_masks": torch.tensor([[1., 0.], [0., 0.]]),
            "dynamic_masks": torch.tensor([[0., 1.], [0., 0.]]),
            "egocar_masks": torch.tensor([[0., 0.], [1., 0.]]),
        }
        np.testing.assert_array_equal(
            evaluation_static_mask(inputs), [[False, False], [False, True]],
        )
        del inputs["dynamic_masks"]
        with self.assertRaisesRegex(ValueError, "dynamic_masks"):
            evaluation_static_mask(inputs)

    def test_help_does_not_require_renderer_imports(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/render_temporal_material.py"), "--help"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--max-history", result.stdout)
        self.assertIn("not an independent temporal metric", result.stdout.replace("\n", " "))

    def test_config_preserves_full_sequence_and_model_topology(self):
        cfg = OmegaConf.create({
            "trainer": {"type": "models.trainers.MultiTrainer_pbr"},
            "data": {
                "start_timestep": 10, "end_timestep": 109, "preload_device": "cuda",
                "pixel_source": {"cameras": [0, 1], "downscale": 1, "load_gbuffer": True,
                                 "load_weather": ["rain"], "test_image_stride": 8,
                                 "downscale_when_loading": [1, 2]},
            },
            "model": {"RigidNodes": {"num_frames": 100}, "Affine": {"n": 200}},
        })
        original = copy.deepcopy(cfg)
        updated = inference_config(cfg, downscale=2)
        self.assertEqual(cfg, original)
        self.assertEqual(updated.data.start_timestep, 10)
        self.assertEqual(updated.data.end_timestep, 109)
        self.assertEqual(updated.data.pixel_source.cameras, [0, 1])
        self.assertEqual(updated.model, cfg.model)
        self.assertEqual(updated.data.preload_device, "cpu")
        self.assertFalse(updated.data.pixel_source.load_gbuffer)
        self.assertEqual(updated.data.pixel_source.load_weather, [])
        self.assertEqual(updated.data.pixel_source.downscale_when_loading, [2, 4])
        self.assertEqual(updated.data.pixel_source.downscale, 1)

    def test_wrong_trainer_and_absent_camera_rejected(self):
        cfg = OmegaConf.create({"trainer": {"type": "weather"},
                                "data": {"pixel_source": {"cameras": [1]}}})
        with self.assertRaisesRegex(ValueError, "raw MultiTrainer_pbr"):
            inference_config(cfg)
        cfg.trainer.type = "models.trainers.MultiTrainer_pbr"
        with self.assertRaisesRegex(ValueError, "camera 0"):
            inference_config(cfg)

    def test_frame_selection_is_relative_and_never_silently_truncated(self):
        self.assertEqual(list(frame_indices(2, 3, 100)), [2, 3, 4])
        self.assertEqual(list(frame_indices(98, None, 100)), [98, 99])
        for args in ((100, None, 100), (-1, 2, 100), (98, 3, 100), (2, 0, 100)):
            with self.assertRaises(ValueError):
                frame_indices(*args)

    def test_references_and_teacher_never_enter_forward(self):
        class NoTensor:
            @staticmethod
            def is_tensor(value):
                return False

        inputs = {key: key for key in ("pixels", "gbuf_normal", "sky_masks", "dynamic_masks",
                                       "lidar_depth_map", "img_idx", "frame_idx", "viewdirs")}
        forwarded = render_inputs(inputs, "cpu", NoTensor)
        self.assertEqual(set(forwarded), {"img_idx", "frame_idx", "viewdirs"})

    def test_static_metrics_share_one_mask_and_empty_region_is_null(self):
        reference = np.zeros((1, 2, 3), np.float32)
        base = np.ones_like(reference)
        improved = base.copy()
        improved[:, 0] = 0.5
        record = quality_stats(base, improved, reference, [[True, False]])
        self.assertAlmostEqual(record["baseline_static_psnr"], 0.0)
        self.assertAlmostEqual(record["transport_static_psnr"], 6.020599913)
        self.assertEqual(record["transport_static_values"], 3)
        self.assertEqual(record["baseline_static_values"], 3)
        empty = quality_stats(base, improved, reference, [[False, False]])
        self.assertIsNone(empty["baseline_static_psnr"])
        self.assertIsNone(empty["transport_static_psnr"])
        self.assertIsNone(empty["static_rgb_delta_mae"])

    def test_summary_aggregates_mse_instead_of_averaging_psnr(self):
        zero = np.zeros((1, 2, 3), np.float32)
        records = []
        for value in (1.0, 0.5):
            record = quality_stats(np.full_like(zero, value), zero, zero, [[True, False]])
            record.update(history_coverage_full=0.5, history_coverage_static=1.,
                          mean_history_weight=0.25, forward_ms=5., split="train")
            records.append(record)
        summary = summarize(records)
        self.assertAlmostEqual(summary["baseline_full_psnr"], -10 * np.log10(0.625))
        self.assertIsNone(summary["temporal_metric"])
        self.assertEqual(summary["train_frames"], 2)

    def test_invalid_parameters_and_occupied_output_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            checkpoint.touch()
            (root / "config.yaml").touch()
            output = root / "result"
            args = make_parser().parse_args(["--checkpoint", str(checkpoint),
                                             "--output-dir", str(output)])
            self.assertEqual(validate_args(args), (checkpoint, output))
            for key, value in (("downscale", 0), ("history_weight", 1),
                               ("history_weight", float("nan")), ("fps", float("inf")),
                               ("max_history", 0), ("num_frames", 0), ("start_frame", -1),
                               ("normal_cos_min", float("nan")), ("depth_rel_tolerance", -1),
                               ("edit_keyframes", 0), ("edit_roi", [.8, .2, .1, .5])):
                invalid = copy.copy(args)
                setattr(invalid, key, value)
                with self.assertRaises(ValueError):
                    validate_args(invalid)
            invalid = copy.copy(args)
            invalid.domain, invalid.edit_affinity = "rgb", "geometry"
            with self.assertRaisesRegex(ValueError, "domain material"):
                validate_args(invalid)
            output.mkdir()
            (output / "keep.txt").write_text("existing")
            with self.assertRaises(FileExistsError):
                validate_args(args)


if __name__ == "__main__":
    unittest.main()
