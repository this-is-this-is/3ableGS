"""Paired native metrics, inference isolation and signed residual exports."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from skimage.metrics import structural_similarity

from tools import eval_base_v5 as evaluator


class LPIPSSpy:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, prediction, target):
        assert prediction.ndim == 4 and prediction.shape[1] == 3
        return (prediction - target).abs().mean()


def outputs(height=9, width=9, active=True):
    pure = torch.full((height, width, 3), .3)
    hybrid = torch.full_like(pure, .4) if active else pure.clone()
    return {
        "rgb": hybrid, "rgb_pure_pbr": pure, "rgb_hybrid": hybrid,
        "opacity": torch.ones(height, width, 1), "rgb_gain": torch.ones_like(pure),
        "rgb_log_gain": torch.zeros_like(pure), "rgb_additive": torch.full_like(pure, -.05),
        "rgb_residual": hybrid - pure,
        "rgb_residual_strength": torch.tensor(1. if active else 0.),
        "rgb_residual_active": torch.tensor(active),
    }


def config():
    return OmegaConf.create({
        "trainer": {"optim": {"num_iters": 30000}, "render": {"antialiased": False},
                    "rgb_joint": {"enabled": True, "warmup_camera_updates": 2000,
                                  "ramp_camera_updates": 1000, "pure_pbr_lambda": .3}},
        "data": {"scene_idx": 788, "start_timestep": 0, "end_timestep": 23,
                 "pixel_source": {"cameras": [0], "test_image_stride": 8}},
    })


class RGBJointEvaluationTests(unittest.TestCase):
    def test_checkpoint_load_discards_training_state_before_recursive_device_move(self):
        from utils import weather_training
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_final.pth"
            torch.save({"step": 6, "rgb_joint_camera_updates": torch.tensor(6),
                        "models": {"RigidNodes": {"nonbuffer": [torch.arange(3.)]}},
                        "rgb_joint": {"camera_updates": 6},
                        "optimizer": {"unused_moments": torch.ones(4)},
                        "rgb_joint_training_state": {"optimizer": torch.ones(5), "rng": torch.ones(2)}}, path)
            real_model_state = weather_training.model_state
            calls = []

            def move_only_model(archive, device):
                self.assertNotIn("rgb_joint_training_state", archive)
                self.assertNotIn("optimizer", archive)
                self.assertEqual(device, "cpu")
                calls.append(True)
                return real_model_state(archive, device)

            with patch.object(torch, "load", wraps=torch.load) as load, \
                    patch.object(weather_training, "model_state", side_effect=move_only_model):
                state = evaluator.read_model_state(path, "cpu")
            load.assert_called_once_with(path, map_location="cpu", weights_only=False, mmap=True)
            self.assertEqual(calls, [True])
            self.assertEqual(int(state["rgb_joint_camera_updates"]), 6)
            torch.testing.assert_close(state["models"]["RigidNodes"]["nonbuffer"][0], torch.arange(3.))

    def test_native_metric_definitions_and_independent_metric_state(self):
        prediction = torch.rand(9, 9, 3) * .4
        target = prediction + .1
        lpips = LPIPSSpy()
        result = evaluator.image_metrics(prediction, target, lpips)
        self.assertAlmostEqual(result["psnr"], 20., places=5)
        self.assertAlmostEqual(result["ssim"], structural_similarity(
            prediction.numpy(), target.numpy(), data_range=1.0, channel_axis=-1), places=7)
        self.assertAlmostEqual(result["lpips"], .1, places=6)
        self.assertEqual(lpips.resets, 2)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            evaluator.image_metrics(prediction * float("nan"), target, lpips)

    def test_statistics_are_pixel_weighted_and_zero_bounds_are_defined(self):
        stats = evaluator.ResidualStatistics(.3, .1)
        one = outputs(1, 1)
        one["rgb_log_gain"].fill_(.3)
        one["rgb_additive"].fill_(-.1)
        stats.update(one)
        three = outputs(1, 3, active=False)
        three["rgb_additive"].zero_()
        stats.update(three)
        ignored = outputs(1, 2)
        ignored["opacity"].fill_(.1)
        stats.update(ignored)
        result = stats.summary()
        self.assertEqual(result["valid_pixels"], 4)
        self.assertAlmostEqual(result["channels"]["r"]["gain_fraction_at_least_95pct_bound"], .25)
        self.assertAlmostEqual(result["channels"]["r"]["mean_additive"], -.025)
        self.assertEqual(result["active_frames"], 2)
        self.assertEqual(result["bypassed_frames"], 1)
        zero = evaluator.ResidualStatistics(0, 0)
        zero.update(outputs(1, 1, active=False))
        self.assertIsNone(zero.summary()["channels"]["r"]["gain_fraction_at_least_95pct_bound"])

    def test_signed_maps_keep_negative_values_and_encoding_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = outputs(1, 1)
            packet["rgb_residual"][0, 0] = torch.tensor([-.6, 0., .6])
            evaluator.save_images(Path(directory), "frame", torch.ones(1, 1, 3), packet, .3, .1)
            with np.load(Path(directory) / "frame_residual.npz") as saved:
                np.testing.assert_array_equal(saved["rgb_additive"], packet["rgb_additive"].numpy())
                self.assertAlmostEqual(float(saved["rgb_residual_png_scale"]), .6, places=6)
                self.assertTrue(saved["rgb_residual_active"])
                self.assertEqual(float(saved["rgb_residual_strength"]), 1.)
            self.assertTrue((Path(directory) / "frame_paired.png").is_file())

    def test_counter_budget_does_not_infer_step_plus_one(self):
        trainer = SimpleNamespace(step=30000, rgb_joint_camera_updates=torch.tensor(30000))
        budget = evaluator.budget_metadata(config(), trainer)
        self.assertEqual(budget["cumulative_training_iterations"], 30000)
        self.assertEqual(budget["checkpoint_step"], 30000)
        self.assertFalse(budget["base_model_frozen"])
        self.assertIsNone(budget["initialization_checkpoint"])

    def test_full_evaluator_uses_only_heldout_views_without_gt_inputs(self):
        class Split:
            def __len__(self):
                return 2

            def get_image(self, index, downscale, weathers):
                self_downscale = float(downscale)
                assert self_downscale == 1.
                frame = [8, 16][index]
                return ({"raw": {"frame_idx": torch.tensor([frame]), "img_idx": torch.tensor([frame]),
                                  "pixels": torch.full((9, 9, 3), .5), "sky_masks": torch.zeros(9, 9),
                                  "gbuffer_basecolor": torch.full((9, 9, 3), 100.),
                                  "lidar_depth_map": torch.ones(9, 9)}},
                        {"raw": {"cam_id": torch.tensor([0])}})

        class Trainer:
            device = torch.device("cpu")
            step = 34999
            rgb_joint_camera_updates = torch.tensor(35000)
            lpips = LPIPSSpy()

            def _get_downscale_factor(self):
                return 1.

            def __call__(self, inputs, cameras, weather):
                assert set(inputs) == {"frame_idx", "img_idx"}
                assert int(inputs["frame_idx"]) in {8, 16}
                assert weather == "raw"
                return outputs()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_final.pth"
            checkpoint.write_bytes(b"trusted mocked checkpoint")
            OmegaConf.save(config(), Path(directory) / "config.yaml")
            args = evaluator.make_parser().parse_args(["--checkpoint", str(checkpoint), "--save-images"])
            dataset = SimpleNamespace(test_image_set=Split(), test_timesteps=[8, 16])
            with patch.object(evaluator, "load_scene", return_value=(config(), dataset, Trainer())):
                result = evaluator.run(args)
            self.assertEqual(result["training_budget"]["completed_camera_updates"], 35000)
            self.assertFalse(result["smoke"])
            self.assertEqual(len(result["frames"]), 2)
            self.assertAlmostEqual(result["delta_hybrid_minus_pure_pbr"]["psnr"], 6.0206, places=3)
            self.assertFalse(result["protocol"]["teacher_inputs"])
            self.assertIn("Current jointly trained", result["pure_pbr_definition"])
            self.assertEqual(result["rgb_residual_statistics"]["valid_pixels"], 162)
            report = (args.output_dir / "report.md").read_text()
            self.assertIn("현재 공동 학습 모델", report)
            self.assertIn("7×7 uniform", report)
            self.assertNotIn("기하 고정", report)
            restored = json.loads((args.output_dir / "evaluation.json").read_text())
            self.assertEqual(restored["image_metrics"], result["image_metrics"])
            # Smoke classification uses the real completed-camera budget, not
            # configured 30k iterations or the checkpoint filename's step.
            Trainer.rgb_joint_camera_updates = torch.tensor(6)
            with patch.object(evaluator, "load_scene", return_value=(config(), dataset, Trainer())):
                smoke = evaluator.run(args)
            self.assertTrue(smoke["smoke"])
            self.assertIn("최종 학습 성능으로 해석하면 안 됩니다", (args.output_dir / "report.md").read_text())
            Trainer.rgb_joint_camera_updates = torch.tensor(35000)
            smoke_directory = Path(directory) / "rgb_joint_smoke"
            smoke_directory.mkdir()
            named_checkpoint = smoke_directory / "checkpoint_final.pth"
            named_checkpoint.write_bytes(b"trusted mocked checkpoint")
            OmegaConf.save(config(), smoke_directory / "config.yaml")
            named_args = evaluator.make_parser().parse_args(["--checkpoint", str(named_checkpoint)])
            with patch.object(evaluator, "load_scene", return_value=(config(), dataset, Trainer())):
                named = evaluator.run(named_args)
            self.assertTrue(named["smoke"])


if __name__ == "__main__":
    unittest.main()
