"""Paired scratch training, unchanged scoring, and queue recovery contracts."""
import contextlib
import copy
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from tools.run_diffreg_experiments import (
    build_plan, cleanup_intermediate_gbuffers, digest, make_parser, run_queue,
    stage_artifacts, training_config,
)


def source_config():
    return {
        "seed": 0, "log_dir": "recorded_directory",
        "trainer": {"type": "models.trainers.MultiTrainer_pbr",
                    "optim": {"num_iters": 30000},
                    "render": {"antialiased": False, "packed": False},
                    "res_schedule": {"double_steps": 250, "downscale_times": 2},
                    "losses": {"rgb": {"w": .7}, "ssim": {"w": .3},
                               "albedo": {"w": .5}, "roughness": {"w": .2},
                               "metallic": {"w": .2}, "normal": {"w": .15}}},
        "model": {"Background": {"ctrl": {"stop_split_at": 25000, "stop_reset_at": 20000}}},
        "render": {"render_test": True, "render_full": False, "render_novel": None},
        "logging": {"saveckpt_freq": 5000},
        "data": {"data_root": "unused", "scene_idx": 788, "start_timestep": 0,
                 "end_timestep": 99, "pixel_source": {"cameras": [0], "load_weather": [],
                 "load_gbuffer": True, "gbuffer_path": "original_teacher", "test_image_stride": 8,
                 "downscale": 1, "downscale_when_loading": [1], "undistort": True}},
    }


class DiffRegQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "config.yaml"
        self.output = self.root / "output"
        OmegaConf.save(OmegaConf.create(source_config()), self.source)
        self.calls = []
        for name in ("code_identity", "input_identity"):
            item = patch("tools.run_diffreg_experiments." + name, return_value={"fixture": "unchanged"})
            item.start()
            self.addCleanup(item.stop)

    def args(self, *extra):
        return make_parser().parse_args(["--source-config", str(self.source),
            "--output-dir", str(self.output), "--steps", "3", "--min-free-gb", "0", *extra])

    def fake_child(self, command, log, lock):
        self.calls.append(command)
        def flag(name):
            return command[command.index(name) + 1]
        if "tools.train" in command:
            self.assertNotIn("--resume_from", command)
            self.assertNotIn("--checkpoint", command)
            cfg = OmegaConf.load(flag("--config_file"))
            output = Path(flag("--output_root")) / flag("--project") / flag("--run_name")
            self.assertFalse(cfg.render.render_test)
            self.assertEqual(cfg.logging.saveckpt_freq, cfg.trainer.optim.num_iters)
            output.mkdir(parents=True)
            for name in ("buffer_maps", "gbuffer_cache"):
                cache = output / name / "raw"
                cache.mkdir(parents=True)
                (cache / "intermediate.bin").write_bytes(b"gbuffer")
            final_render = output / "videos/raw"
            final_render.mkdir(parents=True)
            (final_render / "training_result.mp4").write_bytes(b"render")
            cfg.log_dir = str(output)
            OmegaConf.save(cfg, output / "config.yaml")
            torch.save({"step": int(cfg.trainer.optim.num_iters),
                        "models": {"Background": {"x": torch.ones(1)}}}, output / "checkpoint_final.pth")
        elif "tools.eval" in command:
            checkpoint = Path(flag("--resume_from"))
            cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
            self.assertTrue(cfg.render.render_test)
            self.assertEqual(cfg.log_dir, str(checkpoint.parent))
            self.assertEqual(cfg.data.pixel_source.test_image_stride, 8)
            is_r = cfg.trainer.losses.diffreg_material.enabled
            output = checkpoint.parent / "metrics_eval/raw"
            output.mkdir(parents=True)
            (output / "images_test_fixture.json").write_text(json.dumps({
                "image_metrics/raw/test/psnr": 30.01 if is_r else 30.,
                "image_metrics/raw/test/ssim": .921 if is_r else .92,
                "image_metrics/raw/test/lpips": .129 if is_r else .13,
                "image_metrics/raw/test/occupied_psnr": 29.,
            }))
        else:
            self.assertIn("R_s", flag("--checkpoint"))
            self.assertIn("B0_s", flag("--baseline-checkpoint"))
            output = Path(flag("--output-dir"))
            output.mkdir(parents=True)
            (output / "summary.json").write_text(json.dumps({"temporal": {
                "baseline_temporal_residual_l1": .01, "prediction_temporal_residual_l1": .009,
                "temporal_pairs": 99,
            }}))
            (output / "frames.jsonl").write_text('{"frame_id":0}\n')
            (output / "manifest.json").write_text('{"status":"complete"}')

    def test_preview_is_read_only_and_never_requires_source_weights(self):
        args = self.args()
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "checkpoint_final.pth").exists())
        self.assertFalse(self.calls)
        self.assertEqual(result["updates_per_training"], 4)
        self.assertEqual(result["test_ids"], list(range(8, 100, 8)))
        self.assertEqual(result["train_frames"], 88)
        self.assertEqual(len(result["commands"]), 5)

    def test_config_only_changes_declared_loss_and_common_execution_settings(self):
        source = source_config()
        untouched = copy.deepcopy(source)
        base, regularized = (training_config(source, name, 30000, 0) for name in ("B0", "R"))
        self.assertEqual(source, untouched)
        self.assertEqual(base["data"], source["data"])
        self.assertEqual(base["model"], source["model"])
        self.assertFalse(base["trainer"]["losses"].pop("diffreg_material")["enabled"])
        self.assertTrue(regularized["trainer"]["losses"].pop("diffreg_material")["enabled"])
        self.assertEqual(base, regularized)
        self.assertEqual(base["trainer"], source["trainer"])
        self.assertEqual(base["logging"]["saveckpt_freq"], 30000)

    def test_execution_uses_paired_fresh_training_and_original_metrics(self):
        before = digest(self.source)
        args = self.args("--execute", "--seeds", "0", "1")
        state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(digest(self.source), before)
        trains = [cmd for cmd in self.calls if "tools.train" in cmd]
        self.assertEqual(len(trains), 4)
        self.assertTrue(all("--resume_from" not in cmd for cmd in trains))
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(len(summary["image_metrics"]), 4)
        self.assertEqual(len(summary["paired_comparisons"]), 2)
        self.assertAlmostEqual(summary["paired_comparisons"][0]["image_delta_R_minus_B0"]["psnr"], .01)
        self.assertAlmostEqual(summary["paired_comparisons"][0]["temporal_residual_reduction_percent"], 10.)
        self.assertTrue((self.output / "summary.csv").is_file())
        cleanup = state["intermediate_gbuffer_cleanup"]
        self.assertEqual(cleanup["removed_directories"], [
            "stages/B0_s0_train/attempt_001/buffer_maps",
            "stages/B0_s0_train/attempt_001/gbuffer_cache",
            "stages/R_s0_train/attempt_001/buffer_maps",
            "stages/R_s0_train/attempt_001/gbuffer_cache",
            "stages/B0_s1_train/attempt_001/buffer_maps",
            "stages/B0_s1_train/attempt_001/gbuffer_cache",
            "stages/R_s1_train/attempt_001/buffer_maps",
            "stages/R_s1_train/attempt_001/gbuffer_cache",
        ])
        self.assertEqual(cleanup["removed_files"], 8)
        self.assertEqual(cleanup["removed_bytes"], 8 * len(b"gbuffer"))
        for train_id in ("B0_s0_train", "R_s0_train", "B0_s1_train", "R_s1_train"):
            directory = Path(state["stages"][train_id]["attempts"][0]["directory"])
            self.assertFalse((directory / "buffer_maps").exists())
            self.assertFalse((directory / "gbuffer_cache").exists())
            self.assertEqual((directory / "videos/raw/training_result.mp4").read_bytes(), b"render")
        self.assertEqual(summary["intermediate_gbuffer_cleanup"], cleanup)
        run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(len(self.calls), 10)

    def test_failed_evaluation_does_not_repeat_completed_training(self):
        args = self.args("--execute", "--skip-temporal")
        def fail_image(command, log, lock):
            if "tools.eval" in command:
                raise RuntimeError("simulated image evaluation failure")
            self.fake_child(command, log, lock)
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            run_queue(args, build_plan(args), child_runner=fail_image)
        before = json.loads((self.output / "state.json").read_text())
        self.assertEqual(before["stages"]["B0_s0_train"]["attempts"][0]["status"], "completed")
        train_directory = Path(before["stages"]["B0_s0_train"]["attempts"][0]["directory"])
        self.assertTrue((train_directory / "buffer_maps/raw/intermediate.bin").is_file())
        self.assertTrue((train_directory / "gbuffer_cache/raw/intermediate.bin").is_file())
        state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(state["status"], "completed")
        self.assertNotIn("error", state)
        self.assertEqual(sum("tools.train" in cmd for cmd in self.calls), 2)
        attempts = state["stages"]["B0_s0_images"]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(Path(attempts[1]["directory"]).name, "attempt_002")
        self.assertTrue(Path(attempts[0]["directory"]).is_dir())

    def test_cleanup_refuses_attempts_outside_managed_stage_root(self):
        outside = self.root / "outside"
        cache = outside / "gbuffers"
        cache.mkdir(parents=True)
        (cache / "important.bin").write_bytes(b"keep")
        state = {"stages": {"bad": {"attempts": [{"directory": str(outside)}]}}}
        with self.assertRaisesRegex(ValueError, "outside"):
            cleanup_intermediate_gbuffers(self.output, state)
        self.assertEqual((cache / "important.bin").read_bytes(), b"keep")

    def test_changed_config_options_or_artifacts_are_rejected(self):
        args = self.args("--execute", "--skip-temporal")
        state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        altered = self.args("--execute", "--skip-temporal", "--steps", "4")
        with self.assertRaisesRegex(ValueError, "changed"):
            run_queue(altered, build_plan(altered), child_runner=self.fake_child)
        config = Path(state["stages"]["B0_s0_train"]["attempts"][0]["directory"]) / "config.yaml"
        config.write_text("corrupted")
        with self.assertRaisesRegex(ValueError, "artifact is missing or changed"):
            run_queue(args, build_plan(args), child_runner=self.fake_child)

    def test_cannot_silently_change_split_in_child(self):
        args = self.args("--execute", "--skip-temporal")
        def wrong_split(command, log, lock):
            self.fake_child(command, log, lock)
            if "tools.train" in command:
                get = lambda name: command[command.index(name) + 1]
                path = Path(get("--output_root")) / get("--project") / get("--run_name") / "config.yaml"
                cfg = OmegaConf.load(path)
                cfg.data.pixel_source.test_image_stride = 4
                OmegaConf.save(cfg, path)
        with self.assertRaisesRegex(ValueError, "protocol"):
            run_queue(args, build_plan(args), child_runner=wrong_split)

    def test_locks_and_unrelated_files_are_preserved(self):
        self.output.mkdir()
        unrelated = self.output / "important.txt"
        unrelated.write_text("preserve")
        args = self.args("--execute")
        with self.assertRaisesRegex(ValueError, "not an initialized"):
            run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(unrelated.read_text(), "preserve")
        unrelated.unlink()
        with (self.output / ".queue.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertFalse(self.calls)

    def test_low_disk_fails_before_child(self):
        args = self.args("--execute", "--min-free-gb", "999999")
        with self.assertRaisesRegex(RuntimeError, "GiB free"):
            run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertFalse(self.calls)

    def test_wrong_final_step_and_invalid_metric_are_rejected(self):
        args = self.args("--execute", "--skip-temporal")
        plan = build_plan(args)
        state = run_queue(args, plan, child_runner=self.fake_child)
        train = Path(state["stages"]["B0_s0_train"]["attempts"][0]["directory"])
        torch.save({"step": 2, "models": {"Background": {"x": torch.ones(1)}}}, train / "checkpoint_final.pth")
        with self.assertRaisesRegex(ValueError, "final step"):
            stage_artifacts(plan["stages"][0], train)
        images = Path(state["stages"]["B0_s0_images"]["attempts"][0]["directory"])
        metric = next((images / "metrics_eval/raw").glob("*.json"))
        metric.write_text(json.dumps({"image_metrics/raw/test/psnr": float("nan")}))
        with self.assertRaisesRegex(ValueError, "Invalid original"):
            stage_artifacts(plan["stages"][1], images)

    def test_existing_adapters_and_invalid_arguments_are_rejected(self):
        for modification in ({"rgb_joint": {"enabled": True}},
                             {"teacher_calibration": {"enabled": True}},
                             {"teacher_confidence": {"enabled": True}},
                             {"gbuffer_attention": {"variant": "A2"}}):
            cfg = source_config()
            cfg["trainer"].update(modification)
            OmegaConf.save(OmegaConf.create(cfg), self.source)
            with self.assertRaises(ValueError):
                build_plan(self.args())
        OmegaConf.save(OmegaConf.create(source_config()), self.source)
        for options in (("--steps", "0"), ("--seeds", "0", "0"), ("--min-free-gb", "nan")):
            with self.assertRaises(ValueError):
                build_plan(self.args(*options))


if __name__ == "__main__":
    unittest.main()
