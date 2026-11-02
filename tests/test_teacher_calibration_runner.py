"""CPU planning and fake-child checks; never start a scene training process."""
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

from tests.test_diffreg_runner import source_config
from tools.run_teacher_calibration_experiments import (
    build_plan, make_parser, run_queue, stage_artifacts, training_config,
)


class TeacherCalibrationQueueTests(unittest.TestCase):
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
        return make_parser().parse_args([
            "--source-config", str(self.source), "--output-dir", str(self.output),
            "--min-free-gb", "0", *extra,
        ])

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
            self.assertNotIn("diffreg_material", cfg.trainer.losses)
            self.assertNotIn("temporal_material", cfg.trainer.losses)
            output.mkdir(parents=True)
            cfg.log_dir = str(output)
            OmegaConf.save(cfg, output / "config.yaml")
            checkpoint = {"step": int(cfg.trainer.optim.num_iters),
                          "models": {"Background": {"x": torch.ones(1)}}}
            if cfg.trainer.teacher_calibration.enabled:
                from models.teacher_calibration import TeacherCalibration
                train_ids = [frame for frame in range(100) if frame == 0 or frame % 8]
                head = TeacherCalibration(train_ids, 1, cfg.trainer.teacher_calibration)
                checkpoint["_teacher_calibration"] = head.metadata()
                checkpoint.update({"teacher_calibration_head." + key: value
                                   for key, value in head.state_dict().items()})
            torch.save(checkpoint, output / "checkpoint_final.pth")
            (output / "metrics.json").write_text(json.dumps({
                "iteration": int(cfg.trainer.optim.num_iters),
                "calibration_stats/raw/gain_abs_mean": .01,
                "losses/raw/teacher_calibration_identity_loss": .002,
                "losses/raw/teacher_calibration_smooth_loss": .001,
                "unrelated_stat": 99,
            }) + "\n")
        elif "tools.eval" in command:
            checkpoint = Path(flag("--resume_from"))
            cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
            self.assertTrue(cfg.render.render_test)
            self.assertEqual(cfg.data.pixel_source.test_image_stride, 8)
            self.assertEqual(cfg.log_dir, str(checkpoint.parent))
            enabled = cfg.trainer.teacher_calibration.enabled
            output = checkpoint.parent / "metrics_eval/raw"
            output.mkdir(parents=True)
            (output / "images_test_fixture.json").write_text(json.dumps({
                "image_metrics/raw/test/psnr": 30.02 if enabled else 30.,
                "image_metrics/raw/test/ssim": .921 if enabled else .92,
                "image_metrics/raw/test/lpips": .129 if enabled else .13,
            }))
        else:
            self.assertIn("C_s", flag("--checkpoint"))
            self.assertIn("B0_s", flag("--baseline-checkpoint"))
            output = Path(flag("--output-dir"))
            output.mkdir(parents=True)
            (output / "summary.json").write_text(json.dumps({"temporal": {
                "baseline_temporal_residual_l1": .01,
                "prediction_temporal_residual_l1": .009,
                "temporal_pairs": 99,
            }}))
            (output / "frames.jsonl").write_text('{"frame_id":0}\n')
            (output / "manifest.json").write_text('{"status":"complete"}')

    def complete(self):
        args = self.args("--execute", "--skip-temporal")
        plan = build_plan(args)
        with contextlib.redirect_stdout(io.StringIO()):
            state = run_queue(args, plan, child_runner=self.fake_child)
        stage = plan["stages"][2]
        directory = Path(state["stages"][stage["id"]]["attempts"][0]["directory"])
        return plan, state, stage, directory

    def test_default_preview_is_read_only_and_never_loads_weights(self):
        args = self.args()
        with patch("torch.load", side_effect=AssertionError("no source weights")):
            with contextlib.redirect_stdout(io.StringIO()):
                plan = build_plan(args)
                preview = run_queue(args, plan, child_runner=self.fake_child)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls)
        self.assertEqual(preview["updates_per_training"], 30001)
        self.assertEqual(preview["train_frames"], 88)
        self.assertEqual(plan["identity"]["calibration_parameters"], 880)
        self.assertEqual(preview["test_ids"], list(range(8, 100, 8)))
        self.assertEqual(list(preview["commands"]), [
            "B0_s0_train", "B0_s0_images", "C_s0_train", "C_s0_images", "C_s0_temporal",
        ])
        for key, command in preview["commands"].items():
            if key.endswith("_train"):
                self.assertNotIn("--resume_from", command)
                self.assertNotIn("--checkpoint", command)

    def test_only_enable_flag_differs_and_original_losses_are_preserved(self):
        source = source_config()
        before = copy.deepcopy(source)
        base = training_config(source, "B0", 30000, 0)
        active = training_config(source, "C", 30000, 0)
        self.assertEqual(source, before)
        baseline_cfg = base["trainer"].pop("teacher_calibration")
        active_cfg = active["trainer"].pop("teacher_calibration")
        self.assertFalse(baseline_cfg.pop("enabled"))
        self.assertTrue(active_cfg.pop("enabled"))
        self.assertEqual(baseline_cfg, active_cfg)
        self.assertEqual(active_cfg["start_step"], 3000)
        self.assertEqual(active_cfg["max_gain_delta"], .15)
        self.assertEqual(active_cfg["max_bias"], .05)
        self.assertEqual(base, active)
        self.assertEqual(base["trainer"], source["trainer"])
        self.assertEqual(base["data"], source["data"])
        self.assertEqual(base["model"], source["model"])

    def test_fake_execution_summarizes_c_and_skips_verified_stages(self):
        args = self.args("--execute", "--seeds", "0", "1")
        plan = build_plan(args)
        with contextlib.redirect_stdout(io.StringIO()):
            state = run_queue(args, plan, child_runner=self.fake_child)
            run_queue(args, plan, child_runner=self.fake_child)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(self.calls), 10)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(len(summary["image_metrics"]), 4)
        comparison = summary["paired_comparisons"][0]
        self.assertAlmostEqual(comparison["image_delta_C_minus_B0"]["psnr"], .02)
        self.assertIn("training_stage_wall_time_ratio_C_over_B0", comparison)
        self.assertAlmostEqual(comparison["temporal_residual_reduction_percent"], 10.)
        diagnostics = summary["training_diagnostics"]["C_s0_train"]["last_logged_values"]
        self.assertIn("calibration_stats/raw/gain_abs_mean", diagnostics)
        self.assertIn("losses/raw/teacher_calibration_identity_loss", diagnostics)
        self.assertIn("losses/raw/teacher_calibration_smooth_loss", diagnostics)
        self.assertNotIn("unrelated_stat", diagnostics)

    def test_image_failure_retries_without_repeating_training(self):
        args = self.args("--execute", "--skip-temporal")
        def fail_image(command, log, lock):
            if "tools.eval" in command:
                raise RuntimeError("synthetic evaluation failure")
            self.fake_child(command, log, lock)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                run_queue(args, build_plan(args), child_runner=fail_image)
            state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(sum("tools.train" in command for command in self.calls), 2)
        attempts = state["stages"]["B0_s0_images"]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(Path(attempts[0]["directory"]).is_dir())
        self.assertEqual(Path(attempts[1]["directory"]).name, "attempt_002")

    def test_checkpoint_metadata_split_and_parameter_corruption_are_rejected(self):
        _, _, stage, directory = self.complete()
        path = directory / "checkpoint_final.pth"
        original = torch.load(path, weights_only=False)
        invalid = []
        missing = copy.deepcopy(original)
        missing.pop("_teacher_calibration")
        invalid.append(missing)
        wrong_split = copy.deepcopy(original)
        wrong_split["_teacher_calibration"]["train_indices"][0] = 8
        invalid.append(wrong_split)
        wrong_cfg = copy.deepcopy(original)
        wrong_cfg["_teacher_calibration"]["config"]["max_bias"] = .1
        invalid.append(wrong_cfg)
        wrong_camera = copy.deepcopy(original)
        wrong_camera["_teacher_calibration"]["num_cams"] = 2
        invalid.append(wrong_camera)
        nonfinite = copy.deepcopy(original)
        nonfinite["teacher_calibration_head.raw"][0, 0, 0] = float("nan")
        invalid.append(nonfinite)
        wrong_shape = copy.deepcopy(original)
        wrong_shape["teacher_calibration_head.raw"] = torch.zeros(100, 2, 5)
        invalid.append(wrong_shape)
        wrong_buffer = copy.deepcopy(original)
        wrong_buffer["teacher_calibration_head.train_indices"][0] = 8
        invalid.append(wrong_buffer)
        wrong_step = copy.deepcopy(original)
        wrong_step["step"] = 29999
        invalid.append(wrong_step)
        for checkpoint in invalid:
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                stage_artifacts(stage, directory)
        torch.save(original, path)
        with patch("torch.load", wraps=torch.load) as load:
            stage_artifacts(stage, directory)
        self.assertEqual(load.call_count, 1)

    def test_baseline_cannot_hide_calibration_parameters(self):
        plan, state, _, _ = self.complete()
        stage = plan["stages"][0]
        directory = Path(state["stages"][stage["id"]]["attempts"][0]["directory"])
        path = directory / "checkpoint_final.pth"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["teacher_calibration_head.raw"] = torch.zeros(88, 2, 5)
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "Baseline checkpoint"):
            stage_artifacts(stage, directory)

    def test_saved_arm_mismatch_and_other_regularizer_are_rejected(self):
        _, _, stage, directory = self.complete()
        original = OmegaConf.load(directory / "config.yaml")
        for option in ("trainer.teacher_calibration.enabled=false",
                       "trainer.losses.diffreg_material.enabled=true"):
            OmegaConf.save(OmegaConf.merge(original, OmegaConf.from_dotlist([option])), directory / "config.yaml")
            with self.assertRaises(ValueError):
                stage_artifacts(stage, directory)

    def test_source_adapters_and_invalid_arguments_are_rejected(self):
        for key, value in (("teacher_calibration", {"enabled": True}),
                           ("gbuffer_attention", {"variant": "A2"}),
                           ("rgb_joint", {"enabled": True})):
            cfg = source_config()
            cfg["trainer"][key] = value
            OmegaConf.save(OmegaConf.create(cfg), self.source)
            with self.assertRaises(ValueError):
                build_plan(self.args())
        for key, value in (("diffreg_material", {"enabled": True}),
                           ("temporal_material", {"w": .05})):
            cfg = source_config()
            cfg["trainer"]["losses"][key] = value
            OmegaConf.save(OmegaConf.create(cfg), self.source)
            with self.assertRaises(ValueError):
                build_plan(self.args())
        OmegaConf.save(OmegaConf.create(source_config()), self.source)
        for options in (("--steps", "0"), ("--seeds", "0", "0"), ("--min-free-gb", "nan")):
            with self.assertRaises(ValueError):
                build_plan(self.args(*options))

    def test_input_change_is_rejected_before_child(self):
        args = self.args("--execute")
        plan = build_plan(args)
        with patch("tools.run_diffreg_experiments.input_identity", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                run_queue(args, plan, child_runner=self.fake_child)
        self.assertFalse(self.calls)

    def test_lock_and_unrelated_files_are_preserved(self):
        self.output.mkdir()
        important = self.output / "keep.txt"
        important.write_text("preserve")
        args = self.args("--execute")
        with self.assertRaisesRegex(ValueError, "not an initialized"):
            run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(important.read_text(), "preserve")
        important.unlink()
        with (self.output / ".queue.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
