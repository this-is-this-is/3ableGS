"""No-GPU checks of B0/T planning, protocol preservation and queue recovery."""
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
from tools.run_material_consistency_experiments import (
    build_plan, make_parser, run_queue, stage_artifacts, training_config,
)


class MaterialConsistencyQueueTests(unittest.TestCase):
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
            output.mkdir(parents=True)
            cfg.log_dir = str(output)
            OmegaConf.save(cfg, output / "config.yaml")
            torch.save({"step": int(cfg.trainer.optim.num_iters),
                        "models": {"Background": {"x": torch.ones(1)}}}, output / "checkpoint_final.pth")
            (output / "metrics.json").write_text(json.dumps({
                "iteration": int(cfg.trainer.optim.num_iters),
                "temporal_stats/raw/valid_sample_count": 512,
                "losses/raw/temporal_material_loss": .002,
                "unrelated_stat": 99,
            }) + "\n")
        elif "tools.eval" in command:
            checkpoint = Path(flag("--resume_from"))
            cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
            self.assertTrue(cfg.render.render_test)
            self.assertEqual(cfg.data.pixel_source.test_image_stride, 8)
            self.assertEqual(cfg.log_dir, str(checkpoint.parent))
            is_t = cfg.trainer.losses.temporal_material.w > 0
            output = checkpoint.parent / "metrics_eval/raw"
            output.mkdir(parents=True)
            (output / "images_test_fixture.json").write_text(json.dumps({
                "image_metrics/raw/test/psnr": 30.02 if is_t else 30.,
                "image_metrics/raw/test/ssim": .921 if is_t else .92,
                "image_metrics/raw/test/lpips": .129 if is_t else .13,
            }))
        else:
            self.assertIn("T_s", flag("--checkpoint"))
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

    def test_default_preview_is_read_only_and_never_reads_weights(self):
        args = self.args()
        with patch("torch.load", side_effect=AssertionError("no source weights")):
            with contextlib.redirect_stdout(io.StringIO()):
                preview = run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls)
        self.assertEqual(preview["updates_per_training"], 30001)
        self.assertEqual(preview["train_frames"], 88)
        self.assertEqual(preview["test_ids"], list(range(8, 100, 8)))
        self.assertEqual(list(preview["commands"]), [
            "B0_s0_train", "B0_s0_images", "T_s0_train", "T_s0_images", "T_s0_temporal",
        ])
        for key, command in preview["commands"].items():
            if key.endswith("_train"):
                self.assertNotIn("--resume_from", command)
                self.assertNotIn("--checkpoint", command)

    def test_only_temporal_weight_differs_between_training_arms(self):
        source = source_config()
        before = copy.deepcopy(source)
        base = training_config(source, "B0", 30000, 0)
        treatment = training_config(source, "T", 30000, 0)
        self.assertEqual(source, before)
        control_temporal = base["trainer"]["losses"].pop("temporal_material")
        active_temporal = treatment["trainer"]["losses"].pop("temporal_material")
        self.assertEqual(control_temporal.pop("w"), 0.)
        self.assertEqual(active_temporal.pop("w"), .05)
        self.assertEqual(control_temporal, active_temporal)
        self.assertEqual(active_temporal["interval"], 8)
        self.assertEqual(active_temporal["num_samples"], 4096)
        self.assertEqual(active_temporal["max_gap"], 2)
        self.assertEqual(active_temporal["ramp_steps"], 1000)
        self.assertEqual(base, treatment)
        self.assertEqual(base["trainer"], source["trainer"])
        self.assertEqual(base["data"], source["data"])
        self.assertEqual(base["model"], source["model"])

    def test_fake_execution_produces_t_summary_and_reuses_completed_stages(self):
        args = self.args("--execute", "--seeds", "0", "1")
        plan = build_plan(args)
        state = run_queue(args, plan, child_runner=self.fake_child)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(self.calls), 10)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(len(summary["image_metrics"]), 4)
        self.assertEqual(len(summary["paired_comparisons"]), 2)
        comparison = summary["paired_comparisons"][0]
        self.assertAlmostEqual(comparison["image_delta_T_minus_B0"]["psnr"], .02)
        self.assertIn("training_stage_wall_time_ratio_T_over_B0", comparison)
        self.assertAlmostEqual(comparison["temporal_residual_reduction_percent"], 10.)
        self.assertNotIn("image_delta_R_minus_B0", comparison)
        diagnostics = summary["training_diagnostics"]["T_s0_train"]["last_logged_values"]
        self.assertIn("temporal_stats/raw/valid_sample_count", diagnostics)
        self.assertIn("losses/raw/temporal_material_loss", diagnostics)
        self.assertNotIn("unrelated_stat", diagnostics)
        run_queue(args, plan, child_runner=self.fake_child)
        self.assertEqual(len(self.calls), 10)

    def test_failed_image_stage_retries_without_repeating_training(self):
        args = self.args("--execute", "--skip-temporal")
        def fail_image(command, log, lock):
            if "tools.eval" in command:
                raise RuntimeError("simulated evaluation failure")
            self.fake_child(command, log, lock)
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            run_queue(args, build_plan(args), child_runner=fail_image)
        state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(state["status"], "completed")
        self.assertNotIn("error", state)
        self.assertEqual(sum("tools.train" in cmd for cmd in self.calls), 2)
        attempts = state["stages"]["B0_s0_images"]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(Path(attempts[0]["directory"]).is_dir())
        self.assertEqual(Path(attempts[1]["directory"]).name, "attempt_002")

    def test_saved_temporal_settings_and_diffreg_contamination_are_rejected(self):
        args = self.args("--execute", "--skip-temporal")
        plan = build_plan(args)
        state = run_queue(args, plan, child_runner=self.fake_child)
        stage = plan["stages"][2]
        directory = Path(state["stages"][stage["id"]]["attempts"][0]["directory"])
        original = OmegaConf.load(directory / "config.yaml")
        for override in ("trainer.losses.temporal_material.w=0",
                         "trainer.losses.temporal_material.num_samples=0",
                         "trainer.losses.temporal_material.interval=1",
                         "trainer.losses.diffreg_material.enabled=true"):
            config = OmegaConf.merge(original, OmegaConf.from_dotlist([override]))
            OmegaConf.save(config, directory / "config.yaml")
            with self.assertRaises(ValueError):
                stage_artifacts(stage, directory)

    def test_protocol_change_in_child_is_detected(self):
        args = self.args("--execute", "--skip-temporal")
        def wrong_child(command, log, lock):
            self.fake_child(command, log, lock)
            if "tools.train" in command:
                value = lambda name: command[command.index(name) + 1]
                path = Path(value("--output_root")) / value("--project") / value("--run_name") / "config.yaml"
                cfg = OmegaConf.load(path)
                cfg.data.pixel_source.test_image_stride = 4
                OmegaConf.save(cfg, path)
        with self.assertRaisesRegex(ValueError, "protocol"):
            run_queue(args, build_plan(args), child_runner=wrong_child)

    def test_changed_source_identity_is_detected_before_child(self):
        args = self.args("--execute")
        plan = build_plan(args)
        with patch("tools.run_diffreg_experiments.input_identity", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                run_queue(args, plan, child_runner=self.fake_child)
        self.assertEqual(self.calls, [])

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

    def test_existing_treatments_and_bad_arguments_are_rejected(self):
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


if __name__ == "__main__":
    unittest.main()
