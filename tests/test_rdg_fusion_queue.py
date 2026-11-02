"""CPU-only B0/D queue checks; scene child commands are never executed."""
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
from tools.run_rdg_fusion_experiments import (
    build_plan, make_parser, run_queue, stage_artifacts, training_config,
)


class RDGFusionQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output = self.root / "config.yaml", self.root / "output"
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
            step = int(cfg.trainer.optim.num_iters)
            checkpoint = {"step": step, "models": {"Background": {"x": torch.ones(1)}}}
            if cfg.trainer.rdg_fusion.enabled:
                from models.rdg_fusion import RDGFusion, validate_rdg_config
                config = validate_rdg_config(cfg.trainer.rdg_fusion)
                head = RDGFusion(config)
                checkpoint["_rdg_fusion"] = {"config": config, "rdg_step": step}
                checkpoint.update({"rdg_fusion_head." + key: value
                                   for key, value in head.state_dict().items()})
            torch.save(checkpoint, output / "checkpoint_final.pth")
            (output / "metrics.json").write_text(json.dumps({
                "iteration": step, "rdg_stats/raw/history_weight": .1,
                "losses/raw/rdg_anchor": .002, "losses/raw/rdg_temporal": .001,
                "unrelated_stat": 99,
            }) + "\n")
        elif "tools.eval" in command:
            checkpoint = Path(flag("--resume_from"))
            cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
            self.assertTrue(cfg.render.render_test)
            self.assertEqual(cfg.data.pixel_source.test_image_stride, 8)
            self.assertEqual(cfg.log_dir, str(checkpoint.parent))
            enabled = cfg.trainer.rdg_fusion.enabled
            output = checkpoint.parent / "metrics_eval/raw"
            output.mkdir(parents=True)
            (output / "images_test_fixture.json").write_text(json.dumps({
                "image_metrics/raw/test/psnr": 30.02 if enabled else 30.,
                "image_metrics/raw/test/ssim": .921 if enabled else .92,
                "image_metrics/raw/test/lpips": .129 if enabled else .13,
            }))
        else:
            self.assertIn("D_s", flag("--checkpoint"))
            self.assertIn("B0_s", flag("--baseline-checkpoint"))
            output = Path(flag("--output-dir"))
            output.mkdir(parents=True)
            (output / "summary.json").write_text(json.dumps({"temporal": {
                "baseline_temporal_residual_l1": .01,
                "prediction_temporal_residual_l1": .009, "temporal_pairs": 99,
            }}))
            (output / "frames.jsonl").write_text('{"frame_id":0}\n')
            (output / "manifest.json").write_text('{"status":"complete"}')

    def complete(self, *extra):
        args = self.args("--execute", "--skip-temporal", *extra)
        plan = build_plan(args)
        with contextlib.redirect_stdout(io.StringIO()):
            state = run_queue(args, plan, child_runner=self.fake_child)
        stage = plan["stages"][2]
        directory = Path(state["stages"][stage["id"]]["attempts"][0]["directory"])
        return plan, state, stage, directory

    def test_preview_has_original_protocol_and_does_not_start_children_or_load_weights(self):
        args = self.args()
        with patch("torch.load", side_effect=AssertionError("no source weights")):
            with contextlib.redirect_stdout(io.StringIO()):
                plan = build_plan(args)
                preview = run_queue(args, plan, child_runner=self.fake_child)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls)
        self.assertEqual(preview["updates_per_training"], 30001)
        self.assertEqual(preview["train_frames"], 88)
        self.assertEqual(preview["test_ids"], list(range(8, 100, 8)))
        self.assertEqual(plan["identity"]["fusion"]["history_frames"], 1)
        self.assertEqual(plan["identity"]["fusion"]["feature_downscale"], 4)
        self.assertEqual(list(preview["commands"]), [
            "B0_s0_train", "B0_s0_images", "D_s0_train", "D_s0_images", "D_s0_temporal",
        ])
        for key, command in preview["commands"].items():
            if key.endswith("_train"):
                self.assertNotIn("--resume_from", command)
                self.assertNotIn("--checkpoint", command)

    def test_only_enabled_differs_and_original_losses_and_schedule_are_preserved(self):
        source = source_config()
        before = copy.deepcopy(source)
        base = training_config(source, "B0", 30000, 0)
        active = training_config(source, "D", 30000, 0)
        self.assertEqual(source, before)
        baseline_cfg = base["trainer"].pop("rdg_fusion")
        active_cfg = active["trainer"].pop("rdg_fusion")
        self.assertFalse(baseline_cfg.pop("enabled"))
        self.assertTrue(active_cfg.pop("enabled"))
        self.assertEqual(baseline_cfg, active_cfg)
        self.assertEqual(active_cfg["start_step"], 5000)
        self.assertEqual(active_cfg["interval"], 8)
        self.assertEqual(active_cfg["max_delta_linear"], .05)
        self.assertEqual(base, active)
        self.assertEqual(base["trainer"], source["trainer"])
        self.assertEqual(base["data"], source["data"])
        self.assertEqual(base["model"], source["model"])

    def test_fake_execution_compares_d_and_skips_verified_finished_stages(self):
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
        self.assertAlmostEqual(comparison["image_delta_D_minus_B0"]["psnr"], .02)
        self.assertIn("training_stage_wall_time_ratio_D_over_B0", comparison)
        self.assertAlmostEqual(comparison["temporal_residual_reduction_percent"], 10.)
        diagnostics = summary["training_diagnostics"]["D_s0_train"]["last_logged_values"]
        self.assertIn("rdg_stats/raw/history_weight", diagnostics)
        self.assertIn("losses/raw/rdg_anchor", diagnostics)
        self.assertIn("losses/raw/rdg_temporal", diagnostics)
        self.assertNotIn("unrelated_stat", diagnostics)

    def test_image_failure_retries_without_repeating_finished_training(self):
        args = self.args("--execute", "--skip-temporal")

        def fail_image(command, log, lock):
            if "tools.eval" in command:
                raise RuntimeError("synthetic image failure")
            self.fake_child(command, log, lock)

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                run_queue(args, build_plan(args), child_runner=fail_image)
            state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(sum("tools.train" in command for command in self.calls), 2)
        self.assertEqual(state["status"], "completed")
        attempts = state["stages"]["B0_s0_images"]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(Path(attempts[0]["directory"]).is_dir())

    def test_corrupt_config_step_or_head_tensors_are_rejected_and_loads_once(self):
        _, _, stage, directory = self.complete()
        path = directory / "checkpoint_final.pth"
        original = torch.load(path, weights_only=False)
        first = next(key for key in original if key.startswith("rdg_fusion_head."))
        for mode in ("metadata", "metadata_extra", "config", "config_type", "step", "final_step",
                     "missing", "extra", "shape", "dtype", "nan"):
            with self.subTest(mode=mode):
                checkpoint = copy.deepcopy(original)
                if mode == "metadata":
                    checkpoint.pop("_rdg_fusion")
                elif mode == "metadata_extra":
                    checkpoint["_rdg_fusion"]["unexpected"] = True
                elif mode == "config":
                    checkpoint["_rdg_fusion"]["config"]["max_blend"] = .2
                elif mode == "config_type":
                    checkpoint["_rdg_fusion"]["config"]["weight_decay"] = False
                elif mode == "step":
                    checkpoint["_rdg_fusion"]["rdg_step"] = 29999
                elif mode == "final_step":
                    checkpoint["step"] = 29999
                elif mode == "missing":
                    checkpoint.pop(first)
                elif mode == "extra":
                    checkpoint["rdg_fusion_head.unexpected"] = torch.ones(1)
                elif mode == "shape":
                    checkpoint[first] = torch.ones(1)
                elif mode == "dtype":
                    checkpoint[first] = checkpoint[first].half()
                else:
                    checkpoint[first].flatten()[0] = float("nan")
                torch.save(checkpoint, path)
                with self.assertRaises(ValueError):
                    stage_artifacts(stage, directory)
        torch.save(original, path)
        rng_before = torch.get_rng_state().clone()
        with patch("torch.load", wraps=torch.load) as load:
            stage_artifacts(stage, directory)
        self.assertEqual(load.call_count, 1)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_baseline_cannot_hide_fusion_metadata_or_parameters(self):
        plan, state, _, _ = self.complete()
        stage = plan["stages"][0]
        directory = Path(state["stages"][stage["id"]]["attempts"][0]["directory"])
        path = directory / "checkpoint_final.pth"
        original = torch.load(path, weights_only=False)
        for key, value in (("_rdg_fusion", {}), ("rdg_fusion_head.unexpected", torch.ones(1))):
            torch.save({**original, key: value}, path)
            with self.assertRaisesRegex(ValueError, "Baseline checkpoint"):
                stage_artifacts(stage, directory)

    def test_saved_arm_mismatch_and_other_methods_are_rejected(self):
        _, _, stage, directory = self.complete()
        original = OmegaConf.load(directory / "config.yaml")
        for option in ("trainer.rdg_fusion.enabled=false", "trainer.losses.diffreg_material.enabled=true",
                       "trainer.teacher_calibration.enabled=true", "trainer.teacher_confidence.enabled=true",
                       "trainer.losses.temporal_material.w=0.05"):
            OmegaConf.save(OmegaConf.merge(original, OmegaConf.from_dotlist([option])), directory / "config.yaml")
            with self.assertRaises(ValueError):
                stage_artifacts(stage, directory)

    def test_source_adapters_and_invalid_arguments_are_rejected(self):
        for key, value in (("rdg_fusion", {"enabled": True}),
                           ("teacher_confidence", {"enabled": True}),
                           ("teacher_calibration", {"enabled": True}),
                           ("gbuffer_attention", {"variant": "A2"}),
                           ("rgb_joint", {"enabled": True})):
            cfg = source_config()
            cfg["trainer"][key] = value
            OmegaConf.save(OmegaConf.create(cfg), self.source)
            with self.assertRaises(ValueError):
                build_plan(self.args())
        for key, value in (("diffreg_material", {"enabled": True}), ("temporal_material", {"w": .05})):
            cfg = source_config()
            cfg["trainer"]["losses"][key] = value
            OmegaConf.save(OmegaConf.create(cfg), self.source)
            with self.assertRaises(ValueError):
                build_plan(self.args())
        OmegaConf.save(OmegaConf.create(source_config()), self.source)
        for options in (("--steps", "0"), ("--seeds", "0", "0"), ("--min-free-gb", "nan")):
            with self.assertRaises(ValueError):
                build_plan(self.args(*options))

    def test_live_queue_lock_blocks_children(self):
        args = self.args("--execute")
        self.output.mkdir()
        with (self.output / ".queue.lock").open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(RuntimeError):
                run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
