"""CPU-only B0/V queue checks; child commands are never executed."""
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
from tools.run_teacher_confidence_experiments import (
    build_plan, make_parser, run_queue, source_crop_fraction, stage_artifacts, training_config,
)


def fake_cache(config, step, crop_frac=0.):
    from models.teacher_confidence import validate_teacher_confidence_config
    config = validate_teacher_confidence_config(config)
    train_ids = [frame for frame in range(100) if frame == 0 or frame % 8]
    built = step >= config["build_step"]
    frames = {image_id: torch.full((3, 4, 1), .75, dtype=torch.float16)
              for image_id in train_ids} if built else {}
    return {"schema_version": 1, "config": config, "train_indices": train_ids,
            "num_cams": 1, "crop_frac": crop_frac,
            "built_step": config["build_step"] if built else None,
            "frames": frames, "shapes": {key: list(value.shape) for key, value in frames.items()},
            "stats": {"render_count": len(train_ids),
                      "cache_bytes": sum(value.numel() * 2 for value in frames.values()),
                      "build_seconds": 0.} if built else {}}


class TeacherConfidenceQueueTests(unittest.TestCase):
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
            self.assertNotIn("teacher_calibration", cfg.trainer)
            output.mkdir(parents=True)
            cfg.log_dir = str(output)
            OmegaConf.save(cfg, output / "config.yaml")
            step = int(cfg.trainer.optim.num_iters)
            checkpoint = {"step": step, "models": {"Background": {"x": torch.ones(1)}}}
            if cfg.trainer.teacher_confidence.enabled:
                checkpoint["_teacher_confidence"] = fake_cache(
                    cfg.trainer.teacher_confidence, step,
                    source_crop_fraction(OmegaConf.to_container(cfg, resolve=True)))
            torch.save(checkpoint, output / "checkpoint_final.pth")
            (output / "metrics.json").write_text(json.dumps({
                "iteration": step, "confidence_stats/raw/confidence_mean": .75,
                "confidence_stats/raw/cache_bytes": 2112,
                "losses/raw/albedo_loss": .02,
                "losses/raw/roughness_loss": .01, "losses/raw/metallic_loss": .001,
                "unrelated_stat": 99,
            }) + "\n")
        elif "tools.eval" in command:
            checkpoint = Path(flag("--resume_from"))
            cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
            self.assertTrue(cfg.render.render_test)
            self.assertEqual(cfg.data.pixel_source.test_image_stride, 8)
            self.assertEqual(cfg.log_dir, str(checkpoint.parent))
            enabled = cfg.trainer.teacher_confidence.enabled
            output = checkpoint.parent / "metrics_eval/raw"
            output.mkdir(parents=True)
            (output / "images_test_fixture.json").write_text(json.dumps({
                "image_metrics/raw/test/psnr": 30.02 if enabled else 30.,
                "image_metrics/raw/test/ssim": .921 if enabled else .92,
                "image_metrics/raw/test/lpips": .129 if enabled else .13,
            }))
        else:
            self.assertIn("V_s", flag("--checkpoint"))
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

    def test_preview_is_read_only_and_never_loads_source_weights(self):
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
        self.assertEqual(plan["identity"]["confidence_cache"]["additional_learned_parameters"], 0)
        self.assertEqual(list(preview["commands"]), [
            "B0_s0_train", "B0_s0_images", "V_s0_train", "V_s0_images", "V_s0_temporal",
        ])
        for key, command in preview["commands"].items():
            if key.endswith("_train"):
                self.assertNotIn("--resume_from", command)
                self.assertNotIn("--checkpoint", command)

    def test_only_enable_flag_differs_between_original_training_arms(self):
        source = source_config()
        before = copy.deepcopy(source)
        base = training_config(source, "B0", 30000, 0)
        active = training_config(source, "V", 30000, 0)
        self.assertEqual(source, before)
        baseline_cfg = base["trainer"].pop("teacher_confidence")
        active_cfg = active["trainer"].pop("teacher_confidence")
        self.assertFalse(baseline_cfg.pop("enabled"))
        self.assertTrue(active_cfg.pop("enabled"))
        self.assertEqual(baseline_cfg, active_cfg)
        self.assertEqual(active_cfg["build_step"], 3000)
        self.assertEqual(active_cfg["downscale"], 4)
        self.assertEqual(active_cfg["confidence_floor"], .25)
        self.assertEqual(base, active)
        self.assertEqual(base["trainer"], source["trainer"])
        self.assertEqual(base["data"], source["data"])
        self.assertEqual(base["model"], source["model"])

    def test_fake_execution_summarizes_v_and_skips_verified_stages(self):
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
        self.assertAlmostEqual(comparison["image_delta_V_minus_B0"]["psnr"], .02)
        self.assertIn("training_stage_wall_time_ratio_V_over_B0", comparison)
        self.assertAlmostEqual(comparison["temporal_residual_reduction_percent"], 10.)
        diagnostics = summary["training_diagnostics"]["V_s0_train"]["last_logged_values"]
        self.assertIn("confidence_stats/raw/confidence_mean", diagnostics)
        self.assertIn("confidence_stats/raw/cache_bytes", diagnostics)
        self.assertIn("losses/raw/albedo_loss", diagnostics)
        self.assertNotIn("unrelated_stat", diagnostics)

    def test_image_failure_does_not_repeat_finished_training(self):
        args = self.args("--execute", "--skip-temporal")

        def fail_image(command, log, lock):
            if "tools.eval" in command:
                raise RuntimeError("synthetic evaluation failure")
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

    def test_cache_metadata_and_maps_cannot_disagree_with_saved_protocol(self):
        _, _, stage, directory = self.complete()
        path = directory / "checkpoint_final.pth"
        original = torch.load(path, weights_only=False)
        for mode in ("missing", "split", "crop", "build_step", "missing_map", "map_shape",
                     "dtype", "nan", "below_floor", "bytes", "wrong_final_step"):
            with self.subTest(mode=mode):
                checkpoint = copy.deepcopy(original)
                cache = checkpoint["_teacher_confidence"]
                if mode == "missing":
                    checkpoint.pop("_teacher_confidence")
                elif mode == "split":
                    cache["train_indices"][0] = 8
                elif mode == "crop":
                    cache["crop_frac"] = .1
                elif mode == "build_step":
                    cache["built_step"] = 3001
                elif mode == "missing_map":
                    cache["frames"].pop(0)
                elif mode == "map_shape":
                    cache["shapes"][0] = [4, 3, 1]
                elif mode == "dtype":
                    cache["frames"][0] = cache["frames"][0].float()
                elif mode == "nan":
                    cache["frames"][0][0, 0, 0] = float("nan")
                elif mode == "below_floor":
                    cache["frames"][0][0, 0, 0] = 0.
                elif mode == "bytes":
                    cache["stats"]["cache_bytes"] = 0
                else:
                    checkpoint["step"] = 29999
                torch.save(checkpoint, path)
                with self.assertRaises(ValueError):
                    stage_artifacts(stage, directory)
        torch.save(original, path)
        with patch("torch.load", wraps=torch.load) as load:
            stage_artifacts(stage, directory)
        self.assertEqual(load.call_count, 1)

    def test_short_run_must_remain_unbuilt_and_build_step_requires_cache(self):
        _, _, stage, directory = self.complete("--steps", "100")
        path = directory / "checkpoint_final.pth"
        checkpoint = torch.load(path, weights_only=False)
        cache = checkpoint["_teacher_confidence"]
        self.assertIsNone(cache["built_step"])
        self.assertEqual(cache["frames"], {})
        checkpoint["step"] = 3000
        torch.save(checkpoint, path)
        with self.assertRaises(ValueError):
            stage_artifacts({**stage, "steps": 3000}, directory)
        checkpoint["_teacher_confidence"] = fake_cache(cache["config"], 3000)
        torch.save(checkpoint, path)
        stage_artifacts({**stage, "steps": 3000}, directory)

    def test_baseline_cannot_contain_confidence_cache(self):
        plan, state, _, _ = self.complete()
        stage = plan["stages"][0]
        directory = Path(state["stages"][stage["id"]]["attempts"][0]["directory"])
        path = directory / "checkpoint_final.pth"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["_teacher_confidence"] = {"unexpected": True}
        torch.save(checkpoint, path)
        with self.assertRaises(ValueError):
            stage_artifacts(stage, directory)

    def test_saved_arm_mismatch_and_combined_methods_are_rejected(self):
        _, _, stage, directory = self.complete()
        original = OmegaConf.load(directory / "config.yaml")
        for option in ("trainer.teacher_confidence.enabled=false",
                       "trainer.losses.diffreg_material.enabled=true",
                       "trainer.teacher_calibration.enabled=true"):
            OmegaConf.save(OmegaConf.merge(original, OmegaConf.from_dotlist([option])), directory / "config.yaml")
            with self.assertRaises(ValueError):
                stage_artifacts(stage, directory)

    def test_source_adapters_and_invalid_arguments_are_rejected(self):
        for key, value in (("teacher_confidence", {"enabled": True}),
                           ("teacher_calibration", {"enabled": True}),
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

    def test_crop_fraction_comes_from_original_camera_height(self):
        cfg = source_config()
        cfg["data"]["dataset"] = "waymo"
        cfg["data"]["pixel_source"]["gbuffer_crop_top"] = 64
        cfg["data"]["pixel_source"]["gbuffer_size"] = [768, 1280]
        self.assertEqual(source_crop_fraction(cfg), .05)
        cfg["data"]["pixel_source"]["gbuffer_crop_top"] = 640
        with self.assertRaises(ValueError):
            source_crop_fraction(cfg)

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
