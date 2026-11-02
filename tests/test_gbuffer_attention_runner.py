"""Read-only planning and failure-safe orchestration of matched attention runs."""
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

from tools.run_gbuffer_attention_experiments import (
    build_plan, digest, joint_config, make_parser, run_queue, stage_artifacts,
    validate_source,
)


def source_config():
    return {
        "seed": 0, "log_dir": "old_directory",
        "trainer": {"type": "models.trainers.MultiTrainer_pbr",
                    "optim": {"num_iters": 30000}, "render": {"antialiased": False, "packed": False},
                    "res_schedule": {"double_steps": 250, "downscale_times": 2},
                    "losses": {"rgb": {"w": .7}, "ssim": {"w": .3}, "albedo": {"w": .5}}},
        "model": {"Background": {"optim": {"xyz": {"lr": .00016}},
                                  "ctrl": {"stop_split_at": 25000, "stop_reset_at": 20000}}},
        "render": {"render_test": True, "render_full": False},
        "logging": {"saveckpt_freq": 5000},
        "data": {"data_root": "unused", "scene_idx": 788, "start_timestep": 0,
                 "end_timestep": 99, "pixel_source": {"cameras": [0], "load_weather": [],
                 "load_gbuffer": True, "gbuffer_path": "original_teacher", "test_image_stride": 8,
                 "downscale": 1, "downscale_when_loading": [1], "undistort": True}},
    }


class AttentionQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.checkpoint = self.source / "checkpoint_final.pth"
        self.output = self.root / "output"
        OmegaConf.save(OmegaConf.create(source_config()), self.source / "config.yaml")
        torch.save({"step": 30000, "models": {"Background": {"x": torch.ones(1)}}}, self.checkpoint)
        self.calls = []
        for name in ("code_identity", "input_identity"):
            item = patch("tools.run_gbuffer_attention_experiments." + name,
                         return_value={"fixture": "unchanged"})
            item.start()
            self.addCleanup(item.stop)

    def args(self, *extra):
        return make_parser().parse_args(["--checkpoint", str(self.checkpoint),
            "--output-dir", str(self.output), "--pilot-steps", "2", "--joint-steps", "3", *extra])

    def fake_child(self, command, log, lock):
        self.calls.append(command)
        def flag(name):
            return command[command.index(name) + 1]
        if "tools.train" in command:
            cfg = OmegaConf.load(flag("--config_file"))
            output = Path(flag("--output_root")) / flag("--project") / flag("--run_name")
            step = int(cfg.trainer.optim.num_iters)
        elif "tools.train_gbuffer_attention" in command:
            cfg = OmegaConf.load(Path(flag("--checkpoint")).parent / "config.yaml")
            variant = flag("--variant")
            cfg.trainer.gbuffer_attention = None if variant == "B0" else {"variant": variant}
            cfg.seed = int(flag("--seed"))
            output = Path(flag("--output-dir"))
            step = int(flag("--steps"))
        elif "tools.eval" in command:
            checkpoint = Path(flag("--resume_from"))
            cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
            self.assertEqual(cfg.log_dir, str(checkpoint.parent))
            output = checkpoint.parent / "metrics_eval/raw"
            output.mkdir(parents=True)
            (output / "images_test_fixture.json").write_text(json.dumps({
                "image_metrics/raw/test/psnr": 30., "image_metrics/raw/test/ssim": .92,
                "image_metrics/raw/test/lpips": .13, "image_metrics/raw/test/occupied_psnr": 29.,
            }))
            return
        else:
            output = Path(flag("--output-dir"))
            output.mkdir(parents=True)
            (output / "summary.json").write_text(json.dumps({"temporal_pairs": 99, "residual": .01}))
            (output / "frames.jsonl").write_text('{"frame_id":0}\n')
            (output / "manifest.json").write_text('{"status":"complete"}')
            return
        output.mkdir(parents=True)
        cfg.log_dir = str(output)
        OmegaConf.save(cfg, output / "config.yaml")
        torch.save({"step": step, "models": {"Background": {"x": torch.ones(1)}}},
                   output / "checkpoint_final.pth")
        (output / "manifest.json").write_text('{"status":"complete"}')

    def test_read_only_preview_has_no_output_directory_or_children(self):
        args = self.args()
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(result["status"], "planned")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.calls, [])
        self.assertEqual(result["identity"]["test_ids"], list(range(8, 100, 8)))
        self.assertEqual(len(result["stages"]), 21)

    def test_joint_config_preserves_original_losses_data_and_schedules(self):
        source = source_config()
        original = copy.deepcopy(source)
        cfg = joint_config(source, "A2", steps=30000, seed=0)
        self.assertEqual(source, original)
        cfg.pop("seed")
        attention = cfg["trainer"].pop("gbuffer_attention")
        original.pop("seed")
        original.pop("log_dir")
        self.assertEqual(cfg, original)
        self.assertEqual(attention["temporal_w"], .05)
        self.assertEqual(joint_config(source, "A1", 30000, 0)["trainer"]["gbuffer_attention"]["temporal_w"], 0.)
        self.assertIsNone(joint_config(source, "B0", 30000, 0)["trainer"]["gbuffer_attention"])

    def test_nonmatching_protocol_is_rejected(self):
        for section, key, value in (("pixel", "test_frame_ranges", [[20, 30], [60, 70]]),
                                    ("pixel", "test_image_stride", 4), ("pixel", "downscale", 2),
                                    ("pixel", "cameras", [0, 1]), ("pixel", "load_gbuffer", False),
                                    ("render", "antialiased", True)):
            source = source_config()
            target = source["data"]["pixel_source"] if section == "pixel" else source["trainer"]["render"]
            target[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_source(source)

    def test_frozen_controls_shared_across_head_seeds_joint_controls_matched(self):
        plan = build_plan(self.args("--mode", "both", "--seeds", "0", "1", "--variants", "H0", "A2"))
        prepare = [s for s in plan["stages"] if s["kind"] == "prepare"]
        self.assertEqual([s["id"] for s in prepare],
                         ["pilot_B0_s0_train", "pilot_H0_s0_train", "joint_H0_s0_train", "joint_H0_s1_train"])
        for s in plan["stages"]:
            if s["kind"] == "temporal_eval" and s["mode"] == "joint":
                self.assertEqual(s["baseline_from"], f"joint_B0_s{s['seed']}_train")
        self.assertEqual(plan["joint_updates_per_training"], 4)

    def test_execute_resume_verifies_and_skips_completed_outputs(self):
        args = self.args("--execute", "--variants", "A2")
        plan = build_plan(args)
        state = run_queue(args, plan, child_runner=self.fake_child)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(self.calls), 6)
        original_hash = digest(self.checkpoint)
        run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(digest(self.checkpoint), original_hash)
        self.assertEqual(OmegaConf.load(self.source / "config.yaml").log_dir, "old_directory")
        report = json.loads((self.output / "summary.json").read_text())
        self.assertEqual([row["variant"] for row in report["image_metrics"]], ["B0", "A2"])
        self.assertTrue((self.output / "summary.csv").is_file())

    def test_failed_stage_restarts_in_new_attempt_preserving_completed_dependencies(self):
        args = self.args("--execute", "--variants", "A2", "--skip-temporal")
        def fail_image(command, log, lock):
            self.calls.append(command)
            if "tools.eval" in command:
                raise RuntimeError("simulated evaluation failure")
            self.calls.pop()
            self.fake_child(command, log, lock)
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            run_queue(args, build_plan(args), child_runner=fail_image)
        state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        image_attempts = state["stages"]["pilot_B0_s0_images"]["attempts"]
        self.assertEqual(len(image_attempts), 2)
        self.assertTrue(Path(image_attempts[0]["directory"]).is_dir())
        self.assertEqual(Path(image_attempts[1]["directory"]).name, "attempt_002")
        self.assertEqual(sum("tools.train_gbuffer_attention" in c for c in self.calls), 2)

    def test_changed_completed_artifact_and_options_are_rejected(self):
        args = self.args("--execute", "--variants", "B0", "--skip-temporal")
        state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        changed = self.args("--execute", "--variants", "B0", "--skip-temporal", "--pilot-steps", "4")
        with self.assertRaisesRegex(ValueError, "changed"):
            run_queue(changed, build_plan(changed), child_runner=self.fake_child)
        attempt = state["stages"]["pilot_B0_s0_train"]["attempts"][-1]
        (Path(attempt["directory"]) / "config.yaml").write_text("corrupted")
        with self.assertRaisesRegex(ValueError, "artifact is missing or changed"):
            run_queue(args, build_plan(args), child_runner=self.fake_child)

    def test_unrelated_outputs_and_duplicate_queue_refused(self):
        self.output.mkdir()
        unrelated = self.output / "important.txt"
        unrelated.write_text("preserve")
        args = self.args("--execute", "--variants", "B0")
        with self.assertRaisesRegex(ValueError, "not an initialized"):
            run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(unrelated.read_text(), "preserve")
        unrelated.unlink()
        with (self.output / ".queue.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(self.calls, [])

    def test_joint_and_h0_execute_without_checkpoint_initialization_for_joint(self):
        args = self.args("--execute", "--mode", "joint", "--variants", "H0", "A2", "--skip-temporal")
        state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(state["status"], "completed")
        train_commands = [c for c in self.calls if "tools.train" in c]
        self.assertEqual(len(train_commands), 2)
        self.assertTrue(all("--resume_from" not in c for c in train_commands))
        h0 = next(c for c in self.calls if "tools.train_gbuffer_attention" in c)
        source = h0[h0.index("--checkpoint") + 1]
        self.assertIn("joint_B0_s0_train", source)
        self.assertEqual(h0[h0.index("--steps") + 1], "0")

    def test_child_cannot_silently_change_the_saved_evaluation_split(self):
        args = self.args("--execute", "--variants", "B0", "--skip-temporal")
        def wrong_split(command, log, lock):
            self.fake_child(command, log, lock)
            directory = Path(command[command.index("--output-dir") + 1])
            cfg = OmegaConf.load(directory / "config.yaml")
            cfg.data.pixel_source.test_image_stride = 4
            OmegaConf.save(cfg, directory / "config.yaml")
        with self.assertRaisesRegex(ValueError, "paired training/evaluation protocol"):
            run_queue(args, build_plan(args), child_runner=wrong_split)
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
