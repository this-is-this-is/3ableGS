"""CPU checks for a paired from-scratch training queue and held-out aggregation."""
import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from tools.run_layered_joint_training import (
    build_configs, build_plan, digest, LAYERED_CONFIG, make_parser, run_queue,
    split_evaluation, training_command, verify_checkpoint,
)


def source_config():
    return {
        "seed": 31, "log_dir": "old_training_directory",
        "trainer": {"type": "models.trainers.MultiTrainer_pbr",
                    "optim": {"num_iters": 30000}, "render": {"antialiased": True},
                    "res_schedule": {"double_steps": 250, "downscale_times": 2},
                    "losses": {"rgb": {"w": .7}, "albedo": {"w": .5},
                               "temporal_material": {"w": .3, "max_gap": 1}}},
        "model": {"Background": {"optim": {"xyz": {"lr": .00016}}}},
        "logging": {"print_freq": 250, "saveckpt_freq": 5000},
        "data": {"data_root": "unused", "scene_idx": 788, "start_timestep": 0,
                 "end_timestep": 99, "pixel_source": {"cameras": [0], "load_weather": [],
                 "load_gbuffer": True, "gbuffer_path": "original_static_teacher", "test_image_stride": 0,
                 "test_frame_ranges": [[20, 30], [60, 70]]}},
    }


class JointTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.yaml"
        OmegaConf.save(OmegaConf.create(source_config()), self.source)
        self.output = self.root / "output"
        self.calls = []
        for name in ("code_identity", "input_identity"):
            mock = patch("tools.run_layered_joint_training." + name, return_value={"fixture": "unchanged"})
            mock.start()
            self.addCleanup(mock.stop)

    def args(self, *extra):
        return make_parser().parse_args(["--source-config", str(self.source),
                                         "--output-dir", str(self.output), "--steps", "3", *extra])

    def fake_child(self, command, log, lock):
        self.calls.append(command)
        def flag(name):
            return command[command.index(name) + 1]
        if "tools.train" in command:
            cfg = OmegaConf.load(flag("--config_file"))
            directory = Path(flag("--output_root")) / flag("--project") / flag("--run_name")
            directory.mkdir(parents=True)
            cfg.log_dir = str(directory)
            OmegaConf.save(cfg, directory / "config.yaml")
            torch.save({"step": int(cfg.trainer.optim.num_iters), "models": {"Background": {"x": torch.ones(1)}}},
                       directory / "checkpoint_final.pth")
        else:
            checkpoint, directory = Path(flag("--checkpoint")), Path(flag("--output"))
            self.write_evaluation(checkpoint, directory)

    def write_evaluation(self, checkpoint, directory):
        cfg = OmegaConf.to_container(OmegaConf.load(checkpoint.parent / "config.yaml"))
        renderer = cfg["trainer"]["layered_gbuffer"]
        if renderer is not None:
            renderer["diagnostics"] = True
        identity = {"checkpoint_sha256": digest(checkpoint),
                    "config_sha256": digest(checkpoint.parent / "config.yaml"),
                    "split": "heldout", "downscale": 1., "antialiased": False,
                    "variants": {"trained": renderer}}
        frames = list(range(20, 30)) + list(range(60, 70))
        (directory / "frames").mkdir(parents=True, exist_ok=True)
        for frame in frames:
            # Distinct block values catch accidental val/test mixing. Unequal
            # pixel counts catch averaging per-frame PSNR instead of statistics.
            mse = (.01 if frame < 30 else .04) + .001 * (frame % 10)
            count = 3 * (frame % 10 + 1)
            metrics = {"frame_id": frame, "temporal_has_pair": frame not in (20, 60),
                       "trained_global_rgb_mse_sum": mse * count,
                       "trained_global_rgb_mse_count": count,
                       "trained_fullframe_lpips_sum": .1 if frame < 30 else .2,
                       "trained_fullframe_lpips_count": 1}
            (directory / "frames" / f"{frame:06d}.json").write_text(json.dumps({"frame_id": frame, "metrics": metrics}))
        for name, data in (("manifest", {"frames": frames}), ("summary", {"frames": 20})):
            (directory / f"{name}.json").write_text(json.dumps({"status": "completed", "identity": identity, **data}))

    def test_pair_changes_only_renderer_and_preserves_full_training(self):
        source = source_config()
        pair = build_configs(source, steps=30000, seed=0)
        base, layered = copy.deepcopy(pair["base"]), copy.deepcopy(pair["layered"])
        self.assertIsNone(base["trainer"].pop("layered_gbuffer"))
        self.assertEqual(layered["trainer"].pop("layered_gbuffer"), LAYERED_CONFIG)
        self.assertEqual(base, layered)
        self.assertEqual(base["model"], source["model"])
        self.assertEqual(base["trainer"]["res_schedule"], source["trainer"]["res_schedule"])
        self.assertEqual(base["data"], source["data"])
        self.assertFalse(base["trainer"]["render"]["antialiased"])
        self.assertEqual(base["trainer"]["losses"]["temporal_material"]["w"], 0.)
        self.assertEqual(source["trainer"]["losses"]["temporal_material"]["w"], .3)
        self.assertNotIn("log_dir", base)
        self.assertNotIn("--resume_from", training_command(self.output, "base", "attempt_001"))

    def test_invalid_source_cannot_create_false_holdouts(self):
        for key, value in (("test_frame_ranges", [[0, 10], [90, 100]]), ("test_image_stride", 10),
                           ("load_gbuffer", False), ("cameras", [0, 1]), ("load_weather", ["rain"])):
            source = source_config()
            source["data"]["pixel_source"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_configs(source, steps=3, seed=0)

    def test_preview_then_execute_and_verified_resume_skip(self):
        args = self.args()
        plan = build_plan(args)
        self.assertEqual(run_queue(args, plan, child_runner=self.fake_child)["status"], "planned")
        self.assertEqual(self.calls, [])
        args = self.args("--execute")
        state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.calls[0][self.calls[0].index("--project") + 1], "base")
        self.assertIn("--trained-only", self.calls[1])
        report = json.loads((self.output / "results.json").read_text())
        val = report["arms"]["base"]["val"]
        test = report["arms"]["base"]["test"]
        self.assertAlmostEqual(val["fullframe_lpips"], .1)
        self.assertAlmostEqual(test["fullframe_lpips"], .2)
        self.assertGreater(val["global_rgb_psnr"], test["global_rgb_psnr"])
        self.assertEqual(val["temporal_pairs"], 9)
        self.assertAlmostEqual(val["global_rgb_mse"], .016)
        args = self.args("--execute", "--resume")
        run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(len(self.calls), 4)

    def test_failed_training_restarts_in_new_attempt_never_model_resumes(self):
        args = self.args("--execute")
        def fail_train(command, log, lock):
            self.calls.append(command)
            raise RuntimeError("simulated interruption")
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            run_queue(args, build_plan(args), child_runner=fail_train)
        args = self.args("--execute", "--resume")
        state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        attempts = state["arms"]["base"]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[1]["name"], "attempt_002")
        self.assertTrue(attempts[0]["status"].endswith("restarted_from_scratch"))
        self.assertTrue(all("--resume_from" not in call for call in self.calls))

    def test_failed_evaluation_reuses_verified_training_only(self):
        args = self.args("--execute")
        def fail_eval(command, log, lock):
            self.fake_child(command, log, lock)
            if "tools.layered_gbuffer_experiment" in command:
                raise RuntimeError("evaluation interrupted")
        with self.assertRaisesRegex(RuntimeError, "evaluation interrupted"):
            run_queue(args, build_plan(args), child_runner=fail_eval)
        args = self.args("--execute", "--resume")
        run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(sum("tools.train" in call for call in self.calls), 2)
        self.assertIn("--restart", self.calls[2])

    def test_changed_completed_artifact_fails_closed(self):
        args = self.args("--execute")
        state = run_queue(args, build_plan(args), child_runner=self.fake_child)
        Path(state["arms"]["base"]["attempts"][0]["checkpoint"]).write_bytes(b"corrupted")
        args = self.args("--execute", "--resume")
        with self.assertRaisesRegex(ValueError, "artifact is missing or changed"):
            run_queue(args, build_plan(args), child_runner=self.fake_child)

    def test_changed_options_and_uninitialized_output_rejected(self):
        args = self.args()
        run_queue(args, build_plan(args), child_runner=self.fake_child)
        args = self.args("--execute", "--seed", "7")
        with self.assertRaisesRegex(ValueError, "changed"):
            run_queue(args, build_plan(args), child_runner=self.fake_child)
        (self.output / "state.json").unlink()
        args = self.args("--execute")
        with self.assertRaisesRegex(ValueError, "not an initialized"):
            run_queue(args, build_plan(args), child_runner=self.fake_child)

    def test_existing_queue_lock_prevents_duplicate_training(self):
        self.output.mkdir()
        with (self.output / ".queue.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            args = self.args("--execute")
            with self.assertRaisesRegex(RuntimeError, "already running"):
                run_queue(args, build_plan(args), child_runner=self.fake_child)
        self.assertEqual(self.calls, [])

    def test_checkpoint_step_and_saved_configuration_are_verified(self):
        args = self.args("--execute")
        plan = build_plan(args)
        run_queue(args, plan, child_runner=self.fake_child)
        checkpoint = self.output / "train" / "base" / "attempt_001" / "checkpoint_final.pth"
        with self.assertRaisesRegex(ValueError, "not a completed step"):
            verify_checkpoint(checkpoint, plan["identity"]["configs"]["base"], 4)
        changed = copy.deepcopy(plan["identity"]["configs"]["base"])
        changed["seed"] = 99
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            verify_checkpoint(checkpoint, changed, 3)

    def test_evaluation_cannot_switch_renderer_or_drop_holdout_frames(self):
        args = self.args("--execute")
        plan = build_plan(args)
        run_queue(args, plan, child_runner=self.fake_child)
        directory = self.output / "evaluate" / "base"
        checkpoint = self.output / "train" / "base" / "attempt_001" / "checkpoint_final.pth"
        with self.assertRaisesRegex(ValueError, "renderer differs"):
            split_evaluation(directory, checkpoint, plan["identity"]["configs"]["layered"])
        (directory / "frames" / "000060.json").unlink()
        with self.assertRaisesRegex(ValueError, "exactly the 20"):
            split_evaluation(directory, checkpoint, plan["identity"]["configs"]["base"])


if __name__ == "__main__":
    unittest.main()
