"""Read-only planning and simulated subprocess lifecycle checks."""

import argparse
from contextlib import redirect_stdout
import copy
import fcntl
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from omegaconf import OmegaConf

from tools import run_temporal_experiments as suite


class SuiteTests(unittest.TestCase):
    def args(self, root):
        return suite.make_parser().parse_args([
            "--checkpoint", str(root / "checkpoint_final.pth"),
            "--output-dir", str(root / "suite"),
            "--iterations", "20", "--holdout-ranges", "2:4",
        ])

    @staticmethod
    def small_plan(stages):
        return {
            "signature": "test-signature", "identity": {"full_frames": 8, "render_frames": 2},
            "teacher_plan": None, "scene": 788, "start": 0, "ranges": [[2, 4]], "stages": stages,
        }

    @staticmethod
    def render_stage(name="render"):
        return {"id": name, "kind": "render", "group": "frozen", "args": [], "frames": 2}

    @staticmethod
    def fake_render(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(json.dumps({"status": "completed", "rendered_frames": 2}))
        (output / "summary.json").write_text(json.dumps({
            "frames": 2, "baseline_full_psnr": 30., "transport_full_psnr": 29.,
            "reference_evaluation": {"transport_temporal_residual_l1": .01},
        }))
        (output / "frames.jsonl").write_text("{}\n{}\n")
        (output / "comparison.mp4").write_bytes(b"simulated video")
        return SimpleNamespace(returncode=0)

    def quiet_execute(self, *args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return suite.execute_plan(*args, **kwargs)

    def test_prespecified_variants_cover_all_controls(self):
        controls = dict(suite.variants())
        self.assertEqual(len(controls), 15)
        self.assertEqual(controls["baseline"], ["--history-weight", "0"])
        self.assertIn("--domain", controls["rgb_normal_w030"])
        self.assertIn("--history-source", controls["material_current_w030"])
        self.assertIn("--clamp-history", controls["material_clamp_w030"])
        for ablation in ("shuffle_normal", "shuffle_roughness", "constant_normal", "constant_roughness"):
            self.assertIn(ablation, controls["guide_" + ablation])

    def test_training_configs_preserve_timeline_and_model_and_match_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            saved = OmegaConf.create({
                "seed": 77, "log_dir": "old", "model": {"Background": {"feature_dim": 32}},
                "trainer": {"optim": {"num_iters": 100}, "losses": {"rgb": {"w": 1.}}},
                "data": {"start_timestep": 7, "end_timestep": 14, "preload_device": "cuda",
                         "pixel_source": {"gbuffer_path": "old-static", "cameras": [0]}},
                "logging": {"vis_freq": 4}, "render": {},
            })
            original = OmegaConf.to_container(saved)
            static = suite.training_config(saved, args, [[2, 4]], 3, "static", False)
            video = suite.training_config(saved, args, [[2, 4]], 3, "video", True)
            self.assertEqual(OmegaConf.to_container(saved), original)
            self.assertEqual(static["model"], video["model"])
            self.assertEqual(static["seed"], video["seed"])
            self.assertEqual(video["data"]["start_timestep"], 7)
            self.assertEqual(video["data"]["end_timestep"], 14)
            self.assertEqual(video["data"]["pixel_source"]["test_frame_ranges"], [[2, 4]])
            self.assertEqual(video["trainer"]["losses"]["temporal_material"]["w"], .05)
            self.assertEqual(static["trainer"]["losses"]["temporal_material"]["w"], 0.)
            self.assertEqual(static["trainer"]["optim"]["cache_buffer_freq"], -1)
            self.assertFalse(video["render"]["render_full"])
            self.assertFalse(video["render"]["render_test"])
            self.assertEqual(static["data"]["pixel_source"]["gbuffer_path"], "old-static")

    def test_all_stages_preview_without_writes_then_execute_in_declared_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args(root)
            args.checkpoint.write_bytes(b"checkpoint")
            source = root / "data/waymo/processed/training/788/images/raw"
            source.mkdir(parents=True)
            static = root / "static/raw"
            static.mkdir(parents=True)
            for frame in range(8):
                (source / f"{frame:03d}_0.jpg").write_bytes(b"source")
                for component in ("normal", "basecolor", "roughness", "metallic"):
                    (static / f"0000.{frame:04d}.{component}.png").write_bytes(b"teacher")
            (root / "tools").mkdir()
            for name in ("render_temporal_material.py", "run_video_gbuffers.py", "train.py"):
                (root / "tools" / name).write_text("# fake")
            cfg = OmegaConf.create({
                "seed": 0, "trainer": {"type": "models.trainers.MultiTrainer_pbr",
                                      "optim": {}, "losses": {}},
                "data": {"data_root": str(root / "data/waymo/processed/training"),
                         "scene_idx": 788, "start_timestep": 0, "end_timestep": 7,
                         "pixel_source": {"cameras": [0], "gbuffer_path": str(root / "static"),
                                          "gbuffer_seq_id": 0}},
                "logging": {}, "render": {}, "model": {},
            })
            OmegaConf.save(cfg, root / "config.yaml")
            with mock.patch.object(suite, "ROOT", root), mock.patch(
                    "tools.run_video_gbuffers.build_plan",
                    return_value={"signature": "teacher", "final_png_count": 32}):
                plan = suite.build_plan(args)
            self.assertFalse(args.output_dir.exists())
            self.assertEqual(len(plan["stages"]), 34)
            train = [stage for stage in plan["stages"] if stage["kind"] == "train"]
            self.assertEqual(len(train), 4)
            self.assertEqual(sum(stage["group"] == "surface_edit" for stage in plan["stages"]), 6)
            for stage in train:
                command = suite.stage_command(args, plan, stage, 1, None)
                self.assertNotIn("--resume_from", command)
                self.assertEqual(command[1:3], ["-m", "tools.train"])
            stages = {stage["id"]: stage for stage in plan["stages"]}
            executed = []

            def simulated_pipeline(command, **kwargs):
                if "--work-dir" in command:
                    executed.append("video_teacher")
                    work = Path(command[command.index("--work-dir") + 1])
                    work.mkdir(parents=True)
                    destination = args.output_dir / "video_teacher/data/gbuffer_frames/raw"
                    destination.mkdir(parents=True)
                    from PIL import Image
                    from tools.run_video_gbuffers import SOURCE_SIZE
                    buffer = io.BytesIO()
                    Image.new("RGB", SOURCE_SIZE).save(buffer, format="PNG")
                    content = buffer.getvalue()
                    hashes = {}
                    for frame in range(8):
                        for component in ("normal", "basecolor", "roughness", "metallic"):
                            image = destination / f"0000.{frame:04d}.{component}.png"
                            image.write_bytes(content)
                            hashes[image.name] = suite.digest(image)
                    (work / "generation.json").write_text(json.dumps({
                        "stage": "completed", "signature": "teacher", "final_sha256": hashes,
                    }))
                elif "--config_file" in command:
                    name = command[command.index("--project") + 1]
                    executed.append(name)
                    output = (Path(command[command.index("--output_root") + 1]) / name
                              / command[command.index("--run_name") + 1])
                    output.mkdir(parents=True)
                    config = OmegaConf.load(command[command.index("--config_file") + 1])
                    config.log_dir = str(output)
                    OmegaConf.save(config, output / "config.yaml")
                    (output / "checkpoint_final.pth").write_bytes(b"simulated checkpoint")
                else:
                    output = Path(command[command.index("--output-dir") + 1])
                    name = output.parent.name
                    executed.append(name)
                    frames = stages[name]["frames"]
                    output.mkdir(parents=True)
                    (output / "manifest.json").write_text(json.dumps({
                        "status": "completed", "rendered_frames": frames,
                    }))
                    (output / "summary.json").write_text(json.dumps({"frames": frames}))
                    (output / "frames.jsonl").write_text("{}\n" * frames)
                    (output / "comparison.mp4").write_bytes(b"video")
                return SimpleNamespace(returncode=0)

            self.quiet_execute(args, plan, run=simulated_pipeline)
            self.assertEqual(executed, [stage["id"] for stage in plan["stages"]])
            self.quiet_execute(args, plan, run=simulated_pipeline)
            self.assertEqual(len(executed), 34)

    def test_completed_outputs_are_verified_and_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            plan = self.small_plan([self.render_stage()])
            run = mock.Mock(side_effect=self.fake_render)
            self.quiet_execute(args, plan, run=run)
            self.quiet_execute(args, plan, run=run)
            self.assertEqual(run.call_count, 1)
            state = json.loads((args.output_dir / "suite.json").read_text())
            self.assertEqual(suite.latest(state, "render")["status"], "completed")
            self.assertTrue((args.output_dir / "suite_summary.csv").is_file())
            command = run.call_args.args[0]
            self.assertIsInstance(command, list)
            self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_corrupt_completed_artifact_starts_new_attempt_and_preserves_old(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            plan = self.small_plan([self.render_stage()])
            run = mock.Mock(side_effect=self.fake_render)
            state = self.quiet_execute(args, plan, run=run)
            old = Path(suite.latest(state, "render")["output"])
            (old / "comparison.mp4").write_bytes(b"corrupted")
            state = self.quiet_execute(args, plan, run=run)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(Path(suite.latest(state, "render")["output"]).name, "attempt_002")
            self.assertEqual((old / "comparison.mp4").read_bytes(), b"corrupted")

    def test_failure_aborts_dependents_and_retry_preserves_log(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            plan = self.small_plan([self.render_stage("first"), self.render_stage("second")])
            fail = mock.Mock(return_value=SimpleNamespace(returncode=7))
            with self.assertRaises(RuntimeError):
                self.quiet_execute(args, plan, run=fail)
            self.assertEqual(fail.call_count, 1)
            state = json.loads((args.output_dir / "suite.json").read_text())
            self.assertEqual(suite.latest(state, "first")["status"], "failed")
            self.assertNotIn("second", state["stages"])
            state = self.quiet_execute(args, plan, run=self.fake_render)
            self.assertEqual(len(state["stages"]["first"]["attempts"]), 2)
            self.assertTrue((args.output_dir / "logs/first/attempt_001.log").is_file())
            self.assertEqual(suite.latest(state, "second")["status"], "completed")

    def test_signature_change_and_concurrent_lock_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            plan = self.small_plan([])
            self.quiet_execute(args, plan)
            changed = copy.deepcopy(plan)
            changed["signature"] = "changed"
            with self.assertRaisesRegex(ValueError, "changed"):
                self.quiet_execute(args, changed)
            with (args.output_dir / ".suite.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    self.quiet_execute(args, plan)

    def test_training_retry_uses_new_output_and_dependent_render_follows_it(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            train = {"id": "training", "kind": "train", "group": "heldout",
                     "config": {"data": {}, "seed": 0}}
            render = self.render_stage()
            render.update(dependencies=["training"], checkpoint_stage="training")
            plan = self.small_plan([train, render])
            commands = []

            def run(command, **kwargs):
                commands.append(command)
                if "--config_file" not in command:
                    return self.fake_render(command, **kwargs)
                output = (Path(command[command.index("--output_root") + 1])
                          / command[command.index("--project") + 1] / command[command.index("--run_name") + 1])
                output.mkdir(parents=True)
                config = OmegaConf.load(command[command.index("--config_file") + 1])
                config.log_dir = str(output)
                OmegaConf.save(config, output / "config.yaml")
                (output / "checkpoint_final.pth").write_bytes(b"checkpoint")
                return SimpleNamespace(returncode=0)

            state = self.quiet_execute(args, plan, run=run)
            previous = Path(suite.latest(state, "training")["output"])
            (previous / "checkpoint_final.pth").write_bytes(b"changed")
            state = self.quiet_execute(args, plan, run=run)
            self.assertEqual(len(commands), 4)
            self.assertIn("attempt_002/checkpoint_final.pth",
                          commands[-1][commands[-1].index("--checkpoint") + 1])
            self.assertTrue(previous.is_dir())
            self.assertNotIn("--resume_from", commands[-2])

    def test_surface_edit_summary_omits_raw_psnr(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            stage = self.render_stage("edit")
            stage["group"] = "surface_edit"
            self.quiet_execute(args, self.small_plan([stage]), run=self.fake_render)
            summary = json.loads((args.output_dir / "suite_summary.json").read_text())
            self.assertTrue(all(row["rgb_psnr"] is None for row in summary["rows"]))


if __name__ == "__main__":
    unittest.main()
