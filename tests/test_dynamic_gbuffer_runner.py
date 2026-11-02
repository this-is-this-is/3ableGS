"""D1 queue contracts with fake children; no CUDA scene or training is run."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess

from omegaconf import OmegaConf
import tempfile
import unittest

from tools import run_dynamic_gbuffer_experiments as suite
from utils.config import load_training_config


def args(tmp_path, *extra):
    return suite.make_parser().parse_args(["--output-dir", str(tmp_path / "suite"), *extra])


class FakeChildren:
    def __init__(self):
        self.calls = []
        self.fail_next_eval = False
        self.fail_next_train = False

    def __call__(self, command, log_path):
        self.calls.append(command)

        def flag(name):
            return command[command.index(name) + 1]

        if "--prepare-only" in command:
            assert "--skip-eval" in command
            return
        if command[2].endswith("train_base_v5.py"):
            assert "--skip-eval" in command
            assert "--checkpoint" not in command and "--resume_from" not in command
            directory = Path(flag("--run-dir"))
            directory.mkdir(parents=True, exist_ok=True)
            seed = next(value for value in command if value.startswith("seed="))
            config = load_training_config(flag("--config-file"), [seed])
            config.log_dir = str(directory)
            OmegaConf.save(config, directory / "config.yaml")
            if self.fail_next_train:
                self.fail_next_train = False
                (directory / "checkpoint_training_latest.pth").write_bytes(b"unfinished exact-resume state")
                raise subprocess.CalledProcessError(1, command)
            total = int(flag("--iterations"))
            (directory / "checkpoint_final.pth").write_bytes(f"fake-final-{total}".encode())
            (directory / "training_status.json").write_text(json.dumps({
                "completed_updates": total, "total_updates": total, "camera_updates": total,
                "process_start_update": 0, "loop_elapsed_seconds_this_process": 1.25,
                "peak_allocated_vram_bytes_this_process": 123456}))
        else:
            assert command[2].endswith("eval_base_v5.py")
            assert "--dynamic-metrics" in command
            if self.fail_next_eval:
                self.fail_next_eval = False
                raise subprocess.CalledProcessError(2, command)
            checkpoint = Path(flag("--checkpoint"))
            directory = checkpoint.parent / "rgb_joint_eval"
            directory.mkdir(exist_ok=True)
            completed = json.loads((checkpoint.parent / "training_status.json").read_text())["completed_updates"]
            result = {"checkpoint": str(checkpoint),
                      "training_budget": {"completed_camera_updates": completed},
                      "image_metrics": {"psnr": 31., "ssim": .93, "lpips": .11},
                      "pure_pbr_image_metrics": {"psnr": 27., "ssim": .90, "lpips": .17},
                      "dynamic_object_metrics": {"protocol": {}, "summary": {"dynamic": {"count": 12}}},
                      "elapsed_seconds": 1., "images_saved": "--save-images" in command,
                      "smoke": completed < 100}
            if "--dynamic-temporal" in command:
                result["dynamic_temporal_diagnostic"] = {"heldout_temporal_benchmark": False}
            (directory / "evaluation.json").write_text(json.dumps(result))


def run(parsed, fake):
    with redirect_stdout(io.StringIO()):
        return suite.run_suite(parsed, suite.build_plan(parsed), child_runner=fake)


class DynamicGBufferRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp_path = Path(self.temporary.name)

    def test_preview_does_not_create_files_or_start_children(self):
        tmp_path = self.tmp_path
        parsed, fake = args(tmp_path), FakeChildren()
        result = run(parsed, fake)
        assert result["mode"] == "preview"
        assert not parsed.output_dir.exists() and not fake.calls
        assert result["protocol"]["updates_per_arm"] == 30001
        assert result["protocol"]["test_ids"] == list(range(8, 100, 8))
        assert list(result["commands"]) == ["B0_s0", "D1B_s0", "D1P_s0"]
        for commands in result["commands"].values():
            assert "--dynamic-metrics" in commands["evaluate"]
            assert "--save-images" not in commands["evaluate"]


    def test_overlays_preserve_all_base_v5_training_and_data_settings(self):
        baseline = OmegaConf.to_container(load_training_config(suite.ROOT / "configs/base_v5/scene788_100f.yaml"), resolve=True)
        dynamic = []
        for variant in suite.VARIANTS:
            config = OmegaConf.to_container(suite.resolved_config(variant, 0), resolve=True)
            dynamic.append(config["trainer"].pop("dynamic_gbuffer"))
            assert config == baseline
        assert not dynamic[0]["enabled"] and dynamic[1]["enabled"] and dynamic[2]["enabled"]
        assert dynamic[1]["mode"] == "balanced" and dynamic[2]["mode"] == "purity"
        balanced, purity = deepcopy(dynamic[1]), deepcopy(dynamic[2])
        balanced.pop("mode"); purity.pop("mode")
        assert balanced == purity
        assert dynamic[1]["visibility_downscale"] == 2


    def test_prepare_only_calls_native_validation_without_output_creation(self):
        tmp_path = self.tmp_path
        parsed, fake = args(tmp_path, "--prepare-only"), FakeChildren()
        result = run(parsed, fake)
        assert result["mode"] == "prepare-only"
        assert len(fake.calls) == 3
        assert all("--prepare-only" in command for command in fake.calls)
        assert not parsed.output_dir.exists()


    def test_matched_execution_and_verified_repeat_skip_children(self):
        tmp_path = self.tmp_path
        parsed, fake = args(tmp_path, "--execute"), FakeChildren()
        result = run(parsed, fake)
        assert result["status"] == "completed" and len(fake.calls) == 6
        assert all(row["completed_updates"] == 30001 for row in result["arms"].values())
        assert all(row["evaluation"]["image_metrics"]["psnr"] == 31 for row in result["arms"].values())
        assert result["arms"]["D1P_s0"]["last_training_process"] == {
            "process_start_update": 0, "loop_elapsed_seconds_this_process": 1.25,
            "peak_allocated_vram_bytes_this_process": 123456}
        assert (parsed.output_dir / "report.md").is_file()
        run(parsed, fake)
        assert len(fake.calls) == 6


    def test_failed_training_is_recorded_and_same_directory_is_resumed(self):
        tmp_path = self.tmp_path
        parsed, fake = args(tmp_path, "--execute", "--variants", "D1P"), FakeChildren()
        fake.fail_next_train = True
        with self.assertRaises(subprocess.CalledProcessError):
            run(parsed, fake)
        failed = json.loads((parsed.output_dir / "summary.json").read_text())
        assert failed["status"] == "failed"
        assert failed["arms"]["D1P_s0"]["train_state"] == "incomplete"
        result = run(parsed, fake)
        assert result["status"] == "completed"
        assert len(fake.calls) == 3
        assert fake.calls[0] == fake.calls[1]
        assert len(result["arms"]["D1P_s0"]["train_attempts"]) == 2


    def test_failed_evaluation_does_not_restart_successful_training(self):
        tmp_path = self.tmp_path
        parsed, fake = args(tmp_path, "--execute", "--variants", "B0"), FakeChildren()
        fake.fail_next_eval = True
        with self.assertRaises(subprocess.CalledProcessError):
            run(parsed, fake)
        result = run(parsed, fake)
        assert len(fake.calls) == 3 and fake.calls[1] == fake.calls[2]
        assert result["arms"]["B0_s0"]["status"] == "completed"
        assert len(result["arms"]["B0_s0"]["train_attempts"]) == 1


    def test_eval_options_invalidate_only_evaluation_cache(self):
        tmp_path = self.tmp_path
        fake = FakeChildren()
        run(args(tmp_path, "--execute", "--variants", "D1B"), fake)
        result = run(args(tmp_path, "--execute", "--variants", "D1B", "--dynamic-temporal"), fake)
        assert len(fake.calls) == 3
        assert "--dynamic-temporal" in fake.calls[-1]
        assert result["arms"]["D1B_s0"]["evaluation"]["dynamic_temporal_diagnostic"]["heldout_temporal_benchmark"] is False


    def test_changed_checkpoint_invalidates_training_and_evaluation_cache(self):
        tmp_path = self.tmp_path
        parsed, fake = args(tmp_path, "--execute", "--variants", "B0"), FakeChildren()
        run(parsed, fake)
        checkpoint = parsed.output_dir / "seed_0/B0/checkpoint_final.pth"
        checkpoint.write_bytes(b"externally replaced checkpoint")
        run(parsed, fake)
        assert len(fake.calls) == 4


    def test_resume_rejects_saved_fitting_changes_before_any_child(self):
        tmp_path = self.tmp_path
        parsed, fake = args(tmp_path, "--execute", "--variants", "D1P"), FakeChildren()
        run(parsed, fake)
        path = parsed.output_dir / "seed_0/D1P/config.yaml"
        saved = OmegaConf.load(path)
        saved.trainer.dynamic_gbuffer.w = 9.
        OmegaConf.save(saved, path)
        with self.assertRaisesRegex(ValueError, "Saved fitting settings differ"):
            suite.build_plan(parsed)
        assert len(fake.calls) == 2


    def test_total_budget_extension_uses_same_native_schedule_and_directory(self):
        tmp_path = self.tmp_path
        fake = FakeChildren()
        run(args(tmp_path, "--execute", "--variants", "B0", "--iterations", "6"), fake)
        result = run(args(tmp_path, "--execute", "--variants", "B0", "--iterations", "9"), fake)
        assert len(fake.calls) == 4
        assert result["arms"]["B0_s0"]["completed_updates"] == 9
        config = OmegaConf.load(parsed_path := tmp_path / "suite/seed_0/B0/config.yaml")
        assert config.trainer.optim.num_iters == 30000
        assert parsed_path.is_file()


    def test_skip_eval_is_reported_as_trained_without_evaluation(self):
        tmp_path = self.tmp_path
        parsed, fake = args(tmp_path, "--execute", "--variants", "D1P", "--skip-eval"), FakeChildren()
        result = run(parsed, fake)
        assert result["status"] == "completed_without_evaluation"
        assert result["arms"]["D1P_s0"]["status"] == "trained_evaluation_skipped"
        assert "evaluation" not in result["arms"]["D1P_s0"] and len(fake.calls) == 1

    def test_lower_budget_rejected_instead_of_reusing_overtrained_arm(self):
        fake = FakeChildren()
        run(args(self.tmp_path, "--execute", "--variants", "B0", "--iterations", "9"), fake)
        with self.assertRaisesRegex(ValueError, "above requested"):
            suite.build_plan(args(self.tmp_path, "--execute", "--variants", "B0", "--iterations", "6"))
        self.assertEqual(len(fake.calls), 2)


    def test_invalid_plan_options_fail_before_mutation(self):
        for extra in (["--iterations", "0"], ["--seeds", "-1"], ["--seeds", "0", "0"],
                      ["--variants", "B0", "B0"], ["--skip-eval", "--save-images"],
                      ["--skip-eval", "--dynamic-temporal"]):
            with self.subTest(extra=extra):
                parsed = args(self.tmp_path, *extra)
                with self.assertRaises(ValueError):
                    suite.build_plan(parsed)
                self.assertFalse(parsed.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
