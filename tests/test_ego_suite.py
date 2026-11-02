"""Sequential experiment lifecycle, restart and input-integrity regressions."""
import fcntl
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from omegaconf import OmegaConf
from tools import run_ego_motion_experiments as suite


class EgoSuiteTests(unittest.TestCase):
    def args(self, root):
        return suite.make_parser().parse_args(["--checkpoint", str(root / "checkpoint.pth"),
            "--output-dir", str(root / "results"), "--material-steps", "2", "--head-steps", "2"])

    def source(self, root):
        (root / "checkpoint.pth").write_bytes(b"test scene checkpoint")
        OmegaConf.save(OmegaConf.create({"trainer": {"type": "models.trainers.MultiTrainer_pbr"},
            "data": {"data_root": str(root / "data"), "scene_idx": 788, "start_timestep": 0,
                     "end_timestep": 19, "pixel_source": {"cameras": [0], "test_frame_ranges": [[4, 7], [12, 15]]}}}), root / "config.yaml")

    def simple_plan(self, args, kinds=("head", "head")):
        stages = [{"id": f"stage_{index}", "module": "fake", "args": [],
                   "output": str(args.output_dir / f"stage_{index}"), "kind": kind}
                  for index, kind in enumerate(kinds)]
        return {"signature": "same", "identity": {"code": {"fake": "hash"}}, "stages": stages}

    def execute(self, args, plan, run):
        with mock.patch.object(suite, "code_identity", return_value=plan["identity"]["code"]), \
             mock.patch.object(suite, "write_report"), redirect_stdout(io.StringIO()):
            return suite.execute_plan(args, plan, run=run)

    def test_plan_readonly_order_and_matched_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root)
            args = self.args(root)
            with mock.patch.object(suite, "code_identity", return_value={}):
                plan = suite.build_plan(args)
            self.assertFalse(args.output_dir.exists())
            self.assertEqual(len(plan["stages"]), 36)
            ids = [stage["id"] for stage in plan["stages"]]
            self.assertEqual(ids[:5], ["cache_base", "diagnostics", "eval_base", "cache_aa", "eval_aa"])
            self.assertLess(ids.index("material_albedo_s0"), ids.index("head_current_s0"))
            self.assertLess(ids.index("head_full_s0"), ids.index("head_transformer_s0"))
            self.assertTrue(set(plan["train_frames"]).isdisjoint(plan["val_frames"] + plan["test_frames"]))
            self.assertEqual(plan["val_frames"], [4, 5, 6])
            self.assertEqual(plan["test_frames"], [12, 13, 14])
            full_no_tl = next(stage for stage in plan["stages"] if stage["id"] == "head_full_no_temporal_s0")
            self.assertEqual(full_no_tl["args"][full_no_tl["args"].index("--temporal-weight")+1], "0")

    def test_stride_checkpoint_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root)
            cfg = OmegaConf.load(root / "config.yaml")
            del cfg.data.pixel_source.test_frame_ranges
            cfg.data.pixel_source.test_image_stride = 8
            OmegaConf.save(cfg, root / "config.yaml")
            with self.assertRaisesRegex(ValueError, "explicit"):
                suite.build_plan(self.args(root))

    def test_complete_skip_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            plan = self.simple_plan(args)
            executed = []

            def run(command, **kwargs):
                stage = plan["stages"][len(executed)]
                output = Path(stage["output"])
                output.mkdir(parents=True)
                (output / "manifest.json").write_text('{"status":"completed"}')
                (output / "head_final.pt").write_bytes(b"weights")
                executed.append(command)
                return SimpleNamespace(returncode=0)

            self.execute(args, plan, run)
            self.execute(args, plan, run)
            self.assertEqual(len(executed), 2)
            (Path(plan["stages"][0]["output"]) / "head_final.pt").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "changed"):
                self.execute(args, plan, run)

    def test_failure_stops_dependents_and_optimizer_resume_is_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            plan = self.simple_plan(args, kinds=("material", "head"))
            first = Path(plan["stages"][0]["output"])
            first.mkdir(parents=True)
            (first / "resume_state.pt").write_bytes(b"optimizer")
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=17)

            with self.assertRaisesRegex(RuntimeError, "17"):
                self.execute(args, plan, run)
            self.assertEqual(len(calls), 1)
            self.assertIn("--resume", calls[0])
            state = json.loads((args.output_dir / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertNotIn("stage_1", state["stages"])
            self.assertTrue((args.output_dir / "logs/stage_0.001.log").is_file())

    def test_signature_and_concurrent_worker_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            plan = self.simple_plan(args, kinds=())
            self.execute(args, plan, mock.Mock())
            plan["signature"] = "changed"
            with self.assertRaisesRegex(ValueError, "changed"):
                self.execute(args, plan, mock.Mock())
            with (args.output_dir / ".suite.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    self.execute(args, plan, mock.Mock())

    def test_prior_queue_failure_also_releases_wait(self):
        with tempfile.TemporaryDirectory() as temp:
            status = Path(temp) / "status.json"
            for terminal in ("completed", "failed"):
                status.write_text(json.dumps({"stage": terminal}))
                with mock.patch.object(suite.time, "sleep") as sleep, redirect_stdout(io.StringIO()):
                    suite.wait_for_queue(status)
                sleep.assert_not_called()

    def test_selection_uses_validation_only(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            args.output_dir.mkdir()
            plan = {"stages": []}
            state = {"status": "running", "stages": {}}
            for name, val_temp, test_temp in (("base", .02, .001), ("full", .01, .1)):
                stage = {"id": name, "kind": "evaluate", "family": name, "output": str(args.output_dir / name)}
                output = Path(stage["output"])
                output.mkdir()
                metrics = {"prediction_full_psnr": 30., "prediction_edge_gradient_l1": .01,
                           "prediction_edge_strength_ratio": 1., "by_motion": {"high": {}}}
                (output / "summary.json").write_text(json.dumps({
                    "val": {**metrics, "prediction_temporal_residual_l1": val_temp},
                    "test": {**metrics, "prediction_temporal_residual_l1": test_temp}}))
                plan["stages"].append(stage)
                state["stages"][name] = {"status": "completed"}
            suite.write_report(args, plan, state)
            self.assertEqual(json.loads((args.output_dir / "selection.json").read_text())["selected"], "full")


if __name__ == "__main__":
    unittest.main()
