"""CPU contract checks for frozen pilot optimization and temporal reporting."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from tools import train_gbuffer_attention as train
from tools import eval_gbuffer_attention as evaluate


def config():
    return OmegaConf.create({
        "seed": 0, "trainer": {"type": "models.trainers.MultiTrainer_pbr",
                                  "render": {"antialiased": False, "packed": False}},
        "data": {"preload_device": "cuda", "scene_idx": 788,
                 "start_timestep": 0, "end_timestep": 3,
                 "pixel_source": {"cameras": [0], "test_image_stride": 2,
                                  "load_gbuffer": True, "load_weather": [],
                                  "gbuffer_path": "teacher", "downscale": 1,
                                  "downscale_when_loading": [1]}},
        "log_dir": "old",
    })


class ToySplit:
    def __init__(self):
        self.selected = []

    def next(self, downscale, weathers):
        assert downscale == 1 and weathers == ["raw"]
        image_id = random.choice([0, 1, 3])
        self.selected.append(image_id)
        image = {"img_idx": torch.full((8, 8), image_id),
                 "frame_idx": torch.full((8, 8), image_id),
                 "normed_time": torch.full((8, 8), image_id / 3),
                 "viewdirs": torch.ones(8, 8, 3), "pixels": torch.full((8, 8, 3), .6),
                 "sky_masks": torch.zeros(8, 8), "gbuf_basecolor": torch.ones(8, 8, 3)}
        return {"raw": image}, {"raw": {"cam_id": torch.zeros(8, 8)}}


class ToyTrainer(nn.Module):
    def __init__(self, fail_at=None):
        super().__init__()
        self.device = torch.device("cpu")
        self.models = {"Background": nn.Linear(2, 3), "EnvLight": nn.Linear(3, 3)}
        self.step = 30000
        self.fail_at = fail_at
        self.loss_calls = []

    def configure_gbuffer_attention(self, config):
        self.gbuffer_attention_config = config
        self.gbuffer_attention_head = nn.Linear(3, 3) if config["variant"] not in ("B0", "H0") else None

    def set_eval(self):
        self.eval()
        for model in self.models.values():
            model.eval()

    def forward(self, image_infos, camera_infos, weather):
        # Forward never receives GT RGB/material targets.
        assert "pixels" not in image_infos and "gbuf_basecolor" not in image_infos
        assert self.training and not any(model.training for model in self.models.values())
        prediction = self.gbuffer_attention_head(image_infos["viewdirs"]).sigmoid()
        return {"rgb": prediction, "attention_context_ids": [0]}

    def compute_losses(self, outputs, image_infos, camera_infos, weather):
        if self.attention_step == self.fail_at:
            raise RuntimeError("simulated interruption")
        self.loss_calls.append((self.step, self.attention_step, "gbuf_basecolor" in image_infos))
        difference = outputs["rgb"] - image_infos["pixels"]
        # Consume crop RNG to exercise resume of the original stochastic losses.
        factor = 1 + torch.rand(()) * .01
        return {"rgb_loss": .7 * difference.abs().mean(),
                "ssim_loss": .3 * difference.square().mean() * factor}

    def state_dict(self):
        state = dict(super().state_dict())
        state.update(step=self.step, models={key: module.state_dict() for key, module in self.models.items()})
        return state

    def load_state_dict(self, state, load_only_model=True, strict=True):
        state = dict(state)
        self.step = state.pop("step")
        for key, value in state.pop("models").items():
            self.models[key].load_state_dict(value)
        super().load_state_dict(state, strict=strict)


class WorkflowTests(unittest.TestCase):
    def source(self, root):
        folder = Path(root) / "source"
        folder.mkdir(exist_ok=True)
        checkpoint = folder / "checkpoint_final.pth"
        checkpoint.write_bytes(b"trusted test fixture")
        OmegaConf.save(config(), folder / "config.yaml")
        return checkpoint

    def args(self, checkpoint, output, variant="A2", steps=4, resume=False):
        arguments = ["--checkpoint", str(checkpoint), "--output-dir", str(output),
                     "--variant", variant, "--steps", str(steps), "--save-every", "2"]
        if resume:
            arguments.append("--resume")
        return train.make_parser().parse_args(arguments)

    def mocked_run(self, args, fail_at=None):
        created = []

        def load_scene(*unused, **kwargs):
            self.assertEqual(kwargs["load_gbuffer"], args.steps > 0)
            trainer = ToyTrainer(fail_at)
            dataset = SimpleNamespace(train_indices=[0, 1, 3], test_indices=[2], train_image_set=ToySplit())
            created.append((trainer, dataset))
            return config(), dataset, trainer

        with ExitStack() as stack:
            stack.enter_context(patch.object(train, "load_scene", side_effect=load_scene))
            stack.enter_context(patch.object(train, "code_identity", return_value={"test": "fixed"}))
            for method, value in (("get_rng_state_all", []), ("set_rng_state_all", None),
                                  ("manual_seed_all", None), ("reset_peak_memory_stats", None),
                                  ("max_memory_allocated", 0)):
                stack.enter_context(patch.object(torch.cuda, method, return_value=value))
            result = train.run(args)
        return result, created[0]

    def test_baseline_materialization_and_no_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.source(directory)
            for variant in ("B0", "H0"):
                output = Path(directory) / variant
                result, (trainer, dataset) = self.mocked_run(self.args(checkpoint, output, variant, 0))
                self.assertEqual(result["completed_updates"], 0)
                self.assertEqual(result["trainable_parameters"], 0)
                self.assertEqual(dataset.train_image_set.selected, [])
                self.assertTrue((output / "checkpoint_final.pth").is_file())
                self.assertEqual(OmegaConf.load(output / "config.yaml").trainer.gbuffer_attention.variant, variant)

    def test_only_head_trains_using_original_scene_step_and_training_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.source(directory)
            output = Path(directory) / "pilot"
            result, (trainer, dataset) = self.mocked_run(self.args(checkpoint, output))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_updates"], 4)
            self.assertTrue(all(step == 30000 and has_teacher for step, _, has_teacher in trainer.loss_calls))
            self.assertTrue(set(dataset.train_image_set.selected) <= {0, 1, 3})
            for model in trainer.models.values():
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            self.assertTrue(any(p.grad is not None for p in trainer.gbuffer_attention_head.parameters()))

    def test_interrupted_optimizer_and_rng_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.source(directory)
            full, resumed = Path(directory) / "full", Path(directory) / "resumed"
            self.mocked_run(self.args(checkpoint, full))
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.mocked_run(self.args(checkpoint, resumed), fail_at=3)
            self.mocked_run(self.args(checkpoint, resumed, resume=True))
            left = torch.load(full / "checkpoint_final.pth", weights_only=False)
            right = torch.load(resumed / "checkpoint_final.pth", weights_only=False)
            for key in ("gbuffer_attention_head.weight", "gbuffer_attention_head.bias"):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            self.assertEqual((full / "losses.jsonl").read_text(), (resumed / "losses.jsonl").read_text())
            # Resume identity includes source options, preventing accidental reuse.
            changed = self.args(checkpoint, resumed, steps=5, resume=True)
            with self.assertRaisesRegex(ValueError, "Resume source/code/config/options changed"):
                self.mocked_run(changed)

    def test_nonempty_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.source(directory)
            output = Path(directory) / "used"
            output.mkdir()
            (output / "artifact").write_text("keep")
            with self.assertRaises(FileExistsError):
                train.validate_args(self.args(checkpoint, output))
            self.assertEqual((output / "artifact").read_text(), "keep")

    def test_rdg_streaming_cache_is_rejected_before_loading_or_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.source(directory)
            saved = config()
            saved.trainer.rdg_fusion = {"enabled": True}
            OmegaConf.save(saved, checkpoint.parent / "config.yaml")
            output = Path(directory) / "must_not_exist"
            args = evaluate.make_parser().parse_args([
                "--checkpoint", str(checkpoint), "--output-dir", str(output), "--streaming-cache"])
            with patch.object(evaluate, "load_scene") as load:
                with self.assertRaisesRegex(ValueError, "does not support --streaming-cache"):
                    evaluate.run(args)
            load.assert_not_called()
            self.assertFalse(output.exists())

    def test_invalid_steps_and_finite_optimizer_options(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.source(directory)
            output = Path(directory) / "new"
            for variant, steps in (("B0", 2), ("H0", 2), ("A2", 0)):
                with self.assertRaises(ValueError):
                    train.validate_args(self.args(checkpoint, output, variant, steps))
            args = self.args(checkpoint, output)
            args.lr = float("nan")
            with self.assertRaises(ValueError):
                train.validate_args(args)

    def test_paired_protocol_preserves_resolution_split_aa_affine(self):
        left = config()
        right = copy.deepcopy(left)
        right.data.preload_device = "cpu"
        right.data.pixel_source.load_gbuffer = False
        self.assertEqual(evaluate.protocol_signature(left), evaluate.protocol_signature(right))
        for key, value in (("downscale", 2), ("test_image_stride", 8)):
            changed = copy.deepcopy(left)
            changed.data.pixel_source[key] = value
            self.assertNotEqual(evaluate.protocol_signature(left), evaluate.protocol_signature(changed))
        changed = copy.deepcopy(left)
        changed.trainer.render.antialiased = True
        self.assertNotEqual(evaluate.protocol_signature(left), evaluate.protocol_signature(changed))

    def test_temporal_summary_never_calls_train_to_test_clean_holdout(self):
        records = []
        for frame in range(4):
            records.append({"frame_id": frame, "temporal_has_pair": frame > 0,
                            "prediction_render_ms": 10., "history_coverage": .5,
                            "prediction_temporal_residual_l1_sum": float(frame),
                            "prediction_temporal_residual_l1_count": 2 if frame else 0})
        selected = evaluate.train_to_test_records(records, [0, 1, 3], [2])
        self.assertEqual([record["frame_id"] for record in selected], [2])
        result = evaluate.summarize(records, [0, 1, 3], [2])
        self.assertIn("includes training", result["scope"])
        self.assertEqual(result["temporal"]["temporal_pairs"], 3)
        self.assertEqual(result["train_to_test"]["temporal_pairs"], 1)
        self.assertEqual(result["mean_prediction_render_ms"], 10.)
        self.assertNotIn("psnr", result)

    def test_full_temporal_workflow_preserves_float_packets_and_shared_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.source(directory)
            other = Path(directory) / "variant"
            other.mkdir()
            checkpoint = other / "checkpoint_final.pth"
            checkpoint.write_bytes(b"variant fixture")
            OmegaConf.save(config(), other / "config.yaml")
            output = Path(directory) / "temporal"
            args = evaluate.make_parser().parse_args([
                "--checkpoint", str(checkpoint), "--baseline-checkpoint", str(baseline),
                "--output-dir", str(output)])

            def load(path):
                trainer = ToyTrainer()
                trainer.tag = "baseline" if Path(path) == baseline else "prediction"
                trainer.set_gbuffer_attention_cache = lambda enabled: None
                dataset = SimpleNamespace(train_timesteps=[0, 1, 3], test_timesteps=[2],
                                          full_image_set=list(range(4)))
                return config(), dataset, trainer

            def frame(trainer, dataset, index):
                reference = torch.full((16, 16, 3), .33333334 + index * .01)
                prediction = reference + (.001 * index if trainer.tag == "prediction" else .01 * index)
                info = {"pixels": reference, "sky_masks": torch.zeros(16, 16),
                        "dynamic_masks": torch.zeros(16, 16)}
                return index, info, {"rgb": prediction}, 10.

            with ExitStack() as stack:
                stack.enter_context(patch.object(evaluate, "load_scene", side_effect=load))
                stack.enter_context(patch.object(evaluate, "_frame", side_effect=frame))
                stack.enter_context(patch("utils.temporal_metrics.TemporalEvaluator._flow",
                                          return_value=np.zeros((16, 16, 2), np.float32)))
                stack.enter_context(patch.object(torch.cuda, "empty_cache"))
                stack.enter_context(patch.object(torch.cuda, "reset_peak_memory_stats"))
                stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=0))
                summary = evaluate.run(args)
            self.assertEqual(summary["temporal"]["temporal_pairs"], 3)
            self.assertEqual(summary["train_to_test"]["temporal_pairs"], 1)
            self.assertAlmostEqual(summary["temporal"]["prediction_temporal_residual_l1"], .001, places=6)
            self.assertAlmostEqual(summary["temporal"]["baseline_temporal_residual_l1"], .01, places=6)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "complete")
            self.assertEqual(len((output / "frames.jsonl").read_text().splitlines()), 4)
            self.assertFalse(list(output.glob(".baseline_*")))

    def test_temporal_protocol_mismatch_fails_before_loading_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.source(directory)
            folder = Path(directory) / "bad"
            folder.mkdir()
            checkpoint = folder / "checkpoint_final.pth"
            checkpoint.write_bytes(b"variant fixture")
            changed = config()
            changed.data.pixel_source.downscale = 2
            OmegaConf.save(changed, folder / "config.yaml")
            output = Path(directory) / "untouched"
            args = evaluate.make_parser().parse_args([
                "--checkpoint", str(checkpoint), "--baseline-checkpoint", str(baseline),
                "--output-dir", str(output)])
            with patch.object(evaluate, "load_scene") as loader:
                with self.assertRaisesRegex(ValueError, "protocols differ"):
                    evaluate.run(args)
                loader.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
