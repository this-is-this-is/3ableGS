"""Base v5 config isolation, total budgets and full optimizer continuation."""
import argparse
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from tools.train_base_v5 import prepare
from utils.config import load_training_config
from utils.rgb_joint_training import (STATE_KEY, validate_config, training_signature,
    save_training_state, read_training_state, load_model_state, restore_training_state)
from utils.weather_training import atomic_torch_save

ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "configs/base_v5/scene788_100f.yaml"


class Population(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._means = torch.nn.Parameter(torch.ones(3, 3))
        self.xys_grad_norm = torch.arange(3.)
        self.vis_counts = torch.ones(3)
        self.max_2Dsize = torch.full((3,), .5)
        self.nn_ind = torch.tensor([[1, 2], [0, 2], [0, 1]])

    @property
    def num_points(self):
        return len(self._means)


class Trainer:
    def __init__(self):
        self.step, self.device = 4, torch.device("cpu")
        self.rgb_joint_camera_updates = torch.tensor(5)
        self.gaussian_classes = {"Background": 0}
        self.models = {"Background": Population()}
        self.optimizer = torch.optim.Adam([{"params": self.models["Background"].parameters(), "name": "Background#xyz"}], lr=.01)
        self.grad_scaler = torch.amp.GradScaler("cpu", enabled=False)

    def state_dict(self, only_model=True):
        return {"step": self.step, "rgb_joint_camera_updates": self.rgb_joint_camera_updates,
                "models": {"Background": self.models["Background"].state_dict()}}

    def load_state_dict(self, state, load_only_model=True, strict=True):
        self.step = state["step"]
        self.rgb_joint_camera_updates = state["rgb_joint_camera_updates"]
        self.models["Background"].load_state_dict(state["models"]["Background"], strict=strict)


class RGBJointTrainingTests(unittest.TestCase):
    def test_config_changes_only_opt_in_branch(self):
        before = load_training_config(ROOT / "configs/base_v4/scene788_100f.yaml")
        after = load_training_config(PRESET)
        validate_config(after)
        after.trainer.pop("rgb_joint")
        after.trainer.gaussian_optim_general_cfg.pop("correction_feature")
        for name in ("Background", "RigidNodes", "DeformableNodes", "SMPLNodes"):
            after.model[name].pop("rgb_residual")
        self.assertEqual(OmegaConf.to_container(before, resolve=True), OmegaConf.to_container(after, resolve=True))

    def test_dry_prepare_is_scratch_and_preserves_native_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(config_file=PRESET, run_dir=Path(temp) / "fresh", iterations=None, opts=[])
            with patch("tools.train_base_v5.validate_inputs", return_value={}):
                cfg, path, total, checkpoint, step, _ = prepare(args)
            self.assertFalse(path.exists())
            self.assertEqual((total, step, checkpoint), (30001, 0, None))
            self.assertEqual(cfg.trainer.optim.num_iters, 30000)
            self.assertEqual(cfg.data.end_timestep, 99)
            path.mkdir(); OmegaConf.save(cfg, path / "config.yaml")
            args.opts = ["trainer.rgb_joint.pure_pbr_lambda=0.5"]
            with self.assertRaisesRegex(ValueError, "different fitting"):
                prepare(args)

    def test_complete_optimizer_rng_density_and_knn_resume(self):
        cfg = load_training_config(PRESET)
        dataset = SimpleNamespace(train_image_set=SimpleNamespace(split_indices=[0, 2, 3]))
        trainer = Trainer()
        p = trainer.models["Background"]._means
        p.square().sum().backward(); trainer.optimizer.step(); trainer.optimizer.zero_grad()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint_training_latest.pth"
            save_training_state(trainer, dataset, cfg, path)
            expected_random = (random.random(), np.random.rand(), torch.rand(2))
            p.square().sum().backward(); trainer.optimizer.step()
            expected_next = p.detach().clone()
            state = read_training_state(path, cfg)
            resumed = Trainer()
            load_model_state(resumed, state)
            resumed.models["Background"].nn_ind.zero_()
            self.assertEqual(restore_training_state(resumed, dataset, state, cfg), 5)
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.rand(), expected_random[1])
            torch.testing.assert_close(torch.rand(2), expected_random[2], rtol=0, atol=0)
            for key in ("xys_grad_norm", "vis_counts", "max_2Dsize", "nn_ind"):
                torch.testing.assert_close(getattr(resumed.models["Background"], key), getattr(trainer.models["Background"], key))
            q = resumed.models["Background"]._means
            q.square().sum().backward(); resumed.optimizer.step()
            torch.testing.assert_close(q, expected_next, rtol=0, atol=0)
            changed = deepcopy(cfg); changed.trainer.rgb_joint.pure_pbr_lambda = .6
            with self.assertRaisesRegex(ValueError, "fitting settings"):
                read_training_state(path, changed)
            state.pop(STATE_KEY); atomic_torch_save(state, path)
            with self.assertRaisesRegex(ValueError, "complete optimizer"):
                read_training_state(path, cfg)

    def test_resume_does_not_replay_completed_step_or_change_schedule(self):
        cfg = load_training_config(PRESET)
        dataset = SimpleNamespace(train_image_set=SimpleNamespace(split_indices=[0, 2, 3]))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            OmegaConf.save(cfg, path / "config.yaml")
            save_training_state(Trainer(), dataset, cfg, path / "checkpoint_training_latest.pth")
            args = argparse.Namespace(config_file=PRESET, run_dir=path, iterations=8, opts=[])
            with patch("tools.train_base_v5.validate_inputs", return_value={}):
                config, _, total, checkpoint, next_step, _ = prepare(args)
            self.assertEqual((next_step, total), (5, 8))
            self.assertEqual(config.trainer.optim.num_iters, 30000)
            self.assertEqual(checkpoint.name, "checkpoint_training_latest.pth")


if __name__ == "__main__":
    unittest.main()
