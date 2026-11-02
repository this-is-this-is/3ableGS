"""Stage inheritance, interrupted queues, and optimizer/RNG continuation."""
import argparse
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from tools.run_base_v4_weather import stage_command
from utils.config import load_training_config
from utils.weather_training import (
    load_weather_config, model_state, restore_training_state, save_training_state,
    training_signature, exclusive_run_lock,
)

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "configs/base_v4/weather.yaml"


class WeatherConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw_config = load_training_config(ROOT / "configs/base_v4/scene788_30f.yaml")
        self.raw = self.save_config("raw", self.raw_config)

    def tearDown(self):
        self.temp.cleanup()

    def save_config(self, name, cfg):
        path = self.root / name
        path.mkdir(exist_ok=True)
        OmegaConf.save(cfg, path / "config.yaml")
        return path / "checkpoint_final.pth"

    def test_stage_lengths_lrs_and_topology_inherited(self):
        material = load_weather_config(OVERLAY, self.raw, "rain", "material")
        self.assertEqual(material.data.scene_idx, 788)
        self.assertEqual(material.data.end_timestep, 29)
        self.assertEqual(material.data.pixel_source.gbuffer_path, self.raw_config.data.pixel_source.gbuffer_path)
        self.assertEqual(material.model.Background.feature.feature_dim, 32)
        self.assertEqual(material.trainer.weather_training.material_decoder.hidden_dim, 64)
        self.assertEqual(material.trainer.weather_training.material_decoder.num_hidden_layers, 2)
        self.assertEqual(material.trainer.optim.num_iters, 30000)
        self.assertEqual(material.trainer.weather_training.paired_target_mode, "gt_relative")
        material.trainer.weather_training.material_decoder.hidden_dim = 48
        material.trainer.weather_training.target_albedo_delta_cap = 3.
        env = load_weather_config(OVERLAY, self.save_config("material", material), "rain", "environment")
        self.assertEqual(env.trainer.optim.num_iters, 3000)
        self.assertEqual(env.trainer.weather_training.material_decoder.hidden_dim, 48)
        self.assertEqual(env.trainer.weather_training.target_albedo_delta_cap, 3.)
        joint = load_weather_config(OVERLAY, self.save_config("env", env), "rain", "joint")
        self.assertEqual(joint.trainer.optim.num_iters, 5000)
        self.assertEqual(joint.trainer.weather_training.material_lr, 5e-5)
        self.assertEqual(joint.trainer.weather_training.environment_lr, 5e-4)
        self.assertEqual(joint.weather_run.raw_checkpoint, str(self.raw.resolve()))

    def test_resume_saved_config_authoritative_and_cannot_change_fitting(self):
        cfg = load_weather_config(OVERLAY, self.raw, "snow", "material")
        cfg.trainer.optim.num_iters = 31
        cfg.trainer.weather_training.material_decoder.hidden_dim = 48
        path = self.save_config("resume", cfg)
        resumed = load_weather_config(OVERLAY, path, resume=True)
        self.assertEqual(training_signature(cfg), training_signature(resumed))
        with self.assertRaisesRegex(ValueError, "cannot change"):
            load_weather_config(OVERLAY, path, overrides=["trainer.optim.num_iters=30"], resume=True)
        with self.assertRaisesRegex(ValueError, "stage differs"):
            load_weather_config(OVERLAY, path, stage="joint", resume=True)
        changed_logging = load_weather_config(OVERLAY, path, overrides=["logging.print_freq=1"], resume=True)
        self.assertEqual(changed_logging.logging.print_freq, 1)

    def test_wrong_stage_or_cross_weather_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "completed material"):
            load_weather_config(OVERLAY, self.raw, "rain", "environment")
        mat = load_weather_config(OVERLAY, self.raw, "rain", "material")
        path = self.save_config("rain", mat)
        with self.assertRaisesRegex(ValueError, "independently"):
            load_weather_config(OVERLAY, path, "snow", "material")
        with self.assertRaisesRegex(ValueError, "completed environment"):
            load_weather_config(OVERLAY, path, "rain", "joint")
        with self.assertRaisesRegex(ValueError, "must initialize from raw"):
            load_weather_config(OVERLAY, path, "rain", "material")

    def test_queue_retries_evaluation_instead_of_skipping_final_weights(self):
        args = argparse.Namespace(output_dir=self.root / "runs", config_file=OVERLAY, base_ckpt=self.raw)
        path, command = stage_command(args, "rain", "material", 1, self.raw)
        self.assertIn("--init_from_base", command)
        path.mkdir(parents=True)
        cfg = load_weather_config(OVERLAY, self.raw, "rain", "material")
        OmegaConf.save(cfg, path / "config.yaml")
        (path / "checkpoint_training_latest.pth").touch()
        _, command = stage_command(args, "rain", "material", 1, self.raw)
        self.assertIn("--resume_from", command)
        self.assertNotIn("--evaluate", command)
        (path / "checkpoint_final.pth").touch()
        _, command = stage_command(args, "rain", "material", 1, self.raw)
        self.assertIn("--evaluate", command)
        (path / "stage_status.json").write_text('{"status":"completed","evaluation_done":true}')
        _, command = stage_command(args, "rain", "material", 1, self.raw)
        self.assertIsNone(command)
        args.base_ckpt = self.root / "different_raw.pth"
        with self.assertRaisesRegex(ValueError, "different base"):
            stage_command(args, "rain", "material", 1, self.raw)

    def test_duplicate_stage_lock_is_rejected_and_released(self):
        path = self.root / ".lock"
        with exclusive_run_lock(path):
            with self.assertRaisesRegex(RuntimeError, "already using"):
                with exclusive_run_lock(path):
                    self.fail("Duplicate lock acquired")
        with exclusive_run_lock(path):
            pass


class TinyTrainer:
    def __init__(self):
        self.net = torch.nn.Linear(3, 1)
        self.optimizer = torch.optim.Adam([{"params": list(self.net.parameters()),
                                            "name": "WeatherMaterial#rain", "lr": .01}], eps=1e-15)
        self.step = 0

    def state_dict(self, only_model=True):
        return {"step": self.step, "model": self.net.state_dict(),
                "weather_training_state": {"version": 1, "base_model_step": 30000, "weather": "rain"}}

    def update(self, step):
        self.optimizer.zero_grad()
        x = torch.randn(4, 3) * (random.random() + float(np.random.random()))
        self.net(x).square().mean().backward()
        self.optimizer.step()
        self.step = step


class WeatherResumeTests(unittest.TestCase):
    def test_continuous_matches_save_load_for_weights_adam_and_all_rngs(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        trainer = TinyTrainer()
        cfg = OmegaConf.create({"trainer": {"optim": {"num_iters": 4}}, "seed": 17})
        trainer.update(1)
        trainer.update(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pth"
            save_training_state(trainer, cfg, path)
            trainer.update(3)
            trainer.update(4)
            expected = copy.deepcopy(trainer.net.state_dict())
            expected_opt = copy.deepcopy(trainer.optimizer.state_dict())
            expected_next = (random.random(), np.random.random(), torch.rand(2))
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            state = model_state(loaded)
            self.assertEqual(state["weather_training_state"]["base_model_step"], 30000)
            clone = TinyTrainer()
            clone.net.load_state_dict(state["model"])
            clone.step = state["step"]
            start = restore_training_state(clone, loaded, cfg)
            self.assertEqual(start, 3)
            for step in range(start, 5):
                clone.update(step)
            for key, tensor in clone.net.state_dict().items():
                torch.testing.assert_close(tensor, expected[key], atol=0, rtol=0)
            for key, entry in clone.optimizer.state_dict()["state"].items():
                for name, value in entry.items():
                    torch.testing.assert_close(value, expected_opt["state"][key][name], atol=0, rtol=0)
            self.assertEqual(random.random(), expected_next[0])
            self.assertEqual(np.random.random(), expected_next[1])
            torch.testing.assert_close(torch.rand(2), expected_next[2], atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "optimizer/RNG"):
                restore_training_state(clone, state, cfg)


if __name__ == "__main__":
    unittest.main()
