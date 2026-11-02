"""Keep Base v4's historical overlay isolated from scene/model defaults."""
from pathlib import Path
import unittest

from omegaconf import OmegaConf

from utils.config import load_training_config


ROOT = Path(__file__).resolve().parents[1]


def flatten(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, path))
        else:
            result[path] = value
    return result


class BaseV4ConfigTests(unittest.TestCase):
    def test_only_historical_combo_delta_for_both_data_presets(self):
        # E03/E05: these are the complete config changes, not broad optimizer,
        # geometry, resolution, weather-head or dataset replacements.
        expected = {
            "trainer.losses.rgb.w": .7,
            "trainer.losses.ssim.w": .3,
            "trainer.losses.boundary.enabled": True,
            "trainer.losses.boundary.start_step": 5000,
            "trainer.losses.boundary.ramp_steps": 1000,
            "trainer.losses.boundary.rgb_w": 0.,
            "trainer.losses.boundary.gradient_w": 0.,
            "trainer.losses.boundary.perceptual_w": .02,
            "trainer.losses.boundary.edge_threshold": .05,
            "trainer.losses.boundary.radius": 2,
            "trainer.losses.boundary.patch_size": 128,
            "trainer.losses.boundary.num_patches": 4,
            "trainer.losses.boundary.min_valid_fraction": .99,
            "model.Background.ctrl.stop_split_at": 25000,
            "model.Background.ctrl.stop_reset_at": 20000,
        }
        for preset in ("base.yaml", "scene788_30f.yaml"):
            with self.subTest(preset=preset):
                v3 = load_training_config(ROOT / "configs/base_v3" / preset)
                v4 = load_training_config(ROOT / "configs/base_v4" / preset)
                before = flatten(OmegaConf.to_container(v3, resolve=True))
                after = flatten(OmegaConf.to_container(v4, resolve=True))
                self.assertFalse(set(before) - set(after))
                diff = {k: v for k, v in after.items() if k not in before or before[k] != v}
                self.assertEqual(diff, expected)
                # Class-specific overrides must not leak into shared defaults.
                self.assertEqual(v4.trainer.gaussian_ctrl_general_cfg.stop_split_at, 15000)
                for name in ("RigidNodes", "DeformableNodes"):
                    self.assertEqual(v4.model[name].ctrl.stop_split_at, 20000)
                    self.assertNotIn("stop_reset_at", v4.model[name].ctrl)
                self.assertEqual(OmegaConf.to_container(v3.data), OmegaConf.to_container(v4.data))

    def test_788_thirty_frames_and_cli_test_eval_preserve_training_protocol(self):
        cfg = load_training_config(ROOT / "configs/base_v4/scene788_30f.yaml", [
            "render.render_test=true", "render.render_full=false", "render.render_novel=null",
        ])
        self.assertEqual((cfg.seed, cfg.trainer.optim.num_iters), (0, 30000))
        self.assertEqual((cfg.data.scene_idx, cfg.data.start_timestep, cfg.data.end_timestep),
                         (788, 0, 29))
        self.assertEqual(list(cfg.data.pixel_source.cameras), [0])
        self.assertEqual(list(cfg.data.pixel_source.load_weather), [])
        self.assertEqual(cfg.data.pixel_source.test_image_stride, 8)
        self.assertEqual(cfg.data.pixel_source.gbuffer_path,
                         "data/waymo/render_png_788_single_static_30f/gbuffer_frames")
        self.assertIsNone(cfg.vgg)
        self.assertTrue(cfg.render.render_test)


if __name__ == "__main__":
    unittest.main()
