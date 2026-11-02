import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from utils.config import load_config_with_base, load_training_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, path, contents):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
        return target

    def test_nested_relative_bases_replace_weather_list(self):
        self.write("common.yaml", "data:\n  scene_idx: 788\n  pixel_source:\n    load_weather: [rain, snow]\n    test_image_stride: 0\n")
        self.write("scenes/raw.yaml", "base_config: ../common.yaml\ndata:\n  pixel_source:\n    load_weather: []\n")
        path = self.write("experiment.yaml", "base_config: scenes/raw.yaml\ndata:\n  scene_idx: 94\n  pixel_source:\n    test_image_stride: 8\n")
        cfg = load_config_with_base(path)
        self.assertEqual(cfg.data.scene_idx, 94)
        self.assertEqual(list(cfg.data.pixel_source.load_weather), [])
        self.assertEqual(cfg.data.pixel_source.test_image_stride, 8)
        self.assertNotIn("base_config", cfg)

    def test_cycle_is_rejected_but_shared_parent_is_valid(self):
        path = self.write("a.yaml", "base_config: b.yaml\n")
        self.write("b.yaml", "base_config: a.yaml\n")
        with self.assertRaisesRegex(ValueError, "Circular base_config"):
            load_config_with_base(path)
        self.write("base.yaml", "data: {scene_idx: 94}\n")
        self.write("a.yaml", "base_config: base.yaml\n")
        self.write("b.yaml", "base_config: base.yaml\n")
        path = self.write("both.yaml", "base_config: [a.yaml, b.yaml]\n")
        self.assertEqual(load_config_with_base(path).data.scene_idx, 94)

    def test_dataset_defaults_do_not_overwrite_explicit_data(self):
        self.write("datasets/waymo/one.yaml", "data: {scene_idx: 0, start_timestep: 0, end_timestep: -1}\n")
        self.write("datasets/waymo/two.yaml", "data: {scene_idx: 0, start_timestep: 10, end_timestep: -1}\n")
        path = self.write("run.yaml", "dataset: waymo/one\ndata: {scene_idx: 94, end_timestep: 99}\n")
        cfg = load_training_config(path, ["dataset=waymo/two", "data.scene_idx=23"], config_root=self.root)
        self.assertEqual(dict(cfg.data), {"scene_idx": 23, "start_timestep": 10, "end_timestep": 99})
        self.assertNotIn("dataset", cfg)

    def test_existing_official_default_config_is_unchanged(self):
        original = OmegaConf.load(ROOT / "configs/weathercity.yaml")
        alias = original.pop("dataset")
        expected = OmegaConf.merge(original, OmegaConf.load(ROOT / "configs/datasets" / f"{alias}.yaml"))
        actual = load_training_config(ROOT / "configs/weathercity.yaml")
        self.assertEqual(OmegaConf.to_container(actual), OmegaConf.to_container(expected))

    def test_preparation_config_has_exact_historical_frame_selection(self):
        cfg = load_training_config(ROOT / "configs/reconstruction/data_100f.yaml", ["data.scene_idx=788"])
        self.assertEqual(cfg.seed, 0)
        self.assertEqual(cfg.data.scene_idx, 788)
        self.assertEqual((cfg.data.start_timestep, cfg.data.end_timestep), (0, 99))
        self.assertEqual(list(cfg.data.pixel_source.cameras), [0])
        self.assertEqual(list(cfg.data.pixel_source.load_weather), [])
        self.assertEqual(cfg.data.pixel_source.test_image_stride, 8)
        self.assertEqual(cfg.data.data_root, "data/waymo/processed/training")
        self.assertNotIn("trainer", cfg)

    def test_invalid_base_type_is_rejected(self):
        path = self.write("invalid.yaml", "base_config: {wrong: type}\n")
        with self.assertRaisesRegex(ValueError, "base_config must"):
            load_config_with_base(path)


if __name__ == "__main__":
    unittest.main()
