"""CPU regression checks for raw-only and existing paired-weather loading."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from datasets.waymo.waymo_sourceloader import WaymoCameraData


class CameraLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.scene = Path(self.temp_dir.name)
        for name in ("intrinsics", "extrinsics", "ego_pose"):
            (self.scene / name).mkdir()
        np.savetxt(self.scene / "intrinsics/0.txt", [1000, 1000, 960, 640, 0, 0, 0, 0, 0])
        np.savetxt(self.scene / "extrinsics/0.txt", np.eye(4))
        for frame in range(2):
            np.savetxt(self.scene / f"ego_pose/{frame:03d}.txt", np.eye(4))
        self.write_images("raw", 64)

    def write_images(self, weather, value):
        directory = self.scene / "images" / weather
        directory.mkdir(parents=True)
        for frame in range(2):
            Image.fromarray(np.full((32, 48, 3), value, dtype=np.uint8)).save(
                directory / f"{frame:03d}_0.jpg"
            )

    def camera(self, **kwargs):
        camera = WaymoCameraData(
            dataset_name="waymo",
            data_path=str(self.scene),
            cam_id=0,
            start_timestep=0,
            end_timestep=2,
            downscale_when_loading=40,
            device=torch.device("cpu"),
            **kwargs,
        )
        camera.load_time(torch.tensor([0.0, 1.0]))
        camera.set_unique_ids(0, torch.arange(2))
        return camera

    def test_raw_only_without_weather_files_or_optional_masks(self):
        camera = self.camera(load_weather=[])
        self.assertEqual(list(camera.images), ["raw"])
        self.assertEqual(camera.images["raw"].shape, (2, 32, 48, 3))
        image, calibration = camera.get_image(1)
        self.assertTrue(torch.isfinite(image["pixels"]).all())
        self.assertTrue(torch.allclose(image["pixels"], torch.full((32, 48, 3), 64 / 255)))
        self.assertTrue(torch.allclose(image["viewdirs"].norm(dim=-1), torch.ones(32, 48)))
        self.assertEqual(calibration["camera_to_world"].shape, (4, 4))
        for key in ("dynamic_masks", "human_masks", "vehicle_masks", "sky_masks"):
            self.assertNotIn(key, image)

    def test_existing_rain_snow_and_enabled_masks(self):
        self.write_images("rain", 128)
        self.write_images("snow", 192)
        for name in ("dynamic_masks/all", "dynamic_masks/human", "dynamic_masks/vehicle", "sky_masks"):
            directory = self.scene / name
            directory.mkdir(parents=True)
            for frame in range(2):
                Image.fromarray(np.full((32, 48), 255, dtype=np.uint8)).save(
                    directory / f"{frame:03d}_0.png"
                )
        camera = self.camera(load_weather=["rain", "snow"], load_dynamic_mask=True, load_sky_mask=True)
        self.assertEqual(list(camera.images), ["raw", "rain", "snow"])
        for weather, value in (("raw", 64), ("rain", 128), ("snow", 192)):
            image, _ = camera.get_image(0, weather)
            self.assertTrue(torch.allclose(image["pixels"], torch.full((32, 48, 3), value / 255)))
            for key in ("dynamic_masks", "human_masks", "vehicle_masks", "sky_masks"):
                self.assertTrue(torch.equal(image[key], torch.ones(32, 48)))

    def test_snow_only_does_not_require_rain(self):
        self.write_images("snow", 192)
        camera = self.camera(load_weather=["snow"])
        self.assertEqual(list(camera.images), ["raw", "snow"])
        image, _ = camera.get_image(0, "snow")
        self.assertTrue(torch.allclose(image["pixels"], torch.full((32, 48, 3), 192 / 255)))


if __name__ == "__main__":
    unittest.main()
