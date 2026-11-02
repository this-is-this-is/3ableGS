"""Teacher PNG frame mapping, calibration and native-resolution contracts."""
import unittest

import cv2
import numpy as np
from PIL import Image
import torch

import test_raw_camera_loading as raw_fixture


class GbufferLoadingTests(unittest.TestCase):
    setUp = raw_fixture.CameraLoadingTests.setUp
    write_images = raw_fixture.CameraLoadingTests.write_images
    camera = raw_fixture.CameraLoadingTests.camera

    def targets(self):
        directory = self.scene / "gbuffers" / "raw"
        directory.mkdir(parents=True)
        y, x = np.mgrid[:32, :48]
        rgb = np.stack((x * 5, y * 7, x + y), -1).astype(np.uint8)
        for frame in range(2):
            for kind in ("basecolor", "metallic", "roughness", "normal"):
                value = rgb if kind == "basecolor" else np.full_like(rgb, 100 + frame)
                if kind == "normal":
                    value = np.tile(np.array([128, 128, 255], dtype=np.uint8), (32, 48, 1))
                Image.fromarray(value).save(directory / f"0000.{frame:04d}.{kind}.png")
        return directory.parent, rgb

    def test_frame_channel_and_native_resolution(self):
        root, _ = self.targets()
        camera = self.camera(load_gbuffer=True, gbuffer_path=str(root), gbuffer_size=[32, 48])
        camera.set_downscale_factor(0.5)
        infos, _ = camera.get_image(1)
        self.assertEqual(infos["pixels"].shape, (16, 24, 3))
        self.assertEqual(infos["gbuf_basecolor"].shape, (32, 48, 3))
        self.assertEqual(infos["gbuf_metallic"].shape, (32, 48, 1))
        self.assertEqual(camera.gbuffers["raw"]["normal"].dtype, torch.uint8)
        torch.testing.assert_close(infos["gbuf_roughness"], torch.full((32, 48, 1), 101 / 255))
        torch.testing.assert_close(infos["gbuf_normal"].norm(dim=-1), torch.ones(32, 48))

    def test_undistortion_uses_matching_camera_matrix(self):
        root, source = self.targets()
        distortion = [0.16, -0.03, 0.01, -0.01, 0.0]
        np.savetxt(self.scene / "intrinsics/0.txt", [1000, 1000, 960, 640, *distortion])
        camera = self.camera(load_gbuffer=True, gbuffer_path=str(root),
                             gbuffer_size=[32, 48], undistort=True)
        expected = cv2.undistort(source, camera.intrinsics[0].numpy(), np.array(distortion))
        np.testing.assert_allclose(camera.gbuffer_intrinsics()[0], camera.intrinsics[0].numpy())
        np.testing.assert_array_equal(camera.gbuffers["raw"]["basecolor"][0].numpy(), expected)

    def test_mismatched_target_size_and_missing_frame_fail(self):
        root, _ = self.targets()
        with self.assertRaisesRegex(ValueError, "G-buffer .*size"):
            self.camera(load_gbuffer=True, gbuffer_path=str(root), gbuffer_size=[64, 96])
        (root / "raw/0000.0001.normal.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.camera(load_gbuffer=True, gbuffer_path=str(root), gbuffer_size=[32, 48])


if __name__ == "__main__":
    unittest.main()
