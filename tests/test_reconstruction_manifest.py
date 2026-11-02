"""Input-audit regression tests; do not import or initialize the training stack."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.build_reconstruction_manifest import build_manifest, frame_split


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "dataset"
        self.scene = self.data / "094"
        for camera in range(5):
            self._matrix(f"extrinsics/{camera}.txt")
        path = self.scene / "intrinsics/0.txt"
        path.parent.mkdir(parents=True)
        np.savetxt(path, [1000, 1000, 960, 640, 0, 0, 0, 0, 0])
        rgb = self.root / "template.jpg"
        mask = self.root / "template.png"
        Image.new("RGB", (1920, 1280)).save(rgb)
        Image.new("L", (1920, 1280)).save(mask)
        for frame in range(9):
            self._matrix(f"ego_pose/{frame:03d}.txt")
            lidar = self.scene / f"lidar/{frame:03d}.bin"
            lidar.parent.mkdir(parents=True, exist_ok=True)
            lidar.write_bytes(bytes(14 * 4))
            for weather in ("raw", "rain", "snow"):
                self._link(rgb, f"images/{weather}/{frame:03d}_0.jpg")
            for directory in ("sky_masks", "dynamic_masks/all", "dynamic_masks/human", "dynamic_masks/vehicle"):
                self._link(mask, f"{directory}/{frame:03d}_0.png")
        for relative, content in (
            ("instances/instances_info.json", "{}"),
            ("instances/frame_instances.json", "{" + ",".join(f'"{f}":[]' for f in range(9)) + "}"),
            ("humanpose/smpl.pkl", "not deserialized by this audit"),
        ):
            path = self.scene / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        path = self.root / "smpl_models/SMPL_NEUTRAL.pkl"
        path.parent.mkdir()
        path.write_text("not deserialized by this audit")

    def tearDown(self):
        self.temporary.cleanup()

    def _link(self, source, relative):
        path = self.scene / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.hardlink_to(source)

    def _matrix(self, relative):
        path = self.scene / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, np.eye(4))

    def _manifest(self, scenes=(94,)):
        return build_manifest(self.root, Path("dataset"), scenes, end=8)

    def test_historical_split_and_shifted_start(self):
        split = frame_split()
        self.assertEqual(len(split["train"]), 88)
        self.assertEqual(split["test"], list(range(8, 100, 8)))
        self.assertIn(0, split["train"])
        self.assertEqual(frame_split(10, 26)["test"], [18, 26])
        self.assertEqual(frame_split(0, 8, 0)["test"], [])

    def test_complete_inputs_optional_ego_and_planned_gbuffer(self):
        manifest = self._manifest()
        self.assertEqual(manifest["status"], "ready_for_phase_a")
        scene = manifest["scenes"][0]
        self.assertFalse(scene["ego_mask"]["exists"])
        self.assertFalse(scene["planned_gbuffer"]["required_in_phase_a"])
        self.assertEqual(scene["frames"][8]["split"], "test")
        self.assertEqual(scene["frames"][0]["images"]["rain"]["path"], "dataset/094/images/rain/000_0.jpg")

    def test_missing_weather_frame_is_not_silently_shortened(self):
        path = self.scene / "images/snow/008_0.jpg"
        path.unlink()
        manifest = self._manifest()
        self.assertEqual(manifest["status"], "required_inputs_invalid")
        self.assertEqual(manifest["scenes"][0]["counts"]["frames"], 9)
        self.assertIn({"path": "dataset/094/images/snow/008_0.jpg", "reason": "missing_file"}, manifest["scenes"][0]["required_errors"])

    def test_missing_other_camera_extrinsic_blocks_smpl(self):
        (self.scene / "extrinsics/4.txt").unlink()
        errors = self._manifest()["scenes"][0]["required_errors"]
        self.assertTrue(any(e["path"].endswith("extrinsics/4.txt") for e in errors))

    def test_unavailable_scene_is_not_replaced(self):
        manifest = self._manifest((113,))
        self.assertEqual(manifest["status"], "required_inputs_invalid")
        self.assertEqual(manifest["scenes"][0]["scene_name"], "113")
        self.assertEqual(manifest["scenes"][0]["scene_root"], "dataset/113")

    def test_mismatched_dimensions_are_reported(self):
        path = self.scene / "images/rain/002_0.jpg"
        path.unlink()
        Image.new("RGB", (960, 640)).save(path)
        errors = self._manifest()["scenes"][0]["required_errors"]
        self.assertTrue(any(e["path"].endswith("rain/002_0.jpg") and "size_mismatch" in e["reason"] for e in errors))

    def test_external_data_root_paths_remain_relocatable(self):
        manifest = build_manifest(self.root / "repo", self.data, (94,), end=8)
        self.assertEqual(manifest["data_root"], "../dataset")
        self.assertEqual(manifest["scenes"][0]["frames"][0]["lidar"]["path"], "../dataset/094/lidar/000.bin")


if __name__ == "__main__":
    unittest.main()
