"""Heldout RGB must not enter Gaussian point-color/visibility initialization."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from datasets.driving_dataset import DrivingDataset


class TemporalInitializationTests(unittest.TestCase):
    @staticmethod
    def fixture(heldout_color, ranges):
        class Camera:
            WIDTH = 5
            HEIGHT = 5
            cam_name = "tiny_front"

            def __len__(self):
                return 3

            def load_depth(self, depth):
                self.depth = depth

        camera = Camera()
        camera.intrinsics = torch.tensor(
            [[2., 0., 2.], [0., 2., 2.], [0., 0., 1.]],
        ).expand(3, 3, 3).clone()
        camera.cam_to_worlds = torch.eye(4).expand(3, 4, 4).clone()
        camera.images = {
            "raw": torch.stack([
                torch.full((5, 5, 3), .1),
                torch.full((5, 5, 3), heldout_color),
                torch.full((5, 5, 3), .3),
            ])
        }
        # Each lidar timestamp contributes a separate center-facing point.
        # The heldout point is otherwise valid and visible in its own frame.
        points = torch.tensor([[0., 0., 2.], [0., 0., 3.], [0., 0., 4.]])

        def rays(frame):
            return {
                "lidar_origins": points[frame:frame + 1],
                "lidar_viewdirs": torch.zeros(1, 3),
                "lidar_ranges": torch.ones(1, 1),
                "lidar_mask": torch.arange(3) == frame,
            }

        lidar = SimpleNamespace(
            num_points=3, colors=torch.full((3, 3), -1.),
            visible_masks=torch.zeros(3, dtype=torch.bool),
            find_closest_timestep=lambda normalized: int(round(float(normalized) * 2)),
            get_lidar_rays=rays,
        )
        dataset = SimpleNamespace(
            data_cfg=OmegaConf.create({"pixel_source": {"test_frame_ranges": ranges}}),
            num_img_timesteps=3,
            pixel_source=SimpleNamespace(
                camera_data={0: camera}, normalized_time=torch.tensor([0., .5, 1.]),
            ),
            lidar_source=lidar, device=torch.device("cpu"),
        )
        return dataset, camera, lidar

    @staticmethod
    def project(dataset):
        with patch("datasets.driving_dataset.tqdm", side_effect=lambda values, **_: values):
            DrivingDataset.project_lidar_pts_on_images(dataset, delete_out_of_view_points=False)

    def test_holdout_color_has_no_effect_but_heldout_depth_is_available(self):
        first, camera, lidar = self.fixture(.25, [[1, 2]])
        changed, changed_camera, changed_lidar = self.fixture(.95, [[1, 2]])
        self.project(first)
        self.project(changed)
        self.assertEqual(lidar.visible_masks.tolist(), [True, False, True])
        torch.testing.assert_close(
            lidar.colors, torch.tensor([[.1, .1, .1], [-1., -1., -1.], [.3, .3, .3]]),
        )
        torch.testing.assert_close(changed_lidar.colors, lidar.colors)
        torch.testing.assert_close(changed_lidar.visible_masks, lidar.visible_masks)
        # Evaluation depth still contains all three timestamps, including holdout.
        self.assertEqual(tuple(camera.depth.shape), (3, 5, 5))
        torch.testing.assert_close(camera.depth[:, 1, 1], torch.tensor([2., 3., 4.]))
        self.assertEqual(torch.count_nonzero(camera.depth).item(), 3)
        torch.testing.assert_close(changed_camera.depth, camera.depth)

    def test_no_explicit_holdout_keeps_color_initialization_for_every_frame(self):
        dataset, camera, lidar = self.fixture(.75, None)
        self.project(dataset)
        self.assertTrue(lidar.visible_masks.all())
        torch.testing.assert_close(lidar.colors[1], torch.full((3,), .75))


if __name__ == "__main__":
    unittest.main()
