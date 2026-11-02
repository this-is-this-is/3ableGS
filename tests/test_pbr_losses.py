"""Coordinate, mask and resolution contracts for reconstructed PBR targets."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from torch import nn

from datasets.driving_dataset import DrivingDataset
from models.trainers.scene_graph_pbr import MultiTrainer_pbr, OPENCV_TO_OPENGL


def loss_trainer(crop_frac=0.0):
    # Exercise real loss methods without loading a scene, perceptual weights or
    # HDR cubemap. Rendering/backward/checkpoint integration is tested on GPU.
    trainer = MultiTrainer_pbr.__new__(MultiTrainer_pbr)
    nn.Module.__init__(trainer)
    trainer.crop_frac = crop_frac
    trainer._opencv_to_opengl = OPENCV_TO_OPENGL
    trainer.losses_dict = OmegaConf.create({
        "albedo": {"w": .5}, "metallic": {"w": .2},
        "roughness": {"w": .2}, "normal": {"w": .15},
    })
    return trainer


def targets(height=2, width=2):
    return {
        "sky_masks": torch.zeros(height, width),
        "gbuf_basecolor": torch.zeros(height, width, 3),
        "gbuf_metallic": torch.zeros(height, width, 1),
        "gbuf_roughness": torch.zeros(height, width, 1),
        "gbuf_normal": torch.tensor([0., 0., 1.]).expand(height, width, 3),
    }


def predictions(height=2, width=2):
    return {
        "albedo": torch.ones(height, width, 3, requires_grad=True),
        "metallic": torch.ones(height, width, 1, requires_grad=True),
        "roughness": torch.ones(height, width, 1, requires_grad=True),
        "normal": torch.tensor([1., 0., 0.]).expand(height, width, 3).clone().requires_grad_(),
    }


class PBRLossTests(unittest.TestCase):
    def test_sky_ego_mask_channel_mean_and_dynamic_surface_supervision(self):
        trainer, image, output = loss_trainer(), targets(), predictions()
        image["sky_masks"][0, 0] = 1
        image["egocar_masks"] = torch.tensor([[0., 1.], [0., 0.]])
        image["dynamic_masks"] = torch.ones(2, 2)
        loss = trainer.gbuffer_losses(output, image, {"camera_to_world": torch.eye(4)})
        for name, expected in (("albedo_loss", .5), ("metallic_loss", .2),
                               ("roughness_loss", .2), ("normal_loss", .05)):
            # 90-degree normals: .15 * (1-cos)/(1+(1-cos)/.5) = .05.
            self.assertAlmostEqual(loss[name].item(), expected, places=7)
        sum(loss.values()).backward()
        for key in ("albedo", "metallic", "roughness", "normal"):
            self.assertTrue(torch.equal(output[key].grad[0], torch.zeros_like(output[key].grad[0])))
            self.assertGreater(output[key].grad[1].abs().sum().item(), 0)

    def test_normal_uses_refined_world_camera_rotation_then_opengl_basis(self):
        trainer, image, output = loss_trainer(), targets(), predictions()
        rotation = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        trainer._render_c2w = torch.eye(4)
        trainer._render_c2w[:3, :3] = rotation
        gt = torch.nn.functional.normalize(torch.tensor([1., 2., 3.]), dim=0)
        image["gbuf_normal"] = gt.expand(2, 2, 3)
        world = (gt * OPENCV_TO_OPENGL) @ rotation.T
        output["normal"] = world.expand(2, 2, 3).clone().requires_grad_()
        # Stale original pose is identity; the cached render pose must win.
        loss = trainer.gbuffer_losses(output, image, {"camera_to_world": torch.eye(4)})
        self.assertAlmostEqual(loss["normal_loss"].item(), 0., places=7)
        output["normal"] = -output["normal"]
        loss = trainer.gbuffer_losses(output, image, {"camera_to_world": torch.eye(4)})
        self.assertAlmostEqual(loss["normal_loss"].item(), .06, places=7)

    def test_crop_and_resolution_schedule_match_at_coarser_grid(self):
        trainer, image, output = loss_trainer(1 / 6), targets(4, 4), predictions(6, 4)
        image["sky_masks"] = torch.zeros(6, 4)
        for key, target_key in (("albedo", "gbuf_basecolor"), ("metallic", "gbuf_metallic"),
                                ("roughness", "gbuf_roughness")):
            image[target_key].fill_(1)
            with torch.no_grad():
                output[key][0].fill_(10)
                output[key][-1].fill_(10)
        loss = trainer.gbuffer_losses(output, image, {"camera_to_world": torch.eye(4)})
        self.assertEqual(loss["albedo_loss"].item(), 0)
        self.assertEqual(loss["metallic_loss"].item(), 0)
        self.assertEqual(loss["roughness_loss"].item(), 0)
        # Full target versus quarter-resolution render: no target upsampling.
        trainer.crop_frac = 0
        image, output = targets(8, 12), predictions(2, 3)
        seen = []
        original_resize = trainer._resize_down

        def record_resize(value, target_hw):
            seen.append(target_hw)
            return original_resize(value, target_hw)

        with patch.object(trainer, "_resize_down", side_effect=record_resize):
            trainer.gbuffer_losses(output, image, {"camera_to_world": torch.eye(4)})
        self.assertTrue(seen)
        self.assertEqual(set(seen), {(2, 3)})

    def test_empty_surface_returns_finite_graph_connected_zero(self):
        trainer, image, output = loss_trainer(), targets(), predictions()
        image["sky_masks"].fill_(1)
        loss = trainer.gbuffer_losses(output, image, {"camera_to_world": torch.eye(4)})
        total = sum(loss.values())
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(total.item(), 0)
        total.backward()
        for value in output.values():
            self.assertTrue(torch.equal(value.grad, torch.zeros_like(value.grad)))

    def test_lidar_depth_and_color_sample_use_same_intrinsics_as_image_rays(self):
        class Camera:
            def __len__(self):
                return 1

            def load_depth(self, value):
                self.depth = value

        camera = Camera()
        camera.WIDTH, camera.HEIGHT, camera.cam_name = 10, 8, "tiny_front"
        camera.undistort = True
        camera.intrinsics = torch.tensor([[[10., 0., 5.], [0., 11., 4.], [0., 0., 1.]]])
        camera.distortions = torch.tensor([[-.3, .1, 0., 0., 0.]])
        camera.cam_to_worlds = torch.eye(4)[None]
        camera.images = {"raw": torch.arange(8 * 10 * 3).reshape(1, 8, 10, 3).float() / 240}
        point = torch.tensor([[1.1, -.5, 5.]])
        lidar = SimpleNamespace(
            num_points=1, visible_masks=torch.zeros(1, dtype=torch.bool),
            colors=torch.zeros(1, 3), find_closest_timestep=lambda _: 0,
            get_lidar_rays=lambda _: {"lidar_origins": point, "lidar_viewdirs": torch.zeros_like(point),
                                     "lidar_ranges": torch.ones(1, 1), "lidar_mask": torch.tensor([True])},
        )
        dataset = SimpleNamespace(
            data_cfg=OmegaConf.create({"pixel_source": {}}),
            pixel_source=SimpleNamespace(camera_data={0: camera}, normalized_time=torch.tensor([0.])),
            lidar_source=lidar, device=torch.device("cpu"),
        )
        with patch("datasets.driving_dataset.tqdm", side_effect=lambda values, **_: values):
            DrivingDataset.project_lidar_pts_on_images(dataset, delete_out_of_view_points=False)
        # Original camera K projects the ray to (u=7.2, v=2.9), sampled at (7,2).
        self.assertEqual(camera.depth[0, 2, 7].item(), 5.)
        self.assertEqual(torch.count_nonzero(camera.depth).item(), 1)
        torch.testing.assert_close(lidar.colors[0], camera.images["raw"][0, 2, 7])
        self.assertTrue(lidar.visible_masks[0])


if __name__ == "__main__":
    unittest.main()
