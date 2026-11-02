"""Exercise the real trainer hook with a fixed global visibility accumulator."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from omegaconf import OmegaConf

from models.trainers.scene_graph_pbr import MultiTrainer_pbr
from tests.test_temporal_material_integration import SyntheticTrainer


class LayeredTrainer(SyntheticTrainer):
    def __init__(self):
        super().__init__(size=5)
        self.gaussian_classes = {"Background": 0, "RigidNodes": 1}
        self.models["RigidNodes"] = nn.Identity()
        self.models["RigidNodes"].point_ids = torch.tensor([[0], [1]])
        self.models["RigidNodes"].filter_mask = torch.tensor([True, True])
        self.pts_labels = torch.tensor([0, 1, 1])
        self.render_cfg = OmegaConf.create(dict(
            near_plane=.01, far_plane=100., batch_size=1, packed=True,
            absgrad=False, sparse_grad=False, antialiased=False))
        self.features = nn.Parameter(torch.tensor([
            [.2, .3, .4, .0, .8, 0., 0., -1.],
            [.8, .2, .1, .5, .2, .6, 0., -.8],
            [.1, .7, .4, .1, .6, 0., .6, -.8],
        ]))
        self.weights = torch.tensor([.6, .2, .1])
        self.passes = []

    def collect_gaussians(self, *args):
        return SimpleNamespace(
            rgbs=self.features, means=torch.zeros(3, 3),
            quats=torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1),
            scales=torch.ones(3, 3), opacities=torch.tensor([[.8], [.7], [.5]]))

    render_gaussians = MultiTrainer_pbr.render_gaussians

    def accumulate(self, **kwargs):
        self.passes.append(kwargs)
        colors = kwargs["colors"]
        image = (self.weights[:, None] * colors).sum(0).expand(1, 5, 5, -1)
        if kwargs["render_mode"] == "RGB+ED":
            image = torch.cat((image, torch.full((1, 5, 5, 1), 4.)), -1)
        alpha = self.weights.sum().expand(1, 5, 5, 1)
        return image, alpha, {}


class LayeredIntegrationTests(unittest.TestCase):
    def render(self, trainer, config=None):
        trainer.configure_layered_gbuffer(config)
        images, cameras = trainer.inputs(0)
        with patch("models.trainers.scene_graph_pbr.rasterization", trainer.accumulate):
            return trainer(images, cameras)

    def test_single_group_exactly_preserves_existing_forward(self):
        trainer = LayeredTrainer()
        baseline = self.render(trainer)
        result = self.render(trainer, {"grouping": "single"})
        torch.testing.assert_close(result["rgb"], baseline["rgb"])
        torch.testing.assert_close(result["opacity"], baseline["opacity"])
        torch.testing.assert_close(result["normal"], baseline["normal"])
        self.assertLess(result["layered_diagnostics"]["mass_reconstruction_max_abs"], 1e-6)

    def test_all_group_passes_keep_opacity_and_geometry_and_merge_losses(self):
        trainer = LayeredTrainer()
        baseline = self.render(trainer)
        result = self.render(trainer, {"grouping": "instance", "color_space": "linear"})
        first = trainer.passes[0]
        for call in trainer.passes:
            for key in ("opacities", "means", "scales", "quats", "viewmats", "Ks"):
                torch.testing.assert_close(call[key], first[key], rtol=0, atol=0)
        for key in ("albedo", "metallic", "roughness", "normal", "depth", "opacity"):
            torch.testing.assert_close(result[key], baseline[key], rtol=0, atol=0)
        self.assertEqual(result["layered_diagnostics"]["group_count"], 3)
        self.assertLess(result["layered_diagnostics"]["feature_reconstruction_max_abs"], 1e-6)
        self.assertGreater(float((result["rgb"] - baseline["rgb"]).detach().abs().max()), 1e-4)
        # Group rasterization is recomputed during checkpointed backward.
        with patch("models.trainers.scene_graph_pbr.rasterization", trainer.accumulate):
            result["rgb"].sum().backward()
        self.assertTrue(torch.isfinite(trainer.features.grad).all())
        self.assertGreater(float(trainer.features.grad.abs().sum()), 0.)

    def test_instance_namespaces_filter_alignment_and_fail_closed(self):
        trainer = LayeredTrainer()
        trainer.gaussian_classes["SMPLNodes"] = 2
        node = nn.Identity()
        node.point_ids = torch.tensor([[0], [1], [1]])
        node.filter_mask = torch.tensor([True, False, True])
        trainer.models["SMPLNodes"] = node
        trainer.pts_labels = torch.tensor([0, 1, 1, 2, 2])
        self.assertEqual(trainer._layered_group_ids("instance").tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(trainer._layered_group_ids("dynamic").tolist(), [0, 1, 1, 1, 1])
        node.filter_mask = torch.tensor([True, False, False])
        with self.assertRaisesRegex(ValueError, "align"):
            trainer._layered_group_ids("instance")

    def test_absent_node_reserves_instance_namespace(self):
        trainer = LayeredTrainer()
        trainer.gaussian_classes["SMPLNodes"] = 2
        node = nn.Identity()
        node.point_ids = torch.tensor([[0]])
        node.filter_mask = torch.tensor([True])
        trainer.models["SMPLNodes"] = node
        trainer.pts_labels = torch.tensor([0, 2])
        self.assertEqual(trainer._layered_group_ids("instance").tolist(), [0, 3])

    def test_forbids_history_weather_and_unknown_configuration(self):
        trainer = LayeredTrainer()
        with self.assertRaises(ValueError):
            trainer.configure_layered_gbuffer({"opacity_mask": True})
        trainer.configure_layered_gbuffer({})
        trainer.configure_temporal_material({})
        images, cameras = trainer.inputs(0)
        with self.assertRaisesRegex(ValueError, "without history"):
            trainer(images, cameras)


if __name__ == "__main__":
    unittest.main()
