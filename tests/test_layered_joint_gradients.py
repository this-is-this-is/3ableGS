"""Full-training projection gradients with shading-checkpointed gsplat AbsGrad.

The small custom raster reproduces gsplat's important side effect: backward
sets .absgrad on the projected tensor unpacked from saved_tensors. This tests
the original/recomputed-tensor distinction on CPU without a scene or teacher.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.layered_gbuffer import render_layered_gbuffer
from models.trainers.scene_graph_pbr import MultiTrainer_pbr, _LayeredScreenGradients
from tests.test_layered_cuda import analytic_shader
from tests.test_layered_integration import LayeredTrainer


class SavedTensorRaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, colors):
        ctx.save_for_backward(means, colors)
        weights = means[0, :, :1].sigmoid() * .2
        return (weights * colors).sum(0)[None, None]

    @staticmethod
    def backward(ctx, output_gradient):
        means, colors = ctx.saved_tensors
        sigmoid = means[0, :, :1].sigmoid()
        weights = sigmoid * .2
        gradient = torch.zeros_like(means)
        gradient[0, :, :1] = (
            (colors * output_gradient[0, 0]).sum(-1, keepdim=True)
            * sigmoid * (1 - sigmoid) * .2
        )
        # A distinct statistic proves the hook reads the real attribute,
        # rather than substituting abs(gradient).
        means.absgrad = 2 * gradient.abs()
        return gradient, weights * output_gradient[0, 0]


class LayeredJointGradientTests(unittest.TestCase):
    def render(self, checkpointed, absgrad, space="srgb", cap=0):
        coordinates = torch.tensor([[.2, .3], [-.4, .2], [.5, -.1]],
                                   dtype=torch.double, requires_grad=True)
        features = torch.tensor([
            [.2, .3, .5, .0, .8, 0., 0., 1.],
            [.9, .1, .2, .8, .1, .6, 0., .8],
            [.3, .7, .2, .2, .5, 0., .8, .6],
        ], dtype=torch.double, requires_grad=True)
        primary = coordinates[None] * 1
        primary.retain_grad()
        info = {"means2d": primary}
        global_render = SavedTensorRaster.apply(
            primary, torch.cat((features, features.new_ones(3, 1)), -1))
        collector = _LayeredScreenGradients(info, absgrad)
        pass_count = 0

        def raster(colors):
            nonlocal pass_count
            pass_count += 1
            means = coordinates[None] * 1
            result = SavedTensorRaster.apply(means, colors)
            collector.track({"means2d": means})
            return result

        def eager(function, *args, **kwargs):
            return function(*args)

        context = nullcontext() if checkpointed else patch(
            "models.layered_gbuffer.checkpoint", side_effect=eager)
        with context:
            output = render_layered_gbuffer(
                features, torch.arange(3), raster, analytic_shader,
                color_space=space, max_layers=cap,
                merged_feature_srgb=global_render[..., :8],
                merged_opacity=global_render[..., 8:9],
                checkpoint_raster=False,
            )
            forward_passes = pass_count
            # RGB alone must drive the auxiliary projections; there is no
            # material/depth/normal teacher loss to hide missing gradients.
            loss = output["linear_surface"].square().sum()
            loss.backward()
        collector.finalize(info)
        return (output["linear_surface"].detach(), coordinates.grad, features.grad,
                getattr(primary, "absgrad" if absgrad else "grad"),
                info["layered_gradient_stats"], forward_passes, pass_count)

    def test_checkpoint_preserves_geometry_material_and_densification_gradients(self):
        for absgrad in (False, True):
            for space in ("srgb", "linear"):
                for cap in (0, 2):
                    with self.subTest(absgrad=absgrad, space=space, cap=cap):
                        eager = self.render(False, absgrad, space, cap)
                        actual = self.render(True, absgrad, space, cap)
                        for expected, result in zip(eager[:4], actual[:4]):
                            torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-10)
                            self.assertTrue(torch.isfinite(result).all())
                        self.assertGreater(float(actual[1].abs().sum()), 0)
                        self.assertGreater(float(actual[2].abs().sum()), 0)
                        self.assertGreater(actual[4]["auxiliary_grad_norm"], 0)
                        self.assertEqual(actual[4]["auxiliary_backward_passes"], 3)
                        self.assertEqual(actual[6], actual[5])
                        self.assertEqual(eager[5], eager[6])

    def test_rgb_only_projection_sum_matches_geometry_leaf_gradient(self):
        actual = self.render(True, False)
        torch.testing.assert_close(actual[3][0], actual[1], atol=1e-12, rtol=1e-10)
        # Global attenuation is exactly one for these opaque-enough pixels.
        self.assertLess(actual[4]["primary_grad_norm"], 1e-16)
        self.assertGreater(actual[4]["combined_grad_norm"], 0)

    def test_missing_primary_gradient_is_valid_and_packed_is_rejected(self):
        primary = torch.zeros(1, 3, 2, requires_grad=True)
        info = {"means2d": primary}
        collector = _LayeredScreenGradients(info, False)
        collector.total.fill_(.5)
        collector.finalize(info)
        torch.testing.assert_close(primary.grad, torch.full_like(primary, .5))
        with self.assertRaisesRegex(RuntimeError, "already finalized"):
            collector.finalize(info)
        with self.assertRaisesRegex(ValueError, "packed=false"):
            _LayeredScreenGradients({"means2d": torch.zeros(3, 2)}, False)

    def test_constructor_consumes_saved_renderer_config(self):
        def minimal_parent(trainer, **kwargs):
            self.assertNotIn("layered_gbuffer", kwargs)
            nn.Module.__init__(trainer)
            trainer.dataset = SimpleNamespace(pixel_source=SimpleNamespace())
            trainer.device = "cpu"
            trainer.losses_dict = {}

        with patch("models.trainers.scene_graph_pbr.MultiTrainer_weather.__init__",
                   minimal_parent):
            trainer = MultiTrainer_pbr(layered_gbuffer={"grouping": "dynamic"})
        self.assertEqual(trainer._layered_gbuffer["grouping"], "dynamic")

    def test_trainer_postprocess_uses_all_projection_gradients(self):
        trainer = LayeredTrainer()
        primary = torch.zeros(1, 3, 2, requires_grad=True)
        primary.grad = torch.ones_like(primary)
        trainer.info = {"means2d": primary}
        collector = _LayeredScreenGradients(trainer.info, False)
        collector.total.fill_(2)
        collector.backward_passes = 3
        trainer._layered_screen_gradients = collector

        def check_parent(step):
            self.assertEqual(step, 7)
            torch.testing.assert_close(trainer.info["means2d"].grad,
                                       torch.full_like(primary, 3))

        with patch("models.trainers.scene_graph_pbr.MultiTrainer_weather.postprocess_per_train_step",
                   side_effect=check_parent):
            trainer.postprocess_per_train_step(7)
        self.assertIsNone(trainer._layered_screen_gradients)
        self.assertEqual(trainer.info["layered_gradient_stats"]["auxiliary_backward_passes"], 3)


if __name__ == "__main__":
    unittest.main()
