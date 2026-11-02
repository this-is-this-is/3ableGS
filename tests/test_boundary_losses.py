"""Historical Base v4 crop sampling, masks, schedule and prediction gradients."""

from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.boundary_losses import (
    boundary_rgb_losses, box_mean, sobel, validate_boundary_config,
)


V4_BOUNDARY = {
    "enabled": True, "start_step": 5000, "ramp_steps": 1000,
    "rgb_w": 0., "gradient_w": 0., "perceptual_w": .02,
    "edge_threshold": .05, "radius": 2, "patch_size": 128,
    "num_patches": 4, "min_valid_fraction": .99,
}


class CapturePerceptual(nn.Module):
    """Record the network input contract, without metric state or downloads."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)
        self.calls = []

    def forward(self, prediction, target, normalize=False):
        self.calls.append((prediction, target, normalize, self.training))
        return self.scale * (prediction - target).square().mean((1, 2, 3))


class BoundaryLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_v4_ramp_starts_at_5000_and_reaches_full_weight_at_6000(self):
        prediction = torch.full((40, 48, 3), .8, requires_grad=True)
        target = torch.full_like(prediction, .2)
        valid = torch.ones(40, 48)
        network = CapturePerceptual()
        self.assertEqual(boundary_rgb_losses(prediction, target, valid, 4999,
                                              V4_BOUNDARY, network), {})
        self.assertEqual(network.calls, [])
        for step, ramp in ((5000, 0.), (5500, .5), (6000, 1.), (30000, 1.)):
            loss = boundary_rgb_losses(prediction, target, valid, step,
                                       V4_BOUNDARY, network)
            self.assertEqual(set(loss), {"boundary_perceptual_loss"})
            value = loss["boundary_perceptual_loss"]
            self.assertAlmostEqual(value.item(), .02 * ramp * .6 ** 2, places=7)
            self.assertTrue(value.requires_grad)
        self.assertTrue(all(call[2] and not call[3] for call in network.calls))

    def test_disabled_baseline_skips_all_network_and_edge_work(self):
        with patch("models.boundary_losses.sobel", side_effect=AssertionError("unused")):
            self.assertEqual(boundary_rgb_losses(None, None, None, 30000, {}), {})

    def test_no_eligible_or_small_crop_returns_graph_connected_zero(self):
        for height, width, valid_value in ((40, 48, 0.), (31, 48, 1.)):
            prediction = torch.rand(height, width, 3, requires_grad=True)
            target = torch.rand_like(prediction)
            network = CapturePerceptual()
            loss = boundary_rgb_losses(prediction, target,
                                       torch.full((height, width), valid_value),
                                       6000, V4_BOUNDARY, network)["boundary_perceptual_loss"]
            self.assertEqual(loss.item(), 0.)
            self.assertTrue(loss.requires_grad)
            loss.backward()
            self.assertEqual(prediction.grad.abs().sum().item(), 0.)
            self.assertEqual(network.calls, [])

    def test_invalid_pixels_are_target_filled_and_receive_no_gradient(self):
        target = torch.full((64, 64, 3), .2, requires_grad=True)
        prediction = torch.full((64, 64, 3), .8, requires_grad=True)
        valid = torch.ones(64, 64)
        valid[10:14, 20:24] = 0.  # 0.39% invalid: the sole crop remains eligible.
        network = CapturePerceptual()
        loss = boundary_rgb_losses(prediction, target, valid, 6000,
                                   V4_BOUNDARY, network)["boundary_perceptual_loss"]
        prediction_crops, target_crops, normalize, training = network.calls[0]
        self.assertEqual(prediction_crops.shape, (4, 3, 64, 64))
        torch.testing.assert_close(prediction_crops[:, :, 10:14, 20:24],
                                   target_crops[:, :, 10:14, 20:24], rtol=0, atol=0)
        self.assertTrue(normalize)
        self.assertFalse(training)
        loss.backward()
        self.assertEqual(prediction.grad[valid == 0].abs().sum().item(), 0.)
        self.assertGreater(prediction.grad[valid == 1].abs().sum().item(), 0.)
        self.assertIsNone(target.grad)
        self.assertIsNone(network.scale.grad)

    def test_ego_edges_do_not_contribute_to_sampling_or_gradient_losses(self):
        target = torch.zeros(64, 64, 3)
        target[16:48, 16:48] = 1.
        valid = torch.ones(64, 64)
        valid[16:48, 16:48] = 0.
        prediction = torch.full_like(target, .5, requires_grad=True)
        config = {**V4_BOUNDARY, "rgb_w": .1, "gradient_w": .1,
                  "min_valid_fraction": .7}
        with patch("models.boundary_losses.torch.multinomial", return_value=torch.zeros(4, dtype=torch.long)) as sample:
            losses = boundary_rgb_losses(prediction, target, valid, 6000,
                                         config, CapturePerceptual())
        # Both sides of the invalid-pixel Sobel stencil are excluded before
        # edge dilation, so an ego-only discontinuity leaves just the floor.
        torch.testing.assert_close(sample.call_args.args[0], torch.tensor([.05]), rtol=0, atol=0)
        self.assertEqual(losses["boundary_rgb_loss"].item(), 0.)
        self.assertEqual(losses["boundary_gradient_loss"].item(), 0.)

    def test_sampling_has_historical_edge_density_floor_and_global_rng(self):
        height, width, crop = 40, 80, 32
        target = torch.zeros(height, width, 3)
        target[:, 36:, 0] = 1.
        prediction = torch.full_like(target, .4, requires_grad=True)
        valid = torch.ones(height, width)
        valid[:, :4] = 0.
        config = {**V4_BOUNDARY, "patch_size": crop, "num_patches": 7}
        # A unit vertical RGB step has Sobel magnitude .5 in columns 35/36;
        # radius-two dilation yields exactly columns 33..38.
        edge = torch.zeros(height, width)
        edge[:, 33:39] = 1.
        expected_edge = edge.unfold(0, crop, 1).unfold(1, crop, 1).mean((-1, -2))
        expected_valid = valid.unfold(0, crop, 1).unfold(1, crop, 1).mean((-1, -2))
        expected_scores = (expected_edge + .05) * (expected_valid >= .99)
        selected = []
        original_sample = torch.multinomial

        def record_sample(scores, number, replacement):
            torch.testing.assert_close(scores, expected_scores.flatten(), rtol=0, atol=0)
            self.assertEqual(number, 7)
            self.assertTrue(replacement)
            choice = original_sample(scores, number, replacement=replacement)
            selected.append(choice)
            self.assertTrue((scores[choice] > 0).all())
            return choice

        with torch.random.fork_rng(devices=[]):
            for _ in range(2):
                torch.manual_seed(234)
                with patch("models.boundary_losses.torch.multinomial", side_effect=record_sample):
                    boundary_rgb_losses(prediction, target, valid, 6000,
                                        config, CapturePerceptual())
        self.assertTrue(torch.equal(selected[0], selected[1]))

    def test_sobel_preserves_color_boundaries_with_equal_channel_mean(self):
        image = torch.zeros(8, 12, 3)
        image[:, :6, 0] = 1.
        image[:, 6:, 1] = 1.
        gradient = sobel(image)
        self.assertEqual(gradient.shape, (8, 12, 6))
        self.assertTrue((gradient[:, 5:7, 0] == -.5).all())
        self.assertTrue((gradient[:, 5:7, 2] == .5).all())
        self.assertEqual(gradient[..., 4:].abs().sum().item(), 0.)
        # The historical edge builder suppresses convolution roundoff < 1e-6.
        self.assertLess(sobel(image.mean(-1, keepdim=True)).abs().max().item(), 1e-6)

    def test_integral_box_mean_matches_direct_crop_reduction(self):
        image = torch.arange(7 * 9, dtype=torch.float64).reshape(7, 9) / 10
        direct = torch.stack([
            torch.stack([image[y:y + 3, x:x + 4].mean() for x in range(6)])
            for y in range(5)
        ])
        torch.testing.assert_close(box_mean(image, 3, 4), direct, rtol=1e-14, atol=1e-14)

    def test_configuration_errors_fail_early(self):
        for overrides in ({"unknown_option": 1}, {"enabled": 1}, {"patch_size": 31},
                          {"num_patches": 0}, {"min_valid_fraction": 1.1},
                          {"edge_threshold": 0}, {"perceptual_w": float("nan")},
                          {"radius": 1.5}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_boundary_config({**V4_BOUNDARY, **overrides})
        with self.assertRaisesRegex(ValueError, "frozen LPIPS"):
            boundary_rgb_losses(torch.zeros(32, 32, 3), torch.zeros(32, 32, 3),
                                torch.ones(32, 32), 6000, V4_BOUNDARY)

    def test_frozen_pretrained_alexnet_retains_prediction_gradient_without_metric_state(self):
        # Check the same torchmetrics feature network used by the trainer.
        # Never request a download when this optional local cache is absent.
        cache = Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
        if not cache.is_file():
            self.skipTest("pretrained AlexNet weights are not cached locally")
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        metric = LearnedPerceptualImagePatchSimilarity(normalize=True)
        metric.net.requires_grad_(False)
        metric.net.eval()
        initial_state = {name: value.clone() if torch.is_tensor(value) else list(value)
                         for name, value in metric.metric_state.items()}
        generator = torch.Generator().manual_seed(9)
        prediction = torch.rand(64, 64, 3, generator=generator, requires_grad=True)
        target = torch.rand(64, 64, 3, generator=generator, requires_grad=True)
        loss = boundary_rgb_losses(prediction, target, torch.ones(64, 64),
                                   6000, {**V4_BOUNDARY, "num_patches": 2},
                                   metric.net)["boundary_perceptual_loss"]
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0.)
        self.assertIsNone(target.grad)
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in metric.net.parameters()))
        self.assertEqual(metric._update_count, 0)
        for name, value in metric.metric_state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
            else:
                self.assertEqual(value, initial_state[name])


if __name__ == "__main__":
    unittest.main()
