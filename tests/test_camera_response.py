"""Regression checks for the recovered Base v3 display response."""

import io
import unittest

import torch

from models.camera_response import MonotonicCameraResponse, _StableReinhard


class CameraResponseTests(unittest.TestCase):
    @staticmethod
    def reference_remap(response, compressed):
        # Historical interpolation, before its atomic-scatter backward was
        # replaced. Keep this independent reference to check both derivatives.
        x = compressed.clamp(0.0, 1.0)
        knots = response.knot_x.to(dtype=x.dtype)
        offsets = response.knot_offsets().to(dtype=x.dtype)
        index = (torch.searchsorted(knots, x.contiguous(), right=True) - 1).clamp(
            0, response.num_knots - 2
        )
        fraction = (x - knots[index]) / (knots[index + 1] - knots[index])
        return x + offsets[index] + fraction * (offsets[index + 1] - offsets[index])

    def test_identity_matches_legacy_display_and_hdr_gradient_is_nonnegative(self):
        for knots in (9, 13, 15):
            for dtype in (torch.float32, torch.float64):
                response = MonotonicCameraResponse(num_knots=knots, device="cpu").to(dtype)
                linear = torch.cat((torch.tensor([-1.0, 0.0, 0.001, 1.0, 100.0]),
                                    torch.logspace(-6, 7, 257))).to(dtype).requires_grad_()
                x = (linear / (1 + linear.clamp(min=0))).clamp(min=0)
                legacy = torch.where(x <= 0.0031308, x * 12.92,
                                     1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)
                actual = response(linear)
                torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
                gradient = torch.autograd.grad(actual.sum(), linear)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue((gradient >= 0).all())
        bright = torch.logspace(0, 12, 100, requires_grad=True)
        slope = torch.autograd.grad(_StableReinhard.apply(bright).sum(), bright)[0]
        torch.testing.assert_close(slope, (1 + bright).reciprocal().square(), rtol=0, atol=0)

    def test_reduced_backward_matches_historical_interpolation(self):
        generator = torch.Generator().manual_seed(7)
        for knots in (9, 13, 15):
            reference = MonotonicCameraResponse(num_knots=knots, device="cpu").double()
            optimized = MonotonicCameraResponse(num_knots=knots, device="cpu").double()
            with torch.no_grad():
                reference.interval_logits.normal_(0, 0.7, generator=generator)
                optimized.load_state_dict(reference.state_dict())
            values = torch.cat((torch.tensor([-1., 0., .5, 1., 2.], dtype=torch.float64),
                                reference.knot_x.clone(),
                                torch.rand(257, dtype=torch.float64, generator=generator) * 2 - .5))
            ref_x = values.clone().requires_grad_()
            opt_x = values.clone().requires_grad_()
            weight = torch.randn(values.shape, dtype=values.dtype, generator=generator)
            ref_y = self.reference_remap(reference, ref_x)
            opt_y = optimized.remap(opt_x)
            (ref_y * weight).sum().backward()
            (opt_y * weight).sum().backward()
            torch.testing.assert_close(opt_y, ref_y, rtol=0, atol=0)
            torch.testing.assert_close(opt_x.grad, ref_x.grad, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(optimized.interval_logits.grad,
                                       reference.interval_logits.grad, rtol=1e-12, atol=1e-12)

    def test_monotonicity_anchors_and_slope_floor_with_extreme_logits(self):
        response = MonotonicCameraResponse(device="cpu").double()
        for magnitude in (1., 1000.):
            with torch.no_grad():
                response.interval_logits.copy_(torch.linspace(-magnitude, magnitude, 8).reshape(2, 4))
            x = torch.linspace(0, 1, 1001, dtype=torch.float64, requires_grad=True)
            y = response.remap(x)
            self.assertTrue((y[1:] > y[:-1]).all())
            anchors = torch.tensor([0., .5, 1.], dtype=torch.float64)
            torch.testing.assert_close(response.remap(anchors), anchors, rtol=0, atol=0)
            slope = torch.autograd.grad(y.sum(), x)[0]
            self.assertGreaterEqual(slope.min().item(), .05 - 1e-12)

    def test_shared_channels_parameter_gradients_and_state_roundtrip(self):
        response = MonotonicCameraResponse(device="cpu")
        radiance = torch.tensor([[.03] * 3, [.8] * 3, [4.] * 3])
        prediction = response(radiance)
        self.assertTrue(torch.equal(prediction[:, 0], prediction[:, 1]))
        prediction.square().sum().backward()
        self.assertTrue(torch.isfinite(response.interval_logits.grad).all())
        self.assertGreater(response.interval_logits.grad.abs().sum().item(), 0.)
        self.assertEqual(response.interval_logits.numel(), 8)
        self.assertEqual(list(response.get_param_groups()), ["CameraResponse#all"])
        with torch.no_grad():
            response.interval_logits[0, 1] = 1.5
        buffer = io.BytesIO()
        torch.save(response.state_dict(), buffer)
        buffer.seek(0)
        clone = MonotonicCameraResponse(n=1000, device="cpu")
        clone.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(clone(radiance), response(radiance), rtol=0, atol=0)

    def test_regularization_is_zero_at_identity_and_penalizes_bent_curve(self):
        response = MonotonicCameraResponse(device="cpu")
        config = {"identity": {"w": .001}, "smoothness": {"w": .001}}
        self.assertTrue(all(loss.item() == 0 for loss in response.compute_reg_loss(config).values()))
        with torch.no_grad():
            response.interval_logits[0, 1] = 1.5
        losses = response.compute_reg_loss(config)
        self.assertEqual(set(losses), {"camera_response_identity", "camera_response_smoothness"})
        self.assertTrue(all(loss.item() > 0 for loss in losses.values()))
        sum(losses.values()).backward()
        self.assertGreater(response.interval_logits.grad.abs().sum().item(), 0.)

    def test_invalid_settings_fail_before_render(self):
        for value in (0, 4, 8, 9.5, True):
            with self.assertRaises(ValueError):
                MonotonicCameraResponse(num_knots=value, device="cpu")
        for value in (0., 1., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MonotonicCameraResponse(min_slope=value, device="cpu")
        response = MonotonicCameraResponse(device="cpu")
        for value in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                response.compute_reg_loss({"identity": {"w": value}})


if __name__ == "__main__":
    unittest.main()
