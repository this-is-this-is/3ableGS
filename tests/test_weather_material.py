"""Residual material equations, freezing boundaries, and checkpoint contracts."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from models.gaussians.basics import gaussian_normals
from models.gaussians.feature import FeatureGaussians
from models.weather_material import WeatherMaterialDecoder


def make_weather_model(weather=True, **weather_overrides):
    cfg = {
        "enabled": True, "weathers": ["rain", "snow"], "hidden_dim": 64,
        "num_hidden_layers": 2, "detach_inputs": True,
        "world_up": [0.0, 0.0, 1.0],
    }
    cfg.update(weather_overrides)
    model = FeatureGaussians(
        class_name="Background", ctrl=OmegaConf.create({"sh_degree": 3}),
        feature=OmegaConf.create({"feature_dim": 32}),
        raw_material=OmegaConf.create({
            "spatial_xyz": {"enabled": True},
            "decoder": {"activation": "leaky_relu"},
            "normal_residual": {"enabled": True, "space": "gaussian_local_tangent",
                                "max_tangent": 0.5},
        }),
        weather_material=OmegaConf.create(cfg) if weather else None,
        device="cpu",
    )
    model._means = torch.nn.Parameter(torch.tensor([[0., 0., 1.], [1., 0., 1.]]))
    model._features = torch.nn.Parameter(torch.full((2, 32), 0.1))
    model._quats = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1))
    model._scales = torch.nn.Parameter(torch.tensor([[1., 2., .1], [.1, 1., 2.]]).log())
    model._opacities = torch.nn.Parameter(torch.zeros(2, 1))
    return model


class WeatherMaterialTests(unittest.TestCase):
    def test_canonical_architecture_zero_output_and_first_update_gradient(self):
        decoder = WeatherMaterialDecoder(32, hidden_dim=64, num_hidden_layers=2)
        layers = [m for m in decoder.network if isinstance(m, torch.nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in layers],
                         [(38, 64), (64, 64), (64, 4)])
        inputs = [torch.randn(7, c) for c in (32, 3, 1, 1, 1)]
        output = decoder(*inputs)
        for value in output.values():
            self.assertEqual(torch.count_nonzero(value), 0)
        loss = (output["delta_albedo_logit"] - 1).square().mean()
        loss = loss + (output["delta_roughness_logit"] + .5).square().mean()
        loss.backward()
        self.assertGreater(layers[-1].weight.grad.abs().sum(), 0)
        self.assertGreater(layers[-1].bias.grad.abs().sum(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in decoder.parameters()))

    def test_caps_are_tanh_scaled_and_shape_validation_is_explicit(self):
        decoder = WeatherMaterialDecoder(32, hidden_dim=64, num_hidden_layers=2)
        with torch.no_grad():
            decoder.network[-1].bias.copy_(torch.tensor([-100., -.5, .5, 100.]))
        inputs = [torch.zeros(2, 3, c) for c in (32, 3, 1, 1, 1)]
        outputs = decoder(*inputs)
        expected = torch.tensor([-100., -.5, .5]).tanh() * 4
        torch.testing.assert_close(outputs["delta_albedo_logit"], expected.expand(2, 3, 3))
        torch.testing.assert_close(outputs["delta_roughness_logit"], torch.full((2, 3, 1), 5.))
        with self.assertRaisesRegex(ValueError, "features must end"):
            decoder(torch.zeros(2, 31), *[torch.zeros(2, c) for c in (3, 1, 1, 1)])
        with self.assertRaisesRegex(ValueError, "leading shape"):
            decoder(torch.zeros(2, 32), torch.zeros(3, 3), *[torch.zeros(2, 1)] * 3)

    def test_zero_initialization_is_exact_raw_material_identity(self):
        model = make_weather_model()
        raw = model.materials("raw")
        for weather in ("rain", "snow"):
            for expected, actual in zip(raw, model.materials(weather)):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(list(model.mlp_material), ["raw"])

    def test_strength_zero_skips_decoder_and_clears_stale_residuals(self):
        model = make_weather_model()
        with torch.no_grad():
            model.weather_material_decoders["rain"].network[-1].bias.fill_(.25)
        model.materials("rain")
        self.assertIsNotNone(model.get_weather_material_outputs("rain"))
        raw = model.materials("raw")
        with patch.object(model.weather_material_decoders["rain"], "forward",
                          side_effect=AssertionError("decoder must be bypassed")):
            actual = model.materials("rain", weather_strength=0)
        for expected, value in zip(raw, actual):
            torch.testing.assert_close(expected, value, rtol=0, atol=0)
        self.assertIsNone(model.get_weather_material_outputs("rain"))

    def test_logit_application_preserves_metallic_and_detaches_all_base_inputs(self):
        model = make_weather_model()
        head = model.weather_material_decoders["rain"]
        with torch.no_grad():
            head.network[-1].weight.normal_(std=.03)
            head.network[-1].bias.fill_(.2)
        logits = model.mlp_material["raw"](model._raw_material_input()).detach()
        outputs = model.material_outputs("rain", weather_strength=.4)
        torch.testing.assert_close(outputs["albedo"], torch.sigmoid(
            logits[:, :3] + .4 * outputs["delta_albedo_logit"]))
        torch.testing.assert_close(outputs["roughness"], .99 * torch.sigmoid(
            logits[:, 4:5] + .4 * outputs["delta_roughness_logit"]) + .01)
        torch.testing.assert_close(outputs["metallic"], logits[:, 3:4].sigmoid(), rtol=0, atol=0)
        (outputs["albedo"].sum() + outputs["roughness"].sum()).backward()
        for name, parameter in model.named_parameters():
            if name.startswith("weather_material_decoders.rain."):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertGreater(head.network[0].weight.grad.abs().sum(), 0)

    def test_orientation_is_sign_invariant_geometry_before_tangent_and_camera_flip(self):
        model = make_weather_model(world_up=[0., 0., 7.])
        normals = gaussian_normals(model.get_quats, model.get_scaling)
        expected = torch.tensor([[1.], [0.]])
        torch.testing.assert_close(model._normal_upness(normals), expected)
        torch.testing.assert_close(model._normal_upness(-normals), expected)
        with torch.no_grad():
            model.raw_normal_residual[-1].bias.copy_(torch.tensor([5., 5., 5.]))
        captured = []
        hook = model.weather_material_decoders["rain"].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[-1].clone()))
        camera = torch.eye(4)
        camera[:3, 3] = torch.tensor([-100., -100., -100.])
        packed = model.gbuffer("rain", model.get_quats, model.get_scaling,
                               model._means, SimpleNamespace(camtoworlds=camera))
        hook.remove()
        torch.testing.assert_close(captured[0], expected)
        self.assertFalse(torch.allclose(model._normal_upness(packed[:, 5:]), expected))
        self.assertTrue(torch.all((packed[:, 5:] * (camera[:3, 3] - model._means)).sum(-1) > 0))

    def test_raw_only_preserves_state_keys_rng_and_parameter_coverage(self):
        torch.manual_seed(41)
        raw = make_weather_model(False)
        next_without = torch.rand(5)
        torch.manual_seed(41)
        disabled = make_weather_model(True, enabled=False)
        next_disabled = torch.rand(5)
        self.assertEqual(set(raw.state_dict()), set(disabled.state_dict()))
        self.assertFalse(any("weather" in key for key in raw.state_dict()))
        torch.testing.assert_close(next_without, next_disabled, rtol=0, atol=0)
        for key, value in raw.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
        weather = make_weather_model()
        grouped = [p for values in weather.get_param_groups().values() for p in values]
        self.assertEqual(len(grouped), len({id(p) for p in grouped}))
        self.assertEqual({id(p) for p in weather.parameters()}, {id(p) for p in grouped})

    def test_warm_start_allows_only_new_complete_branch_and_roundtrip_is_exact(self):
        raw = make_weather_model(False)
        weather = make_weather_model()
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            weather.load_state_dict(raw.state_dict(), strict=True)
        weather.load_state_dict(raw.state_dict(), strict=True, allow_new_weather=True)
        for expected, actual in zip(raw.materials(), weather.materials("rain")):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        with torch.no_grad():
            weather.weather_material_decoders["rain"].network[-1].bias.fill_(.7)
        buffer = io.BytesIO()
        torch.save(weather.state_dict(), buffer)
        buffer.seek(0)
        restored = make_weather_model()
        state = torch.load(buffer, weights_only=True)
        restored.load_state_dict(state, strict=True)
        for expected, actual in zip(weather.materials("rain"), restored.materials("rain")):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        partial = dict(state)
        partial.pop("weather_material_decoders.rain.network.0.weight")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            restored.load_state_dict(partial, strict=True, allow_new_weather=True)
        damaged_base = dict(raw.state_dict())
        damaged_base.pop("mlp_material.raw.0.weight")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            restored.load_state_dict(damaged_base, strict=True, allow_new_weather=True)

    def test_invalid_weather_configuration_and_strength_fail_explicitly(self):
        for up in ([0., 0., 0.], [0., float("nan"), 1.], [1., 2.]):
            with self.assertRaisesRegex(ValueError, "world_up"):
                make_weather_model(world_up=up)
        for weathers in (["raw"], [], ["rain", "rain"]):
            with self.assertRaises(ValueError):
                make_weather_model(weathers=weathers)
        model = make_weather_model()
        for strength in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                model.set_weather_material_strength(strength)
        with self.assertRaisesRegex(KeyError, "not configured"):
            model.materials("hail")


if __name__ == "__main__":
    unittest.main()
