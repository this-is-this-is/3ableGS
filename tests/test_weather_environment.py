"""Weather HDR identity, freezing, filtering and CUDA shading regression checks."""
import copy
import unittest

import torch
from omegaconf import OmegaConf

from models.pbr import HDREnvLight, LOG_RADIANCE_CLAMP


def environment(weathers=("rain", "snow"), device="cpu"):
    return HDREnvLight(
        "EnvLight", resolution=8, n_mips=3, brdf_lut_size=8,
        ggx_tail_resolution=4, weather_residuals=weathers,
        device=torch.device(device),
    )


class WeatherEnvironmentTests(unittest.TestCase):
    def test_optional_residual_preserves_raw_state_and_rng(self):
        torch.manual_seed(714)
        raw = environment(())
        rng = torch.random.get_rng_state().clone()
        weather = environment()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertEqual(set(raw.state_dict()), {
            "_log_env", "brdf_lut", "_sh_basis", "_solid_angles"
        })
        for name, tensor in raw.state_dict().items():
            torch.testing.assert_close(weather.state_dict()[name], tensor, rtol=0, atol=0)
        for name in (None, "raw", "rain", "snow"):
            torch.testing.assert_close(weather.radiance_for(name), raw.radiance, rtol=0, atol=0)

    def test_log_radiance_scalar_strength_and_independent_maps(self):
        env = environment()
        with torch.no_grad():
            env._log_env.fill_(0.3)
            env.weather_log_residuals.rain.fill_(0.7)
            env.weather_log_residuals.snow.fill_(-0.5)
        for weather, delta in (("rain", 0.7), ("snow", -0.5)):
            for strength in (0.0, 0.4, 1.0, 2.0):
                expected = torch.full_like(env._log_env, 0.3 + strength * delta).exp()
                torch.testing.assert_close(env.radiance_for(weather, strength), expected)
            torch.testing.assert_close(env.radiance_for(weather, 0), env.radiance, rtol=0, atol=0)
        with torch.no_grad():
            env.weather_log_residuals.rain.fill_(100)
        torch.testing.assert_close(
            env.radiance_for("rain"), torch.full_like(env._log_env, LOG_RADIANCE_CLAMP).exp()
        )
        for strength in (float("nan"), float("inf"), torch.ones(2)):
            with self.assertRaises(ValueError):
                env.radiance_for("rain", strength)
        with self.assertRaises(KeyError):
            env.radiance_for("unconfigured")

    def test_stage_freezing_and_residual_regularization(self):
        env = environment()
        for parameter in env.parameters():
            parameter.requires_grad_(False)
        env.get_weather_parameters("rain")[0].requires_grad_(True)
        before = copy.deepcopy(env.state_dict())
        optimizer = torch.optim.Adam(env.get_weather_parameters("rain"), lr=0.02)
        optimizer.zero_grad()
        env.sh_coeffs("rain").square().mean().backward()
        self.assertIsNone(env._log_env.grad)
        self.assertIsNone(env.weather_log_residuals.snow.grad)
        self.assertTrue(torch.isfinite(env.weather_log_residuals.rain.grad).all())
        self.assertGreater(float(env.weather_log_residuals.rain.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before["weather_log_residuals.rain"], env.weather_log_residuals.rain))
        torch.testing.assert_close(env._log_env, before["_log_env"], rtol=0, atol=0)
        torch.testing.assert_close(env.weather_log_residuals.snow, before["weather_log_residuals.snow"], rtol=0, atol=0)
        cfg = OmegaConf.create({"weather_residual": {"magnitude": {"w": 0.001}, "tv": {"w": 0.001}}})
        with torch.no_grad():
            residual = env.weather_log_residuals.rain
            residual.copy_(torch.arange(residual.numel()).reshape_as(residual) / 100)
        losses = env.compute_weather_reg_loss("rain", cfg)
        expected_tv = (residual[:, 1:] - residual[:, :-1]).abs().mean()
        expected_tv += (residual[:, :, 1:] - residual[:, :, :-1]).abs().mean()
        torch.testing.assert_close(losses["envmap_weather_rain_magnitude"], residual.abs().mean() * 0.001)
        torch.testing.assert_close(losses["envmap_weather_rain_tv"], expected_tv * 0.001)
        self.assertEqual(env.compute_weather_reg_loss("raw", cfg), {})
        self.assertEqual(env.get_weather_parameters("raw"), [])
        self.assertIs(env.get_base_parameters()[0], env._log_env)
        env.requires_grad_(False)
        self.assertFalse(any(p.requires_grad for p in env.parameters()))

    def test_prefilter_has_fresh_weather_graph_and_keeps_fixed_kernels(self):
        env = environment()
        env._log_env.requires_grad_(False)
        optimizer = torch.optim.SGD(env.get_weather_parameters("rain"), lr=0.1)
        kernels = {key: value.clone() for key, value in env.named_buffers() if key.startswith("_ggx_")}
        values = []
        for _ in range(2):
            optimizer.zero_grad()
            mip = env._mip_chain("rain")[-1]
            values.append(mip.detach().clone())
            mip.sum().backward()
            self.assertGreater(float(env.weather_log_residuals.rain.grad.abs().sum()), 0)
            optimizer.step()
        self.assertGreater(float((values[0] - values[1]).abs().max()), 0)
        for key, tensor in kernels.items():
            torch.testing.assert_close(getattr(env, key), tensor, rtol=0, atol=0)
        for raw, zero in zip(env._mip_chain(), env._mip_chain("rain", 0)):
            torch.testing.assert_close(raw, zero, rtol=0, atol=0)

    def test_checkpoint_raw_identity_and_weather_roundtrip_are_strict(self):
        raw, restored = environment(()), environment()
        with torch.no_grad():
            raw._log_env.normal_(0, 0.2)
            restored.weather_log_residuals.rain.fill_(4)
        restored.load_state_dict(raw.state_dict(), strict=True)
        for weather in restored.weather_names:
            self.assertEqual(float(restored.weather_log_residuals[weather].detach().abs().sum()), 0)
        broken = dict(raw.state_dict())
        broken.pop("_log_env")
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(broken, strict=True)
        broken = dict(raw.state_dict(), unrelated=torch.zeros(1))
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(broken, strict=True)
        with torch.no_grad():
            restored.weather_log_residuals.rain.normal_(0, 0.1)
            restored.weather_log_residuals.snow.normal_(0, 0.1)
        roundtrip = environment()
        roundtrip.load_state_dict(restored.state_dict(), strict=True)
        for weather in ("raw", "rain", "snow"):
            torch.testing.assert_close(roundtrip.radiance_for(weather), restored.radiance_for(weather), rtol=0, atol=0)

    def test_external_hdr_overrides_residual_and_clear_recovers_weather(self):
        env = environment()
        with torch.no_grad():
            env.weather_log_residuals.rain.fill_(1)
        expected = env.radiance_for("rain").detach().clone()
        env.external = torch.full_like(env._log_env, 5)
        self.assertIs(env.radiance_for("rain", 0.4), env.external)
        self.assertIs(env.radiance_for("unconfigured"), env.external)
        env.clear_external()
        torch.testing.assert_close(env.radiance_for("rain"), expected, rtol=0, atol=0)

    def test_invalid_weather_names_fail_early(self):
        for names in (("raw",), ("rain", "rain"), ("",), ("snow.a",)):
            with self.assertRaises(ValueError):
                environment(names)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA nvdiffrast required")
    def test_cuda_surface_sky_strength_gradient_and_weather_switch(self):
        env = environment(device="cuda")
        with torch.no_grad():
            env._log_env.normal_(0, 0.15)
            env.weather_log_residuals.rain.fill_(torch.log(torch.tensor(2.0)).item())
        env._log_env.requires_grad_(False)
        albedo = torch.tensor([[0.3, 0.5, 0.7], [0.6, 0.4, 0.2]], device="cuda")
        metallic = torch.tensor([[0.1], [0.8]], device="cuda")
        roughness = torch.tensor([[0.3], [0.7]], device="cuda")
        normal = torch.tensor([[0.2, 0.1, 1.0], [0.5, 0.3, 1.0]], device="cuda")
        view = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], device="cuda")
        raw = env.shade(albedo, metallic, roughness, normal, view)
        raw_sky = env.background(-view)
        wet = env.shade(albedo, metallic, roughness, normal, view, "rain")
        wet_sky = env.background(-view, "rain")
        for key in ("rgb", "diffuse", "specular"):
            torch.testing.assert_close(wet[key], raw[key] * 2, rtol=2e-6, atol=1e-6)
        torch.testing.assert_close(wet_sky, raw_sky * 2, rtol=2e-6, atol=1e-6)
        for weather, strength in (("rain", 0), ("snow", 1), ("raw", 1)):
            identity = env.shade(albedo, metallic, roughness, normal, view, weather, strength)
            torch.testing.assert_close(identity["rgb"], raw["rgb"], rtol=0, atol=0)
            torch.testing.assert_close(env.background(-view, weather, strength), raw_sky, rtol=0, atol=0)
        wet["rgb"].sum().backward(retain_graph=True)
        surface_grad = env.weather_log_residuals.rain.grad.detach().clone()
        self.assertTrue(torch.isfinite(surface_grad).all())
        self.assertGreater(float(surface_grad.abs().sum()), 0)
        env.zero_grad(set_to_none=True)
        wet_sky.sum().backward()
        self.assertGreater(float(env.weather_log_residuals.rain.grad.abs().sum()), 0)
        self.assertIsNone(env._log_env.grad)
        self.assertIsNone(env.weather_log_residuals.snow.grad)


if __name__ == "__main__":
    unittest.main()
