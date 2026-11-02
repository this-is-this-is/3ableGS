"""Math and CUDA integration checks for the recovered raw HDR lighting.

The dense-reference GGX check is recovered from the historical regression
script; it catches the broad-lobe energy/direction issue that led to the
final directional-tail implementation.
"""
import io
import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.pbr import (
    HDREnvLight, build_ggx_prefilter, cubemap_dirs, cubemap_solid_angles,
    direction_to_cube_index, linear_to_srgb, srgb_to_linear, tonemap,
)

BASE_RESOLUTION = 32
OUTPUT_RESOLUTION = 16
TAIL_RESOLUTION = 16
ROUGHNESS = 0.5
# Same kept-core fraction as the production R64, roughness-0.5 level.
BUDGET = 500_000


def dense_kernel():
    r = F.normalize(cubemap_dirs(OUTPUT_RESOLUTION, "cpu").reshape(-1, 3), dim=-1)
    base_dirs = cubemap_dirs(BASE_RESOLUTION, "cpu")
    light_dirs = F.normalize(base_dirs.reshape(-1, 3), dim=-1)
    solid_angles = cubemap_solid_angles(base_dirs).reshape(-1)
    alpha = ROUGHNESS**2
    rows = []
    for lo in range(0, r.shape[0], 128):
        reflection = r[lo:lo + 128]
        n_dot_l = (reflection @ light_dirs.T).clamp(min=0)
        half = F.normalize(reflection[:, None, :] + light_dirs[None, :, :], dim=-1)
        n_dot_h = (reflection[:, None, :] * half).sum(-1).clamp(min=0)
        denom = n_dot_h.square() * (alpha * alpha - 1) + 1
        ndf = (alpha * alpha) / (torch.pi * denom.square() + 1e-12)
        weight = ndf * n_dot_l * solid_angles[None]
        rows.append(weight / weight.sum(-1, keepdim=True).clamp(min=1e-12))
    return torch.cat(rows), solid_angles, light_dirs


def coarse_radiance(radiance, solid_angles):
    block = BASE_RESOLUTION // TAIL_RESOLUTION
    weighted = (radiance.reshape(6, BASE_RESOLUTION, BASE_RESOLUTION, 3)
                * solid_angles.reshape(6, BASE_RESOLUTION, BASE_RESOLUTION, 1))
    numerator = weighted.reshape(
        6, TAIL_RESOLUTION, block, TAIL_RESOLUTION, block, 3
    ).sum((2, 4))
    denominator = solid_angles.reshape(
        6, TAIL_RESOLUTION, block, TAIL_RESOLUTION, block
    ).sum((2, 4))[..., None]
    return (numerator / denominator).reshape(-1, 3)


def normalized_rmse(actual, expected):
    return float((actual - expected).square().mean().sqrt()
                 / expected.square().mean().sqrt().clamp(min=1e-12))


def check_directional_ggx():
    idx, core_weight, tail_weight = build_ggx_prefilter(
        OUTPUT_RESOLUTION,
        ROUGHNESS,
        BASE_RESOLUTION,
        budget=BUDGET,
        tail_resolution=TAIL_RESOLUTION,
        device="cpu",
    )
    dense, solid_angles, light_dirs = dense_kernel()

    mass_error = (core_weight.sum(-1) + tail_weight.sum(-1) - 1).abs().max()
    assert mass_error < 2e-6, mass_error
    assert tail_weight.min() >= 0

    uniform = torch.ones(light_dirs.shape[0], 3)
    uniform_approx = (
        (uniform[idx.long()] * core_weight[..., None]).sum(1)
        + tail_weight @ coarse_radiance(uniform, solid_angles)
    )
    uniform_error = float((uniform_approx - 1).abs().max())
    assert uniform_error < 2e-6, uniform_error

    # Unequal RGB lobes at non-cardinal directions expose both lost direction
    # and channel mixing while staying smooth enough for a 32px reference map.
    a = F.normalize(torch.tensor([0.73, -0.41, 0.55]), dim=0)
    b = F.normalize(torch.tensor([-0.22, 0.91, 0.35]), dim=0)
    lobe_a = torch.exp(55 * ((light_dirs @ a) - 1))[:, None]
    lobe_b = torch.exp(18 * ((light_dirs @ b) - 1))[:, None]
    radiance = (
        torch.tensor([0.04, 0.06, 0.10])
        + lobe_a * torch.tensor([45.0, 12.0, 3.0])
        + lobe_b * torch.tensor([2.0, 9.0, 28.0])
    )
    reference = dense @ radiance
    core = (radiance[idx.long()] * core_weight[..., None]).sum(1)
    directional = core + tail_weight @ coarse_radiance(radiance, solid_angles)
    env_mean = (radiance * solid_angles[:, None]).sum(0) / solid_angles.sum()
    global_tail = core + (1 - core_weight.sum(-1))[:, None] * env_mean

    directional_error = normalized_rmse(directional, reference)
    global_error = normalized_rmse(global_tail, reference)
    assert directional.min() >= 0
    assert directional_error < 0.05, directional_error
    assert directional_error < global_error * 0.5, (directional_error, global_error)

    # A compact HDR source is the case that motivated the directional tail.
    # It is intentionally a few source texels wide rather than an adversarial
    # delta, matching what supersampled equirectangular conversion produces.
    sharp = 0.02 + torch.exp(300 * ((light_dirs @ a) - 1))[:, None] * torch.tensor(
        [120.0, 35.0, 8.0]
    )
    sharp_reference = dense @ sharp
    sharp_directional = (
        (sharp[idx.long()] * core_weight[..., None]).sum(1)
        + tail_weight @ coarse_radiance(sharp, solid_angles)
    )
    sharp_error = normalized_rmse(sharp_directional, sharp_reference)
    assert sharp_error < 0.08, sharp_error

    differentiable = radiance.clone().requires_grad_(True)
    result = ((differentiable[idx.long()] * core_weight[..., None]).sum(1)
              + tail_weight @ coarse_radiance(differentiable, solid_angles))
    result.square().mean().backward()
    assert differentiable.grad is not None
    assert torch.isfinite(differentiable.grad).all()
    assert differentiable.grad.abs().sum() > 0


class PBRMathTests(unittest.TestCase):
    def test_cubemap_coordinates_are_inverse(self):
        dirs = cubemap_dirs(8, "cpu")
        expected = torch.arange(6 * 8 * 8).reshape(6, 8, 8)
        torch.testing.assert_close(direction_to_cube_index(dirs, 8), expected)
        self.assertLess(abs(float(cubemap_solid_angles(dirs).sum()) - 4 * torch.pi), 0.05)

    def test_display_transfer_has_finite_zero_gradient(self):
        x = torch.tensor([0.0, 0.001, 0.1, 0.5, 1.0], requires_grad=True)
        torch.testing.assert_close(srgb_to_linear(linear_to_srgb(x)), x, atol=2e-7, rtol=2e-6)
        y = linear_to_srgb(tonemap(x))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad > 0).all())

    def test_directional_tail_against_dense_integral(self):
        check_directional_ggx()

    def test_uniform_lighting_checkpoint_and_gradient(self):
        env = HDREnvLight("EnvLight", resolution=8, n_mips=3,
                          brdf_lut_size=8, ggx_tail_resolution=4, device="cpu")
        normals = F.normalize(torch.tensor([[1., 2., 3.], [-1., 0., 0.], [0., 0., 1.]]), dim=-1)
        irradiance = env.irradiance(normals)
        torch.testing.assert_close(irradiance, torch.full_like(irradiance, torch.pi), rtol=0.004, atol=0.001)
        mips = env._mip_chain()
        for mip in mips:
            torch.testing.assert_close(mip, torch.ones_like(mip), rtol=2e-6, atol=2e-6)
        (irradiance.mean() + sum(m.mean() for m in mips)).backward()
        self.assertTrue(torch.isfinite(env._log_env.grad).all())
        self.assertGreater(float(env._log_env.grad.abs().sum()), 0.)
        with torch.no_grad():
            env._log_env[:, 1:] += 0.1
        loss = env.compute_reg_loss(OmegaConf.create({"tv": {"w": 0.001}}))
        self.assertGreater(float(loss["envmap_tv"].detach()), 0.)
        saved = io.BytesIO()
        torch.save(env.state_dict(), saved)
        saved.seek(0)
        clone = HDREnvLight("EnvLight", resolution=8, n_mips=3,
                            brdf_lut_size=8, ggx_tail_resolution=4, device="cpu")
        clone.load_state_dict(torch.load(saved, weights_only=True), strict=True)
        torch.testing.assert_close(clone.irradiance(normals), env.irradiance(normals))
        self.assertEqual(set(env.get_param_groups()), {"EnvLight#all"})
        self.assertEqual(len(list(env.parameters())), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for nvdiffrast texture")
    def test_cuda_surface_sky_energy_and_backward(self):
        env = HDREnvLight("EnvLight", resolution=8, n_mips=3,
                          brdf_lut_size=16, ggx_tail_resolution=4, device="cuda")
        normals = F.normalize(torch.randn(4, 5, 3, device="cuda"), dim=-1).requires_grad_()
        albedo = torch.full((4, 5, 3), 0.7, device="cuda", requires_grad=True)
        metallic = torch.full((4, 5, 1), 0.2, device="cuda", requires_grad=True)
        roughness = torch.full((4, 5, 1), 0.5, device="cuda", requires_grad=True)
        view = F.normalize(normals.detach() + torch.tensor([0.1, 0., 0.], device="cuda"), dim=-1)
        out = env.shade(albedo, metallic, roughness, normals, view)
        sky = env.background(-view)
        torch.testing.assert_close(sky, torch.ones_like(sky), rtol=1e-6, atol=1e-6)
        for value in out.values():
            self.assertEqual(value.shape, (4, 5, 3))
            self.assertTrue(torch.isfinite(value).all())
        rgb = linear_to_srgb(tonemap(0.8 * out["rgb"] + 0.2 * sky))
        rgb.square().mean().backward()
        for value in (albedo, metallic, roughness, normals, env._log_env):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.)
        white = env.shade(torch.ones_like(albedo), torch.zeros_like(metallic),
                          roughness.detach(), normals.detach(), view)["rgb"]
        self.assertLess(float(white.max().detach()), 1.005)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
