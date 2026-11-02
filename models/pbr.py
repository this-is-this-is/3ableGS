"""Physically-based shading pieces for relightable gaussian splatting.

The scene's lighting lives in a single learnable HDR cubemap. Surfaces are
shaded from it with the split-sum approximation, so the same parameter that
paints the sky background also illuminates every gaussian -- which is what
makes swapping in an external .hdr a coherent relight rather than just a new
backdrop.

No visibility/occlusion term: every point is lit as if unoccluded, so
relighting changes tint and highlight direction but does not cast shadows.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger()

# Radiance is stored as log values; exp() keeps it positive and unbounded
# above, which is what separates a sun from a bright patch of sky.
LOG_RADIANCE_CLAMP = 8.0

# Ramamoorthi & Hanrahan irradiance-from-SH constants.
_C1, _C2, _C3, _C4, _C5 = 0.429043, 0.511664, 0.743125, 0.886227, 0.247708


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


def tonemap(linear: torch.Tensor) -> torch.Tensor:
    """Reinhard. Rolls highlights off instead of letting them clip.

    Without it an HDR sun lands far outside the display range -- a radiance of
    723 leaves linear_to_srgb at 16.3 -- and every highlight saturates to flat
    white the moment a bright probe is swapped in.
    """
    return linear / (1.0 + linear.clamp(min=0.0))


def to_display(linear: torch.Tensor) -> torch.Tensor:
    """Linear radiance to display-referred sRGB."""
    return linear_to_srgb(tonemap(linear))


def cubemap_dirs(resolution: int, device) -> torch.Tensor:
    """Unnormalized direction through the centre of every cubemap texel.

    Faces follow the OpenGL order (+X, -X, +Y, -Y, +Z, -Z) that
    nvdiffrast's boundary_mode='cube' expects.
    """
    t = (torch.arange(resolution, device=device, dtype=torch.float32) + 0.5) / resolution * 2 - 1
    v, u = torch.meshgrid(t, t, indexing="ij")
    one = torch.ones_like(u)
    faces = [
        torch.stack([one, -v, -u], -1),
        torch.stack([-one, -v, u], -1),
        torch.stack([u, one, v], -1),
        torch.stack([u, -one, -v], -1),
        torch.stack([u, -v, one], -1),
        torch.stack([-u, -v, -one], -1),
    ]
    return torch.stack(faces, 0)                                   # [6, R, R, 3]


def cubemap_solid_angles(dirs: torch.Tensor) -> torch.Tensor:
    """Solid angle each texel subtends; sums to 4*pi over the whole cube."""
    resolution = dirs.shape[1]
    r = dirs.norm(dim=-1)
    return (2.0 / resolution) ** 2 / r.pow(3)                      # [6, R, R]


def direction_to_cube_index(d: torch.Tensor, resolution: int) -> torch.Tensor:
    """Flat index into a [6, R, R] cubemap for each direction.

    Faces and their uv conventions mirror cubemap_dirs exactly, so this is its
    inverse.
    """
    ax = d.abs()
    major = ax.argmax(-1)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    denom = ax.gather(-1, major[..., None]).squeeze(-1).clamp(min=1e-8)
    face = major * 2 + (d.gather(-1, major[..., None]).squeeze(-1) < 0).long()
    u = torch.where(major == 0, torch.where(x >= 0, -z, z),
        torch.where(major == 1, x, torch.where(z >= 0, x, -x)))
    v = torch.where(major == 0, -y,
        torch.where(major == 1, torch.where(y >= 0, z, -z), -y))
    u, v = u / denom, v / denom
    col = (((u + 1) * 0.5) * resolution).floor().long().clamp(0, resolution - 1)
    row = (((v + 1) * 0.5) * resolution).floor().long().clamp(0, resolution - 1)
    return face * resolution * resolution + row * resolution + col


def build_ggx_prefilter(
    resolution: int,
    roughness: float,
    base_resolution: int,
    budget: int = 2_000_000,
    tail_resolution: int = 16,
    device="cuda",
):
    """Gather indices and weights that convolve a cubemap with the GGX lobe.

    The convolution is linear in radiance, so the kernel depends only on
    geometry and is built once; at runtime prefiltering is a gather and a
    weighted sum, which keeps the learnable environment differentiable.

    Integrated deterministically over every base texel rather than by
    importance sampling. Monte Carlo missed a small bright source entirely at
    high roughness -- a sun vanished from the roughest mip -- and left the
    energy off by more than a factor of two elsewhere.

    The largest weights are kept exactly.  The omitted, broad tail is stored
    over a coarse cubemap instead of being replaced by one global average.
    That preserves which side of the scene the missing light came from while
    remaining deterministic, positive, and exactly energy preserving for a
    uniform environment.

    Uses the usual split-sum simplification N = V = R.
    """
    if tail_resolution <= 0 or base_resolution % tail_resolution != 0:
        raise ValueError(
            "tail_resolution must be a positive divisor of base_resolution "
            f"(got {tail_resolution} for {base_resolution})"
        )
    r = F.normalize(cubemap_dirs(resolution, device).reshape(-1, 3), dim=-1)     # [M, 3]
    base_dirs = cubemap_dirs(base_resolution, device)
    l = F.normalize(base_dirs.reshape(-1, 3), dim=-1)                            # [B, 3]
    d_omega = cubemap_solid_angles(base_dirs).reshape(-1)                        # [B]
    block = base_resolution // tail_resolution
    face, row, col = torch.meshgrid(
        torch.arange(6, device=device),
        torch.arange(base_resolution, device=device),
        torch.arange(base_resolution, device=device),
        indexing="ij",
    )
    bin_idx = (
        face * tail_resolution * tail_resolution
        + (row // block) * tail_resolution
        + col // block
    ).reshape(-1)
    num_bins = 6 * tail_resolution * tail_resolution
    a = max(roughness, 1e-3) ** 2
    # Keep a bounded exact core; the directional coarse-cube term below
    # accounts for all remaining mass instead of silently renormalizing it.
    k = min(l.shape[0], max(64, budget // max(r.shape[0], 1)))

    idx_all, w_all, tail_all = [], [], []
    for lo in range(0, r.shape[0], 256):
        rc = r[lo:lo + 256]
        n_dot_l = (rc @ l.T).clamp(min=0)                                        # [c, B]
        h = F.normalize(rc[:, None, :] + l[None, :, :], dim=-1)
        n_dot_h = (rc[:, None, :] * h).sum(-1).clamp(min=0)
        denom = n_dot_h**2 * (a * a - 1) + 1
        ndf = (a * a) / (np.pi * denom**2 + 1e-12)
        w = ndf * n_dot_l * d_omega[None, :]
        total = w.sum(-1, keepdim=True).clamp(min=1e-12)
        top_w, top_i = w.topk(k, dim=-1)
        normalized = w / total
        top_normalized = top_w / total

        # Sum the full lobe and its exact top-k core into the same directional
        # bins. Their difference is the omitted tail. Renormalize the tiny
        # floating-point subtraction error so every row still sums to exactly
        # the mass missing from the core.
        full_bins = torch.zeros(
            rc.shape[0], num_bins, device=device, dtype=w.dtype
        ).scatter_add_(1, bin_idx[None].expand(rc.shape[0], -1), normalized)
        top_bins = torch.zeros_like(full_bins).scatter_add_(
            1, bin_idx[top_i], top_normalized
        )
        tail_bins = (full_bins - top_bins).clamp(min=0.0)
        missing_mass = (1.0 - top_normalized.sum(-1, keepdim=True)).clamp(min=0.0)
        tail_bins = tail_bins * (
            missing_mass / tail_bins.sum(-1, keepdim=True).clamp(min=1e-12)
        )

        idx_all.append(top_i.int())
        # Normalized against the *full* lobe, so the weights sum to the
        # fraction actually kept and the rest can be accounted for separately.
        w_all.append(top_normalized)
        tail_all.append(tail_bins)
    return torch.cat(idx_all), torch.cat(w_all), torch.cat(tail_all)


def build_brdf_lut(resolution: int = 64, num_samples: int = 1024, device="cuda") -> torch.Tensor:
    """Karis' split-sum environment BRDF, indexed by (NoV, roughness).

    Returns [resolution, resolution, 2] holding the (A, B) scale/bias that
    the specular term applies to F0. Computed once; it depends only on the
    BRDF, not on the scene.
    """
    n_dot_v = (torch.arange(resolution, device=device, dtype=torch.float32) + 0.5) / resolution
    rough = (torch.arange(resolution, device=device, dtype=torch.float32) + 0.5) / resolution
    n_dot_v, rough = torch.meshgrid(n_dot_v, rough, indexing="ij")
    n_dot_v, rough = n_dot_v.reshape(-1), rough.reshape(-1)

    # View vector in tangent space, normal is +Z.
    v = torch.stack([torch.sqrt(1 - n_dot_v**2), torch.zeros_like(n_dot_v), n_dot_v], -1)
    a = rough**2

    # Hammersley sequence, shared across every (NoV, roughness) cell.
    i = torch.arange(num_samples, device=device, dtype=torch.float32)
    bits = i.to(torch.int64)
    radical = torch.zeros_like(i)
    for b in range(24):
        radical += ((bits >> b) & 1).float() * (2.0 ** -(b + 1))
    xi1, xi2 = (i / num_samples)[None, :], radical[None, :]

    # GGX importance sampling around the normal.
    cos_t = torch.sqrt((1 - xi2) / (1 + (a[:, None] ** 2 - 1) * xi2 + 1e-12))
    sin_t = torch.sqrt((1 - cos_t**2).clamp(min=0))
    phi = 2 * np.pi * xi1
    h = torch.stack([sin_t * torch.cos(phi), sin_t * torch.sin(phi), cos_t], -1)

    v_dot_h = (v[:, None, :] * h).sum(-1, keepdim=True)
    l = 2 * v_dot_h * h - v[:, None, :]
    n_dot_l, n_dot_h = l[..., 2], h[..., 2]
    v_dot_h = v_dot_h.squeeze(-1)
    nv = n_dot_v[:, None]

    # Smith geometry term with the IBL remapping k = alpha/2, alpha = roughness^2.
    k = a[:, None] / 2
    g = (n_dot_l / (n_dot_l * (1 - k) + k + 1e-12)) * (nv / (nv * (1 - k) + k + 1e-12))
    g_vis = g * v_dot_h / (n_dot_h * nv + 1e-12)
    fc = (1 - v_dot_h).clamp(min=0) ** 5

    valid = (n_dot_l > 0).float()
    lut_a = ((1 - fc) * g_vis * valid).sum(1) / num_samples
    lut_b = (fc * g_vis * valid).sum(1) / num_samples
    return torch.stack([lut_a, lut_b], -1).reshape(resolution, resolution, 2)


def equirect_to_cubemap(equirect: torch.Tensor, resolution: int, rotation_deg: float = 0.0,
                        supersample: int = 8) -> torch.Tensor:
    """Resample a lat-long HDR panorama (H, W, 3) into a [6, R, R] cubemap.

    Each texel is integrated over its own footprint rather than sampled at its
    centre. A sun is far smaller than a texel at these resolutions -- point
    sampling either lands on it or misses, so a half-degree sun kept 16% of its
    energy and a one-degree sun swung between 0.84 and 1.65 of it depending on
    how the probe happened to be rotated.
    """
    device = equirect.device
    fine = resolution * supersample
    dirs = cubemap_dirs(fine, device)
    weights = cubemap_solid_angles(dirs)                          # [6, F, F]
    d = F.normalize(dirs.reshape(-1, 3), dim=-1)
    yaw = np.deg2rad(rotation_deg)
    x = d[:, 0] * np.cos(yaw) - d[:, 2] * np.sin(yaw)
    z = d[:, 0] * np.sin(yaw) + d[:, 2] * np.cos(yaw)
    u = (torch.atan2(x, -z) / (2 * np.pi) + 0.5) * 2 - 1
    v = (torch.acos(d[:, 1].clamp(-1, 1)) / np.pi) * 2 - 1
    grid = torch.stack([u, v], -1)[None, None]                    # [1, 1, N, 2]
    src = equirect.permute(2, 0, 1)[None]                         # [1, 3, H, W]
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)
    fine_cube = out[0, :, 0].T.reshape(6, fine, fine, 3)

    # Solid-angle weighted average within each output texel.
    w = weights[..., None]
    num = (fine_cube * w).reshape(6, resolution, supersample, resolution, supersample, 3)
    den = w.reshape(6, resolution, supersample, resolution, supersample, 1)
    return (num.sum((2, 4)) / den.sum((2, 4)).clamp(min=1e-12)).contiguous()


class HDREnvLight(nn.Module):
    """A learnable HDR environment map that both lights the scene and is the sky.

    Replaces the LDR sky backdrop of models.modules.EnvLight. That one only
    ever received gradient from sky pixels, so on a single forward camera
    97% of its texels never moved from their init; here every shaded surface
    acts as a light probe, which reaches directions the camera never looks at.
    """

    def __init__(
        self,
        class_name: str,
        resolution: int = 64,
        n_mips: int = 5,
        brdf_lut_size: int = 64,
        background_resolution: int = 512,
        ggx_tail_resolution: int = 16,
        weather_residuals: Optional[list] = None,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.resolution = resolution
        self.n_mips = n_mips
        if ggx_tail_resolution <= 0:
            raise ValueError(
                f"ggx_tail_resolution must be positive, got {ggx_tail_resolution}"
            )
        self.ggx_tail_resolution = min(ggx_tail_resolution, resolution)
        if resolution % self.ggx_tail_resolution != 0:
            raise ValueError(
                "ggx_tail_resolution must divide the environment resolution "
                f"(got {self.ggx_tail_resolution} for {resolution})"
            )
        # These are non-persistent so old checkpoints remain strict-loadable,
        # but registering them makes a later module.to(device) move every
        # tensor used by lighting together.
        self.register_buffer(
            "to_opengl",
            torch.tensor(
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )
        # exp(0) = 1: a uniform unit-radiance environment to start from.
        self._log_env = nn.Parameter(torch.zeros(6, resolution, resolution, 3, device=device))
        # A weather condition changes the scene-wide illumination relative to
        # the raw capture.  Store that change in log-radiance so it composes
        # multiplicatively with the learned base map:
        #
        #     radiance(weather, s) = exp(_log_env + s * residual[weather])
        #
        # There is one map per scene/module and weather -- deliberately no
        # image or frame embedding.  Zero initialization makes enabling the
        # module an exact identity for an existing raw checkpoint.
        if weather_residuals is None:
            weather_names = []
        elif isinstance(weather_residuals, str):
            weather_names = [weather_residuals]
        else:
            weather_names = list(weather_residuals)
        weather_names = [str(name) for name in weather_names]
        if len(weather_names) != len(set(weather_names)):
            raise ValueError(f"weather_residuals contains duplicates: {weather_names}")
        for name in weather_names:
            if not name or name == "raw":
                raise ValueError(
                    "weather residual names must be non-empty and cannot be 'raw'; "
                    "raw lighting is stored in _log_env"
                )
            if "." in name:
                raise ValueError(
                    f"weather residual name cannot contain '.': {name!r}"
                )
        self.weather_log_residuals = nn.ParameterDict({
            name: nn.Parameter(torch.zeros_like(self._log_env))
            for name in weather_names
        })
        self.register_buffer("brdf_lut", build_brdf_lut(brdf_lut_size, device=device))
        # One GGX kernel per mip, built once because the convolution is linear
        # in radiance. Mip m stands for roughness m / (n_mips - 1).
        for m in range(1, n_mips):
            res_m = max(resolution >> m, 1)
            rough_m = m / (n_mips - 1)
            idx, w, tail = build_ggx_prefilter(
                res_m,
                rough_m,
                resolution,
                tail_resolution=self.ggx_tail_resolution,
                device=device,
            )
            self.register_buffer(f"_ggx_idx_{m}", idx, persistent=False)
            self.register_buffer(f"_ggx_w_{m}", w, persistent=False)
            self.register_buffer(f"_ggx_tail_{m}", tail, persistent=False)
        self.register_buffer("_base_solid_angles",
                             cubemap_solid_angles(cubemap_dirs(resolution, device)).reshape(-1),
                             persistent=False)
        dirs = cubemap_dirs(resolution, device)
        self.register_buffer("_sh_basis", self._sh_basis_from_dirs(dirs))
        self.register_buffer("_solid_angles", cubemap_solid_angles(dirs))
        # Set by load_external_hdr() to relight with a map we did not learn.
        self.register_buffer("external", None, persistent=False)
        # The background is a direct cube fetch and needs no prefilter kernel,
        # so an external probe can be kept at a resolution the shading path
        # cannot afford. At 64 the sky quantizes to about 1.4 degrees, which
        # diffuse never notices but a viewer looking straight at it does.
        self.register_buffer("external_bg", None, persistent=False)
        self.background_resolution = background_resolution

    # ---------------------------------------------------------------- lighting

    @property
    def device(self) -> torch.device:
        """Current module device, including after a later ``module.to(...)``."""
        return self._log_env.device

    @property
    def radiance(self) -> torch.Tensor:
        """Active raw/external radiance (legacy weather-agnostic API)."""
        return self.radiance_for()


    @property
    def weather_names(self) -> Tuple[str, ...]:
        """Configured learned weather residuals, in stable parameter order."""
        return tuple(self.weather_log_residuals.keys())


    def get_base_parameters(self) -> list:
        """Parameters of the raw learned environment, for staged freezing."""
        return [self._log_env]


    def get_weather_parameters(self, weather: Optional[str] = None) -> list:
        """Return all weather maps, or the single map for ``weather``.

        ``raw`` has no residual parameters by construction.  This helper lets
        staged weather training freeze the raw map without relying on state
        dict name matching.
        """
        if weather is None:
            return list(self.weather_log_residuals.parameters())
        if weather == "raw":
            return []
        if weather not in self.weather_log_residuals:
            raise KeyError(
                f"no environment residual configured for weather {weather!r}; "
                f"available: {list(self.weather_log_residuals.keys())}"
            )
        return [self.weather_log_residuals[weather]]


    def learned_radiance(
        self,
        weather: Optional[str] = None,
        weather_strength=1.0,
    ) -> torch.Tensor:
        """Learned base radiance with an optional weather log-residual.

        This method intentionally ignores an external HDR.  Use
        :meth:`radiance_for` for rendering; it guarantees that an external
        probe bypasses every learned weather residual.
        """
        log_radiance = self._log_env
        if weather is None or weather == "raw":
            return torch.exp(log_radiance.clamp(-LOG_RADIANCE_CLAMP, LOG_RADIANCE_CLAMP))
        if weather not in self.weather_log_residuals:
            raise KeyError(
                f"no environment residual configured for weather {weather!r}; "
                f"available: {list(self.weather_log_residuals.keys())}"
            )
        strength = torch.as_tensor(
            weather_strength, device=log_radiance.device, dtype=log_radiance.dtype
        )
        if strength.numel() != 1:
            raise ValueError(
                "weather_strength must be a scene-level scalar, not a per-frame "
                f"or per-pixel tensor (got shape {tuple(strength.shape)})"
            )
        if not bool(torch.isfinite(strength).all()):
            raise ValueError(f"weather_strength must be finite, got {weather_strength}")
        residual = self.weather_log_residuals[weather]
        combined = log_radiance + strength.reshape(()) * residual
        return torch.exp(combined.clamp(-LOG_RADIANCE_CLAMP, LOG_RADIANCE_CLAMP))


    def radiance_for(
        self,
        weather: Optional[str] = None,
        weather_strength=1.0,
    ) -> torch.Tensor:
        """Radiance used for rendering.

        External HDR relighting is authoritative: once a probe is loaded,
        ``weather`` and ``weather_strength`` are ignored completely.  With no
        external probe this returns ``exp(base + strength * weather_delta)``.
        """
        if self.external is not None:
            return self.external
        return self.learned_radiance(weather, weather_strength)


    @staticmethod
    def _sh_basis_from_dirs(dirs: torch.Tensor) -> torch.Tensor:
        """First 9 real spherical harmonics evaluated per texel. [6, R, R, 9]"""
        d = F.normalize(dirs, dim=-1)
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        return torch.stack([
            torch.full_like(x, 0.282095),
            0.488603 * y, 0.488603 * z, 0.488603 * x,
            1.092548 * x * y, 1.092548 * y * z,
            0.315392 * (3 * z * z - 1),
            1.092548 * x * z, 0.546274 * (x * x - y * y),
        ], -1)

    def sh_coeffs(
        self,
        weather: Optional[str] = None,
        weather_strength=1.0,
    ) -> torch.Tensor:
        """Project the current radiance onto SH-9. [9, 3]"""
        w = (self._sh_basis * self._solid_angles[..., None]).reshape(-1, 9)
        return w.T @ self.radiance_for(weather, weather_strength).reshape(-1, 3)


    def irradiance(
        self,
        normal: torch.Tensor,
        weather: Optional[str] = None,
        weather_strength=1.0,
    ) -> torch.Tensor:
        """Cosine-convolved irradiance E(n) arriving at a surface.

        For a uniform environment of radiance 1 this returns pi, so the
        diffuse term below reduces to the albedo exactly.
        """
        L = self.sh_coeffs(weather, weather_strength)
        # The SH basis lives in the cubemap's frame, so the normal has to be
        # rotated into it exactly like _sample_cube does for specular and sky.
        # Without this diffuse peaks 90 degrees away from the reflections.
        n = F.normalize(normal, dim=-1) @ self.to_opengl.T
        x, y, z = n[..., :1], n[..., 1:2], n[..., 2:3]
        return (
            _C1 * L[8] * (x * x - y * y)
            + _C3 * L[6] * z * z
            + _C4 * L[0]
            - _C5 * L[6]
            + 2 * _C1 * (L[4] * x * y + L[7] * x * z + L[5] * y * z)
            + 2 * _C2 * (L[3] * x + L[1] * y + L[2] * z)
        ).clamp(min=0.0)


    def _mip_chain(
        self,
        weather: Optional[str] = None,
        weather_strength=1.0,
    ) -> list:
        """GGX-prefiltered copies of the environment, one per roughness level.

        Box-filtered mips were standing in for this, which is the wrong kernel
        and blurs across nothing at the cube's face seams. Gathering through a
        precomputed kernel is both the right lobe and differentiable, so the
        environment still learns through it.
        """
        radiance = self.radiance_for(weather, weather_strength)
        flat = radiance.reshape(-1, 3)
        tail_res = self.ggx_tail_resolution
        block = self.resolution // tail_res
        solid_angles = self._base_solid_angles.reshape(
            6, self.resolution, self.resolution
        )
        # Representative radiance for each directional tail bin. Solid-angle
        # weighting is required because cubemap texels near a corner cover less
        # of the sphere than texels near a face centre.
        tail_num = (radiance * solid_angles[..., None]).reshape(
            6, tail_res, block, tail_res, block, 3
        ).sum((2, 4))
        tail_den = solid_angles.reshape(
            6, tail_res, block, tail_res, block
        ).sum((2, 4))[..., None]
        tail_radiance = (tail_num / tail_den.clamp(min=1e-12)).reshape(-1, 3)
        mips = []
        for m in range(1, self.n_mips):
            idx = getattr(self, f"_ggx_idx_{m}")
            w = getattr(self, f"_ggx_w_{m}")
            res_m = max(self.resolution >> m, 1)
            # Keep the sharp core exact and integrate the omitted broad tail
            # directionally. A global mean has the right uniform-light energy
            # but sends light from behind the surface into every reflection.
            tail = getattr(self, f"_ggx_tail_{m}")
            level = (flat[idx.long()] * w[..., None]).sum(1) + tail @ tail_radiance
            mips.append(level.reshape(1, 6, res_m, res_m, 3).contiguous())
        return mips


    def _sample_cube(
        self,
        directions: torch.Tensor,
        roughness: Optional[torch.Tensor] = None,
        weather: Optional[str] = None,
        weather_strength=1.0,
    ):
        prefix = directions.shape[:-1]
        l = F.normalize(directions.reshape(-1, 3), dim=-1) @ self.to_opengl.T
        l = l.reshape(1, 1, -1, 3).contiguous()
        tex = self.radiance_for(weather, weather_strength)[None].contiguous()
        if roughness is None:
            out = dr.texture(tex, l, filter_mode="linear", boundary_mode="cube")
        else:
            # Rougher surfaces gather from a wider lobe -> blurrier mip.
            bias = (roughness.reshape(1, 1, -1) * (self.n_mips - 1)).contiguous()
            out = dr.texture(
                tex, l, mip=self._mip_chain(weather, weather_strength), mip_level_bias=bias,
                filter_mode="linear-mipmap-linear", boundary_mode="cube",
            )
        return out.reshape(*prefix, 3)


    def background(
        self,
        viewdirs: torch.Tensor,
        weather: Optional[str] = None,
        weather_strength=1.0,
    ) -> torch.Tensor:
        """Sky radiance, in linear. Compositing happens before tonemapping."""
        if self.external_bg is None:
            return self._sample_cube(
                viewdirs, weather=weather, weather_strength=weather_strength
            )
        prefix = viewdirs.shape[:-1]
        l = F.normalize(viewdirs.reshape(-1, 3), dim=-1) @ self.to_opengl.T
        out = dr.texture(self.external_bg[None].contiguous(),
                         l.reshape(1, 1, -1, 3).contiguous(),
                         filter_mode="linear", boundary_mode="cube")
        return out.reshape(*prefix, 3)


    def _sample_brdf_lut(self, n_dot_v: torch.Tensor, roughness: torch.Tensor):
        grid = torch.stack([roughness * 2 - 1, n_dot_v * 2 - 1], -1).reshape(1, 1, -1, 2)
        lut = self.brdf_lut.permute(2, 0, 1)[None]                 # [1, 2, S, S]
        ab = F.grid_sample(lut, grid, mode="bilinear", padding_mode="border", align_corners=False)
        # keep the trailing channel axis of n_dot_v so this broadcasts against
        # f0 as (..., 1) rather than inflating it into an N x N grid
        ab = ab[0, :, 0].T.reshape(*n_dot_v.shape[:-1], 2)
        return ab[..., :1], ab[..., 1:]

    def shade(
        self,
        albedo: torch.Tensor,
        metallic: torch.Tensor,
        roughness: torch.Tensor,
        normal: torch.Tensor,
        viewdir: torch.Tensor,
        weather: Optional[str] = None,
        weather_strength=1.0,
    ) -> Dict[str, torch.Tensor]:
        """Split-sum IBL. ``albedo`` is sRGB; returned radiance is linear.

        `viewdir` points from the surface towards the camera.
        """
        n = F.normalize(normal, dim=-1)
        v = F.normalize(viewdir, dim=-1)
        # Gaussian normals have no inherent orientation; face them at the camera.
        n = n * torch.sign((n * v).sum(-1, keepdim=True) + 1e-8)
        n_dot_v = (n * v).sum(-1, keepdim=True).clamp(1e-4, 1.0)

        albedo_lin = srgb_to_linear(albedo)
        f0 = 0.04 * (1 - metallic) + albedo_lin * metallic

        lut_a, lut_b = self._sample_brdf_lut(n_dot_v, roughness)
        specular_weight = f0 * lut_a + lut_b
        # Light taken by the specular lobe is not also available to scatter
        # diffusely. Without this a white dielectric reflected up to 1.19 of
        # what reached it under uniform light, which is energy from nowhere.
        k_d = (1 - specular_weight).clamp(min=0.0) * (1 - metallic)
        diffuse = k_d * albedo_lin / np.pi * self.irradiance(
            n, weather, weather_strength
        )
        specular = self._sample_cube(
            2 * n_dot_v * n - v,
            roughness,
            weather,
            weather_strength,
        ) * specular_weight

        # Linear, so the caller can composite against the sky before tonemapping;
        # blending display-referred values darkens every partially covered pixel.
        return {
            "rgb": diffuse + specular,
            "diffuse": diffuse,
            "specular": specular,
            "normal_oriented": n,
        }


    # ------------------------------------------------------------- relighting

    def load_external_hdr(self, path: str, rotation_deg: float = 0.0, scale: float = 1.0) -> None:
        """Relight from a .hdr/.exr panorama, bypassing the learned map.

        `scale` matches its exposure to the learned lighting; without it a
        panorama authored at a different exposure blows out or goes flat.
        """
        import cv2

        scale = float(scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(f"external HDR scale must be finite and non-negative, got {scale}")

        # imageio routes .hdr through OpenCV's 8-bit path and silently returns
        # uint8, which would flatten the sun into the sky. Read it ourselves.
        raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(f"could not read environment map: {path}")
        raw = np.ascontiguousarray(raw[..., ::-1], dtype=np.float32)   # BGR -> RGB
        if not np.isfinite(raw).all():
            raise ValueError(f"environment map contains non-finite radiance: {path}")
        if raw.max() <= 1.0 + 1e-6:
            logger.warning(
                f"{path} peaks at {raw.max():.3f}: this looks like an LDR image, "
                "not an HDR probe. Relighting will be washed out with no sun."
            )
        equirect = torch.from_numpy(raw).to(
            device=self.device, dtype=self._log_env.dtype
        )
        cube = equirect_to_cubemap(equirect[..., :3], self.resolution, rotation_deg) * scale
        self.external = cube.clamp(min=0.0)
        if self.background_resolution > self.resolution:
            bg = equirect_to_cubemap(equirect[..., :3], self.background_resolution,
                                     rotation_deg, supersample=2) * scale
            self.external_bg = bg.clamp(min=0.0)
        else:
            self.external_bg = None
        logger.info(
            f"Relighting from {path} (rot {rotation_deg} deg, scale {scale}); "
            f"mean radiance {self.external.mean():.3f}"
        )

    def _solid_angle_mean(self, cube: torch.Tensor) -> float:
        """Mean radiance weighted by the solid angle each texel subtends."""
        sa = self._base_solid_angles
        return float(((cube.reshape(-1, 3).mean(-1) * sa).sum() / sa.sum()).detach())

    def scale_external(self, factor: float) -> float:
        """Scale both external-lighting copies by the same linear factor."""
        if self.external is None:
            raise RuntimeError("no external HDR loaded")
        factor = float(factor)
        if not np.isfinite(factor) or factor < 0.0:
            raise ValueError(f"external HDR scale must be finite and non-negative, got {factor}")
        scaled = self.external * factor
        scaled_bg = self.external_bg * factor if self.external_bg is not None else None
        if not torch.isfinite(scaled).all() or (
            scaled_bg is not None and not torch.isfinite(scaled_bg).all()
        ):
            raise ValueError(f"external HDR scale {factor} produces non-finite radiance")
        self.external = scaled
        self.external_bg = scaled_bg
        return factor

    def apply_exposure_stops(self, stops: float) -> float:
        """Adjust the loaded external HDR by photographic exposure stops."""
        stops = float(stops)
        if not np.isfinite(stops):
            raise ValueError(f"exposure stops must be finite, got {stops}")
        try:
            factor = 2.0 ** stops
        except OverflowError as exc:
            raise ValueError(f"exposure stops are outside the supported range: {stops}") from exc
        return self.scale_external(factor)

    def match_exposure(self, target_mean: Optional[float] = None) -> float:
        """Rescale the external map to the learned map's average radiance.

        Weighted by solid angle: cubemap texels do not subtend equal angles, so
        a plain texel mean swung by up to 45% as the same probe was rotated,
        changing the exposure of a relight that should only change direction.
        """
        if self.external is None:
            raise RuntimeError("no external HDR loaded")
        if target_mean is None:
            learned = torch.exp(self._log_env.clamp(-LOG_RADIANCE_CLAMP, LOG_RADIANCE_CLAMP))
            target_mean = self._solid_angle_mean(learned)
        source_mean = self._solid_angle_mean(self.external)
        if not np.isfinite(source_mean) or source_mean <= 0.0:
            raise ValueError(
                "cannot match exposure of an external HDR with zero or non-finite radiance"
            )
        factor = target_mean / source_mean
        return self.scale_external(factor)

    def clear_external(self) -> None:
        self.external = None
        self.external_bg = None

    # ------------------------------------------------------------ boilerplate

    def forward(
        self,
        image_infos: Dict[str, torch.Tensor],
        weather: Optional[str] = None,
        weather_strength=1.0,
    ) -> torch.Tensor:
        """Drop-in for the old EnvLight so `rgb_sky` keeps working."""
        return to_display(self.background(
            image_infos["viewdirs"], weather, weather_strength
        ))


    def get_param_groups(self) -> Dict[str, list]:
        return {self.class_prefix + "all": list(self.parameters())}

    def compute_reg_loss(self, reg_cfg) -> Dict[str, torch.Tensor]:
        """Smoothness, so observed directions inform unobserved neighbours."""
        loss_dict = {}
        tv_cfg = reg_cfg.get("tv", None) if reg_cfg is not None else None
        if tv_cfg is not None:
            e = self._log_env
            tv = (e[:, 1:] - e[:, :-1]).abs().mean() + (e[:, :, 1:] - e[:, :, :-1]).abs().mean()
            loss_dict["envmap_tv"] = tv * tv_cfg.w
        return loss_dict

    def compute_weather_reg_loss(
        self,
        weather: Optional[str],
        reg_cfg,
    ) -> Dict[str, torch.Tensor]:
        """Magnitude/TV regularization for one learned weather residual.

        Configuration lives under ``EnvLight.reg.weather_residual`` with
        optional ``magnitude.w`` and ``tv.w`` entries.  Raw lighting is not a
        residual, so ``None``/``raw`` intentionally returns no losses.  This
        is separate from :meth:`compute_reg_loss` to keep the legacy base-map
        regularizer bit-compatible and to avoid charging every weather map on
        every training branch.
        """
        if weather is None or weather == "raw":
            return {}
        if weather not in self.weather_log_residuals:
            raise KeyError(
                f"no environment residual configured for weather {weather!r}; "
                f"available: {list(self.weather_log_residuals.keys())}"
            )
        weather_cfg = (
            reg_cfg.get("weather_residual", None)
            if reg_cfg is not None else None
        )
        if weather_cfg is None:
            return {}

        residual = self.weather_log_residuals[weather]
        prefix = f"envmap_weather_{weather}"
        loss_dict = {}
        magnitude_cfg = weather_cfg.get("magnitude", None)
        if magnitude_cfg is not None:
            loss_dict[f"{prefix}_magnitude"] = (
                residual.abs().mean() * magnitude_cfg.w
            )
        tv_cfg = weather_cfg.get("tv", None)
        if tv_cfg is not None:
            tv = (
                (residual[:, 1:] - residual[:, :-1]).abs().mean()
                + (residual[:, :, 1:] - residual[:, :, :-1]).abs().mean()
            )
            loss_dict[f"{prefix}_tv"] = tv * tv_cfg.w
        return loss_dict


    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        """Strict-load raw checkpoints into a configured weather module.

        Weather residuals did not exist in older checkpoints.  Their required
        identity initialization is known exactly, so fill only those missing
        keys with zeros; every other missing/unexpected key retains PyTorch's
        normal strict-load error handling.
        """
        missing_weather_keys = [
            f"weather_log_residuals.{name}"
            for name in self.weather_log_residuals
            if f"weather_log_residuals.{name}" not in state_dict
        ]
        if missing_weather_keys:
            patched = state_dict.copy()
            if hasattr(state_dict, "_metadata"):
                patched._metadata = state_dict._metadata
            for key in missing_weather_keys:
                name = key.split(".", 1)[1]
                patched[key] = torch.zeros_like(self.weather_log_residuals[name])
            state_dict = patched
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

