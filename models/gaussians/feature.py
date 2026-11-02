"""
Filename: 3dgs.py

Author: Ziyu Chen (ziyu.sjtu@gmail.com)

Description:
Unofficial implementation of 3DGS based on the work by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
This implementation is modified from the nerfstudio GaussianSplattingModel.

- Original work by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
- Codebase reference: nerfstudio GaussianSplattingModel (https://github.com/nerfstudio-project/nerfstudio/blob/gaussian-splatting/nerfstudio/models/gaussian_splatting.py)

Original paper: https://arxiv.org/abs/2308.04079
"""

from typing import Dict, List, Tuple
from contextlib import contextmanager
from omegaconf import OmegaConf
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter

from models.gaussians.basics import *

logger = logging.getLogger()


class FeatureGaussians(nn.Module):

    def __init__(
            self,
            class_name: str,
            ctrl: OmegaConf,
            reg: OmegaConf = None,
            networks: OmegaConf = None,
            feature: OmegaConf = None,
            raw_material: OmegaConf = None,
            weather_material: OmegaConf = None,
            rgb_residual: OmegaConf = None,
            weathers: list[str] = ["raw"],
            scene_scale: float = 30.,
            scene_origin: torch.Tensor = torch.zeros(3),
            num_train_images: int = 300,
            device: torch.device = torch.device("cuda"),
            **kwargs
    ):
        super().__init__()
        self.class_name = class_name
        self.class_prefix = class_name + "#"
        self.ctrl_cfg = ctrl
        self.reg_cfg = reg
        self.networks_cfg = networks
        self.feature_cfg = feature
        self.scene_scale = scene_scale
        self.scene_origin = scene_origin
        self.num_train_images = num_train_images
        self.step = 0

        self.device = torch.device(device)
        self.ball_gaussians = self.ctrl_cfg.get("ball_gaussians", False)
        self.gaussian_2d = self.ctrl_cfg.get("gaussian_2d", False)
        self.feature_dim = self.feature_cfg.get("feature_dim", 32)
        self.weathers = weathers

        # for evaluation
        self.in_test_set = False

        # init models
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self._means = torch.zeros(1, 3, device=self.device)
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)
        else:
            if self.gaussian_2d:
                self._scales = torch.zeros(1, 2, device=self.device)
            else:
                self._scales = torch.zeros(1, 3, device=self.device)
        self._quats = torch.zeros(1, 4, device=self.device)
        self._opacities = torch.zeros(1, 1, device=self.device)
        self._features = torch.zeros(1, self.feature_dim, device=self.device)

        # A raw_material block selects the reconstructed PBR contract. Existing
        # WeatherCity RGB configurations retain their original decoder and RNG.
        self.pbr_enabled = raw_material is not None
        self.raw_material_cfg = raw_material if raw_material is not None else {}
        self.raw_normal_residual = None
        self._raw_normal_residual_penalty = None
        self.raw_normal_residual_regularization_weight = 0.0
        if self.pbr_enabled:
            self._init_raw_material()
        else:
            self.mlp_color = nn.ModuleDict({w: nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.ReLU(True),
                nn.Linear(self.feature_dim, 3),
                nn.Sigmoid()
            ) for w in self.weathers}).to(self.device)

        self._init_weather_material(weather_material)
        residual_cfg = rgb_residual if rgb_residual is not None else {}
        self.rgb_residual_enabled = residual_cfg.get("enabled", False)
        if type(self.rgb_residual_enabled) is not bool:
            raise ValueError("rgb_residual.enabled must be boolean")
        self.correction_feature_dim = residual_cfg.get("feature_dim", 8)
        if type(self.correction_feature_dim) is not int or self.correction_feature_dim < 1:
            raise ValueError("rgb_residual.feature_dim must be a positive integer")
        if self.rgb_residual_enabled:
            if not self.pbr_enabled:
                raise ValueError("RGB residual features require a raw PBR material model")
            self._initialize_correction_features()

        # self._features_rest = torch.zeros(1, num_sh_bases(self.sh_degree) - 1, 3, device=self.device)
        # self._features_rest = torch.zeros(1, 3, device=self.device)

    @property
    def sh_degree(self):
        return self.ctrl_cfg.sh_degree

    def _initialize_correction_features(self):
        if self.rgb_residual_enabled:
            self._correction_features = Parameter(
                self._means.new_zeros(self.num_points, self.correction_feature_dim))

    def _densify_correction_features(self, splits, dups, nsamps):
        if self.rgb_residual_enabled:
            values = self._correction_features.detach()
            self._correction_features = Parameter(torch.cat(
                [values, values[splits].repeat(nsamps, 1), values[dups]], dim=0))

    def _filter_correction_features(self, keep):
        if self.rgb_residual_enabled:
            self._correction_features = Parameter(self._correction_features[keep].detach())

    def _add_correction_payload(self, gs_dict, mask):
        if self.rgb_residual_enabled:
            gs_dict["_correction_features"] = self._correction_features[mask]

    @contextmanager
    def _fork_isolated_rng(self, seed: int):
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None
                       else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            yield

    def _init_raw_material(self):
        """Restore Base v3's raw decoder and zero-initialized normal head."""
        if self.weathers != ["raw"]:
            raise ValueError("Phase 3 PBR reconstruction supports weathers=['raw']")
        # Construct the legacy PBR MLP first, preserving historical global RNG
        # consumption; spatial and normal heads use isolated streams below.
        self.mlp_material = nn.ModuleDict({"raw": nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(True),
            nn.Linear(self.feature_dim, 5),
        )}).to(self.device)
        spatial_cfg = self.raw_material_cfg.get("spatial_xyz", {})
        self.raw_spatial_xyz_enabled = bool(spatial_cfg.get("enabled", False))
        if spatial_cfg.get("fourier", {}).get("enabled", False):
            raise ValueError("Fourier material inputs are outside Base v3 reconstruction")
        decoder_cfg = self.raw_material_cfg.get("decoder", {})
        if (int(decoder_cfg.get("hidden_dim", self.feature_dim)) != self.feature_dim
                or int(decoder_cfg.get("num_hidden_layers", 1)) != 1
                or decoder_cfg.get("branch_mode", "shared") != "shared"):
            raise ValueError("Base v3 uses a shared decoder with one feature_dim hidden layer")
        activation = str(decoder_cfg.get("activation", "relu"))
        negative_slope = float(decoder_cfg.get("negative_slope", 0.01))
        if activation not in {"relu", "leaky_relu"}:
            raise ValueError(f"Unsupported raw material activation: {activation}")
        if not math.isfinite(negative_slope) or negative_slope <= 0:
            raise ValueError("Material negative_slope must be finite and positive")
        def make_activation():
            return (nn.LeakyReLU(negative_slope, inplace=True)
                    if activation == "leaky_relu" else nn.ReLU(True))
        if self.raw_spatial_xyz_enabled:
            with self._fork_isolated_rng(int(spatial_cfg.get("seed", 12012))):
                self.mlp_material["raw"] = nn.Sequential(
                    nn.Linear(self.feature_dim + 3, self.feature_dim),
                    make_activation(),
                    nn.Linear(self.feature_dim, 5),
                ).to(self.device)
        elif activation == "leaky_relu":
            self.mlp_material["raw"][1] = make_activation()

        normal_cfg = self.raw_material_cfg.get("normal_residual", {})
        self.raw_normal_residual_enabled = bool(normal_cfg.get("enabled", False))
        self.raw_normal_residual_space = str(normal_cfg.get("space", "world_tangent"))
        if self.raw_normal_residual_space not in {"world_tangent", "gaussian_local_tangent"}:
            raise ValueError(f"Unsupported raw normal space: {self.raw_normal_residual_space}")
        max_angle_deg = float(normal_cfg.get("max_angle_deg", 10.0))
        if not 0 < max_angle_deg < 90:
            raise ValueError("Normal residual max_angle_deg must be in (0, 90)")
        self.raw_normal_residual_max_tangent = float(normal_cfg.get(
            "max_tangent", math.tan(math.radians(max_angle_deg))))
        if (not math.isfinite(self.raw_normal_residual_max_tangent)
                or self.raw_normal_residual_max_tangent <= 0):
            raise ValueError("Normal residual max_tangent must be finite and positive")
        self.raw_normal_residual_regularization_weight = float(
            normal_cfg.get("regularization_weight", 0.0))
        if (not math.isfinite(self.raw_normal_residual_regularization_weight)
                or self.raw_normal_residual_regularization_weight < 0):
            raise ValueError("Normal residual regularization must be finite and nonnegative")
        self._raw_normal_residual_active = True
        if self.raw_normal_residual_enabled:
            hidden_dim = int(normal_cfg.get("hidden_dim", self.feature_dim))
            if hidden_dim <= 0:
                raise ValueError("Normal residual hidden_dim must be positive")
            with self._fork_isolated_rng(int(normal_cfg.get("seed", 15015))):
                self.raw_normal_residual = nn.Sequential(
                    nn.Linear(self.feature_dim, hidden_dim),
                    nn.ReLU(True),
                    nn.Linear(hidden_dim, 3),
                ).to(self.device)
                nn.init.zeros_(self.raw_normal_residual[-1].weight)
                nn.init.zeros_(self.raw_normal_residual[-1].bias)

    @staticmethod
    def _activate_material_logits(logits):
        return (torch.sigmoid(logits[..., :3]),
                torch.sigmoid(logits[..., 3:4]),
                torch.sigmoid(logits[..., 4:5]) * 0.99 + 0.01)

    def _raw_material_input(self, means=None):
        if not self.raw_spatial_xyz_enabled:
            return self._features
        means = self._means if means is None else means
        if means.shape != self._means.shape:
            raise ValueError("Spatial XYZ must match Gaussian feature rows")
        origin = torch.as_tensor(self.scene_origin, device=means.device, dtype=means.dtype)
        scale = max(float(self.scene_scale), 1e-6)
        xyz = ((means.detach() - origin) / scale).clamp(-1.0, 1.0)
        return torch.cat([self._features, xyz], dim=-1)

    def _init_weather_material(self, weather_material=None):
        """Attach residual decoders without changing the raw scene topology."""
        cfg = weather_material if weather_material is not None else {}
        self.weather_material_enabled = bool(cfg.get("enabled", False))
        self.weather_material_decoders = nn.ModuleDict()
        if self.weather_material_enabled:
            if not self.pbr_enabled or self.class_name != "Background":
                raise ValueError("Weather material residuals require PBR Background")
            from models.weather_material import WeatherMaterialDecoder
            residual_weathers = list(cfg.get("weathers", []))
            if not residual_weathers or "raw" in residual_weathers:
                raise ValueError("Weather residuals require non-raw weather names")
            if len(set(residual_weathers)) != len(residual_weathers):
                raise ValueError("Weather residual names must be unique")
            for weather in residual_weathers:
                self.weather_material_decoders[weather] = WeatherMaterialDecoder(
                    feature_dim=self.feature_dim,
                    hidden_dim=cfg.get("hidden_dim", self.feature_dim),
                    num_hidden_layers=cfg.get("num_hidden_layers", 1),
                    albedo_delta_cap=cfg.get("albedo_delta_cap", 4.0),
                    roughness_delta_cap=cfg.get("roughness_delta_cap", 5.0),
                )
        self.weather_material_decoders.to(self.device)
        world_up = torch.as_tensor(
            cfg.get("world_up", [0.0, 0.0, 1.0]),
            dtype=torch.float32, device=self.device,
        )
        if world_up.shape != (3,) or not torch.isfinite(world_up).all():
            raise ValueError("world_up must be a finite 3-vector")
        if world_up.norm() <= 0:
            raise ValueError("world_up must be non-zero")
        self.register_buffer("_weather_world_up", world_up / world_up.norm(), persistent=False)
        self._weather_detach_inputs = bool(cfg.get("detach_inputs", True))
        self._weather_material_strength = 1.0
        self._weather_material_outputs = {}

    def _normal_upness(self, preflip_normal):
        """Sign-invariant geometry alignment, before tangent/camera correction."""
        if preflip_normal.shape[-1] != 3:
            raise ValueError("preflip_normal must end in 3 channels")
        world_up = self._weather_world_up.to(dtype=preflip_normal.dtype)
        return torch.abs((preflip_normal * world_up).sum(-1, keepdim=True)).clamp(0.0, 1.0)

    @staticmethod
    def _validate_weather_material_strength(strength):
        strength = float(strength)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("Weather material strength must be finite and nonnegative")
        return strength

    def set_weather_material_strength(self, strength=1.0):
        self._weather_material_strength = self._validate_weather_material_strength(strength)

    def material_outputs(self, weather="raw", preflip_normal=None,
                         weather_strength=None, means=None):
        if not self.pbr_enabled:
            raise RuntimeError("Material decoding requires a raw_material configuration")
        if weather not in self.weathers and weather not in self.weather_material_decoders:
            raise KeyError(f"Weather {weather!r} is not configured")
        self._weather_material_outputs.pop(weather, None)
        base_logits = self.mlp_material["raw"](self._raw_material_input(means))
        base_albedo, base_metallic, base_roughness = self._activate_material_logits(base_logits)
        outputs = {
            "base_albedo": base_albedo, "base_metallic": base_metallic,
            "base_roughness": base_roughness,
            "albedo": base_albedo, "metallic": base_metallic,
            "roughness": base_roughness,
            "delta_albedo_logit": torch.zeros_like(base_albedo),
            "delta_roughness_logit": torch.zeros_like(base_roughness),
        }
        if weather == "raw" or weather not in self.weather_material_decoders:
            return outputs
        strength = self._validate_weather_material_strength(
            self._weather_material_strength if weather_strength is None else weather_strength)
        if strength == 0.0:
            return outputs
        if preflip_normal is None:
            preflip_normal = gaussian_normals(self.get_quats, self.get_scaling)
        upness = self._normal_upness(preflip_normal)
        decoder_inputs = (self._features, base_albedo, base_metallic, base_roughness, upness)
        if self._weather_detach_inputs:
            decoder_inputs = tuple(value.detach() for value in decoder_inputs)
        residual = self.weather_material_decoders[weather](*decoder_inputs)
        # Applying to the original logits retains exact identity at zero delta.
        residual_base_logits = base_logits.detach() if self._weather_detach_inputs else base_logits
        outputs.update({
            "albedo": torch.sigmoid(residual_base_logits[..., :3]
                                    + strength * residual["delta_albedo_logit"]),
            "metallic": base_metallic.detach() if self._weather_detach_inputs else base_metallic,
            "roughness": torch.sigmoid(residual_base_logits[..., 4:5]
                                       + strength * residual["delta_roughness_logit"]) * 0.99 + 0.01,
            **residual, "upness": upness,
        })
        self._weather_material_outputs[weather] = residual
        return outputs

    def materials(self, weather="raw", means=None, preflip_normal=None, weather_strength=None):
        outputs = self.material_outputs(weather, preflip_normal, weather_strength, means)
        return outputs["albedo"], outputs["metallic"], outputs["roughness"]

    def get_weather_material_outputs(self, weather):
        return self._weather_material_outputs.get(weather)

    def pop_weather_material_outputs(self, weather):
        return self._weather_material_outputs.pop(weather, None)

    def clear_weather_material_outputs(self, weather=None):
        if weather is None:
            self._weather_material_outputs.clear()
        else:
            self._weather_material_outputs.pop(weather, None)

    def get_weather_material_parameters(self, weather):
        if weather not in self.weather_material_decoders:
            return tuple()
        return tuple(self.weather_material_decoders[weather].parameters())

    def _apply_raw_normal_residual(self, preflip_normal, quats=None, scales=None):
        if self.raw_normal_residual is None or not self._raw_normal_residual_active:
            self._raw_normal_residual_penalty = None
            return preflip_normal
        raw_delta = self.raw_normal_residual(self._features)
        if self.raw_normal_residual_space == "gaussian_local_tangent":
            if quats is None or scales is None:
                raise ValueError("gaussian_local_tangent requires quats and scales")
            rotation = quat_to_rotmat(quats)
            normal_axis = torch.nn.functional.one_hot(
                scales.argmin(dim=-1), num_classes=3).to(raw_delta)
            bounded = self.raw_normal_residual_max_tangent * raw_delta / torch.sqrt(
                1.0 + raw_delta.square().sum(dim=-1, keepdim=True))
            local_tangent = bounded - normal_axis * (bounded * normal_axis).sum(-1, keepdim=True)
            tangent = torch.bmm(rotation, local_tangent.unsqueeze(-1)).squeeze(-1)
            self._raw_normal_residual_penalty = local_tangent.square().sum(-1).mean()
        else:
            tangent = raw_delta - (raw_delta * preflip_normal).sum(-1, keepdim=True) * preflip_normal
            factor = (self.raw_normal_residual_max_tangent
                      / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.0)
            tangent = tangent * factor
            self._raw_normal_residual_penalty = tangent.square().sum(-1).mean()
        return torch.nn.functional.normalize(preflip_normal + tangent, dim=-1)

    def set_raw_normal_residual_active(self, active):
        previous = self._raw_normal_residual_active
        self._raw_normal_residual_active = bool(active)
        return previous

    def gbuffer(self, weather, quats, scales, means=None, cam=None, weather_strength=None):
        """Pack A3/M1/R1/N3 using final world-space node rotations."""
        if not self.pbr_enabled:
            return self.colors(weather)
        preflip_normal = gaussian_normals(quats, scales)
        albedo, metallic, roughness = self.materials(
            weather, means=means, preflip_normal=preflip_normal,
            weather_strength=weather_strength)
        normals = self._apply_raw_normal_residual(
            preflip_normal, quats=quats, scales=scales)
        if means is not None and cam is not None:
            towards_cam = cam.camtoworlds[:3, 3].reshape(1, 3) - means
            normals = normals * torch.sign((normals * towards_cam).sum(-1, keepdim=True) + 1e-8)
        return torch.cat([albedo, metallic, roughness, normals], dim=-1)

    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        self._means = Parameter(init_means)

        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        if self.ball_gaussians:
            self._scales = Parameter(torch.log(avg_dist.repeat(1, 1)))
        else:
            if self.gaussian_2d:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 2)))
            else:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))

        self._features = Parameter(torch.zeros((self.num_points, self.feature_dim)).to(self.device))
        self._initialize_correction_features()
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))

    def colors(self, weather):
        if self.pbr_enabled:
            return self.materials(weather)[0]
        return self.mlp_color[weather](self._features)
        # if self.sh_degree > 0:
        #     return SH2RGB(self._features_dc)
        # else:
        #     return torch.sigmoid(self._features_dc)

    # @property
    # def shs_0(self):
    #     return self._features_dc
    #
    # @property
    # def shs_rest(self):
    #     return self._features_rest

    @property
    def num_points(self):
        return self._means.shape[0]

    @property
    def get_scaling(self):
        if self.ball_gaussians:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales).repeat(1, 2)
                scaling = torch.cat([scaling, torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales).repeat(1, 3)
        else:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales)
                scaling = torch.cat([scaling[..., :2], torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacities)

    @property
    def get_quats(self):
        return self.quat_act(self._quats)

    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    def preprocess_per_train_step(self, step: int):
        self.step = step
        # Some node classes can be absent in the current frame. Never carry a
        # normal-head graph from the previous frame into their regularizers.
        self._raw_normal_residual_penalty = None

    def postprocess_per_train_step(
            self,
            step: int,
            optimizer: torch.optim.Optimizer,
            radii: torch.Tensor,
            xys_grad: torch.Tensor,
            last_size: int,
    ) -> None:
        self.after_train(radii, xys_grad, last_size)
        if step % self.ctrl_cfg.refine_interval == 0:
            self.refinement_after(step, optimizer)

    def after_train(
            self,
            radii: torch.Tensor,
            xys_grad: torch.Tensor,
            last_size: int,
    ) -> None:
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (radii > 0).flatten()
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
            full_mask[self.filter_mask] = visible_mask

            grads = xys_grad.norm(dim=-1)
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask] = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        decoders = self.mlp_material if self.pbr_enabled else self.mlp_color
        groups = {
            self.class_prefix + "xyz": [self._means],
            self.class_prefix + "feature": [self._features],
            self.class_prefix + "opacity": [self._opacities],
            self.class_prefix + "scaling": [self._scales],
            self.class_prefix + "rotation": [self._quats],
            **{self.class_prefix + "mlp." + w: decoder.parameters()
               for w, decoder in decoders.items()},
        }
        if self.raw_normal_residual is not None:
            groups[self.class_prefix + "mlp.raw.normal_residual"] = self.raw_normal_residual.parameters()
        if self.rgb_residual_enabled:
            groups[self.class_prefix + "correction_feature"] = [self._correction_features]
        groups.update({self.class_prefix + "weather_mlp." + weather: decoder.parameters()
                       for weather, decoder in self.weather_material_decoders.items()})
        return groups

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()

    def refinement_after(self, step, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                    self.step < self.ctrl_cfg.stop_split_at
                    and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None

                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()

                splits = (
                        self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature,
                    split_opacities,
                    split_scales,
                    split_quats,
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                        self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                ) = self.dup_gaussians(dups)

                self._densify_correction_features(splits, dups, nsamps)

                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))

                self._features = Parameter(
                    torch.cat([self._features.detach(), split_feature, dup_feature], dim=0))
                self._opacities = Parameter(
                    torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")

            # reset opacity
            if (
                self.step % reset_interval == self.ctrl_cfg.refine_interval
                and self.opacity_reset_is_active()
            ):
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix + "opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def opacity_reset_is_active(self) -> bool:
        """Whether scheduled opacity resets are still enabled at this step."""
        stop_reset_at = self.ctrl_cfg.get("stop_reset_at", None)
        return stop_reset_at is None or self.step < stop_reset_at

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                    torch.exp(self._scales).max(dim=-1).values >
                    self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features = Parameter(self._features[~culls].detach())
        self._filter_correction_features(~culls)

        self._opacities = Parameter(self._opacities[~culls].detach())

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
                self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
            # torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature = self._features[split_mask].repeat(samps, 1)

        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        return new_means, new_feature, new_opacities, new_scales, new_quats

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature = self._features[dup_mask]

        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        return dup_means, dup_feature, dup_opacities, dup_scales, dup_quats

    def get_gaussians(self, cam: dataclass_camera, weather: str) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask

        # get colors of gaussians
        # colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        # if self.sh_degree > 0:
        #     viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
        #     viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        #     n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
        #     rgbs = spherical_harmonics(n, viewdirs, colors)
        #     rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        # else:
        #     rgbs = torch.sigmoid(colors[:, 0, :])
        ### TODO: add viewpoint encoding

        activated_opacities = self.get_opacity
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        actovated_colors = self.gbuffer(
            weather, activated_rotations, activated_scales, self._means, cam)

        # collect gaussians information
        gs_dict = dict(
            _means=self._means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )

        self._add_correction_payload(gs_dict, filter_mask)
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")

        return gs_dict

    def compute_reg_loss(
        self,
        weather: str
    ):
        loss_dict = {}
        sharp_shape_reg_cfg = self.reg_cfg.get("sharp_shape_reg", None)
        if sharp_shape_reg_cfg is not None:
            w = sharp_shape_reg_cfg.w
            max_gauss_ratio = sharp_shape_reg_cfg.max_gauss_ratio
            step_interval = sharp_shape_reg_cfg.step_interval
            if self.step % step_interval == 0:
                # scale regularization
                scale_exp = self.get_scaling
                scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                                          torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                scale_reg = scale_reg.mean() * w
                loss_dict["sharp_shape_reg"] = scale_reg

        flatten_reg = self.reg_cfg.get("flatten", None)
        if flatten_reg is not None:
            sclaings = self.get_scaling
            min_scale, _ = torch.min(sclaings, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss_dict["flatten"] = flatten_loss * flatten_reg.w

        sparse_reg = self.reg_cfg.get("sparse_reg", None)
        if sparse_reg:
            if (self.cur_radii > 0).sum():
                opacity = torch.sigmoid(self._opacities)
                opacity = opacity.clamp(1e-6, 1 - 1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1 - opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.cur_radii > 0].mean()
                loss_dict["sparse_reg"] = sparse_loss * sparse_reg.w

        # compute the max of scaling
        max_s_square_reg = self.reg_cfg.get("max_s_square_reg", None)
        if max_s_square_reg is not None and not self.ball_gaussians:
            loss_dict["max_s_square"] = torch.mean((self.get_scaling.max(dim=1).values) ** 2) * max_s_square_reg.w
        if (self.raw_normal_residual_regularization_weight > 0
                and self._raw_normal_residual_penalty is not None):
            loss_dict["normal_residual"] = (
                self.raw_normal_residual_regularization_weight * self._raw_normal_residual_penalty)
        return loss_dict

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        allow_new_weather = bool(kwargs.pop("allow_new_weather", False))
        N = state_dict["_means"].shape[0]
        self._means = Parameter(torch.zeros((N,) + self._means.shape[1:], device=self.device))
        self._scales = Parameter(torch.zeros((N,) + self._scales.shape[1:], device=self.device))
        self._quats = Parameter(torch.zeros((N,) + self._quats.shape[1:], device=self.device))
        self._features = Parameter(torch.zeros((N,) + self._features.shape[1:], device=self.device))
        self._opacities = Parameter(torch.zeros((N,) + self._opacities.shape[1:], device=self.device))
        if self.rgb_residual_enabled:
            values = state_dict.get("_correction_features")
            if not isinstance(values, torch.Tensor) or values.shape != (N, self.correction_feature_dim):
                raise ValueError("RGB joint checkpoint requires matching correction feature rows")
            self._initialize_correction_features()

        decoder_prefix = 'mlp_material.' if self.pbr_enabled else 'mlp_color.'
        current_mlp_keys = {key for key in self.state_dict().keys() if key.startswith(decoder_prefix)}

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(decoder_prefix):
                if k in current_mlp_keys:
                    filtered_state_dict[k] = v
            else:
                filtered_state_dict[k] = v

        if allow_new_weather:
            current = self.state_dict()
            for weather in self.weather_material_decoders:
                prefix = f"weather_material_decoders.{weather}."
                expected = {key for key in current if key.startswith(prefix)}
                present = expected.intersection(filtered_state_dict)
                if not present:
                    # Only an entirely new branch may be absent on warm start.
                    filtered_state_dict.update({key: current[key] for key in expected})

        msg = super().load_state_dict(filtered_state_dict, **kwargs)
        return msg

    def export_gaussians_to_ply(self, alpha_thresh: float, weather: str) -> Dict:
        means = self._means
        direct_color = self.colors(weather)

        activated_opacities = self.get_opacity
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }
