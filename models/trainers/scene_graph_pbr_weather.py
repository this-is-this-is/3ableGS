"""Frozen-base Rain/Snow material, environment, and joint training.

Equations and confidence masks are recovered from the October 7 historical
source; stage setup is cross-checked against October 13 and E09/E10.
"""
import logging
import math
import time
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.trainers.scene_graph_pbr import MultiTrainer_pbr

logger = logging.getLogger(__name__)

def _bounded_logit(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Numerically stable logit for pseudo material maps in [0, 1]."""
    if not 0.0 < eps < 0.5:
        raise ValueError(f"logit eps must be in (0, 0.5), got {eps}")
    return torch.logit(x.clamp(eps, 1.0 - eps))


def _roughness_logit(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Map the renderer's [0.01, 1] roughness interval to logit space."""
    normalized = ((x - 0.01) / 0.99).clamp(0.0, 1.0)
    return _bounded_logit(normalized, eps)


def _masked_charbonnier(
    difference: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Channel-balanced robust mean which is exactly zero at zero residual."""
    if eps <= 0.0:
        raise ValueError(f"Charbonnier eps must be positive, got {eps}")
    if mask.ndim == difference.ndim - 1:
        mask = mask[..., None]
    if mask.shape[-1] == 1 and difference.shape[-1] != 1:
        mask = mask.expand_as(difference)
    elif mask.shape != difference.shape:
        mask = mask.expand_as(difference)
    robust = torch.sqrt(difference.square() + eps * eps) - eps
    return (robust * mask).sum() / mask.sum().clamp(min=1.0)


def _erode_binary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Erode an HWC one-channel mask without adding a morphology dependency."""
    if radius <= 0:
        return mask
    invalid = 1.0 - mask.permute(2, 0, 1)[None]
    invalid = F.max_pool2d(invalid, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (1.0 - invalid[0].permute(1, 2, 0)).clamp(0.0, 1.0)


class MultiTrainer_pbr_weather(MultiTrainer_pbr):
    """Frozen-base trainer for a paired raw -> weather material/environment fit.

    Geometry, opacity, Gaussian features, the raw material decoder, the base
    environment, camera poses, and affine correction are immutable.  Only the
    explicitly selected per-Gaussian weather decoder and/or the weather
    environment residual enter Adam.  This makes a model-only base checkpoint
    a weight initializer rather than an accidental continuation of its step,
    optimizer, densification, or camera fitting state.
    """

    VALID_STAGES = {"material", "environment", "joint"}
    VALID_PAIRED_TARGET_MODES = {"gt_relative", "reconstruction"}
    _GAUSSIAN_NAMES = {
        "Background", "RigidNodes", "DeformableNodes", "SMPLNodes",
    }

    def __init__(self, weather_training=None, **kwargs):
        weather_cfg = OmegaConf.create(weather_training or {})
        if not weather_cfg.get("enabled", False):
            raise ValueError("MultiTrainer_pbr_snow requires weather_training.enabled=true")
        weather = str(weather_cfg.get("weather", "snow"))
        if weather == "raw":
            raise ValueError("weather residual training requires a non-raw weather name")
        stage = str(weather_cfg.get("stage", "material"))
        if stage not in self.VALID_STAGES:
            raise ValueError(f"weather stage must be one of {sorted(self.VALID_STAGES)}, got {stage}")
        paired_target_mode = str(
            weather_cfg.get("paired_target_mode", "gt_relative")
        )
        if paired_target_mode not in self.VALID_PAIRED_TARGET_MODES:
            raise ValueError(
                "weather paired_target_mode must be one of "
                f"{sorted(self.VALID_PAIRED_TARGET_MODES)}, got "
                f"{paired_target_mode}"
            )

        model_config = kwargs.get("model_config")
        if model_config is None:
            raise ValueError("model_config is required")
        material_classes = list(weather_cfg.get("material_classes", ["Background"]))
        unknown_classes = sorted(set(material_classes).difference(self._GAUSSIAN_NAMES))
        if unknown_classes:
            raise ValueError(f"unknown weather material classes: {unknown_classes}")

        # Inject the decoder contract before MultiTrainer_weather constructs the
        # Gaussian models.  This keeps the small overlay config independent of
        # whether the chosen base checkpoint used the lightweight or hybrid
        # scene topology.
        decoder_cfg = OmegaConf.to_container(
            weather_cfg.get("material_decoder", {}), resolve=True
        )
        for class_name in self._GAUSSIAN_NAMES:
            if class_name not in model_config:
                continue
            per_model = dict(decoder_cfg)
            per_model.update({
                "enabled": class_name in material_classes,
                "weathers": [weather],
            })
            model_config[class_name]["weather_material"] = OmegaConf.create(per_model)
            # FeatureGaussians recreates its point Parameters while loading a
            # checkpoint.  These flags ensure those replacements start frozen;
            # configure_weather_stage() freezes all remaining node parameters.
            ctrl = model_config[class_name].get("ctrl", OmegaConf.create({}))
            for key in ("freeze_x", "freeze_o", "freeze_q", "freeze_s", "freeze_feature"):
                ctrl[key] = True
            model_config[class_name]["ctrl"] = ctrl

        if "EnvLight" not in model_config:
            raise ValueError("weather PBR training requires model.EnvLight")
        env_params = model_config.EnvLight.get("params", OmegaConf.create({}))
        residuals = list(env_params.get("weather_residuals", []))
        if weather not in residuals:
            residuals.append(weather)
        env_params.weather_residuals = residuals
        model_config.EnvLight.params = env_params

        kwargs["model_config"] = model_config
        # Gaussian base heads remain raw-only; residual decoders are separate.
        kwargs["weathers"] = ["raw"]
        super().__init__(**kwargs)
        self.weather_training_cfg = weather_cfg
        self.weather_cfg = weather_cfg
        self.weather_name = weather
        self.weather_stage = stage
        self.paired_target_mode = paired_target_mode
        self.material_classes = material_classes
        self.base_model_step = 0
        self.configure_weather_stage(stage)
        self.weathers = ["raw", weather]

    @property
    def weather_training_enabled(self) -> bool:
        return True

    @property
    def requires_paired_raw(self) -> bool:
        return self.weather_stage in {"material", "joint"}

    def weather_strength(self, weather: str) -> float:
        if weather == "raw":
            return 0.0
        return float(self.weather_training_cfg.get("weather_strength", 1.0))

    def _weather_material_parameters(self) -> List[torch.nn.Parameter]:
        params = []
        for class_name in self.material_classes:
            model = self.models.get(class_name)
            if model is None:
                continue
            getter = getattr(model, "get_weather_material_parameters", None)
            if getter is None:
                raise RuntimeError(
                    f"{class_name} does not expose get_weather_material_parameters(); "
                    "the WeatherMaterialDecoder integration is missing"
                )
            params.extend(list(getter(self.weather_name)))
        return params

    def _weather_environment_parameters(self) -> List[torch.nn.Parameter]:
        getter = getattr(self.env, "get_weather_parameters", None)
        if getter is None:
            raise RuntimeError(
                "EnvLight does not expose get_weather_parameters(); "
                "the weather environment residual integration is missing"
            )
        return list(getter(self.weather_name))

    @staticmethod
    def _unique_parameters(params: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
        unique, seen = [], set()
        for parameter in params:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            unique.append(parameter)
        return unique

    def configure_weather_stage(self, stage: str = None) -> None:
        """Freeze the base and enable only the parameters selected by the stage."""
        stage = self.weather_stage if stage is None else str(stage)
        if stage not in self.VALID_STAGES:
            raise ValueError(f"weather stage must be one of {sorted(self.VALID_STAGES)}, got {stage}")
        self.weather_stage = stage

        for model in self.models.values():
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        material_params = self._weather_material_parameters()
        env_params = self._weather_environment_parameters()
        selected = []
        if stage in {"material", "joint"}:
            if not material_params:
                raise RuntimeError(
                    f"stage={stage} selected no {self.weather_name} material parameters"
                )
            selected.extend(material_params)
        if stage in {"environment", "joint"}:
            if not env_params:
                raise RuntimeError(
                    f"stage={stage} selected no {self.weather_name} environment parameters"
                )
            selected.extend(env_params)
        for parameter in self._unique_parameters(selected):
            parameter.requires_grad_(True)

        self._stage_material_params = self._unique_parameters(material_params)
        self._stage_env_params = self._unique_parameters(env_params)
        logger.info(
            "Weather stage %s: %d material and %d environment parameter tensors selected",
            stage,
            len(self._stage_material_params) if stage in {"material", "joint"} else 0,
            len(self._stage_env_params) if stage in {"environment", "joint"} else 0,
        )

    def initialize_optimizer(self) -> None:
        """Build Adam solely from weather parameters; base config groups are ignored."""
        self.configure_weather_stage(self.weather_stage)
        cfg = self.weather_training_cfg
        groups = []
        self.param_groups = {}
        if self.weather_stage in {"material", "joint"}:
            name = f"WeatherMaterial#{self.weather_name}"
            params = self._stage_material_params
            self.param_groups[name] = params
            groups.append({
                "params": params,
                "name": name,
                "lr": float(cfg.get("material_lr", 1e-4)),
                "weight_decay": float(cfg.get("weight_decay", 0.0)),
            })
        if self.weather_stage in {"environment", "joint"}:
            name = f"WeatherEnv#{self.weather_name}"
            params = self._stage_env_params
            self.param_groups[name] = params
            groups.append({
                "params": params,
                "name": name,
                "lr": float(cfg.get("environment_lr", 1e-3)),
                "weight_decay": float(cfg.get("weight_decay", 0.0)),
            })
        if not groups:
            raise RuntimeError(f"weather stage {self.weather_stage} has no trainable parameters")
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.lr_schedulers = {}
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self.optim_general.get("use_grad_scaler", False)
        )

    def preprocess_per_train_step(self, step: int) -> None:
        """Advance weather optimization without changing frozen base behavior."""
        self.step = step
        for class_name in self.gaussian_classes:
            # Preserve the trained deformation schedule without refinement hooks.
            self.models[class_name].step = self.base_model_step
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            self.tic = time.time()

    def postprocess_per_train_step(self, step: int) -> None:
        """Deliberately skip Gaussian split/cull/reset for a frozen base scene."""
        self.step = step
        if self.viewer is not None:
            num_rays = self.render_cfg.batch_size * self.info["width"] * self.info["height"]
            self.viewer.lock.release()
            steps_per_second = 1.0 / (time.time() - self.tic)
            self.viewer.state.num_train_rays_per_sec = num_rays * steps_per_second
            self.viewer.update(step, num_rays)

    def render_raw_reference(
        self,
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Render paired raw material maps without building an autograd graph."""
        was_training = self.training
        self._suppress_eval_layers = True
        try:
            self.set_eval()
            with torch.no_grad():
                outputs = self(image_infos, camera_infos, "raw")
                # Paired material supervision only consumes these maps.  Drop
                # shaded RGB/lobes, normals, sky, and depth before the weather
                # forward so a full-resolution raw reference does not retain
                # hundreds of MB of otherwise unused tensors.
                outputs = {
                    key: outputs[key]
                    for key in ("albedo", "roughness", "opacity")
                }
        finally:
            self._suppress_eval_layers = False
            if was_training:
                self.set_train()
        return outputs

    def supports_weather(self, weather):
        return weather in {"raw", self.weather_name}

    @staticmethod
    def _validate_render_weather_strength(name, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite scene-level scalar")
        return value

    def set_render_weather_strengths(self, material=None, environment=None):
        self._render_material_weather_strength = (
            None if material is None else self._validate_render_weather_strength("material", material)
        )
        self._render_environment_weather_strength = (
            None if environment is None else self._validate_render_weather_strength("environment", environment)
        )

    def material_weather_strength(self, weather):
        if weather == "raw":
            return 0.0
        value = getattr(self, "_render_material_weather_strength", None)
        return self.weather_strength(weather) if value is None else value

    def environment_weather_strength(self, weather):
        if weather == "raw":
            return 0.0
        value = getattr(self, "_render_environment_weather_strength", None)
        return self.weather_strength(weather) if value is None else value

    def prepare_weather_render(self, weather):
        for name in self.gaussian_classes:
            model = self.models[name]
            model.step = self.base_model_step
            setter = getattr(model, "set_weather_material_strength", None)
            if setter is not None:
                setter(self.material_weather_strength(weather))

    def gaussian_weather(self, class_name, weather):
        # Only Background carries a material edit. Dynamic geometry/material
        # remains raw while shared weather lighting still affects its shading.
        return weather if class_name in self.material_classes else "raw"

    def state_dict(self, only_model=True):
        if not only_model:
            raise ValueError("Use the weather training-state utility for optimizer/RNG snapshots")
        state = super().state_dict(only_model=True)
        state["weather_training_state"] = {
            "version": 1, "weather": self.weather_name,
            "base_model_step": int(self.base_model_step),
        }
        return state

    def load_state_dict(self, state_dict, load_only_model=True, strict=True):
        if not strict:
            raise ValueError("Weather checkpoints require strict model loading")
        state = dict(state_dict)
        if load_only_model:
            state.pop("weather_optimizer_state", None)
        # Existing Rigid/SMPL loaders pop their data-dependent metadata.
        # Copy the per-model mappings so loading does not consume the caller's
        # checkpoint, which may also be needed for another stage or validation.
        state["models"] = {name: dict(values) for name, values in state["models"].items()}
        required_models = (set(self.models) - set(self.gaussian_classes)) | {"Background"}
        absent_models = required_models.difference(state["models"])
        if absent_models:
            raise ValueError(f"Checkpoint is missing required base models: {sorted(absent_models)}")
        weather_state = state.pop("weather_training_state", None)
        if weather_state is not None:
            if weather_state.get("version") != 1:
                raise ValueError("Unsupported weather_training_state version")
            if weather_state.get("weather") != self.weather_name:
                raise ValueError(
                    f"weather checkpoint/config mismatch: {weather_state.get('weather')} != {self.weather_name}"
                )
            base_step = int(weather_state["base_model_step"])
            # A weather checkpoint must contain every residual; raw initializers
            # alone may omit newly constructed zero residual parameters.
            for name, model in self.models.items():
                if name not in state["models"]:
                    continue
                # Dynamic node state_dict wrappers require point_ids/template
                # fields restored only during load. Inspect registered state
                # directly before that data-dependent initialization exists.
                expected = {k for k in torch.nn.Module.state_dict(model)
                            if k.startswith(("weather_material_decoders.", "weather_log_residuals."))}
                missing = expected.difference(state["models"][name])
                if missing:
                    raise ValueError(f"Weather checkpoint is missing residual weights in {name}: {sorted(missing)}")
        else:
            base_step = int(state["step"])
            saved_models = {name: dict(values) for name, values in state["models"].items()}
            for name, model in self.models.items():
                if name not in saved_models:
                    continue
                current = torch.nn.Module.state_dict(model)
                for prefix in (
                    f"weather_material_decoders.{self.weather_name}.",
                    f"weather_log_residuals.{self.weather_name}",
                ):
                    expected = {k for k in current if k.startswith(prefix)}
                    present = expected.intersection(saved_models[name])
                    if present and present != expected:
                        raise ValueError(f"Partially missing weather branch in {name}: {prefix}")
                    if not present:
                        saved_models[name].update({k: current[k] for k in expected})
            state["models"] = saved_models
        result = super().load_state_dict(state, load_only_model=load_only_model, strict=True)
        self.base_model_step = base_step
        for model in self.models.values():
            if hasattr(model, "step"):
                model.step = self.base_model_step
        self.configure_weather_stage(self.weather_stage)
        return result

    def init_from_base_state(self, state_dict):
        """Start a new stage from weights, retaining frozen base behavior."""
        state = {k: v for k, v in state_dict.items()
                 if k not in {"optimizer", "training_state", "weather_optimizer_state"}}
        self.load_state_dict(state, load_only_model=True, strict=True)
        self.step = 0
        for attr in ("optimizer", "grad_scaler", "param_groups", "lr_schedulers"):
            if hasattr(self, attr):
                delattr(self, attr)
        for model in self.models.values():
            for attr in ("xys_grad_norm", "vis_counts", "max_2Dsize", "t_grad_accum"):
                if hasattr(model, attr):
                    setattr(model, attr, None)
        self.configure_weather_stage(self.weather_stage)

    def init_from_base_checkpoint(self, ckpt_path):
        self.init_from_base_state(torch.load(ckpt_path, map_location=self.device, weights_only=False))

    def compute_losses(self, outputs, image_infos, cam_infos, weather="raw"):
        raise RuntimeError("Weather stages require compute_weather_losses with the paired raw reference")

    def _weather_material_mask(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        raw_infos: Dict[str, torch.Tensor],
        weather_infos: Dict[str, torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        cfg = self.weather_training_cfg.get("mask", {})
        base = torch.ones_like(raw_infos["sky_masks"])
        base = base * (1.0 - raw_infos["sky_masks"])
        if "egocar_masks" in raw_infos:
            base = base * (1.0 - raw_infos["egocar_masks"])
        if cfg.get("exclude_dynamic", True) and "dynamic_masks" in raw_infos:
            base = base * (1.0 - raw_infos["dynamic_masks"])
        mask = (self._crop_and_resize(base[..., None], target_hw) > 0.99).float()

        min_opacity = float(cfg.get("min_opacity", 0.5))
        opacity = self._crop_and_resize(raw_outputs["opacity"], target_hw)
        mask = mask * (opacity >= min_opacity).float()

        normal_cos_min = cfg.get("normal_cos_min", None)
        if normal_cos_min is not None:
            raw_n = F.normalize(self._resize_down(raw_infos["gbuf_normal"], target_hw), dim=-1)
            weather_n = F.normalize(
                self._resize_down(weather_infos["gbuf_normal"], target_hw), dim=-1
            )
            pair_cos = (raw_n * weather_n).sum(-1, keepdim=True)
            mask = mask * (pair_cos >= float(normal_cos_min)).float()

        raw_albedo = self._resize_down(raw_infos["gbuf_basecolor"], target_hw)
        pred_albedo = self._crop_and_resize(raw_outputs["albedo"], target_hw)
        max_a_error = cfg.get("base_albedo_error_max", None)
        if max_a_error is not None:
            error = (pred_albedo - raw_albedo).abs().mean(-1, keepdim=True)
            mask = mask * (error <= float(max_a_error)).float()

        raw_roughness = self._resize_down(raw_infos["gbuf_roughness"], target_hw)
        pred_roughness = self._crop_and_resize(raw_outputs["roughness"], target_hw)
        max_r_error = cfg.get("base_roughness_error_max", None)
        if max_r_error is not None:
            error = (pred_roughness - raw_roughness).abs()
            mask = mask * (error <= float(max_r_error)).float()

        return _erode_binary_mask(mask, int(cfg.get("erode_px", 1)))

    def _pop_delta_regularizer(self, weather: str) -> Tuple[torch.Tensor, torch.Tensor]:
        albedo_terms, roughness_terms = [], []
        albedo_saturated, roughness_saturated = [], []
        albedo_elements = roughness_elements = 0
        decoder_cfg = self.weather_training_cfg.get("material_decoder", {})
        albedo_cap = float(decoder_cfg.get("albedo_delta_cap", 4.0))
        roughness_cap = float(decoder_cfg.get("roughness_delta_cap", 5.0))
        for class_name in self.material_classes:
            model = self.models.get(class_name)
            if model is None:
                continue
            pop = getattr(model, "pop_weather_material_outputs", None)
            if pop is not None:
                values = pop(weather)
            else:
                get = getattr(model, "get_weather_material_outputs", None)
                values = get(weather) if get is not None else None
                clear = getattr(model, "clear_weather_material_outputs", None)
                if clear is not None:
                    clear(weather)
            if not values:
                continue
            delta_albedo = values["delta_albedo_logit"]
            delta_roughness = values["delta_roughness_logit"]
            albedo_terms.append(delta_albedo.abs().mean())
            roughness_terms.append(delta_roughness.abs().mean())
            albedo_saturated.append(
                (delta_albedo.detach().abs() > 0.95 * albedo_cap).sum()
            )
            roughness_saturated.append(
                (delta_roughness.detach().abs() > 0.95 * roughness_cap).sum()
            )
            albedo_elements += delta_albedo.numel()
            roughness_elements += delta_roughness.numel()
        self._last_albedo_cap_saturation = (
            torch.stack(albedo_saturated).sum().float() / albedo_elements
            if albedo_saturated else None
        )
        self._last_roughness_cap_saturation = (
            torch.stack(roughness_saturated).sum().float() / roughness_elements
            if roughness_saturated else None
        )
        zero = torch.zeros((), device=self.device)
        return (
            torch.stack(albedo_terms).mean() if albedo_terms else zero,
            torch.stack(roughness_terms).mean() if roughness_terms else zero,
        )

    def _material_losses(
        self,
        weather_outputs: Dict[str, torch.Tensor],
        raw_outputs: Dict[str, torch.Tensor],
        weather_infos: Dict[str, torch.Tensor],
        raw_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.weather_training_cfg
        loss_cfg = cfg.get("losses", {})
        target = weather_infos["gbuf_basecolor"]
        gbuf_h, gbuf_w = target.shape[:2]
        render_w = weather_outputs["albedo"].shape[1]
        width = min(render_w, gbuf_w)
        target_hw = (round(width * gbuf_h / gbuf_w), width)
        mask = self._weather_material_mask(raw_outputs, raw_infos, weather_infos, target_hw)
        valid_pixels = int(mask.sum().detach().item())
        valid_fraction = mask.mean().detach()
        min_valid_pixels = int(cfg.get("mask", {}).get("min_valid_pixels", 64))
        if min_valid_pixels < 1:
            raise ValueError(
                f"weather mask.min_valid_pixels must be positive, got {min_valid_pixels}"
            )
        if valid_pixels < min_valid_pixels:
            raise RuntimeError(
                "weather material mask has too few valid pixels "
                f"({valid_pixels} < {min_valid_pixels}, fraction="
                f"{float(valid_fraction):.6f}); relax confidence thresholds or "
                "verify paired raw/weather G-buffer alignment"
            )
        self._last_weather_material_valid_fraction = valid_fraction

        eps = float(cfg.get("logit_eps", 1e-4))
        robust_eps = float(cfg.get("charbonnier_eps", 1e-3))
        cap_a = float(cfg.get("target_albedo_delta_cap", 4.0))
        cap_r = float(cfg.get("target_roughness_delta_cap", 5.0))

        pred_raw_a = self._crop_and_resize(raw_outputs["albedo"], target_hw)
        pred_weather_a = self._crop_and_resize(weather_outputs["albedo"], target_hw)
        gt_raw_a = self._resize_down(raw_infos["gbuf_basecolor"], target_hw)
        gt_weather_a = self._resize_down(weather_infos["gbuf_basecolor"], target_hw)
        pred_delta_a = _bounded_logit(pred_weather_a, eps) - _bounded_logit(pred_raw_a, eps)
        paired_target_mode = str(cfg.get("paired_target_mode", "gt_relative"))
        if paired_target_mode == "gt_relative":
            target_raw_a = gt_raw_a
        elif paired_target_mode == "reconstruction":
            # Compensate for the frozen raw model's residual error so paired and
            # absolute supervision have the same optimum: pred_weather == GT.
            target_raw_a = pred_raw_a.detach()
        else:
            raise ValueError(
                "weather paired_target_mode must be one of "
                f"{sorted(self.VALID_PAIRED_TARGET_MODES)}, got "
                f"{paired_target_mode}"
            )
        gt_delta_a = (
            _bounded_logit(gt_weather_a, eps) - _bounded_logit(target_raw_a, eps)
        ).clamp(-cap_a, cap_a)

        pred_raw_r = self._crop_and_resize(raw_outputs["roughness"], target_hw)
        pred_weather_r = self._crop_and_resize(weather_outputs["roughness"], target_hw)
        gt_raw_r = self._resize_down(raw_infos["gbuf_roughness"], target_hw)
        gt_weather_r = self._resize_down(weather_infos["gbuf_roughness"], target_hw)
        pred_delta_r = _roughness_logit(pred_weather_r, eps) - _roughness_logit(pred_raw_r, eps)
        target_raw_r = (
            gt_raw_r
            if paired_target_mode == "gt_relative"
            else pred_raw_r.detach()
        )
        gt_delta_r = (
            _roughness_logit(gt_weather_r, eps) - _roughness_logit(target_raw_r, eps)
        ).clamp(-cap_r, cap_r)

        losses = {}
        paired_a = loss_cfg.get("paired_albedo", None)
        if paired_a is not None and paired_a.get("w", 0.0) > 0:
            losses["weather_paired_albedo_loss"] = float(paired_a.w) * _masked_charbonnier(
                pred_delta_a - gt_delta_a, mask, robust_eps
            )
        paired_r = loss_cfg.get("paired_roughness", None)
        if paired_r is not None and paired_r.get("w", 0.0) > 0:
            losses["weather_paired_roughness_loss"] = float(paired_r.w) * _masked_charbonnier(
                pred_delta_r - gt_delta_r, mask, robust_eps
            )
        absolute_a = loss_cfg.get("absolute_albedo", None)
        if absolute_a is not None and absolute_a.get("w", 0.0) > 0:
            losses["weather_absolute_albedo_loss"] = float(absolute_a.w) * _masked_charbonnier(
                pred_weather_a - gt_weather_a, mask, robust_eps
            )
        absolute_r = loss_cfg.get("absolute_roughness", None)
        if absolute_r is not None and absolute_r.get("w", 0.0) > 0:
            losses["weather_absolute_roughness_loss"] = float(absolute_r.w) * _masked_charbonnier(
                pred_weather_r - gt_weather_r, mask, robust_eps
            )
        return losses

    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Report mask coverage beside the ordinary RGB metric, not as a loss."""
        metrics = super().compute_metrics(outputs, image_infos)
        valid_fraction = getattr(
            self, "_last_weather_material_valid_fraction", None
        )
        if valid_fraction is not None:
            metrics["weather_material_valid_fraction"] = valid_fraction.to(
                device=outputs["rgb"].device
            )
        for name in ("albedo_cap_saturation", "roughness_cap_saturation"):
            value = getattr(self, f"_last_{name}", None)
            if value is not None:
                metrics[name] = value.to(device=outputs["rgb"].device)
        return metrics

    def _rgb_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.weather_training_cfg.get("losses", {})
        valid = torch.ones_like(image_infos["sky_masks"])
        if "egocar_masks" in image_infos:
            valid = valid * (1.0 - image_infos["egocar_masks"])
        surface = valid * (1.0 - image_infos["sky_masks"])
        mask_cfg = self.weather_training_cfg.get("mask", {})
        if mask_cfg.get("exclude_dynamic", True) and "dynamic_masks" in image_infos:
            surface = surface * (1.0 - image_infos["dynamic_masks"])
        sky = valid * image_infos["sky_masks"]
        gt = image_infos["pixels"]
        pred = outputs["rgb"]
        robust_eps = float(self.weather_training_cfg.get("charbonnier_eps", 1e-3))
        losses = {}
        surface_cfg = cfg.get("rgb_surface", cfg.get("rgb", None))
        if surface_cfg is not None and surface_cfg.get("w", 0.0) > 0:
            losses["weather_rgb_surface_loss"] = float(surface_cfg.w) * _masked_charbonnier(
                pred - gt, surface[..., None], robust_eps
            )
        sky_cfg = cfg.get("rgb_sky", None)
        if sky_cfg is not None and sky_cfg.get("w", 0.0) > 0:
            losses["weather_rgb_sky_loss"] = float(sky_cfg.w) * _masked_charbonnier(
                pred - gt, sky[..., None], robust_eps
            )
        ssim_cfg = cfg.get("ssim", None)
        if ssim_cfg is not None and ssim_cfg.get("w", 0.0) > 0:
            # Keep SSIM optional and global: masking images creates an artificial
            # high-contrast contour at every sky boundary.  Its default weight
            # is deliberately much lower than the surface reconstruction term.
            gt_ssim = gt * valid[..., None]
            pred_ssim = pred * valid[..., None]
            similarity = self.ssim(
                gt_ssim.permute(2, 0, 1)[None], pred_ssim.permute(2, 0, 1)[None]
            )
            losses["weather_ssim_loss"] = float(ssim_cfg.w) * (1.0 - similarity)
        return losses

    def compute_weather_losses(
        self,
        weather_outputs: Dict[str, torch.Tensor],
        raw_outputs: Dict[str, torch.Tensor],
        weather_infos: Dict[str, torch.Tensor],
        raw_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
        weather: str,
    ) -> Dict[str, torch.Tensor]:
        """Weather-only objective; metallic/normal/depth/base losses stay off."""
        del cam_infos  # kept in the public contract for future geometric confidence.
        if weather != self.weather_name:
            raise ValueError(f"expected weather={self.weather_name}, got {weather}")
        losses = {}
        if self.weather_stage in {"material", "joint"}:
            if raw_outputs is None:
                raise ValueError(f"stage={self.weather_stage} requires a paired raw render")
            losses.update(self._material_losses(
                weather_outputs, raw_outputs, weather_infos, raw_infos
            ))
        if self.weather_stage in {"environment", "joint"}:
            losses.update(self._rgb_losses(weather_outputs, weather_infos))
            reg = getattr(self.env, "compute_weather_reg_loss", None)
            if reg is None:
                raise RuntimeError("EnvLight is missing compute_weather_reg_loss")
            env_cfg = self.model_config.EnvLight.get("reg", None)
            losses.update(reg(weather, env_cfg))

        delta_a, delta_r = self._pop_delta_regularizer(weather)
        loss_cfg = self.weather_training_cfg.get("losses", {})
        delta_a_cfg = loss_cfg.get("delta_albedo_reg", None)
        if delta_a_cfg is not None and delta_a_cfg.get("w", 0.0) > 0:
            losses["weather_delta_albedo_reg"] = float(delta_a_cfg.w) * delta_a
        delta_r_cfg = loss_cfg.get("delta_roughness_reg", None)
        if delta_r_cfg is not None and delta_r_cfg.get("w", 0.0) > 0:
            losses["weather_delta_roughness_reg"] = float(delta_r_cfg.w) * delta_r

        differentiable = [value for value in losses.values() if value.requires_grad]
        if not differentiable:
            raise RuntimeError(
                f"weather stage {self.weather_stage} produced no differentiable loss; "
                "check its loss weights and enabled modules"
            )
        return losses


# Historical public name retained for saved configuration compatibility.
MultiTrainer_pbr_snow = MultiTrainer_pbr_weather
