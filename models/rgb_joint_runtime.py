"""Opt-in display RGB residual training for the native Base-v4 PBR trainer.

This module keeps the original Base-v4 defaults and all scene modules intact.
The residual is a display-space reconstruction term, bypassed for external HDR.
"""
import math

import torch
import torch.nn.functional as F


DEFAULTS = {
    "enabled": True, "feature_dim": 8, "hidden_dim": 32,
    "warmup_camera_updates": 2000, "ramp_camera_updates": 1000,
    "pure_pbr_lambda": .3, "gain_lambda": .001, "additive_lambda": .001,
    "max_log_gain": .3, "max_additive": .1, "lr": .001,
}


def validate_rgb_joint_config(config):
    cfg = {**DEFAULTS, **dict(config)}
    if set(cfg) - set(DEFAULTS):
        raise ValueError(f"Unknown RGB joint options: {sorted(set(cfg) - set(DEFAULTS))}")
    if type(cfg["enabled"]) is not bool:
        raise ValueError("rgb_joint.enabled must be boolean")
    for key in ("feature_dim", "hidden_dim", "ramp_camera_updates"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"rgb_joint.{key} must be a positive integer")
    if type(cfg["warmup_camera_updates"]) is not int or cfg["warmup_camera_updates"] < 0:
        raise ValueError("rgb_joint.warmup_camera_updates must be a nonnegative integer")
    for key in ("pure_pbr_lambda", "gain_lambda", "additive_lambda", "max_log_gain", "max_additive", "lr"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"rgb_joint.{key} must be finite and nonnegative")
    if cfg["lr"] <= 0:
        raise ValueError("rgb_joint.lr must be positive")
    return cfg


class RGBJointMixin:
    def configure_rgb_joint(self, config=None):
        self.rgb_joint_config = None if config is None else validate_rgb_joint_config(config)
        if self.rgb_joint_config is not None and not self.rgb_joint_config["enabled"]:
            self.rgb_joint_config = None
        self.rgb_residual_head = None
        self._rgb_joint_pending_camera_step = None
        if self.rgb_joint_config is None:
            return
        if (getattr(self, "_layered_gbuffer", None) is not None
                or getattr(self, "gbuffer_attention_config", None) is not None
                or getattr(self, "_temporal_material", None) is not None):
            raise ValueError("RGB joint training requires the ordinary Base-v4 PBR renderer")
        cfg = self.rgb_joint_config
        for name in self.gaussian_classes:
            model = self.models[name]
            if (not getattr(model, "rgb_residual_enabled", False)
                    or getattr(model, "correction_feature_dim", None) != cfg["feature_dim"]):
                raise ValueError(f"{name} must enable matching Gaussian rgb_residual features")
        from models.rgb_residual import RGBResidual
        self.rgb_residual_head = RGBResidual(
            feature_dim=cfg["feature_dim"], hidden_dim=cfg["hidden_dim"],
            max_log_gain=cfg["max_log_gain"], max_additive=cfg["max_additive"],
        ).to(self.device)
        self.register_buffer("rgb_joint_camera_updates", torch.zeros((), dtype=torch.int64, device=self.device))

    def rgb_residual_strength(self):
        cfg = getattr(self, "rgb_joint_config", None)
        if cfg is None:
            return 0.
        return min(1., max(0., (int(self.rgb_joint_camera_updates) - cfg["warmup_camera_updates"])
                           / cfg["ramp_camera_updates"]))

    def initialize_optimizer(self):
        super().initialize_optimizer()
        head = getattr(self, "rgb_residual_head", None)
        if head is not None:
            parameters = list(head.parameters())
            self.param_groups["RGBResidual#all"] = parameters
            self.optimizer.add_param_group({"params": parameters, "name": "RGBResidual#all",
                                            "lr": self.rgb_joint_config["lr"], "eps": 1e-15})

    def preprocess_per_train_step(self, step):
        super().preprocess_per_train_step(step)
        self._rgb_joint_pending_camera_step = None

    def complete_camera_update(self, step=None):
        """Advance once after a successful optimizer/density update, never on render."""
        if getattr(self, "rgb_joint_config", None) is None:
            return
        step = self.step if step is None else step
        if self._rgb_joint_pending_camera_step == step:
            self.rgb_joint_camera_updates.add_(1)
            self._rgb_joint_pending_camera_step = None

    def collect_gaussians(self, cam, weather, image_ids):
        if getattr(self, "rgb_joint_config", None) is None:
            return super().collect_gaussians(cam, weather, image_ids)
        from models.gaussians.basics import dataclass_gs
        keys = ("_means", "_scales", "_quats", "_rgbs", "_opacities", "_correction_features")
        rows = {name: [] for name in keys}
        labels = []
        for name, label in self.gaussian_classes.items():
            values = self.models[name].get_gaussians(cam, self.gaussian_weather(name, weather))
            if values is None:
                continue
            for key in keys:
                if key not in values:
                    raise ValueError(f"RGB joint Gaussian output missing {name}.{key}")
                rows[key].append(values[key])
            labels.append(torch.full((len(values["_means"]),), label, device=self.device))
        if not labels:
            raise ValueError("RGB joint render contains no Gaussian population")
        joined = {key: torch.cat(value, dim=0) for key, value in rows.items()}
        self.pts_labels = torch.cat(labels)
        if self.render_dynamic_mask:
            self.dynamic_pts_mask = (self.pts_labels != 0).float()
        feature = joined.pop("_correction_features")
        return dataclass_gs(**joined, detach_keys=[], extras={"correction_features": feature})

    def apply_rgb_joint(self, outputs, gbuffer, viewdirs, weather="raw"):
        cfg = getattr(self, "rgb_joint_config", None)
        if cfg is None:
            return
        pure = outputs["rgb"]
        strength = self.rgb_residual_strength()
        active = strength > 0 and weather == "raw" and getattr(self.env, "external", None) is None
        features = outputs.pop("_rgb_correction_features")
        if active:
            maps = self.rgb_residual_head(
                features, outputs["normal"], gbuffer["roughness"], outputs["depth"],
                outputs["opacity"], -F.normalize(viewdirs, dim=-1), pure, strength=strength)
        else:
            maps = {"rgb_hybrid": pure, "rgb_gain": torch.ones_like(pure),
                    "rgb_log_gain": torch.zeros_like(pure), "rgb_additive": torch.zeros_like(pure),
                    "rgb_residual": torch.zeros_like(pure)}
        outputs.update(maps)
        outputs["rgb_pure_pbr"] = pure
        outputs["rgb"] = maps["rgb_hybrid"]
        outputs["rgb_residual_strength"] = pure.new_tensor(strength)
        outputs["rgb_residual_active"] = torch.tensor(active, device=pure.device)

    def rgb_joint_losses(self, outputs, image_infos):
        cfg = getattr(self, "rgb_joint_config", None)
        if cfg is None or "rgb_pure_pbr" not in outputs:
            return {}
        if self.training:
            self._rgb_joint_pending_camera_step = self.step
        if not bool(outputs["rgb_residual_active"]):
            return {}
        strength = self.rgb_residual_strength()
        valid = 1. - image_infos.get("egocar_masks", torch.zeros_like(image_infos["pixels"][..., 0]))
        valid = valid.detach().to(outputs["rgb_pure_pbr"])
        target = image_infos["pixels"] * valid[..., None]
        pure = outputs["rgb_pure_pbr"] * valid[..., None]
        rgb_loss = self.losses_dict.rgb.w * (pure - target).abs().mean()
        rgb_loss = rgb_loss + self.losses_dict.ssim.w * (
            1 - self.ssim(pure.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]))
        weight = outputs["opacity"].detach().clamp(0, 1) * valid[..., None]
        denom = weight.sum().clamp_min(1)
        losses = {"rgb_joint_pure_pbr_loss": cfg["pure_pbr_lambda"] * strength * rgb_loss}
        for name, key, bound in (("gain", "rgb_log_gain", "max_log_gain"),
                                 ("additive", "rgb_additive", "max_additive")):
            value = outputs[key]
            penalty = (value / cfg[bound]).square() if cfg[bound] else value * 0
            losses[f"rgb_residual_{name}_reg"] = (
                cfg[f"{name}_lambda"] * strength * (penalty.mean(-1, keepdim=True) * weight).sum() / denom)
        return losses

    def state_dict(self, only_model=True):
        state = super().state_dict(only_model=only_model)
        if getattr(self, "rgb_joint_config", None) is not None:
            state["rgb_joint"] = {"config": dict(self.rgb_joint_config),
                                  "camera_updates": int(self.rgb_joint_camera_updates)}
        return state

    def load_state_dict(self, state_dict, load_only_model=True, strict=True):
        state = dict(state_dict)
        metadata = state.pop("rgb_joint", None)
        cfg = getattr(self, "rgb_joint_config", None)
        if metadata is not None:
            if cfg is None or validate_rgb_joint_config(metadata["config"]) != cfg:
                raise ValueError("RGB joint checkpoint requires its unchanged trainer.rgb_joint config")
            counter = state.get("rgb_joint_camera_updates")
            if (not isinstance(counter, torch.Tensor) or counter.shape != torch.Size([])
                    or counter.dtype != torch.int64 or int(counter) < 0
                    or int(counter) != metadata.get("camera_updates")):
                raise ValueError("RGB joint checkpoint requires its completed-camera counter")
        elif cfg is not None:
            raise ValueError("RGB joint scratch model cannot initialize from a pretrained Base-v4 checkpoint")
        result = super().load_state_dict(state, load_only_model=load_only_model, strict=strict)
        self._rgb_joint_pending_camera_step = None
        return result
