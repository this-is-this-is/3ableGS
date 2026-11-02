"""Train-only D1 supervision on the unchanged Base v5 material renderer."""
import torch
import torch.nn.functional as F

from models.dynamic_gbuffer import (
    VisibleInstanceAccumulator, balanced_instance_loss,
    dynamic_gbuffer_weight, validate_dynamic_gbuffer_config,
)


class DynamicGBufferMixin:
    def configure_dynamic_gbuffer(self, config=None):
        self.dynamic_gbuffer_config = validate_dynamic_gbuffer_config(config)
        self._last_dynamic_gbuffer_stats = {}
        if self.dynamic_gbuffer_config["enabled"]:
            if getattr(self, "rgb_joint_config", None) is None:
                raise ValueError("D1 dynamic_gbuffer requires Base v5 RGB joint training")
            for name in ("teacher_calibration_config", "teacher_confidence_config", "rdg_fusion_config"):
                other = getattr(self, name, None)
                if other and other.get("enabled", False):
                    raise ValueError(f"D1 must be isolated from {name}")
            if getattr(self, "_diffreg_config", {}).get("enabled", False):
                raise ValueError("D1 must be isolated from diffreg_material")

    def dynamic_gbuffer_strength(self):
        config = getattr(self, "dynamic_gbuffer_config", None)
        if not config or not config["enabled"] or not self.training or not torch.is_grad_enabled():
            return 0.
        # Absolute completed-update clock: sampling/rendering never advances it.
        step = int(self.rgb_joint_camera_updates)
        return dynamic_gbuffer_weight(step, config)

    def prepare_dynamic_gbuffer(self, outputs, render_fn, gs, image_infos):
        """Accumulate visibility with all occluders present, without a graph.

        Only continuous mass channels are resized. Instance IDs are selected
        afterwards at exactly the ordinary teacher-loss crop and resolution.
        Low-resolution indicator mass and alpha come from the SAME raster pass;
        using the main high-resolution alpha would break their decomposition.
        """
        self._last_dynamic_gbuffer_stats = {}
        strength = self.dynamic_gbuffer_strength()
        if not strength:
            return
        cfg = self.dynamic_gbuffer_config
        gh, gw = image_infos["gbuf_basecolor"].shape[:2]
        height, width = outputs["opacity"].shape[:2]
        target_width = min(width, gw)
        target_hw = (round(target_width * gh / gw), target_width)
        factor = cfg["visibility_downscale"]
        guide_hw = ((height + factor - 1) // factor, (width + factor - 1) // factor)
        with torch.no_grad():
            group_ids = self._layered_group_ids("instance")
            instance_ids = torch.unique(group_ids[group_ids > 0], sorted=True)
            accumulator = None
            passes = 0
            for begin in range(0, instance_ids.numel(), cfg["chunk_size"]):
                ids = instance_ids[begin:begin + cfg["chunk_size"]]
                indicator = (group_ids[:, None] == ids[None, :]).to(gs.rgbs)
                features = torch.cat((indicator, torch.ones_like(indicator[:, :1])), dim=-1)
                masses = render_fn(features=features, feature_size=guide_hw)
                # Match the existing rounded native crop, including odd sizes.
                masses = self._resize_down(masses, (height, width))
                masses = self._crop_and_resize(masses, target_hw)
                if accumulator is None:
                    accumulator = VisibleInstanceAccumulator(masses[..., -1:])
                accumulator.add(masses[..., :-1], ids)
                passes += 1
            if accumulator is None:
                accumulator = VisibleInstanceAccumulator(
                    self._crop_and_resize(outputs["opacity"].detach(), target_hw))
            outputs["_dynamic_gbuffer_ownership"] = accumulator.finalize()
            outputs["_dynamic_gbuffer_weight"] = strength
            outputs["_dynamic_gbuffer_render_stats"] = {
                "guide_passes": passes, "candidate_instances": instance_ids.numel(),
                "guide_height": guide_hw[0], "guide_width": guide_hw[1],
            }

    def dynamic_gbuffer_losses(self, outputs, image_infos, camera_infos):
        strength = self.dynamic_gbuffer_strength()
        if not strength:
            self._last_dynamic_gbuffer_stats = {}
            return {}
        if "_dynamic_gbuffer_ownership" not in outputs:
            raise RuntimeError("D1 active training requires visibility from the current full-scene render")
        if outputs.get("_dynamic_gbuffer_weight") != strength:
            raise RuntimeError("D1 visibility and loss must use the same completed-update clock")
        ownership = outputs["_dynamic_gbuffer_ownership"]
        target_hw = ownership["owner"].shape
        valid = image_infos.get("egocar_masks")
        valid = torch.ones_like(image_infos["sky_masks"]) if valid is None else 1 - valid
        mask = self._crop_and_resize(
            ((1 - image_infos["sky_masks"]) * valid)[..., None], target_hw) > .99
        mask = mask.detach()
        config = self.dynamic_gbuffer_config
        losses, stats = {}, {}
        for name, target in (("albedo", "basecolor"), ("metallic", "metallic"),
                             ("roughness", "roughness"), ("normal", "normal")):
            channel_cfg = self.losses_dict.get(name)
            if channel_cfg is None:
                continue
            prediction = self._crop_and_resize(outputs[name], target_hw)
            teacher = self._resize_down(image_infos[f"gbuf_{target}"].detach(), target_hw)
            if name == "normal":
                c2w = getattr(self, "_render_c2w", camera_infos["camera_to_world"])
                r_wc = torch.linalg.inv(c2w)[:3, :3]
                prediction = F.normalize(prediction @ r_wc.T * self._opencv_to_opengl, dim=-1)
                teacher = F.normalize(teacher, dim=-1)
                residual = 1 - (prediction * teacher).sum(-1, keepdim=True)
                residual = residual / (1 + residual / channel_cfg.get("robust_c", .5))
            else:
                residual = (prediction - teacher).abs()
            value, channel_stats = balanced_instance_loss(residual, ownership, mask, config)
            losses[f"dynamic_gbuffer_{name}_loss"] = strength * channel_cfg.w * value
            stats.update({f"{name}/{key}": value for key, value in channel_stats.items()})
        self._last_dynamic_gbuffer_stats = {
            **outputs.get("_dynamic_gbuffer_render_stats", {}), **stats, "weight": strength,
        }
        return losses
