"""Train-only teacher calibration wiring; rendering never calls this module."""
from collections.abc import Mapping

import torch

from models.teacher_calibration import (
    TeacherCalibration, teacher_calibration_strength, validate_teacher_calibration_config,
)


class TeacherCalibrationMixin:
    def configure_teacher_calibration(self, config=None):
        cfg = validate_teacher_calibration_config(config)
        self.teacher_calibration_config = cfg if cfg["enabled"] else None
        self.teacher_calibration_head = None
        self._last_teacher_calibration_stats = {}
        if cfg["enabled"]:
            # Zero initialization consumes no random numbers and does not alter
            # Gaussian initialization, scene sampling or perceptual sampling.
            self.teacher_calibration_head = TeacherCalibration(
                self.dataset.train_indices, self.dataset.pixel_source.num_cams, cfg,
            ).to(self.device)
            self.teacher_calibration_head.train(self.training)

    def initialize_optimizer(self):
        super().initialize_optimizer()
        head = getattr(self, "teacher_calibration_head", None)
        if head is not None:
            parameters = list(head.parameters())
            name = "TeacherCalibration#all"
            self.param_groups[name] = parameters
            self.optimizer.add_param_group({
                "params": parameters, "name": name,
                "lr": self.teacher_calibration_config["lr"], "eps": 1e-8,
                "weight_decay": 0.,
            })

    def calibrated_teacher_targets(self, outputs, image_infos, target_hw, surface_mask):
        """Correct aligned static teacher values, preserving original loss masks.

        Sky/ego masking and material denominators remain those of Base v4.
        Dynamic or uncertain pixels still receive their original supervision.
        Dataset tensors, rendered outputs and the normal teacher are untouched.
        """
        self._last_teacher_calibration_stats = {}
        head = getattr(self, "teacher_calibration_head", None)
        cfg = getattr(self, "teacher_calibration_config", None)
        if (head is None or not self.training
                or not teacher_calibration_strength(self.step, cfg)):
            return None, {}
        image_id = image_infos.get("img_idx")
        if (not isinstance(image_id, torch.Tensor) or not image_id.numel()
                or image_id.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("Teacher calibration requires an integer current training image ID")
        index = int(image_id.reshape(-1)[0])
        dynamic = image_infos.get("dynamic_masks")
        if dynamic is None:
            raise ValueError("Teacher calibration requires dynamic_masks for static support")
        with torch.no_grad():
            if dynamic.ndim == 3 and dynamic.shape[-1] == 1:
                dynamic = dynamic[..., 0]
            if dynamic.ndim != 2:
                raise ValueError("Teacher calibration dynamic_masks must have shape H x W")
            static = (torch.isfinite(dynamic) & (dynamic <= 0)).float()[..., None]
            static = self._crop_and_resize(static, target_hw)[..., 0] > .99
            opacity = self._crop_and_resize(outputs["opacity"].detach(), target_hw)[..., 0]
            valid = (surface_mask.detach()[..., 0] > .99) & static
            valid &= torch.isfinite(opacity) & (opacity >= cfg["min_opacity"])
        teacher = {
            key: self._resize_down(image_infos[f"gbuf_{target}"], target_hw)
            for key, target in (("albedo", "basecolor"), ("roughness", "roughness"),
                                ("metallic", "metallic"))
        }
        corrected, regularization, stats = head.correct(teacher, valid, index, self.step)
        self._last_teacher_calibration_stats = stats
        return corrected, regularization

    def state_dict(self, only_model=True):
        state = super().state_dict(only_model=only_model)
        head = getattr(self, "teacher_calibration_head", None)
        if head is not None:
            state["_teacher_calibration"] = head.metadata()
        return state

    def load_state_dict(self, state_dict, load_only_model=True, strict=True):
        state = dict(state_dict)
        metadata = state.pop("_teacher_calibration", None)
        head = getattr(self, "teacher_calibration_head", None)
        if head is None:
            if metadata is not None or any(key.startswith("teacher_calibration_head.") for key in state):
                raise ValueError("Teacher calibration checkpoint requires its saved trainer.teacher_calibration config")
        else:
            # Verify the split before nn.Module can replace its ID buffer. A
            # reordered or different split must not silently reassign rows.
            if not isinstance(metadata, Mapping) or dict(metadata) != head.metadata():
                raise ValueError("Teacher calibration checkpoint config/train split mismatch")
            saved_ids = state.get("teacher_calibration_head.train_indices")
            if (not isinstance(saved_ids, torch.Tensor) or saved_ids.dtype != torch.int64
                    or not torch.equal(saved_ids.cpu(), head.train_indices.cpu())):
                raise ValueError("Teacher calibration checkpoint train ID buffer mismatch")
            raw = state.get("teacher_calibration_head.raw")
            if (not isinstance(raw, torch.Tensor) or raw.shape != head.raw.shape
                    or not raw.is_floating_point() or not bool(torch.isfinite(raw).all())):
                raise ValueError("Teacher calibration checkpoint has invalid parameters")
        result = super().load_state_dict(state, load_only_model=load_only_model, strict=strict)
        self._last_teacher_calibration_stats = {}
        return result
