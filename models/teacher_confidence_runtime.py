"""One-time, train-only confidence preprocessing from a fresh scene's geometry.

The frozen CPU maps never participate in inference rendering. They are built
before the warmup boundary's main update, using training images only, and are
serialized to make the training confidence source inspectable and repeatable.
"""
from collections.abc import Mapping
import math
from numbers import Integral, Real
import time

import torch

from models.gbuffer_attention_runtime import RENDER_IMAGE_KEYS, preserve_render_state
from models.teacher_confidence import (
    TeacherConfidenceMoments, normalized_teacher_confidence,
    teacher_confidence_strength, validate_teacher_confidence_config,
)


def _ids(train_indices, num_cams):
    if isinstance(num_cams, bool) or not isinstance(num_cams, Integral) or num_cams < 1:
        raise ValueError("Teacher confidence requires a positive integer camera count")
    if isinstance(train_indices, torch.Tensor):
        if train_indices.ndim != 1 or train_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Teacher confidence training IDs must be integer indices")
        train_indices = train_indices.tolist()
    try:
        values = list(train_indices)
    except TypeError as exc:
        raise ValueError("Teacher confidence requires training image IDs") from exc
    if (not values or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("Teacher confidence training IDs must be unique nonnegative integers")
    return sorted(int(v) for v in values)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"Teacher confidence {name} must be an integer >= {minimum}")
    return int(value)


def _crop(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < .5:
        raise ValueError("Teacher confidence crop fraction must be finite in [0, .5)")
    return float(value)


def confidence_crop_fraction(trainer):
    """Recover saved teacher cropping even when inference disables teacher I/O."""
    pixel = trainer.dataset.pixel_source
    cameras = getattr(pixel, "camera_data", {})
    values = [float(camera.gbuffer_crop_frac) for camera in cameras.values()
              if hasattr(camera, "gbuffer_crop_frac")]
    if values:
        if any(not math.isclose(value, values[0], abs_tol=1e-12) for value in values):
            raise ValueError("Teacher confidence requires matching camera crop fractions")
        return _crop(values[0])
    return _crop(getattr(pixel, "gbuffer_crop_frac", getattr(trainer, "crop_frac", 0.)))


def validate_teacher_confidence_state(saved, *, config, train_indices, num_cams,
                                      crop_frac, checkpoint_step):
    """Validate an envelope without a renderer; return detached CPU cache data.

    GPU-mapped torch.load tensors are accepted and copied to CPU. Unbuilt
    checkpoints are valid only before the declared preprocessing boundary.
    """
    cfg = validate_teacher_confidence_config(config)
    step = _integer(checkpoint_step, "checkpoint step")
    if not cfg["enabled"]:
        if saved is not None:
            raise ValueError("Baseline checkpoint contains teacher confidence state")
        return None
    ids = _ids(train_indices, num_cams)
    crop = _crop(crop_frac)
    required = {"schema_version", "config", "train_indices", "num_cams", "crop_frac",
                "built_step", "frames", "shapes", "stats"}
    if not isinstance(saved, Mapping) or set(saved) != required:
        raise ValueError("Teacher confidence checkpoint has missing or unknown state fields")
    if (_integer(saved["schema_version"], "schema version", 1) != 1
            or saved["config"] != cfg or not isinstance(saved["train_indices"], list)
            or _ids(saved["train_indices"], num_cams) != ids or saved["train_indices"] != ids
            or _integer(saved["num_cams"], "saved camera count", 1) != int(num_cams)
            or _crop(saved["crop_frac"]) != crop):
        raise ValueError("Teacher confidence checkpoint config/train split/crop mismatch")
    frames, shapes, stats = saved["frames"], saved["shapes"], saved["stats"]
    if not all(isinstance(value, Mapping) for value in (frames, shapes, stats)):
        raise ValueError("Teacher confidence frames, shapes and stats must be mappings")
    result = {"schema_version": 1, "config": dict(cfg), "train_indices": ids,
              "num_cams": int(num_cams), "crop_frac": crop,
              "built_step": saved["built_step"], "frames": {}, "shapes": {}, "stats": dict(stats)}
    if saved["built_step"] is None:
        if step >= cfg["build_step"] or frames or shapes or stats:
            raise ValueError("Teacher confidence checkpoint is missing its warmup-boundary cache")
        return result
    built = _integer(saved["built_step"], "built step")
    if built != cfg["build_step"] or step < built:
        raise ValueError("Teacher confidence checkpoint has an incompatible build step")
    if (any(isinstance(key, bool) or not isinstance(key, Integral) for key in (*frames, *shapes))
            or set(frames) != set(ids) or set(shapes) != set(ids)):
        raise ValueError("Teacher confidence cache must contain exactly the training image IDs")
    byte_count = 0
    for index in ids:
        frame, shape = frames[index], shapes[index]
        if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in shape)
                or shape[-1] != 1 or not isinstance(frame, torch.Tensor)
                or frame.dtype != torch.float16 or tuple(frame.shape) != tuple(shape)):
            raise ValueError("Teacher confidence cache has invalid map shape or dtype")
        frame = frame.detach().to(device="cpu").contiguous().clone()
        # Casting is monotone: a valid fp32 confidence cannot fall below the
        # same floor rounded to fp16. Use that exact bound, not a loose epsilon.
        rounded_floor = float(torch.tensor(cfg["confidence_floor"], dtype=torch.float16))
        if (not bool(torch.isfinite(frame).all())
                or bool((frame <= 0.).any()) or bool((frame.float() < rounded_floor).any())
                or bool((frame > 1.).any())):
            raise ValueError("Teacher confidence cache contains nonfinite or out-of-range values")
        byte_count += frame.numel() * frame.element_size()
        if byte_count > cfg["max_cache_mb"] * 1024 ** 2:
            raise ValueError("Teacher confidence cache exceeds max_cache_mb")
        result["frames"][index], result["shapes"][index] = frame, list(shape)
    for key, value in stats.items():
        if (not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value)):
            raise ValueError("Teacher confidence build statistics must be finite scalars")
    if (_integer(stats.get("render_count"), "render count") != len(ids)
            or _integer(stats.get("cache_bytes"), "cache byte count") != byte_count
            or stats.get("build_seconds", -1) < 0):
        raise ValueError("Teacher confidence build statistics do not match the cache")
    return result


def _image_index(image):
    value = image.get("img_idx")
    if (not isinstance(value, torch.Tensor) or not value.numel()
            or value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError("Teacher confidence requires an integer image ID")
    return int(value.reshape(-1)[0])


class TeacherConfidenceMixin:
    def configure_teacher_confidence(self, config=None):
        cfg = validate_teacher_confidence_config(config)
        self.teacher_confidence_config = cfg if cfg["enabled"] else None
        self._teacher_confidence_cache = None
        self._last_teacher_confidence_stats = {}
        if cfg["enabled"]:
            self._teacher_confidence_ids = _ids(
                self.dataset.train_indices, self.dataset.pixel_source.num_cams)
            self._teacher_confidence_num_cams = int(self.dataset.pixel_source.num_cams)
            self._teacher_confidence_crop_frac = confidence_crop_fraction(self)

    def _empty_teacher_confidence_state(self):
        return {"schema_version": 1, "config": dict(self.teacher_confidence_config),
                "train_indices": list(self._teacher_confidence_ids),
                "num_cams": self._teacher_confidence_num_cams,
                "crop_frac": self._teacher_confidence_crop_frac,
                "built_step": None, "frames": {}, "shapes": {}, "stats": {}}

    @torch.no_grad()
    def _confidence_static_mask(self, image, target_hw, device):
        valid = torch.ones(target_hw, dtype=torch.bool, device=device)
        for key in ("sky_masks", "dynamic_masks", "egocar_masks"):
            value = image.get(key)
            if value is None:
                if key != "egocar_masks":
                    raise ValueError(f"Teacher confidence requires {key}")
                continue
            value = value.detach().to(device)
            if value.ndim == 3 and value.shape[-1] == 1:
                value = value[..., 0]
            if value.ndim != 2:
                raise ValueError(f"Teacher confidence {key} must have shape H x W")
            mask = (torch.isfinite(value) & (value <= 0)).float()[..., None]
            valid &= self._crop_and_resize(mask, target_hw)[..., 0] > .99
        return valid

    @torch.no_grad()
    def prepare_teacher_confidence(self, step):
        cfg = getattr(self, "teacher_confidence_config", None)
        if cfg is None or not self.training:
            return
        step = _integer(step, "training step")
        if self._teacher_confidence_cache is not None or step < cfg["build_step"]:
            return
        if step != cfg["build_step"]:
            raise ValueError("Teacher confidence cache was not built at build_step; restart this arm from scratch")
        if _integer(self.step, "model step") != step:
            raise ValueError("Teacher confidence build step does not match the current model step")
        if (self.dataset.pixel_source.num_cams != self._teacher_confidence_num_cams
                or _ids(self.dataset.train_indices, self.dataset.pixel_source.num_cams) != self._teacher_confidence_ids):
            raise ValueError("Teacher confidence training split changed before cache construction")
        if float(self.crop_frac) != self._teacher_confidence_crop_frac:
            raise ValueError("Teacher confidence training crop differs from the configured teacher crop")
        positions = {int(index): position for position, index in enumerate(self.dataset.train_indices)}
        num_cams = self._teacher_confidence_num_cams
        order = sorted(self._teacher_confidence_ids, key=lambda index: (index % num_cams, index // num_cams))
        pending = self._empty_teacher_confidence_state()
        pending["built_step"] = step
        byte_count, count, sums = 0, 0, {}
        started = time.perf_counter()
        tracker, prior_camera = None, None
        # Includes loader calls: SplitWrapper's temporary downscale stack must
        # be restored even when loading fails before its own reset call.
        with preserve_render_state(self):
            self._suppress_eval_layers = True
            for index in order:
                camera_id, frame_id = index % num_cams, index // num_cams
                if camera_id != prior_camera:
                    tracker = TeacherConfidenceMoments(cfg)
                    prior_camera = camera_id
                images, cameras = self.dataset.train_image_set.get_image(positions[index], cfg["downscale"], ["raw"])
                image = images["raw"]
                if _image_index(image) != index:
                    raise ValueError("Teacher confidence loader returned a different training image")
                inputs = {key: value.to(self.device) if torch.is_tensor(value) else value
                          for key, value in image.items() if key in RENDER_IMAGE_KEYS}
                camera = {key: value.to(self.device) if torch.is_tensor(value) else value
                          for key, value in cameras["raw"].items()}
                outputs = self(inputs, camera, "raw")
                rh, rw = outputs["depth"].shape[:2]
                gh, gw = image["gbuf_basecolor"].shape[:2]
                width = min(rw, gw)
                target_hw = (round(width * gh / gw), width)
                top = round(rh * self.crop_frac)
                if min(target_hw) < 1 or rh - 2 * top < 1:
                    raise ValueError("Teacher confidence crop leaves no valid image")
                intrinsics = self._render_intrinsics.detach().clone()
                intrinsics[1, 2] -= top
                intrinsics[0, :] *= target_hw[1] / rw
                intrinsics[1, :] *= target_hw[0] / (rh - 2 * top)
                material = {key: self._resize_down(image[f"gbuf_{target}"].to(self.device), target_hw).detach()
                            for key, target in (("albedo", "basecolor"), ("roughness", "roughness"), ("metallic", "metallic"))}
                material["normal"] = self._crop_and_resize(outputs["normal"], target_hw).detach()
                packet = {"gbuffer": material,
                          "depth": self._crop_and_resize(outputs["depth"], target_hw).detach(),
                          "opacity": self._crop_and_resize(outputs["opacity"], target_hw).detach(),
                          "static_mask": self._confidence_static_mask(image, target_hw, self.device),
                          "camera_to_world": self._render_c2w.detach().clone(), "intrinsics": intrinsics}
                confidence, stats = tracker.update(packet, frame_id, camera_id)
                stored = confidence.detach().to(device="cpu", dtype=torch.float16).contiguous()
                byte_count += stored.numel() * stored.element_size()
                if byte_count > cfg["max_cache_mb"] * 1024 ** 2:
                    raise ValueError("Teacher confidence cache exceeds max_cache_mb; increase downscale or cache budget")
                pending["frames"][index], pending["shapes"][index] = stored, list(stored.shape)
                count += 1
                for key, value in stats.items():
                    scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
                    sums[key] = sums.get(key, 0.) + scalar
        pending["stats"] = {"render_count": count, "cache_bytes": byte_count,
                            "build_seconds": time.perf_counter() - started,
                            **{f"mean_{key}": value / count for key, value in sums.items()}}
        # No partial cache survives a failed frame or validation. The sampler,
        # node caches and main render state were restored by the context above.
        self._teacher_confidence_cache = validate_teacher_confidence_state(
            pending, config=cfg, train_indices=self._teacher_confidence_ids,
            num_cams=num_cams, crop_frac=self._teacher_confidence_crop_frac, checkpoint_step=step)

    def teacher_confidence_weights(self, image_infos, target_hw, surface_mask):
        self._last_teacher_confidence_stats = {}
        cfg = getattr(self, "teacher_confidence_config", None)
        if cfg is None or not self.training:
            return None
        strength = teacher_confidence_strength(self.step, cfg)
        if not strength:
            return None
        cache = self._teacher_confidence_cache
        if cache is None:
            raise ValueError("Call prepare_teacher_confidence at build_step before applying confidence")
        index = _image_index(image_infos)
        if index not in cache["frames"]:
            raise ValueError("Teacher confidence cannot use a held-out current image")
        with torch.no_grad():
            confidence = cache["frames"][index].to(device=surface_mask.device, dtype=surface_mask.dtype)
            # Cached maps are already in teacher-crop coordinates: resize only.
            confidence = self._resize_down(confidence, target_hw)
            static = self._confidence_static_mask(image_infos, target_hw, surface_mask.device)
            weights = normalized_teacher_confidence(confidence, static, surface_mask, strength)
            self._last_teacher_confidence_stats = {
                "strength": float(strength), "confidence_mean": confidence.mean().detach(),
                "weight_mean": weights.mean().detach(), "weight_min": weights.min().detach(),
                "weight_max": weights.max().detach(), "static_fraction": static.float().mean().detach(),
                **cache["stats"],
            }
        return weights.detach()

    def state_dict(self, only_model=True):
        state = super().state_dict(only_model=only_model)
        if getattr(self, "teacher_confidence_config", None) is not None:
            cache = self._teacher_confidence_cache or self._empty_teacher_confidence_state()
            state["_teacher_confidence"] = {
                **cache, "config": dict(cache["config"]), "train_indices": list(cache["train_indices"]),
                "frames": {index: value.detach().cpu().clone() for index, value in cache["frames"].items()},
                "shapes": {index: list(value) for index, value in cache["shapes"].items()},
                "stats": dict(cache["stats"]),
            }
        return state

    def load_state_dict(self, state_dict, load_only_model=True, strict=True):
        state = dict(state_dict)
        saved = state.pop("_teacher_confidence", None)
        cfg = getattr(self, "teacher_confidence_config", None)
        if cfg is None:
            if saved is not None:
                raise ValueError("Teacher confidence checkpoint requires its saved trainer.teacher_confidence config")
            cache = None
        else:
            cache = validate_teacher_confidence_state(
                saved, config=cfg, train_indices=self._teacher_confidence_ids,
                num_cams=self._teacher_confidence_num_cams, crop_frac=self._teacher_confidence_crop_frac,
                checkpoint_step=state.get("step"))
            if cache["built_step"] is None:
                cache = None
        result = super().load_state_dict(state, load_only_model=load_only_model, strict=strict)
        self._teacher_confidence_cache = cache
        self._last_teacher_confidence_stats = {}
        return result
