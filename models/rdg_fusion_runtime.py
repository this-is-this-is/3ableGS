"""Opt-in RDG-inspired material fusion with one generated history image.

History uses a recorded same-camera timestamp at a reduced raster resolution.
It is detached, contains no teacher/target inputs, and never recursively fuses
its own history. The current render retains ownership of refinement state.
"""
from collections.abc import Mapping
from numbers import Integral

import torch

from models.gbuffer_attention_runtime import (
    RENDER_IMAGE_KEYS, history_image_ids, preserve_render_state,
)


def _integer(value, label):
    if isinstance(value, torch.Tensor):
        if (not value.numel() or value.dtype not in
                (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError(f"RDG {label} must be a nonempty integer ID")
        first = value.reshape(-1)[0]
        if not bool((value == first).all()):
            raise ValueError(f"RDG {label} must identify one image")
        value = int(first)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"RDG {label} must be a nonnegative integer")
    return int(value)


def _actual_camera_id(pixel_source, unique_camera):
    # A source may select cameras [1, 3], whose global image IDs still use
    # camera indices [0, 1]. Packet stream IDs contain the actual camera IDs.
    for camera_id, camera in getattr(pixel_source, "camera_data", {}).items():
        if getattr(camera, "unique_cam_idx", None) == unique_camera:
            return int(camera_id)
    camera_list = getattr(pixel_source, "camera_list", None)
    return int(camera_list[unique_camera]) if camera_list is not None else unique_camera


def _shape(image_infos):
    value = image_infos.get("viewdirs")
    if (not isinstance(value, torch.Tensor) or value.ndim != 3
            or value.shape[-1] != 3 or min(value.shape[:2]) < 1):
        raise ValueError("RDG viewdirs must have nonempty H x W x 3 shape")
    return tuple(value.shape[:2])


def _validate_time(trainer, image_infos, frame_id, label):
    timestamps = getattr(trainer, "normalized_timestamps", None)
    if timestamps is None:
        return
    time = image_infos.get("normed_time")
    if (not isinstance(time, torch.Tensor) or not time.numel() or not time.is_floating_point()
            or frame_id >= len(timestamps)):
        raise ValueError(f"RDG {label} requires normalized time for its recorded frame")
    expected = torch.as_tensor(timestamps[frame_id], device=time.device, dtype=time.dtype)
    if not bool(torch.isfinite(time).all() & ((time - expected).abs() <= 1e-6).all()):
        raise ValueError(f"RDG {label} normalized time/frame ID mismatch")


class RDGFusionMixin:
    """Add a small independent fusion head without changing Base-v4 defaults."""

    def _configure_rdg_fusion(self, config=None):
        from models.rdg_fusion import RDGFusion, validate_rdg_config
        cfg = validate_rdg_config(config)
        self.rdg_fusion_config = cfg if cfg["enabled"] else None
        self.rdg_fusion_head = None
        if not cfg["enabled"]:
            return
        self.assert_rdg_compatible()
        # Isolate initialization from scene creation and subsequent view sampling.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(
                (torch.initial_seed() + 309107) % (2 ** 63))
            torch.set_rng_state(generator.get_state())
            self.rdg_fusion_head = RDGFusion(cfg).to(self.device)
        self.rdg_fusion_head.train(self.training)

    def assert_rdg_compatible(self):
        """Check at configuration and render time, including later opt-ins."""
        attention = getattr(self, "gbuffer_attention_config", None)
        conflicts = (attention is not None and attention.get("variant") != "B0")
        conflicts |= any(getattr(self, name, None) is not None for name in (
            "_layered_gbuffer", "_temporal_material", "rgb_joint_config",
            "teacher_calibration_config", "teacher_confidence_config",
        ))
        conflicts |= bool((getattr(self, "_diffreg_config", None) or {}).get("enabled", False))
        losses = getattr(self, "losses_dict", None) or {}
        temporal_loss = losses.get("temporal_material", None) or {}
        conflicts |= temporal_loss.get("w", 0.) > 0
        if conflicts:
            raise ValueError("RDG fusion requires the ordinary Base-v4 renderer without other experiments")

    def rdg_strength(self):
        from models.rdg_fusion import rdg_strength
        cfg = getattr(self, "rdg_fusion_config", None)
        return 0. if cfg is None else rdg_strength(cfg, self.step, training=self.training)

    def rdg_history(self, image_infos):
        """Render at most one raw previous packet, with no references or cache."""
        cfg = getattr(self, "rdg_fusion_config", None)
        if cfg is None or getattr(self, "_rdg_context", False) or self.rdg_strength() <= 0:
            return None
        dataset = self.dataset
        pixel_source = dataset.pixel_source
        num_cams = _integer(pixel_source.num_cams, "camera count")
        total = len(dataset.full_image_set)
        if num_cams < 1 or total % num_cams:
            raise ValueError("RDG requires a complete interleaved camera sequence")
        image_id = _integer(image_infos.get("img_idx"), "current image ID")
        frame_id = _integer(image_infos.get("frame_idx"), "current frame ID")
        if frame_id != image_id // num_cams:
            raise ValueError("RDG current frame/image IDs disagree")
        _validate_time(self, image_infos, frame_id, "current image")
        shape = _shape(image_infos)
        train_ids = frozenset(_integer(value, "training image ID") for value in dataset.train_indices)
        offsets = tuple(range(1, cfg["max_gap"] + 1)) if self.training else (1,)
        ids = history_image_ids(image_id, num_cams, total, train_ids, self.training, offsets)
        if not ids:
            return None
        index = ids[0]
        expected_frame = index // num_cams
        expected_camera = _actual_camera_id(pixel_source, index % num_cams)
        expected_shape = tuple(size // cfg["feature_downscale"] for size in shape)
        if min(expected_shape) < 1:
            raise ValueError("RDG history resolution would be empty")
        # full_image_set is indexed by global image ID, unlike train_image_set.
        full_ids = getattr(dataset.full_image_set, "split_indices", None)
        if full_ids is not None and int(full_ids[index]) != index:
            raise ValueError("RDG full_image_set must preserve global image indices")
        with preserve_render_state(self), torch.no_grad():
            self._rdg_context = True
            self._suppress_eval_layers = True
            images, cameras = dataset.full_image_set.get_image(
                index, self._get_downscale_factor() * cfg["feature_downscale"], ["raw"])
            inputs = {key: value.to(self.device) if torch.is_tensor(value) else value
                      for key, value in images["raw"].items() if key in RENDER_IMAGE_KEYS}
            camera = {key: value.to(self.device) if torch.is_tensor(value) else value
                      for key, value in cameras["raw"].items()}
            if (_integer(inputs.get("img_idx"), "history image ID") != index
                    or _integer(inputs.get("frame_idx"), "history frame ID") != expected_frame
                    or _integer(camera.get("cam_id"), "history camera ID") != expected_camera):
                raise ValueError("RDG loaded history camera/frame/image IDs disagree")
            _validate_time(self, inputs, expected_frame, "history image")
            if (_shape(inputs) != expected_shape
                    or (_integer(camera.get("height"), "history height"),
                        _integer(camera.get("width"), "history width")) != expected_shape):
                raise ValueError("RDG history resolution differs from the requested feature resolution")
            packet = self(inputs, camera, "raw")["_rdg_packet"]
            from models.ego_temporal import _validate_packet
            if _validate_packet(packet) != expected_shape:
                raise ValueError("RDG generated history resolution is inconsistent")
            if (packet.get("image_id") != index or packet.get("frame_id") != expected_frame
                    or packet.get("stream_id") != ("raw", expected_camera)):
                raise ValueError("RDG generated history identity is inconsistent")
            return {key: value.detach() if torch.is_tensor(value) else value
                    for key, value in packet.items()}

    def initialize_optimizer(self):
        super().initialize_optimizer()
        head = getattr(self, "rdg_fusion_head", None)
        if head is not None:
            parameters = list(head.parameters())
            name = "RDGFusion#all"
            self.param_groups[name] = parameters
            self.optimizer.add_param_group({
                "params": parameters, "name": name, "eps": 1e-8,
                "lr": self.rdg_fusion_config["lr"],
                "weight_decay": self.rdg_fusion_config["weight_decay"],
            })

    def optimizer_step(self):
        head = getattr(self, "rdg_fusion_head", None)
        if head is not None:
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), self.rdg_fusion_config["grad_clip"], error_if_nonfinite=True)
        super().optimizer_step()

    def rdg_losses(self, outputs):
        # The core applies configuration weights and the activation ramp once.
        if not self.training or getattr(self, "rdg_fusion_config", None) is None:
            return {}
        return {f"rdg_{name}": outputs[f"_rdg_loss_{name}"]
                for name in ("anchor", "temporal") if f"_rdg_loss_{name}" in outputs}

    def state_dict(self, only_model=True):
        state = super().state_dict(only_model=only_model)
        cfg = getattr(self, "rdg_fusion_config", None)
        if cfg is not None:
            state["_rdg_fusion"] = {"config": dict(cfg), "rdg_step": _integer(self.step, "checkpoint step")}
        return state

    def load_state_dict(self, state_dict, load_only_model=True, strict=True):
        state = dict(state_dict)
        metadata = state.pop("_rdg_fusion", None)
        cfg = getattr(self, "rdg_fusion_config", None)
        prefix = "rdg_fusion_head."
        saved_keys = {key for key in state if key.startswith(prefix)}
        if cfg is None:
            if metadata is not None or saved_keys:
                raise ValueError("RDG checkpoint requires its saved trainer.rdg_fusion config")
        else:
            from models.rdg_fusion import validate_rdg_config
            if (not isinstance(metadata, Mapping) or set(metadata) != {"config", "rdg_step"}
                    or validate_rdg_config(metadata["config"]) != cfg):
                raise ValueError("RDG checkpoint/config mismatch; enabled RDG requires its own scratch checkpoint")
            if _integer(metadata["rdg_step"], "saved step") != _integer(state.get("step"), "checkpoint step"):
                raise ValueError("RDG checkpoint step metadata mismatch")
            expected = self.rdg_fusion_head.state_dict()
            if saved_keys != {prefix + key for key in expected}:
                raise ValueError("RDG checkpoint fusion head is incomplete")
            for key, target in expected.items():
                value = state[prefix + key]
                if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                        or value.dtype != target.dtype or not bool(torch.isfinite(value).all())):
                    raise ValueError(f"RDG checkpoint contains invalid fusion parameters: {key}")
        return super().load_state_dict(state, load_only_model=load_only_model, strict=strict)
