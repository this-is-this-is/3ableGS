"""Recorded-camera context and trainer plumbing for G-buffer attention.

Context is rendered before the differentiable current view. It never consumes
reference images and never owns the current view's Gaussian refinement state.
The ordinary image evaluator needs no changes to its split or metric code.
"""
from collections import OrderedDict
from contextlib import contextmanager
import random

import numpy as np
import torch


RENDER_IMAGE_KEYS = frozenset({
    "origins", "viewdirs", "direction_norm", "pixel_coords", "normed_time",
    "img_idx", "frame_idx",
})
_MISSING = object()


def history_image_ids(image_id, num_cams, num_images, train_ids, training,
                      offsets=(1, 2, 4)):
    """Actual preceding timestamps in the same camera, never preceding test IDs."""
    if not 0 <= image_id < num_images or num_cams < 1:
        raise ValueError("Image index outside recorded camera sequence")
    if training and image_id not in train_ids:
        raise ValueError("Attention training target must belong to the saved train split")
    return [image_id - int(delta) * num_cams for delta in offsets
            if image_id // num_cams >= delta
            and (not training or image_id - int(delta) * num_cams in train_ids)]


def _move_packet(packet, device, clone=False):
    return {key: (value.detach().to(device).clone() if clone else value.detach().to(device))
            if torch.is_tensor(value) else value for key, value in packet.items()}


@contextmanager
def preserve_render_state(trainer):
    """Restore mutable render caches and RNG, including exception paths."""
    names = ("cur_frame", "info", "pts_labels", "dynamic_pts_mask",
             "_render_c2w", "_render_intrinsics", "_layered_screen_gradients",
             "_gbuffer_attention_context", "_rdg_context", "_suppress_eval_layers")
    node_names = ("cur_frame", "in_test_set", "filter_mask", "_gs_cache",
                  "_raw_normal_residual_penalty", "train_frame_ids")
    owners = [(trainer, names)] + [(model, node_names) for model in trainer.models.values()]
    pixel_source = getattr(getattr(trainer, "dataset", None), "pixel_source", None)
    if pixel_source is not None:
        owners.append((pixel_source, ("_downscale_factor", "_old_downscale_factor")))
        owners.extend((camera, ("downscale_factor",))
                      for camera in getattr(pixel_source, "camera_data", {}).values())
    snapshots = [(obj, {key: getattr(obj, key, _MISSING) for key in keys})
                 for obj, keys in owners]
    # SplitWrapper pushes to this stack in place before it loads a frame.
    for _, state in snapshots:
        if isinstance(state.get("_old_downscale_factor"), list):
            state["_old_downscale_factor"] = list(state["_old_downscale_factor"])
    py_state, np_state = random.getstate(), np.random.get_state()
    device = torch.device(trainer.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        for obj, state in snapshots:
            for key, value in state.items():
                if value is _MISSING:
                    if hasattr(obj, key):
                        delattr(obj, key)
                else:
                    setattr(obj, key, value)


class AttentionHistory:
    """Optional CPU cache of unrefined generated packets; default random access."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.enabled = False
        self.cache = OrderedDict()
        self.limit = 4

    def clear(self):
        self.cache.clear()

    def key(self, image_id, shape):
        return int(image_id), tuple(shape), int(self.trainer.step)

    def remember(self, image_id, packet):
        if not self.enabled or self.trainer.training:
            return
        key = self.key(image_id, packet["albedo"].shape[:2])
        self.cache[key] = _move_packet(packet, "cpu", clone=True)
        self.cache.move_to_end(key)
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)

    def prepare(self, image_infos):
        trainer, dataset = self.trainer, self.trainer.dataset
        cfg = trainer.gbuffer_attention_config
        image_id = int(image_infos["img_idx"].flatten()[0])
        shape = image_infos["viewdirs"].shape[:2]
        num_cams = int(dataset.pixel_source.num_cams)
        train_ids = frozenset(map(int, dataset.train_indices))
        total = len(dataset.full_image_set)
        ids = history_image_ids(image_id, num_cams, total, train_ids, trainer.training,
                                cfg["history_offsets"])
        local = {}

        def packet(index):
            if index in local:
                return local[index]
            key = self.key(index, shape)
            if self.enabled and not trainer.training and key in self.cache:
                # Reading t-4 must not evict t-3 when current t is inserted.
                # FIFO retains the last four generated timestamps in a stream.
                value = _move_packet(self.cache[key], trainer.device)
            else:
                with preserve_render_state(trainer), torch.no_grad():
                    trainer._gbuffer_attention_context = True
                    trainer._suppress_eval_layers = True
                    images, cameras = dataset.full_image_set.get_image(
                        index, trainer._get_downscale_factor(), ["raw"])
                    inputs = {k: v.to(trainer.device) if torch.is_tensor(v) else v
                              for k, v in images["raw"].items() if k in RENDER_IMAGE_KEYS}
                    camera = {k: v.to(trainer.device) if torch.is_tensor(v) else v
                              for k, v in cameras["raw"].items()}
                    value = trainer(inputs, camera, "raw")["_attention_packet"]
                if value["albedo"].shape[:2] != shape:
                    raise ValueError("History resolution differs from the current render")
                value = _move_packet(value, trainer.device)
                self.remember(index, value)
            local[index] = value
            return value

        histories = [packet(index) for index in ids] if cfg["variant"] != "C0" else []
        previous = None
        from models.gbuffer_attention import temporal_weight
        step = trainer.attention_step if trainer.attention_step is not None else trainer.step
        if trainer.training and ids and temporal_weight(cfg, step) > 0:
            # A private RNG makes pair selection independent of scene sampling.
            index = random.Random(int(step) * 1000003 + image_id).choice(ids)
            previous = packet(index)
            older = history_image_ids(index, num_cams, total, train_ids, True,
                                      cfg["history_offsets"])
            context = [packet(i) for i in older] if cfg["variant"] != "C0" else []
            from models.gbuffer_attention import attention_strength
            with torch.no_grad():
                result = trainer.gbuffer_attention_head(
                    previous, context, strength=attention_strength(cfg, step, True))
            previous = {**previous, "albedo": result["albedo"].detach()}
        return histories, previous, ids


class GBufferAttentionMixin:
    """Explicit opt-in; old Base v4 checkpoints retain their original contract."""
    def configure_gbuffer_attention(self, config=None):
        from models.gbuffer_attention import GBufferAttentionHead, validate_attention_config
        if config is not None and (getattr(self, "_layered_gbuffer", None) is not None
                                   or getattr(self, "_temporal_material", None) is not None):
            raise ValueError("G-buffer attention cannot be combined with other history/layered renderers")
        self.gbuffer_attention_config = None if config is None else validate_attention_config(config)
        self.attention_step = None
        self.gbuffer_attention_head = None
        self._attention_history = AttentionHistory(self)
        if self.gbuffer_attention_config is not None:
            # CPU construction within fork_rng preserves the scene's seed stream.
            with torch.random.fork_rng(devices=[]):
                generator = torch.Generator(device="cpu").manual_seed(
                    (torch.initial_seed() + 82103) % (2 ** 63))
                torch.set_rng_state(generator.get_state())
                self.gbuffer_attention_head = GBufferAttentionHead(self.gbuffer_attention_config).to(self.device)
            self.gbuffer_attention_head.train(self.training)

    def set_gbuffer_attention_cache(self, enabled=True):
        history = getattr(self, "_attention_history", None)
        if history is None:
            raise ValueError("Configure G-buffer attention before enabling its cache")
        history.clear()
        history.enabled = bool(enabled)

    def initialize_optimizer(self):
        super().initialize_optimizer()
        head = getattr(self, "gbuffer_attention_head", None)
        parameters = list(head.parameters()) if head is not None else []
        if parameters:
            name = "GBufferAttention#all"
            self.param_groups[name] = parameters
            self.optimizer.add_param_group({"params": parameters, "name": name,
                                            "lr": self.gbuffer_attention_config["lr"], "eps": 1e-8})

    def optimizer_step(self):
        head = getattr(self, "gbuffer_attention_head", None)
        if head is not None:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        super().optimizer_step()
        if getattr(self, "_attention_history", None) is not None:
            self._attention_history.clear()

    def attention_losses(self, outputs):
        current = outputs.get("_attention_current")
        if current is None or not self.training:
            return {}
        from models.gbuffer_attention import attention_material_loss, temporal_weight
        from models.ego_temporal import srgb_to_linear
        cfg = self.gbuffer_attention_config
        mask = outputs["_attention_surface"].detach().to(current["albedo"].dtype)
        error = (srgb_to_linear(current["albedo"]) -
                 srgb_to_linear(outputs["albedo"].detach())).abs().mean(-1, keepdim=True)
        losses = {"attention_anchor_loss": cfg["anchor_w"] * (error * mask).sum() / mask.sum().clamp_min(1)}
        step = self.attention_step if self.attention_step is not None else self.step
        weight = temporal_weight(cfg, step)
        previous = outputs.get("_attention_previous")
        if weight > 0 and previous is not None:
            losses["attention_temporal_loss"] = weight * attention_material_loss(current, previous)["loss"]
        return losses

    def state_dict(self, only_model=True):
        state = super().state_dict(only_model=only_model)
        if getattr(self, "gbuffer_attention_config", None) is not None:
            state["_gbuffer_attention"] = {"config": dict(self.gbuffer_attention_config),
                                            "attention_step": self.attention_step}
        return state

    def load_state_dict(self, state_dict, load_only_model=True, strict=True):
        state = dict(state_dict)
        metadata = state.pop("_gbuffer_attention", None)
        configured = getattr(self, "gbuffer_attention_config", None)
        if metadata is not None:
            from models.gbuffer_attention import validate_attention_config
            saved = validate_attention_config(metadata["config"])
            if configured is None:
                raise ValueError("Attention checkpoint requires its saved trainer.gbuffer_attention config")
            if saved != configured:
                raise ValueError("Attention checkpoint/config mismatch")
        elif configured is not None:
            raise ValueError("Load the base checkpoint before configuring a new attention head")
        result = super().load_state_dict(state, load_only_model=load_only_model, strict=strict)
        self.attention_step = None if metadata is None else metadata.get("attention_step")
        if getattr(self, "_attention_history", None) is not None:
            self._attention_history.clear()
        return result
