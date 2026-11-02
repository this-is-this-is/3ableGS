"""Optional train-only material consistency baseline, without inference fusion.

A previous training view is rendered under no_grad before the main render.
This leaves the main render's densification/visibility buffers intact. Pair
selection never reads a held-out target image. Camera/depth/normal routing and
the previous material target are detached. The current rendered material branch
receives gradients, including its rasterization weights and geometry.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch
import torch.nn.functional as F

from models.temporal_material import TemporalMaterialResolver
from utils.temporal_split import previous_train_image_index


def validate_temporal_training_config(config=None):
    # Legacy callers retain dense, every-step behavior unless sampling is
    # explicitly enabled. Experiment T uses interval=8 and num_samples=4096.
    defaults = {"w": 0., "start_step": 3000, "max_gap": 1,
                "interval": 1, "ramp_steps": 0, "num_samples": 0, "seed": 20261101}
    if config is None:
        return defaults
    if not isinstance(config, Mapping):
        raise ValueError("temporal_material training loss config must be a mapping.")
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown temporal training options: {sorted(unknown)}")
    result = {**defaults, **config}
    weight = result["w"]
    if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
        raise ValueError("temporal_material.w must be finite and nonnegative.")
    for name, minimum in (("start_step", 0), ("max_gap", 1), ("interval", 1),
                          ("ramp_steps", 0), ("num_samples", 0), ("seed", 0)):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"temporal_material.{name} must be an integer >= {minimum}.")
    return result


def temporal_training_weight(step, config):
    """Scheduled active-step coefficient; it is not multiplied by interval."""
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("Temporal training step must be a nonnegative integer")
    elapsed = step - config["start_step"]
    if config["w"] == 0 or elapsed < 0 or elapsed % config["interval"]:
        return 0.
    ramp = min(1., elapsed / config["ramp_steps"]) if config["ramp_steps"] else 1.
    return float(config["w"]) * ramp


@torch.no_grad()
def static_training_mask(image_infos, reference):
    """Exclude annotated sky, dynamic objects and ego car in both pair views."""
    h, w = reference.shape[:2]
    valid = torch.ones((h, w), dtype=torch.bool, device=reference.device)
    for name in ("sky_masks", "dynamic_masks", "egocar_masks"):
        value = image_infos.get(name)
        if value is None:
            if name != "egocar_masks":
                raise ValueError(f"Temporal material training requires {name}.")
            continue
        value = value.detach().to(device=reference.device, dtype=torch.float32)
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value[..., 0]
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape H x W or H x W x 1.")
        if value.shape != (h, w):
            value = F.interpolate(value[None, None], (h, w), mode="nearest")[0, 0]
        valid &= torch.isfinite(value) & (value <= 0.)
    return valid


def material_temporal_loss(current, current_static, current_camera, current_intrinsics,
                           previous, weight=1.):
    """Channel-balanced masked linear-albedo/roughness/metallic L1 loss.

    ``previous`` contains ``gbuffer``, ``depth``, ``opacity``, ``static_mask``,
    ``camera_to_world`` and ``intrinsics``. Geometry is used only to accept and
    warp detached targets; no gradient enters the correspondence branch or
    previous view. The current material render remains differentiable through
    its splat weights, geometry and any camera parameters used to render it.
    """
    zero = sum(torch.nan_to_num(current[key]).sum() * 0.
               for key in ("albedo", "roughness", "metallic"))
    if previous is None or weight == 0:
        return zero
    resolver = TemporalMaterialResolver({
        "mode": "normal", "history_weight": .5, "max_history": 2,
        "erode_radius": 1,
    })
    resolver.resolve(**previous, frame_id=0, stream_id="train-pair")
    current_gbuffer = {key: current[key] for key in ("albedo", "roughness", "metallic", "normal")}
    _, info = resolver.resolve(
        gbuffer=current_gbuffer, depth=current["depth"], opacity=current["opacity"],
        static_mask=current_static, camera_to_world=current_camera,
        intrinsics=current_intrinsics, frame_id=1, stream_id="train-pair",
    )
    mask = info["temporal_valid"].detach()
    denominator = mask.sum().clamp_min(1)
    terms = []
    for key in ("albedo", "roughness", "metallic"):
        # Linearize before warping so bilinear sampling of albedo agrees with
        # the inference material-transport experiment.
        target = previous["gbuffer"][key].detach().to(dtype=info["temporal_grid"].dtype)
        value = torch.nan_to_num(current[key])
        if key == "albedo":
            target = resolver._linear(target)
            value = resolver._linear(value)
        with torch.no_grad():
            target = resolver._sample(torch.nan_to_num(target), info["temporal_grid"])
        difference = torch.where(mask, (value - target).abs(), torch.zeros_like(value))
        terms.append(difference.mean(dim=-1, keepdim=True).sum() / denominator)
    return float(weight) * sum(terms) / len(terms)


class TemporalMaterialTraining:
    """Pair acquisition and loss hook used by tools/train.py."""

    def __init__(self, dataset, config=None):
        self.config = validate_temporal_training_config(config)
        self.dataset = dataset
        self.train_indices = frozenset(int(value) for value in dataset.train_indices)
        self.train_positions = {int(value): index for index, value in enumerate(dataset.train_indices)}
        self.num_cams = int(dataset.pixel_source.num_cams)
        self.last_stats = {}
        self.counters = {"scheduled_updates": 0, "neighbor_renders": 0, "missing_pairs": 0}

    def active(self, step, weather="raw"):
        return weather == "raw" and temporal_training_weight(step, self.config) > 0

    @torch.no_grad()
    def prepare_previous(self, trainer, image_infos, camera_downscale, weather, step):
        """Render only an allowed train neighbor before the differentiable view."""
        self.last_stats = {}
        if not self.active(step, weather):
            return None
        if not getattr(trainer, "training", True):
            return None
        image_idx = int(image_infos["img_idx"].flatten()[0].item())
        previous_idx = previous_train_image_index(
            image_idx, self.train_indices, self.num_cams, self.config["max_gap"],
        )
        if self.config["num_samples"]:
            self.counters["scheduled_updates"] += 1
        if previous_idx is None:
            if self.config["num_samples"]:
                self.counters["missing_pairs"] += 1
            return None
        if self.config["num_samples"]:
            return self._prepare_sparse_previous(
                trainer, image_infos, previous_idx, camera_downscale, weather, step,
            )
        image_by_weather, camera_by_weather = self.dataset.train_image_set.get_image(
            self.train_positions[previous_idx], camera_downscale, [weather],
        )
        device = image_infos["pixels"].device

        def to_device(mapping):
            return {key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor)
                    else value for key, value in mapping.items()}

        previous_image = to_device(image_by_weather[weather])
        previous_camera = to_device(camera_by_weather[weather])
        previous_outputs = trainer(previous_image, previous_camera, weather)
        if not hasattr(trainer, "_render_c2w") or not hasattr(trainer, "_render_intrinsics"):
            raise ValueError("Temporal material training requires a PBR trainer with rendered camera metadata.")
        return {
            "gbuffer": {
                key: previous_outputs[key].detach().clone()
                for key in ("albedo", "roughness", "metallic", "normal")
            },
            "depth": previous_outputs["depth"].detach().clone(),
            "opacity": previous_outputs["opacity"].detach().clone(),
            "static_mask": static_training_mask(previous_image, previous_outputs["depth"]),
            "camera_to_world": trainer._render_c2w.detach().clone(),
            "intrinsics": trainer._render_intrinsics.detach().clone(),
        }

    @torch.no_grad()
    def _prepare_sparse_previous(self, trainer, current_image, previous_idx,
                                 camera_downscale, weather, step):
        from models.gbuffer_attention_runtime import preserve_render_state, RENDER_IMAGE_KEYS
        # Restore frame/camera/node caches, downscale stacks and all RNG streams
        # even if loading or the neighbor render raises. No stale model cache
        # is reused across optimization updates.
        with preserve_render_state(trainer):
            trainer._suppress_eval_layers = True
            image_by_weather, camera_by_weather = self.dataset.train_image_set.get_image(
                self.train_positions[previous_idx], camera_downscale, [weather],
            )
            image = image_by_weather[weather]
            inputs = {key: value.to(trainer.device) if torch.is_tensor(value) else value
                      for key, value in image.items() if key in RENDER_IMAGE_KEYS}
            camera = {key: value.to(trainer.device) if torch.is_tensor(value) else value
                      for key, value in camera_by_weather[weather].items()}
            outputs = trainer(inputs, camera, weather)
            if not hasattr(trainer, "_render_c2w") or not hasattr(trainer, "_render_intrinsics"):
                raise ValueError("Sparse material training requires rendered PBR camera metadata")
            packet = {
                "gbuffer": {key: outputs[key].detach() for key in
                            ("albedo", "roughness", "metallic", "normal")},
                "depth": outputs["depth"].detach(), "opacity": outputs["opacity"].detach(),
                "static_mask": static_training_mask(image, outputs["depth"]),
                "camera_to_world": trainer._render_c2w.detach().clone(),
                "intrinsics": trainer._render_intrinsics.detach().clone(),
                "_image_idx": previous_idx,
                "_current_image_idx": int(current_image["img_idx"].reshape(-1)[0]),
                "_step": int(step),
            }
        self.counters["neighbor_renders"] += 1
        return packet

    def compute(self, trainer, outputs, image_infos, previous, weather, step):
        self.last_stats = {}
        if not self.active(step, weather) or not getattr(trainer, "training", True):
            return {}
        weight = temporal_training_weight(step, self.config)
        if self.config["num_samples"]:
            from models.sparse_material_consistency import sparse_material_temporal_loss
            current_idx = int(image_infos["img_idx"].reshape(-1)[0])
            if current_idx not in self.train_indices:
                raise ValueError("Sparse material training cannot use a held-out current image")
            if previous is not None:
                expected = previous_train_image_index(
                    current_idx, self.train_indices, self.num_cams, self.config["max_gap"],
                )
                if (previous.get("_step") != int(step)
                        or previous.get("_current_image_idx") != current_idx
                        or previous.get("_image_idx") != expected):
                    raise ValueError("Sparse material target belongs to a different step or training pair")
            value, self.last_stats = sparse_material_temporal_loss(
                outputs, static_training_mask(image_infos, outputs["depth"]),
                trainer._render_c2w, trainer._render_intrinsics, previous,
                weight=weight, num_samples=self.config["num_samples"],
                seed=self.config["seed"], step=int(step), sample_key=current_idx,
            )
            return {"temporal_material_loss": value}
        value = material_temporal_loss(
            outputs, static_training_mask(image_infos, outputs["depth"]),
            trainer._render_c2w, trainer._render_intrinsics,
            previous, weight,
        )
        return {"temporal_material_loss": value}
