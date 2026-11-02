"""Frozen raw-PBR scene adapter for the high-ego-motion experiments.

GPU-dependent imports stay inside ``SceneRuntime`` so split validation and
contract tests run on CPU. Inference inputs never contain reference RGB,
reference masks, LiDAR or teacher material buffers.
"""

import copy
import math
from numbers import Integral
import os
from pathlib import Path
import time

import torch

from tools.render_temporal_material import (
    ROOT, evaluation_static_mask, inference_config, render_inputs, to_device,
)
from utils.temporal_split import resolve_test_frames


def experiment_splits(num_frames, pixel_config, require_holdout=True):
    """Use the first saved held-out block for validation, later blocks for test.

    Never manufacture a new holdout from frames already used by the base
    checkpoint. Adjacent ranges are one block and cannot serve as independent
    validation/test clips. Each evaluation block needs at least one pair.
    """
    ranges = pixel_config.get("test_frame_ranges")
    heldout = resolve_test_frames(
        num_frames, ranges, int(pixel_config.get("test_image_stride", 0)))
    if require_holdout:
        if ranges is None or len(ranges) < 2:
            raise ValueError(
                "Ego-motion experiments require at least two explicit saved "
                "test_frame_ranges; a stride checkpoint is not a clean block holdout.")
        for index, (start, end) in enumerate(ranges):
            if end - start < 2:
                raise ValueError("Each held-out block must contain at least two frames.")
            if index and start <= ranges[index - 1][1]:
                raise ValueError("Validation/test blocks need a separating training interval.")
    heldout_set = set(heldout)
    train = [frame for frame in range(num_frames) if frame not in heldout_set]
    if ranges is not None and len(ranges) >= 2:
        validation = list(range(int(ranges[0][0]), int(ranges[0][1])))
        validation_set = set(validation)
        test = [frame for frame in heldout if frame not in validation_set]
    else:
        validation, test = [], heldout
    return train, validation, test


def packet_to(packet, device):
    """Move nested packets without changing integers or reference separation."""
    if torch.is_tensor(packet):
        return packet.to(device)
    if isinstance(packet, dict):
        return {key: packet_to(value, device) for key, value in packet.items()}
    if isinstance(packet, tuple):
        return tuple(packet_to(value, device) for value in packet)
    if isinstance(packet, list):
        return [packet_to(value, device) for value in packet]
    return packet


def detach_packet(packet, device="cpu"):
    """Create a graph-free packet suitable for a trusted local torch cache."""
    if torch.is_tensor(packet):
        return packet.detach().to(device).clone()
    if isinstance(packet, dict):
        return {key: detach_packet(value, device) for key, value in packet.items()}
    if isinstance(packet, tuple):
        return tuple(detach_packet(value, device) for value in packet)
    if isinstance(packet, list):
        return [detach_packet(value, device) for value in packet]
    return packet


def select_material_parameters(trainer, update_features=True):
    """Freeze geometry, lighting, response and normals; return named materials.

    Learned normal residuals share the Gaussian feature vectors. Their features
    therefore stay frozen, including after saving/reloading the ordinary scene
    checkpoint. Raw material decoder weights remain trainable. Nodes without
    that normal branch may also update their material feature vectors.
    """
    trainer.requires_grad_(False)
    for model in trainer.models.values():
        model.requires_grad_(False)
    selected, notes = [], []
    seen = set()
    for name in trainer.gaussian_classes:
        model = trainer.models[name]
        decoders = getattr(model, "mlp_material", None)
        if decoders is None or "raw" not in decoders:
            continue
        normal_branch = getattr(model, "raw_normal_residual", None)
        features = getattr(model, "_features", None)
        # Conservatively freeze features whenever the normal module exists;
        # its ephemeral active flag is not persisted in a checkpoint.
        include_features = bool(update_features and normal_branch is None)
        candidates = [(f"{name}.mlp_material.raw.{key}", value)
                      for key, value in decoders["raw"].named_parameters()]
        if include_features and isinstance(features, torch.nn.Parameter):
            candidates.insert(0, (f"{name}._features", features))
        for key, parameter in candidates:
            if id(parameter) not in seen:
                parameter.requires_grad_(True)
                selected.append((key, parameter))
                seen.add(id(parameter))
        notes.append({
            "node": name,
            "features_trainable": include_features and isinstance(features, torch.nn.Parameter),
            "features_frozen_reason": (
                "shared_with_learned_normal" if normal_branch is not None
                else "feature_updates_disabled" if not update_features else None),
        })
    if not selected:
        raise ValueError("No raw material decoder parameters were found in this checkpoint.")
    summary = {
        "trainable_names": [name for name, _ in selected],
        "trainable_elements": sum(parameter.numel() for _, parameter in selected),
        "nodes": notes,
        "frozen": ["geometry", "camera", "lighting", "normal", "camera_response", "affine"],
    }
    return selected, summary


class SceneRuntime:
    """Camera-0 rendering and differentiable current-view PBR reshading.

    ``render`` returns generated HWC RGB/material/depth maps plus camera
    matrices. ``target_rgb`` and ``evaluation_static`` are reference-only
    fields; callers must never feed these into the learned head. ``normal``
    is world-space before PBR orientation, with camera-dependent sign inherited
    from Gaussian packing. Normal comparisons should use absolute cosine.
    """

    def __init__(self, checkpoint, downscale=2., antialiased=False, device="cuda",
                 require_holdout=True, load_gbuffer=False):
        from omegaconf import OmegaConf

        if isinstance(downscale, bool) or not math.isfinite(float(downscale)) or downscale < 1:
            raise ValueError("downscale must be finite and >= 1.")
        if not isinstance(antialiased, bool):
            raise ValueError("antialiased must be a boolean.")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = torch.device(device)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        saved_path = self.checkpoint.parent / "config.yaml"
        if not saved_path.is_file():
            raise FileNotFoundError(saved_path)
        self.saved_config = OmegaConf.load(saved_path)
        self.cfg = inference_config(self.saved_config, float(downscale))
        if load_gbuffer:
            # Adaptation retains the checkpoint's existing teacher paths and
            # objectives; inference remains entirely independent of teachers.
            self.cfg.data.pixel_source.load_gbuffer = True
        self.config = self.cfg
        self.cfg.trainer.render.antialiased = antialiased
        # Reject unclean holdouts before touching CUDA or loading scene assets.
        expected_frames = int(self.cfg.data.end_timestep) - int(self.cfg.data.start_timestep) + 1
        experiment_splits(expected_frames, self.cfg.data.pixel_source, require_holdout)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SceneRuntime requires CUDA and the project's gsplat environment.")
        from datasets.driving_dataset import DrivingDataset
        from utils.misc import import_str

        os.chdir(ROOT)
        self.dataset = DrivingDataset(data_cfg=self.cfg.data)
        self.camera = self.dataset.pixel_source.camera_data[0]
        self.camera.set_downscale_factor(self.dataset.pixel_source.downscale_factor)
        self.frame_ids = list(range(len(self.camera)))
        self.train_frames, self.val_frames, self.test_frames = experiment_splits(
            len(self.camera), self.cfg.data.pixel_source, require_holdout)
        if set(map(int, self.dataset.train_timesteps)) != set(self.train_frames):
            raise ValueError("Dataset training split disagrees with the saved checkpoint ranges.")
        if set(map(int, self.dataset.test_timesteps)) != set(self.val_frames + self.test_frames):
            raise ValueError("Dataset held-out split disagrees with the saved checkpoint ranges.")
        self.trainer = import_str(self.cfg.trainer.type)(
            **copy.deepcopy(self.cfg.trainer), dataset=self.dataset, weathers=["raw"],
            vgg_config=self.cfg.get("vgg"), model_config=copy.deepcopy(self.cfg.model),
            device=self.device,
        )
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.trainer.load_state_dict(state, load_only_model=True, strict=True)
        del state
        self.trainer.set_eval()
        self.trainer.requires_grad_(False)
        for model in self.trainer.models.values():
            model.requires_grad_(False)
        self.trainer._suppress_eval_layers = True
        self.trainer.configure_temporal_material(None)
        # Apply the same mean affine for train-frame fitting and all evaluation.
        self.trainer._temporal_shared_affine = True
        self.trainer._export_ego_inputs = True
        self.material_parameter_summary = None
        self.reset()

    def reset(self):
        self.trainer.reset_temporal_material()

    def freeze_material_parameters(self, update_features=True):
        selected, self.material_parameter_summary = select_material_parameters(
            self.trainer, update_features=update_features)
        return selected

    def render(self, frame_id, grad=False, return_outputs=False):
        if isinstance(frame_id, bool) or not isinstance(frame_id, Integral) or frame_id not in self.frame_ids:
            raise ValueError(f"Frame {frame_id!r} is outside the saved sequence.")
        if grad and frame_id not in self.train_frames:
            raise ValueError("Gradient scene rendering is restricted to saved training frames.")
        if return_outputs and frame_id not in self.train_frames:
            raise ValueError("Training loss packets are restricted to saved training frames.")
        self.trainer.set_eval()
        infos, cameras = self.camera.get_image(int(frame_id), "raw")
        inputs = render_inputs(infos, self.device, torch)
        cameras = to_device(cameras, self.device, torch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        dynamic_loss = getattr(self.trainer, "losses_dict", {}).get("dynamic_region")
        needs_dynamic = (return_outputs and dynamic_loss is not None and
                         self.trainer.step > dynamic_loss.get("start_from", 0))
        old_suppress = getattr(self.trainer, "_suppress_eval_layers", True)
        if needs_dynamic:
            # Original dynamic-region loss consumes the auxiliary opacity mask.
            self.trainer._suppress_eval_layers = False
        try:
            with torch.enable_grad() if grad else torch.no_grad():
                outputs = self.trainer(inputs, cameras, "raw")
        finally:
            if needs_dynamic:
                self.trainer._suppress_eval_layers = old_suppress
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        forward_ms = (time.perf_counter() - started) * 1000
        packet = {key: outputs[key] for key in
                  ("rgb", "albedo", "roughness", "metallic", "depth", "opacity")}
        packet.update({
            "normal": outputs["ego_normal"],
            "static_mask": outputs["ego_static_mask"],
            "camera_to_world": outputs["ego_camera_to_world"],
            "intrinsics": outputs["ego_intrinsics"],
            "viewdirs": outputs["ego_viewdirs"],
            "background_linear": outputs["ego_background_linear"],
            "frame_id": int(frame_id),
            "forward_ms": forward_ms,
            "image_infos": inputs,
            "target_rgb": infos["pixels"].detach().to(self.device),
            "evaluation_static": torch.as_tensor(
                evaluation_static_mask(infos), device=self.device, dtype=torch.bool),
        })
        # Reference-only evaluation masks. They never enter render_inputs().
        dynamic = infos.get("dynamic_masks")
        packet["evaluation_dynamic"] = (None if dynamic is None else
            torch.as_tensor(dynamic, device=self.device).reshape(packet["rgb"].shape[:2]) > .5)
        ego = infos.get("egocar_masks")
        packet["evaluation_valid"] = (torch.ones(packet["rgb"].shape[:2], device=self.device,
                                                 dtype=torch.bool) if ego is None else
            torch.as_tensor(ego, device=self.device).reshape(packet["rgb"].shape[:2]) < .5)
        if "lidar_depth_map" in infos:
            packet["evaluation_depth"] = infos["lidar_depth_map"].detach()
        if "layered_diagnostics" in outputs:
            packet["layered_diagnostics"] = outputs["layered_diagnostics"]
            packet["layered_diagnostic_maps"] = outputs.get("layered_diagnostic_maps", {})
            packet["layered_diffuse_linear"] = outputs["layered_diffuse_linear"]
            packet["layered_specular_linear"] = outputs["layered_specular_linear"]
        if return_outputs:
            packet["training_outputs"] = outputs
            packet["training_image_infos"] = to_device(infos, self.device, torch)
            packet["training_camera_infos"] = cameras
        return packet

    def shade(self, packet, albedo, roughness=None, metallic=None):
        """Shade current-view materials, preserving gradients to supplied maps."""
        materials = {
            "albedo": albedo,
            "roughness": packet["roughness"] if roughness is None else roughness,
            "metallic": packet["metallic"] if metallic is None else metallic,
            "normal": packet["normal"],
        }
        shaded = self.trainer.shade_gbuffer(materials, packet["viewdirs"], "raw")
        opacity = packet["opacity"]
        linear = shaded["rgb"] * opacity + packet["background_linear"] * (1 - opacity)
        return self.trainer.output_transform(
            self.trainer.display_transform(linear), packet["image_infos"])
