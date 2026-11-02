"""Raw Base v3/v4: material splatting, HDR lighting and camera response.

Recovered equations and reductions are documented in
docs/reconstruction/phase3_integration_provenance.md. Base v4 enables the
historical boundary LPIPS objective through its configuration overlay.
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization

from models.pbr import to_display
from models.gbuffer_attention_runtime import GBufferAttentionMixin
from models.rgb_joint_runtime import RGBJointMixin
from models.teacher_calibration_runtime import TeacherCalibrationMixin
from models.teacher_confidence_runtime import TeacherConfidenceMixin
from models.rdg_fusion_runtime import RDGFusionMixin
from models.dynamic_gbuffer_runtime import DynamicGBufferMixin
from models.trainers.scene_graph_weather import MultiTrainer_weather


OPENCV_TO_OPENGL = torch.tensor([1.0, -1.0, -1.0])


class _LayeredScreenGradients:
    """Collect every auxiliary projection's densification signal once.

    Full joint training checkpoints shading only: gsplat sets ``absgrad``
    on its saved projected tensor, an attribute that raster checkpointing
    would lose on detached replay tensors. Each hook consumes its original
    projection and releases the extra reference immediately. Only the summed
    statistic is retained, not one gradient array per group.

    Signed gradients sum before taking the norm. With AbsGrad, the sum of
    per-pass absolute gradients is a conservative statistic: cancellation
    between render passes is intentionally not recovered.
    """

    def __init__(self, info, absgrad):
        means = info["means2d"]
        if means.ndim != 3:
            raise ValueError("Joint layered densification requires packed=false")
        self.total = torch.zeros_like(means, requires_grad=False)
        self.absgrad = bool(absgrad)
        self.backward_passes = 0
        self._finalized = False
        self._pending_slots = []

    def track(self, info):
        means = info["means2d"]
        if not means.requires_grad:
            return
        if means.shape != self.total.shape:
            raise ValueError("Layered projections must preserve the full Gaussian order")
        slot = {"tensor": means}
        self._pending_slots.append(slot)

        def accumulate(gradient):
            tensor = slot.pop("tensor")
            value = getattr(tensor, "absgrad", None) if self.absgrad else gradient
            if value is None:
                raise RuntimeError("Missing layered projection AbsGrad after raster backward")
            with torch.no_grad():
                self.total.add_(value.detach())
            self.backward_passes += 1
            return gradient

        means.register_hook(accumulate)

    def finalize(self, info):
        if self._finalized:
            raise RuntimeError("Layered densification gradients were already finalized")
        self._finalized = True
        means = info["means2d"]
        attribute = "absgrad" if self.absgrad else "grad"
        primary = getattr(means, attribute, None)
        if primary is None:
            primary = torch.zeros_like(self.total)
        combined = primary.detach() + self.total
        setattr(means, attribute, combined)
        # Some diagnostic/global-color passes do not contribute to the loss.
        # Release their slots too, avoiding a tensor -> hook -> tensor cycle.
        for slot in self._pending_slots:
            slot.pop("tensor", None)
        self._pending_slots.clear()
        info["layered_gradient_stats"] = {
            "auxiliary_backward_passes": self.backward_passes,
            "auxiliary_grad_norm": float(self.total.norm()),
            "primary_grad_norm": float(primary.norm()),
            "combined_grad_norm": float(combined.norm()),
            "absgrad": self.absgrad,
        }


class MultiTrainer_pbr(DynamicGBufferMixin, RDGFusionMixin, TeacherConfidenceMixin, TeacherCalibrationMixin, RGBJointMixin, GBufferAttentionMixin, MultiTrainer_weather):
    def __init__(self, **kwargs):
        layered_config = kwargs.pop("layered_gbuffer", None)
        attention_config = kwargs.pop("gbuffer_attention", None)
        rgb_joint_config = kwargs.pop("rgb_joint", None)
        teacher_calibration_config = kwargs.pop("teacher_calibration", None)
        teacher_confidence_config = kwargs.pop("teacher_confidence", None)
        rdg_config = kwargs.pop("rdg_fusion", None)
        dynamic_gbuffer_config = kwargs.pop("dynamic_gbuffer", None)
        if kwargs.get("vgg_config") is not None:
            raise ValueError("Raw Base v3 uses its explicit PBR losses; set vgg: null")
        if list(kwargs.get("weathers") or ["raw"]) != ["raw"]:
            raise ValueError("This reconstruction stage supports raw Base v3/v4 only")
        super().__init__(**kwargs)
        data_cfg = getattr(self.dataset, "data_cfg", None)
        pixel_cfg = getattr(data_cfg, "pixel_source", None)
        self._block_holdout_train_ids = (
            tuple(int(frame) for frame in self.dataset.train_timesteps)
            if pixel_cfg is not None and pixel_cfg.get("test_frame_ranges") is not None else None
        )
        self.crop_frac = float(getattr(self.dataset.pixel_source, "gbuffer_crop_frac", 0.0))
        self._opencv_to_opengl = OPENCV_TO_OPENGL.to(self.device)
        from models.boundary_losses import validate_boundary_config
        boundary_cfg = validate_boundary_config(self.losses_dict.get("boundary", {}))
        from models.diffreg_material import validate_diffreg_config
        self._diffreg_config = validate_diffreg_config(self.losses_dict.get("diffreg_material"))
        self._last_diffreg_stats = {}
        if boundary_cfg["enabled"] and boundary_cfg["perceptual_w"]:
            # Use the existing frozen AlexNet feature network directly. Calling
            # the stateful evaluation Metric would accumulate training samples.
            self.lpips.net.requires_grad_(False)
            self.lpips.net.eval()
        # Stored in the ordinary trainer config, so fresh training and normal
        # checkpoint evaluation reconstruct the same rendering computation.
        self.configure_layered_gbuffer(layered_config)
        self.configure_gbuffer_attention(attention_config)
        self.configure_rgb_joint(rgb_joint_config)
        self.configure_teacher_calibration(teacher_calibration_config)
        self.configure_teacher_confidence(teacher_confidence_config)
        self._configure_rdg_fusion(rdg_config)
        self.configure_dynamic_gbuffer(dynamic_gbuffer_config)

    def _init_models(self):
        # null removes inherited RGB Sky. PBR uses the same HDR for sky/surface.
        for name in list(self.model_config):
            if self.model_config[name] is None:
                del self.model_config[name]
        if "Sky" in self.model_config:
            raise ValueError("PBR requires model.Sky=null and model.EnvLight")
        self.misc_classes_keys = list(self.misc_classes_keys) + ["EnvLight", "CameraResponse"]
        allowed = set(self.misc_classes_keys) | {
            "Background", "RigidNodes", "DeformableNodes", "SMPLNodes"
        }
        unknown = set(self.model_config) - allowed
        if unknown:
            raise ValueError(f"Unsupported Base v3 model classes: {sorted(unknown)}")
        super()._init_models()

    @property
    def env(self):
        return self.models["EnvLight"]

    def render_gaussians(self, gs, cam, update_info=True, **kwargs):
        """Rasterize signed material8 + expected camera-Z, without RGB clipping."""
        if gs.rgbs.shape[-1] != 8:
            raise ValueError(f"PBR requires 8 material channels, got {gs.rgbs.shape[-1]}")
        if self.render_cfg.batch_size != 1:
            raise ValueError("PBR currently supports batch_size=1")
        screen_gradients = None
        correction = (getattr(gs, "extras", None) or {}).get("correction_features")
        # Use one projection for material and correction features. Even during
        # warmup this gives the feature optimizer zero gradients/Adam moments,
        # so the native density split/prune operations retain their contract.
        payload = gs.rgbs if correction is None else torch.cat((gs.rgbs, correction), dim=-1)
        correction_image = None

        def render_fn(opacity_mask=None, return_info=False, features=None, feature_size=None):
            nonlocal correction_image
            # The layered reference changes colors only. Masking opacity here
            # would change the global transmittance and invalidate the comparison.
            if features is not None and (opacity_mask is not None or return_info):
                raise ValueError("Feature accumulation must preserve all Gaussian opacities")
            opacity = gs.opacities.reshape(-1)
            if opacity_mask is not None:
                opacity = opacity * opacity_mask
            options = dict(kwargs)
            width, height, intrinsics = int(cam.W), int(cam.H), cam.Ks
            if feature_size is not None:
                if (features is None or len(feature_size) != 2
                        or any(type(value) is not int or value < 1 for value in feature_size)):
                    raise ValueError("feature_size requires auxiliary features and positive (height, width)")
                height, width = feature_size
                intrinsics = cam.Ks.clone()
                intrinsics[0] *= width / int(cam.W)
                intrinsics[1] *= height / int(cam.H)
                if "radius_clip" in options:
                    options["radius_clip"] *= min(width / int(cam.W), height / int(cam.H))
            if features is not None:
                if features.ndim != 2 or features.shape[0] != gs.means.shape[0]:
                    raise ValueError("Auxiliary features must align with all collected Gaussians")
                options["render_mode"] = "RGB"
            renders, alphas, info = rasterization(
                means=gs.means, quats=gs.quats, scales=gs.scales,
                opacities=opacity, colors=payload if features is None else features,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None], Ks=intrinsics[None],
                width=width, height=height, packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad, sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **options,
            )
            if features is not None:
                if screen_gradients is not None:
                    screen_gradients.track(info)
                return renders[0]
            if renders.shape[-1] != payload.shape[-1] + 1:
                raise RuntimeError("Expected material8 + optional correction features + depth1 from gsplat")
            material, depth = renders[0, ..., :8], renders[0, ..., -1:]
            if return_info and correction is not None:
                coverage = alphas[0].detach()
                correction_image = renders[0, ..., 8:-1] / coverage.clamp_min(1e-4)
                correction_image = torch.where(coverage > 1e-4, correction_image, torch.zeros_like(correction_image))
            result = (material, depth, alphas[0])
            return (*result, info) if return_info else result

        material, depth, opacity, info = render_fn(return_info=True)
        if update_info:
            self.info = info
            if self.training and info["means2d"].requires_grad:
                info["means2d"].retain_grad()
            if (self.training and torch.is_grad_enabled()
                    and getattr(self, "_layered_gbuffer", None) is not None):
                screen_gradients = _LayeredScreenGradients(info, self.render_cfg.absgrad)
            self._layered_screen_gradients = screen_gradients
        outputs = {"rgb_gaussians": material, "depth": depth, "opacity": opacity}
        if correction_image is not None:
            outputs["_rgb_correction_features"] = correction_image
        return outputs, render_fn

    def postprocess_per_train_step(self, step):
        screen_gradients = getattr(self, "_layered_screen_gradients", None)
        if screen_gradients is not None:
            screen_gradients.finalize(self.info)
        try:
            result = super().postprocess_per_train_step(step)
            self.complete_camera_update(step)
            return result
        finally:
            self._layered_screen_gradients = None

    def unpack_gbuffer(self, feat, opacity):
        g = feat / opacity.clamp(min=1e-4)
        return {
            "albedo": g[..., :3].clamp(0, 1),
            "metallic": g[..., 3:4].clamp(0, 1),
            "roughness": g[..., 4:5].clamp(0.01, 1),
            "normal": F.normalize(g[..., 5:8], dim=-1),
        }

    def refined_viewdirs(self, image_infos, camera_infos, cam):
        r_ref = cam.camtoworlds[:3, :3]
        r_gt = camera_infos["camera_to_world"][:3, :3]
        return image_infos["viewdirs"] @ (r_ref @ torch.linalg.inv(r_gt)).T

    def supports_weather(self, weather):
        return weather == "raw"

    def configure_layered_gbuffer(self, config=None):
        """Enable grouped shading for inference or full joint training.

        This renderer adds no learned parameters. The ordinary GS, material,
        motion and lighting parameters remain trainable. Persist this setting
        as ``trainer.layered_gbuffer`` when saving a training configuration.
        Legacy material/depth outputs remain the original merged buffers, so
        existing material supervision is identical across the four conditions.
        """
        if config is None:
            self._layered_gbuffer = None
            return
        defaults = {"grouping": "instance", "color_space": "srgb",
                    "max_layers": 0, "ablation": "none", "diagnostics": True}
        unknown = set(config) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown layered G-buffer settings: {sorted(unknown)}")
        defaults.update(config)
        if defaults["grouping"] not in ("single", "dynamic", "instance"):
            raise ValueError("Layer grouping must be single, dynamic or instance")
        if defaults["color_space"] not in ("srgb", "linear"):
            raise ValueError("Layer color_space must be srgb or linear")
        layers = defaults["max_layers"]
        if isinstance(layers, bool) or not isinstance(layers, int) or layers < 0 or layers == 1:
            raise ValueError("max_layers must be 0 (unlimited) or an integer >= 2")
        if defaults["ablation"] not in ("none", "normal", "brdf", "albedo"):
            raise ValueError("Unknown layered component ablation")
        if not isinstance(defaults["diagnostics"], bool):
            raise ValueError("diagnostics must be boolean")
        if getattr(self, "_temporal_material", None) is not None:
            raise ValueError("Layered experiment must run without temporal history")
        if getattr(self, "gbuffer_attention_config", None) is not None:
            raise ValueError("Layered rendering cannot be combined with G-buffer attention")
        self._layered_gbuffer = defaults

    def _layered_group_ids(self, grouping):
        """Integer identities aligned with collection order, never blended IDs.

        Local point_ids identify instances, not persistent Gaussian identities.
        Namespacing by node prevents rigid/deformable/SMPL instance collisions.
        """
        labels = self.pts_labels
        if grouping == "single":
            return torch.zeros_like(labels, dtype=torch.long)
        background = self.gaussian_classes["Background"]
        if grouping == "dynamic":
            return (labels != background).long()
        groups = torch.zeros_like(labels, dtype=torch.long)
        offset = 1
        for name, label in self.gaussian_classes.items():
            if name == "Background":
                continue
            node = self.models[name]
            target = labels == label
            count = int(target.sum().item())
            ids = getattr(node, "point_ids", None)
            if ids is None:
                if count:
                    raise ValueError(f"{name} has no instance point_ids for layered rendering")
                continue
            ids = ids.reshape(-1)
            if ids.dtype not in (torch.int32, torch.int64) or bool((ids < 0).any()):
                raise ValueError(f"{name} instance point_ids must be nonnegative integers")
            namespace_size = int(ids.max().item()) + 1 if ids.numel() else 0
            # SMPL can omit an entire node for a frame while retaining its
            # canonical points. Reserve its namespace even when absent.
            if count == 0:
                offset += namespace_size
                continue
            mask = getattr(node, "filter_mask", None)
            if mask is not None:
                if mask.dtype != torch.bool or mask.shape != ids.shape:
                    raise ValueError(f"{name} filter_mask does not align with instance IDs")
                ids = ids[mask]
            if ids.numel() != count:
                raise ValueError(f"{name} instance IDs do not align with collected Gaussians")
            groups[target] = ids.to(device=labels.device, dtype=torch.long) + offset
            offset += namespace_size
        return groups

    def prepare_weather_render(self, weather):
        """Hook for residual trainers; the raw renderer has no weather state."""

    def environment_weather_strength(self, weather):
        return 0.0

    def shade_gbuffer(self, gbuf, viewdirs, weather="raw"):
        return self.env.shade(
            gbuf["albedo"], gbuf["metallic"], gbuf["roughness"],
            gbuf["normal"], -F.normalize(viewdirs, dim=-1),
            weather=weather, weather_strength=self.environment_weather_strength(weather),
        )

    def display_transform(self, linear_rgb):
        response = self.models.get("CameraResponse")
        return to_display(linear_rgb) if response is None else response(linear_rgb)

    def diagnostic_display_transform(self, linear_rgb):
        # Historical optimized path: diagnostic layers need no response graph.
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                return self.display_transform(linear_rgb)
        return self.display_transform(linear_rgb)

    def output_transform(self, display_rgb, image_infos):
        if getattr(self.env, "external", None) is not None:
            return display_rgb
        if getattr(self, "_temporal_shared_affine", False) and not self.training:
            # AffineTransform uses its mean embedding without img_idx. Apply
            # exactly the same response to the unfiltered and transported
            # renders, including transitions across the original train split.
            image_infos = {key: value for key, value in image_infos.items()
                           if key != "img_idx"}
        return self.affine_transformation(display_rgb, image_infos)

    def configure_temporal_material(self, config=None, *, shared_affine=True, domain="material"):
        """Enable an ephemeral, inference-only static material history.

        Pass None to disable. No parameters or checkpoint buffers are added;
        old checkpoints and the default training/rendering path are unchanged.
        Both paired RGB outputs use a shared affine by default, so these
        comparisons should not be mixed with the original image-specific
        affine evaluation scores.
        """
        if config is None:
            self._temporal_material = None
            self._temporal_shared_affine = False
            self._temporal_domain = "material"
            return
        if domain not in ("material", "rgb"):
            raise ValueError("Temporal transport domain must be material or rgb")
        if getattr(self, "gbuffer_attention_config", None) is not None:
            raise ValueError("Temporal transport cannot be combined with G-buffer attention")
        from models.temporal_material import TemporalMaterialResolver
        self._temporal_material = TemporalMaterialResolver(config)
        self._temporal_shared_affine = bool(shared_affine)
        self._temporal_domain = domain

    def configure_surface_editor(self, editor=None):
        """Attach a finalized static edit bank; never update it during rendering."""
        self._surface_editor = editor
        self.reset_temporal_material()

    def reset_temporal_material(self):
        """Clear history before a different sequence or checkpoint."""
        resolver = getattr(self, "_temporal_material", None)
        if resolver is not None:
            resolver.reset()

    def _resolve_temporal_material(self, gbuf, outputs, render_fn, cam,
                                   image_infos, camera_infos, weather, signal=None):
        resolver = self._temporal_material
        # Standalone dynamic opacity also rejects occluded dynamic nodes.
        # This is deliberately conservative: only confident background
        # interiors may reuse history. No target image/segmentation is read.
        dynamic_mask = self.pts_labels != self.gaussian_classes["Background"]
        if bool(dynamic_mask.any()):
            _, _, dynamic_opacity = render_fn(dynamic_mask)
        else:
            dynamic_opacity = torch.zeros_like(outputs["opacity"])
        static = (
            (outputs["opacity"][..., 0] >= resolver.config["min_opacity"])
            & torch.isfinite(dynamic_opacity[..., 0])
            & (dynamic_opacity[..., 0] <= 0.01)
        )
        frame_id = int(image_infos["frame_idx"].flatten()[0].item())
        camera_id = int(camera_infos["cam_id"].flatten()[0].item())
        resolved, diagnostics = resolver.resolve(
            gbuffer=gbuf, depth=outputs["depth"], opacity=outputs["opacity"],
            static_mask=static, camera_to_world=cam.camtoworlds,
            intrinsics=cam.Ks, frame_id=frame_id,
            stream_id=(weather, camera_id),
            signal=signal,
        )
        outputs.update(diagnostics)
        # Independent of history acceptance; useful for paired static-region
        # image metrics, never a per-method temporal evaluation mask.
        outputs["temporal_static_mask"] = static[..., None]
        return resolved

    def forward(self, image_infos, camera_infos, weather="raw", novel_view=False):
        attention_cfg = getattr(self, "gbuffer_attention_config", None)
        attention_context = getattr(self, "_gbuffer_attention_context", False)
        rdg_context = getattr(self, "_rdg_context", False)
        context_render = attention_context or rdg_context
        rdg_cfg = getattr(self, "rdg_fusion_config", None)
        rdg_active = bool(rdg_cfg and rdg_cfg["enabled"])
        rdg_previous, rdg_strength_value = None, 0.
        if rdg_active:
            self.assert_rdg_compatible()
            if novel_view or weather != "raw":
                raise ValueError("RDG fusion requires raw recorded camera sequences")
            if "frame_idx" not in image_infos or "cam_id" not in camera_infos:
                raise ValueError("RDG fusion requires frame_idx and cam_id")
            rdg_strength_value = self.rdg_strength()
            rdg_active = rdg_context or rdg_strength_value > 0
            if rdg_active and not context_render:
                # One low-resolution, detached context completes before the
                # current view takes ownership of raster/refinement state.
                rdg_previous = self.rdg_history(image_infos)
        attention = attention_cfg is not None and attention_cfg["variant"] != "B0"
        attention_histories, attention_previous, attention_ids = [], None, []
        attention_strength_value = 0.
        if attention:
            if novel_view or weather != "raw":
                raise ValueError("G-buffer attention requires raw recorded camera sequences")
            if "frame_idx" not in image_infos or "cam_id" not in camera_infos:
                raise ValueError("G-buffer attention requires frame_idx and cam_id")
            from models.gbuffer_attention import attention_strength
            attention_step = self.attention_step if self.attention_step is not None else self.step
            attention_strength_value = attention_strength(attention_cfg, attention_step, self.training)
            attention = context_render or attention_strength_value > 0
            if attention and not context_render:
                # These no-grad renders finish before the current view takes
                # ownership of raster/normal regularization and AbsGrad state.
                attention_histories, attention_previous, attention_ids = self._attention_history.prepare(image_infos)
        temporal = getattr(self, "_temporal_material", None) is not None
        layered_config = getattr(self, "_layered_gbuffer", None)
        if layered_config is not None and (temporal or weather != "raw"):
            raise ValueError("Layered G-buffer experiment requires raw rendering without history")
        editor = getattr(self, "_surface_editor", None)
        if editor is not None and not temporal:
            raise ValueError("Surface editing requires the temporal static-surface guard")
        if temporal:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("Temporal material transport requires eval mode and torch.no_grad()")
            if novel_view or weather != "raw":
                raise ValueError("Temporal material transport currently supports raw recorded camera sequences only")
            if "frame_idx" not in image_infos or "cam_id" not in camera_infos:
                raise ValueError("Temporal material transport requires frame_idx and cam_id for history isolation")
        if not self.supports_weather(weather):
            raise ValueError(f"Unsupported weather for this trainer: {weather}")
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(torch.abs(self.normalized_timestamps - normed_time))
        for model in self.models.values():
            if hasattr(model, "in_test_set"):
                model.in_test_set = self.in_test_set
        for name in self.gaussian_classes:
            if getattr(self, "_block_holdout_train_ids", None) is not None:
                self.models[name].train_frame_ids = self._block_holdout_train_ids
            if hasattr(self.models[name], "set_cur_frame"):
                self.models[name].set_cur_frame(self.cur_frame)
        cam = self.process_camera(camera_infos, image_infos["img_idx"].flatten()[0], novel_view)
        self.prepare_weather_render(weather)
        gs = self.collect_gaussians(cam, weather, image_infos["img_idx"].flatten()[0])
        outputs, render_fn = self.render_gaussians(
            gs, cam, near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane, render_mode="RGB+ED",
            radius_clip=self.render_cfg.get("radius_clip", 0.0),
            update_info=not context_render,
        )
        self.prepare_dynamic_gbuffer(outputs, render_fn, gs, image_infos)
        viewdirs = self.refined_viewdirs(image_infos, camera_infos, cam)
        self._render_c2w = cam.camtoworlds
        self._render_intrinsics = cam.Ks
        raw_material = outputs.pop("rgb_gaussians")
        gbuf = self.unpack_gbuffer(raw_material, outputs["opacity"])
        if getattr(self, "_export_ego_inputs", False) or attention or rdg_active:
            # Opt-in surface packets for motion experiments. Keep the world
            # normal before the shader's final view-facing orientation. The
            # Gaussian packer itself can flip its sign towards the camera,
            # so correspondence checks must still use sign-invariant cosine.
            outputs["ego_normal"] = gbuf["normal"]
            outputs["ego_viewdirs"] = viewdirs
            outputs["ego_camera_to_world"] = cam.camtoworlds
            outputs["ego_intrinsics"] = cam.Ks
            # A predicted static-surface mask must not read target masks.
            # This auxiliary pass is detached even during material fitting.
            with torch.no_grad():
                dynamic = self.pts_labels != self.gaussian_classes["Background"]
                if bool(dynamic.any()):
                    _, _, dynamic_opacity = render_fn(dynamic)
                else:
                    dynamic_opacity = torch.zeros_like(outputs["opacity"])
                outputs["ego_static_mask"] = (
                    (outputs["opacity"][..., 0] >= .95)
                    & torch.isfinite(dynamic_opacity[..., 0])
                    & (dynamic_opacity[..., 0] <= .01)
                )
        original_attention_gbuf = None
        if rdg_active:
            packet = {
                **gbuf, "rgb": gbuf["albedo"], "depth": outputs["depth"],
                "opacity": outputs["opacity"], "viewdirs": viewdirs,
                "static_mask": outputs["ego_static_mask"],
                "camera_to_world": cam.camtoworlds, "intrinsics": cam.Ks,
                "frame_id": int(image_infos["frame_idx"].flatten()[0]),
                "image_id": int(image_infos["img_idx"].flatten()[0]),
                "stream_id": (weather, int(camera_infos["cam_id"].flatten()[0])),
            }
            # rgb is a generated-albedo proxy required by the shared geometry
            # packet contract. RDG neither shades nor encodes an RGB guide.
            if rdg_context:
                return {"_rdg_packet": packet}
            refined = self.rdg_fusion_head(packet, rdg_previous, strength=rdg_strength_value)
            original_attention_gbuf = gbuf
            gbuf = {**gbuf, "albedo": refined["albedo"]}
            for name in ("albedo", "history_weight", "valid", "residual_linear"):
                outputs[f"rdg_{name}"] = refined[name]
            outputs["rdg_context_ids"] = [] if rdg_previous is None else [rdg_previous["image_id"]]
            if self.training:
                outputs["_rdg_loss_anchor"] = refined["loss_anchor"]
                outputs["_rdg_loss_temporal"] = refined["loss_temporal"]
        if attention:
            # Generate guide RGB before affine, with no target image input.
            with torch.no_grad():
                guide_shaded = self.shade_gbuffer(gbuf, viewdirs, weather)
                guide_sky = self.env.background(
                    viewdirs, weather=weather,
                    weather_strength=self.environment_weather_strength(weather))
                guide_rgb = self.display_transform(
                    guide_shaded["rgb"] * outputs["opacity"] +
                    guide_sky * (1 - outputs["opacity"]))
            frame_id = int(image_infos["frame_idx"].flatten()[0])
            packet = {
                **gbuf, "rgb": guide_rgb.detach(), "depth": outputs["depth"],
                "opacity": outputs["opacity"], "viewdirs": viewdirs,
                "static_mask": outputs["ego_static_mask"],
                "camera_to_world": cam.camtoworlds, "intrinsics": cam.Ks,
                "frame_id": frame_id,
                "stream_id": (weather, int(camera_infos["cam_id"].flatten()[0])),
            }
            if context_render:
                return {"_attention_packet": packet}
            self._attention_history.remember(int(image_infos["img_idx"].flatten()[0]), packet)
            refined = self.gbuffer_attention_head(
                packet, attention_histories, strength=attention_strength_value)
            original_attention_gbuf = gbuf
            gbuf = {**gbuf, "albedo": refined["albedo"]}
            outputs["attention_albedo"] = refined["albedo"]
            outputs["attention_history_weight"] = refined["history_weight"]
            outputs["attention_valid"] = refined["valid"]
            outputs["attention_residual_linear"] = refined["residual_linear"]
            outputs["attention_context_ids"] = attention_ids
            if self.training:
                outputs["_attention_current"] = {**packet, "albedo": refined["albedo"]}
                outputs["_attention_previous"] = attention_previous
                outputs["_attention_surface"] = refined["surface_valid"]
        if temporal:
            unfiltered_shaded = self.shade_gbuffer(gbuf, viewdirs, weather)
            original_gbuf = gbuf
            gbuf = self._resolve_temporal_material(
                gbuf, outputs, render_fn, cam, image_infos, camera_infos, weather,
                signal=(unfiltered_shaded["rgb"]
                        if getattr(self, "_temporal_domain", "material") == "rgb" else None),
            )
            if getattr(self, "_export_temporal_inputs", False):
                outputs.update({f"unfiltered_{key}": value for key, value in original_gbuf.items()})
                outputs["unfiltered_surface_rgb"] = self.display_transform(unfiltered_shaded["rgb"])
            if getattr(self, "_temporal_domain", "material") == "rgb":
                gbuf = original_gbuf
            if editor is not None:
                if getattr(self, "_temporal_domain", "material") != "material":
                    raise ValueError("Surface editing cannot be combined with RGB history")
                gbuf, edit_diagnostics = editor.apply(
                    gbuffer=gbuf, depth=outputs["depth"], opacity=outputs["opacity"],
                    static_mask=outputs["temporal_static_mask"][..., 0],
                    camera_to_world=cam.camtoworlds, intrinsics=cam.Ks,
                    raw_rgb=self.display_transform(unfiltered_shaded["rgb"]),
                )
                outputs.update(edit_diagnostics)
        layered = None
        if layered_config is not None:
            from models.layered_gbuffer import render_layered_gbuffer
            layered = render_layered_gbuffer(
                gs.rgbs, self._layered_group_ids(layered_config["grouping"]),
                lambda colors: render_fn(features=colors),
                lambda material: self.shade_gbuffer(material, viewdirs, weather),
                color_space=layered_config["color_space"],
                max_layers=layered_config["max_layers"],
                ablation=layered_config["ablation"],
                merged_feature_srgb=raw_material,
                merged_opacity=outputs["opacity"],
                checkpoint_raster=getattr(self, "_layered_screen_gradients", None) is None,
            )
            if layered_config["diagnostics"]:
                outputs["layered_diagnostics"] = layered["diagnostics"]
                outputs["layered_diagnostic_maps"] = layered.get("diagnostic_maps", {})
        shaded = self.shade_gbuffer(gbuf, viewdirs, weather)
        if temporal and getattr(self, "_temporal_domain", "material") == "rgb":
            shaded["rgb"] = outputs.pop("temporal_signal")
        # Original teacher supervision remains attached to unfused material.
        outputs.update(gbuf if original_attention_gbuf is None else original_attention_gbuf)
        outputs["normal"] = shaded["normal_oriented"]
        outputs["diffuse_linear"] = shaded["diffuse"]
        outputs["specular_linear"] = shaded["specular"]
        outputs["n_dot_v"] = (
            outputs["normal"] * -F.normalize(viewdirs, dim=-1)
        ).sum(-1, keepdim=True).clamp(1e-4, 1)
        for name in ("diffuse", "specular"):
            outputs[f"rgb_{name}"] = self.diagnostic_display_transform(
                shaded[name] * outputs["opacity"] if layered is None
                else layered[f"{name}_surface"]
            )
        sky = self.env.background(
            viewdirs, weather=weather,
            weather_strength=self.environment_weather_strength(weather),
        )
        if getattr(self, "_export_ego_inputs", False):
            outputs["ego_background_linear"] = sky
        surface_linear = (shaded["rgb"] * outputs["opacity"] if layered is None
                          else layered["linear_surface"])
        linear = surface_linear + sky * (1 - outputs["opacity"])
        if layered is not None:
            # These are premultiplied by the original global visibility weights.
            outputs["layered_diffuse_linear"] = layered["diffuse_surface"]
            outputs["layered_specular_linear"] = layered["specular_surface"]
        outputs["rgb_sky"] = self.diagnostic_display_transform(sky)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1 - outputs["opacity"])
        outputs["rgb"] = self.output_transform(self.display_transform(linear), image_infos)
        self.apply_rgb_joint(outputs, gbuf, viewdirs, weather)
        if temporal:
            unfiltered_linear = (
                unfiltered_shaded["rgb"] * outputs["opacity"]
                + sky * (1 - outputs["opacity"])
            )
            outputs["rgb_unfiltered"] = self.output_transform(
                self.display_transform(unfiltered_linear), image_infos,
            )
        if getattr(self, "_export_linear_rgb", False):
            outputs["rgb_linear"] = linear

        # These auxiliary renders never overwrite the full render's means2d.
        if not getattr(self, "_suppress_eval_layers", False):
            masks = {}
            if not self.training and self.render_each_class:
                masks.update({name: self.pts_labels == label
                              for name, label in self.gaussian_classes.items()})
            if not self.training or self.render_dynamic_mask:
                masks["Dynamic"] = self.pts_labels != self.gaussian_classes["Background"]
            with torch.no_grad():
                for name, mask in masks.items():
                    feat, depth, opacity = render_fn(mask)
                    layer = self.shade_gbuffer(self.unpack_gbuffer(feat, opacity), viewdirs, weather)
                    outputs[name + "_rgb"] = self.output_transform(
                        self.display_transform(layer["rgb"] * opacity), image_infos
                    )
                    outputs[name + "_opacity"] = opacity
                    outputs[name + "_depth"] = depth
        return outputs

    def _crop_and_resize(self, x: torch.Tensor, target_hw: Tuple[int, int]):
        h = x.shape[0]
        top = round(h * self.crop_frac)
        return self._resize_down(x[top:h - top], target_hw)

    @staticmethod
    def _resize_down(x, target_hw):
        if x.shape[:2] == target_hw:
            return x
        return F.interpolate(
            x.permute(2, 0, 1)[None].float(), size=target_hw,
            mode="bilinear", antialias=True, align_corners=False,
        )[0].permute(1, 2, 0)

    def gbuffer_losses(self, outputs, image_infos, cam_infos):
        cfg = self.losses_dict
        gh, gw = image_infos["gbuf_basecolor"].shape[:2]
        w = min(outputs["albedo"].shape[1], gw)
        target_hw = (round(w * gh / gw), w)
        valid = image_infos.get("egocar_masks")
        valid = torch.ones_like(image_infos["sky_masks"]) if valid is None else 1 - valid
        mask = self._crop_and_resize(
            ((1 - image_infos["sky_masks"]) * valid)[..., None], target_hw
        )
        mask = (mask > 0.99).float()
        denom = mask.sum().clamp(min=1)
        calibrated, losses = self.calibrated_teacher_targets(outputs, image_infos, target_hw, mask)
        confidence = self.teacher_confidence_weights(image_infos, target_hw, mask)
        material_mask = mask if confidence is None else mask * confidence
        for name, target in (("albedo", "basecolor"), ("metallic", "metallic"),
                             ("roughness", "roughness")):
            if cfg.get(name) is not None:
                pred = self._crop_and_resize(outputs[name], target_hw)
                gt = (calibrated[name] if calibrated is not None else
                      self._resize_down(image_infos[f"gbuf_{target}"], target_hw))
                losses[f"{name}_loss"] = cfg[name].w * (material_mask * (pred - gt).abs()).sum() / (
                    denom * pred.shape[-1]
                )
        normal_cfg = cfg.get("normal")
        if normal_cfg is not None:
            c2w = getattr(self, "_render_c2w", cam_infos["camera_to_world"])
            r_wc = torch.linalg.inv(c2w)[:3, :3]
            pred = self._crop_and_resize(outputs["normal"], target_hw)
            pred = F.normalize(pred @ r_wc.T * self._opencv_to_opengl, dim=-1)
            gt = F.normalize(self._resize_down(image_infos["gbuf_normal"], target_hw), dim=-1)
            cos = (pred * gt).sum(-1, keepdim=True)
            residual = 1 - cos
            c = normal_cfg.get("robust_c", 0.5)
            losses["normal_loss"] = normal_cfg.w * (
                mask * residual / (1 + residual / c)
            ).sum() / denom
            self._last_normal_cos = (mask * cos).sum().item() / denom.item()
        return losses

    def diffreg_losses(self, outputs, image_infos, weather="raw"):
        """Train-only material relations using the current view's existing pass.

        Align with the original teacher crop/resize convention. Sky, ego car,
        moving objects and low-opacity surfaces never connect to another pixel.
        This adds no renderer state, model parameters or inference computation.
        """
        from models.diffreg_material import (
            diffreg_weight, material_relation_loss, validate_diffreg_config,
        )
        cfg = getattr(self, "_diffreg_config", None)
        if cfg is None:
            cfg = validate_diffreg_config(self.losses_dict.get("diffreg_material"))
        self._last_diffreg_stats = {}
        if not self.training or weather != "raw" or not diffreg_weight(self.step, cfg):
            return {}
        for key in ("sky_masks", "dynamic_masks"):
            if key not in image_infos:
                raise ValueError(f"DiffReg static support requires {key}")
        gh, gw = image_infos["gbuf_basecolor"].shape[:2]
        width = min(outputs["albedo"].shape[1], gw)
        target_hw = (round(width * gh / gw), width)
        prediction, teacher = {}, {}
        for key, reference in (("albedo", "basecolor"), ("roughness", "roughness"),
                               ("metallic", "metallic")):
            prediction[key] = self._crop_and_resize(outputs[key], target_hw)
            teacher[key] = self._resize_down(image_infos[f"gbuf_{reference}"], target_hw)
        with torch.no_grad():
            sky = image_infos["sky_masks"]
            valid = torch.ones_like(sky[..., 0] if sky.ndim == 3 else sky)
            for key in ("sky_masks", "dynamic_masks", "egocar_masks"):
                if key in image_infos:
                    mask = image_infos[key]
                    if mask.ndim == 3:
                        mask = mask[..., 0]
                    valid = valid * (torch.isfinite(mask) & (mask <= 0)).to(valid)
            valid = self._crop_and_resize(valid[..., None], target_hw)[..., 0] > .99
            opacity = self._crop_and_resize(outputs["opacity"].detach(), target_hw)[..., 0]
            valid &= torch.isfinite(opacity) & (opacity >= .95)
            depth = self._crop_and_resize(outputs["depth"].detach(), target_hw)
            normal = self._crop_and_resize(outputs["normal"].detach(), target_hw)
        # img_idx identifies this current training image; no neighboring image,
        # held-out target or model-generated history is accessed.
        sample_key = int(image_infos["img_idx"].reshape(-1)[0])
        loss, stats = material_relation_loss(
            prediction, teacher, valid, step=self.step, config=cfg,
            sample_key=sample_key, depth=depth, normal=normal,
        )
        self._last_diffreg_stats = stats
        return {"diffreg_material_loss": loss}

    def compute_losses(self, outputs, image_infos, cam_infos, weather="raw"):
        # Missing teacher targets must not silently turn PBR into RGB-only fit.
        required = [f"gbuf_{k}" for k in ("basecolor", "metallic", "roughness", "normal")]
        if any(key not in image_infos for key in required):
            raise ValueError("Base v3/v4 training requires all four G-buffer targets")
        losses = super().compute_losses(outputs, image_infos, cam_infos, weather)
        boundary_cfg = self.losses_dict.get("boundary", {})
        if boundary_cfg.get("enabled", False) and weather == "raw":
            from models.boundary_losses import boundary_rgb_losses
            valid = 1.0 - image_infos.get(
                "egocar_masks", torch.zeros_like(image_infos["pixels"][..., 0])
            )
            losses.update(boundary_rgb_losses(
                outputs["rgb"], image_infos["pixels"], valid, self.step,
                boundary_cfg, self.lpips.net,
            ))
        losses.update(self.gbuffer_losses(outputs, image_infos, cam_infos))
        losses.update(self.dynamic_gbuffer_losses(outputs, image_infos, cam_infos))
        losses.update(self.diffreg_losses(outputs, image_infos, weather))
        losses.update(self.env.compute_reg_loss(self.model_config.EnvLight.get("reg")))
        if "CameraResponse" in self.models:
            losses.update(self.models["CameraResponse"].compute_reg_loss(
                self.model_config.CameraResponse.get("reg")
            ))
        losses.update(self.attention_losses(outputs))
        losses.update(self.rdg_losses(outputs))
        losses.update(self.rgb_joint_losses(outputs, image_infos))
        return losses

    def load_state_dict(self, state_dict, load_only_model=True, strict=True):
        self.reset_temporal_material()
        # Anchors describe the old checkpoint's geometry/materials.
        self._surface_editor = None
        # The saved config predates data-dependent removal of empty node
        # classes. Recreate that topology before the inherited weight loader.
        saved = state_dict["models"]
        for name in list(self.gaussian_classes):
            if name not in saved:
                del self.gaussian_classes[name]
                del self.models[name]
                del self.model_config[name]
        if strict and set(saved) != set(self.models):
            raise ValueError(
                f"Checkpoint model mismatch: saved={sorted(saved)}, "
                f"configured={sorted(self.models)}"
            )
        # BasicTrainer pops its envelope keys; do not mutate the caller's file.
        return super().load_state_dict(dict(state_dict), load_only_model, strict)
