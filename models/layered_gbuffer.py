"""Visibility-preserving, grouped deferred shading reference implementation.

The rasterizer callback MUST keep every Gaussian, its opacity, and its depth
ordering unchanged. Only the supplied feature colors may change. Every group
mass is then ``sum_i T_i * alpha_i`` with the original global transmittance;
the shaded groups are summed directly, without a second alpha composition.

This module deliberately imports neither gsplat nor the CUDA PBR module. A
linear feature accumulator is sufficient for CPU correctness tests. The
multi-pass reference prioritizes attribution and correctness over speed.
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Match models.pbr.srgb_to_linear without importing its CUDA dependency."""
    return torch.where(x <= 0.04045, x / 12.92,
                       ((x.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Match models.pbr.linear_to_srgb, including finite derivatives at zero."""
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92,
                       1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


def _validate_color_space(color_space: str) -> None:
    if color_space not in ("srgb", "linear"):
        raise ValueError("color_space must be 'srgb' or 'linear'")


def unpack_features(feat: torch.Tensor, mass: torch.Tensor,
                    color_space: str = "srgb") -> Dict[str, torch.Tensor]:
    """Unpack premultiplied material8, using the trainer's 1e-4 denominator.

    Layout is albedo[3], metallic[1], roughness[1], signed normal[3]. The
    returned albedo is always sRGB because that is the current shader API.
    ``linear`` means that albedo was linearized *before* rasterization.
    """
    _validate_color_space(color_space)
    if feat.shape[-1] != 8 or mass.shape != (*feat.shape[:-1], 1):
        raise ValueError("Expected material (..., 8) and matching mass (..., 1)")
    g = feat / mass.clamp(min=1e-4)
    albedo = g[..., :3].clamp(0, 1)
    if color_space == "linear":
        albedo = linear_to_srgb(albedo)
    return {
        "albedo": albedo,
        "metallic": g[..., 3:4].clamp(0, 1),
        "roughness": g[..., 4:5].clamp(0.01, 1),
        "normal": F.normalize(g[..., 5:8], dim=-1),
    }


def render_layered_gbuffer(
    features: torch.Tensor,
    group_ids: torch.Tensor,
    rasterize_features: Callable[[torch.Tensor], torch.Tensor],
    shade: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    *,
    color_space: str = "srgb",
    max_layers: int = 0,
    ablation: str = "none",
    merged_feature_srgb: Optional[torch.Tensor] = None,
    merged_opacity: Optional[torch.Tensor] = None,
    checkpoint_raster: bool = True,
) -> Dict[str, object]:
    """Shade surface groups with unchanged global visibility weights.

    ``rasterize_features`` maps (N,C) colors to premultiplied (...,C) features
    with a zero feature background. It must neither drop non-group Gaussians
    nor mask their opacities. ``shade`` returns linear ``rgb``, ``diffuse`` and
    ``specular``; ``normal_oriented`` is optional. No sky, camera response,
    exposure, or tone mapping is applied here.

    ``max_layers=0`` streams and shades every group. K>0 keeps K-1 strongest
    groups independently at each pixel, and sums *all* remaining groups into
    one overflow layer. K=1 is therefore a merged-buffer control. This is a
    contribution-based compression, not a depth ordering or top-K Gaussian
    operation; depth ordering is already in the global raster weights.

    ``ablation`` replaces normal, BRDF (metallic+roughness), or albedo in each
    layer with the corresponding merged-buffer value. Grouping and weights
    remain unchanged. The supplied merged sRGB buffer/opacity can reuse the
    trainer's original render and are retained for existing material losses.

    Tiny groups are normalized by their own actual nonzero mass, then scaled
    by the *global* A/clamp(A,1e-4). Thus splitting one material into fading
    groups does not introduce a per-group 1e-4 attenuation threshold, while
    preserving the baseline's global low-opacity behavior.

    Full-group inference uses image-sized working buffers rather than a
    GxHxW stack. Training checkpoints each complete group raster/shade using
    shared Gaussian features, recomputing image-sized intermediates backward.
    Compressed variants checkpoint the raster and final shading separately.
    Callbacks must be deterministic and retain the same geometry/lighting
    until backward completes; no persistent rendering state is mutated here.
    Full joint training sets ``checkpoint_raster=False`` and checkpoints only
    shading. Keeping the original raster graphs preserves gsplat's projection
    ``absgrad`` attributes for densification; raster recomputation detaches
    saved tensors and loses those attributes on the original projection.
    """
    _validate_color_space(color_space)
    if features.ndim != 2 or features.shape[1] != 8 or not features.is_floating_point():
        raise ValueError("features must be a floating-point (N,8) tensor")
    if group_ids.shape != features.shape[:1] or group_ids.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    ):
        raise ValueError("group_ids must be an integer (N,) tensor")
    if group_ids.device != features.device:
        raise ValueError("group_ids and features must be on the same device")
    if not isinstance(max_layers, int) or isinstance(max_layers, bool) or max_layers < 0:
        raise ValueError("max_layers must be a nonnegative integer")
    if ablation not in ("none", "normal", "brdf", "albedo"):
        raise ValueError("ablation must be none, normal, brdf, or albedo")
    if not isinstance(checkpoint_raster, bool):
        raise ValueError("checkpoint_raster must be boolean")

    selected = features
    if color_space == "linear":
        selected = torch.cat((srgb_to_linear(features[:, :3]), features[:, 3:]), -1)
    colors = torch.cat((selected, torch.ones_like(selected[:, :1])), -1)

    def raster(colors_to_render):
        rendered = rasterize_features(colors_to_render)
        if not isinstance(rendered, torch.Tensor) or rendered.shape[-1] != colors_to_render.shape[-1]:
            raise ValueError("rasterize_features must return accumulated features with unchanged channel count")
        return rendered

    if color_space == "srgb" and merged_feature_srgb is not None and merged_opacity is not None:
        global_render = torch.cat((merged_feature_srgb, merged_opacity), -1)
    else:
        global_render = raster(colors)
    merged_feature = global_render[..., :8]
    raster_mass = global_render[..., 8:9]
    opacity = raster_mass if merged_opacity is None else merged_opacity
    if opacity.shape != raster_mass.shape:
        raise ValueError("merged_opacity does not match rasterized image shape")
    if merged_feature_srgb is None:
        merged_feature_srgb = merged_feature if color_space == "srgb" else raster(features)
    if merged_feature_srgb.shape != merged_feature.shape:
        raise ValueError("merged_feature_srgb does not match rasterized image shape")
    global_gbuffer = unpack_features(merged_feature, opacity, color_space)
    global_attenuation = opacity / opacity.clamp(min=1e-4)
    global_keys = tuple(global_gbuffer)
    global_fields = tuple(global_gbuffer[key] for key in global_keys)
    checkpoint_training = torch.is_grad_enabled() and any(
        value.requires_grad for value in (colors, global_render, global_attenuation, *global_fields))

    unique_ids = torch.unique(group_ids, sorted=True)
    group_count = unique_ids.numel()

    # Reconstruction buffers are diagnostic only; don't retain a second graph.
    reconstructed = torch.zeros_like(global_render, requires_grad=False)
    dominant_mass = torch.zeros_like(opacity)
    dominant_id = torch.full(opacity.shape, -1, dtype=torch.long, device=opacity.device)
    visible_group_count = torch.zeros_like(opacity)
    tiny_group_mass = torch.zeros_like(opacity)
    zero_rgb = torch.zeros_like(merged_feature[..., :3])
    surface = {"rgb": zero_rgb, "diffuse": zero_rgb, "specular": zero_rgb,
               "normal_oriented": zero_rgb}
    shaded_layer_count = 0

    def shade_layer(layer, attenuation, *merged_fields):
        """Pure tensor operation: safe to recompute during backward."""
        mass = layer[..., 8:9]
        # Empty pixels use denominator one, avoiding hidden 0/0 derivatives.
        denominator = torch.where(mass > 0, mass.clamp_min(torch.finfo(mass.dtype).tiny),
                                  torch.ones_like(mass))
        normalized = layer[..., :8] / denominator * attenuation
        gbuffer = unpack_features(normalized, torch.ones_like(mass), color_space)
        replace = {"normal": ("normal",), "brdf": ("metallic", "roughness"),
                   "albedo": ("albedo",), "none": ()}[ablation]
        merged = dict(zip(global_keys, merged_fields))
        for key in replace:
            gbuffer[key] = merged[key]
        shaded = shade(gbuffer)
        for key in ("rgb", "diffuse", "specular"):
            if key not in shaded or shaded[key].shape != zero_rgb.shape:
                raise ValueError(f"shade must return {key} with shape {tuple(zero_rgb.shape)}")
        oriented = shaded.get("normal_oriented", gbuffer["normal"])
        return (mass * shaded["rgb"], mass * shaded["diffuse"],
                mass * shaded["specular"], mass * oriented)

    def raster_group(shared_colors, group_mask):
        # Masks are explicit checkpoint inputs, so backward cannot accidentally
        # reuse the final loop iteration's group identity.
        return raster(shared_colors * group_mask)

    def render_shade_group(shared_colors, group_mask, attenuation, *merged_fields):
        layer = raster_group(shared_colors, group_mask)
        return (layer, *shade_layer(layer, attenuation, *merged_fields))

    def accumulate_shaded(weighted):
        nonlocal shaded_layer_count
        for key, value in zip(surface, weighted):
            surface[key] = surface[key] + value
        shaded_layer_count += 1

    def add_layer(layer):
        args = (layer, global_attenuation, *global_fields)
        weighted = (checkpoint(shade_layer, *args, use_reentrant=False)
                    if checkpoint_training else shade_layer(*args))
        accumulate_shaded(weighted)

    # No compression is needed when every group fits in the layer budget.
    compress = max_layers > 0 and group_count > max_layers
    kept = [torch.zeros_like(global_render) for _ in range(max_layers - 1)] if compress else []
    overflow = torch.zeros_like(global_render)
    for group_id in unique_ids:
        weighted = None
        if group_count == 1:
            layer = global_render
        else:
            group_mask = (group_ids == group_id)[:, None]
            # Crucially, this mask applies ONLY to colors, including a unit
            # indicator used to recover the original visibility contribution.
            if checkpoint_training and checkpoint_raster and not compress:
                result = checkpoint(render_shade_group, colors, group_mask,
                                    global_attenuation, *global_fields, use_reentrant=False)
                layer, weighted = result[0], result[1:]
            elif checkpoint_training and checkpoint_raster:
                layer = checkpoint(raster_group, colors, group_mask, use_reentrant=False)
            else:
                layer = raster_group(colors, group_mask)
        if layer.shape != global_render.shape:
            raise ValueError("rasterizer changed image shape between groups")
        with torch.no_grad():
            reconstructed.add_(layer.detach())
            mass = layer[..., 8:9].detach()
            stronger = mass > dominant_mass
            dominant_mass = torch.maximum(dominant_mass, mass)
            dominant_id = torch.where(stronger, group_id.long(), dominant_id)
            visible_group_count.add_((mass > 1e-6).to(mass.dtype))
            tiny_group_mass.add_(torch.where(mass < 1e-4, mass, torch.zeros_like(mass)))
        if not compress:
            if weighted is None:
                add_layer(layer)
            else:
                accumulate_shaded(weighted)
            continue
        # Streaming insertion into descending per-pixel contribution slots.
        # The losing candidate is accumulated in overflow without subtraction,
        # preserving even tiny positive contributions and signed normals.
        candidate = layer
        for slot in range(len(kept)):
            stronger = candidate[..., 8:9].detach() > kept[slot][..., 8:9].detach()
            winner = torch.where(stronger, candidate, kept[slot])
            candidate = torch.where(stronger, kept[slot], candidate)
            kept[slot] = winner
        overflow = overflow + candidate
    if compress:
        for layer in kept:
            add_layer(layer)
        add_layer(overflow)

    with torch.no_grad():
        feature_error = (reconstructed[..., :8] - merged_feature.detach()).abs().amax(-1, keepdim=True)
        mass_error = (reconstructed[..., 8:9] - opacity.detach()).abs()
        opacity_safe = opacity.detach().clamp(min=1e-12)
        mixing = torch.where(opacity.detach() > 0,
                             1 - dominant_mass / opacity_safe, torch.zeros_like(opacity))
        overflow_mass = overflow[..., 8:9].detach() if compress else torch.zeros_like(opacity)
        total_mass = opacity.detach().sum().clamp(min=1e-12)
        diagnostics = {
            "group_count": group_count,
            "shaded_layer_count": shaded_layer_count,
            "max_layers": max_layers,
            "color_space": color_space,
            "ablation": ablation,
            "feature_reconstruction_max_abs": float(feature_error.max()),
            "mass_reconstruction_max_abs": float(mass_error.max()),
            "raster_mass_max_abs": float((raster_mass.detach() - opacity.detach()).abs().max()),
            "overflow_mass_fraction": float(overflow_mass.sum() / total_mass),
            "tiny_group_mass_fraction": float(tiny_group_mass.sum() / total_mass),
        }
    return {
        "linear_surface": surface["rgb"],
        "diffuse_surface": surface["diffuse"],
        "specular_surface": surface["specular"],
        "normal_oriented": F.normalize(surface["normal_oriented"], dim=-1),
        "merged_feature_srgb": merged_feature_srgb,
        "merged_feature": merged_feature,
        "opacity": opacity,
        "diagnostics": diagnostics,
        "diagnostic_maps": {
            "mass_reconstruction_error": mass_error,
            "feature_reconstruction_error": feature_error,
            "mixing_fraction": mixing,
            "visible_group_count": visible_group_count,
            "dominant_group_id": dominant_id,
            "dominant_group_mass": dominant_mass,
            "overflow_mass": overflow_mass,
            "tiny_group_mass": tiny_group_mass,
        },
    }
