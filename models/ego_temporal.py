"""Differentiable, surface-aligned temporal material refinement.

Packets contain generated HWC images only; supervision is deliberately absent
from this module. Depth is camera Z, normals and view directions are world
space, albedo is sRGB, and intrinsics use pixel centers at ``index + .5``.
Visibility decisions and camera geometry are detached. Only bounded residual
sampling offsets and the learned fusion/residual branches receive gradients.
"""

from collections.abc import Mapping
import math

import torch
from torch import nn
import torch.nn.functional as F


PAYLOAD = ("rgb", "albedo", "roughness", "metallic", "normal", "opacity")
ABLATIONS = ("none", "constant_normal", "shuffle_normal", "constant_roughness", "shuffle_roughness")


def srgb_to_linear(value):
    value = value.clamp_min(0)
    return torch.where(value <= .04045, value / 12.92, ((value + .055) / 1.055).pow(2.4))


def linear_to_srgb(value):
    value = value.clamp_min(0)
    return torch.where(value <= .0031308, value * 12.92,
                       1.055 * value.clamp_min(.0031308).pow(1 / 2.4) - .055)


def sample_image(value, grid):
    """Bilinearly sample an HWC map with an HWC2 align_corners=False grid."""
    return F.grid_sample(value.permute(2, 0, 1)[None], grid[None],
                         mode="bilinear", padding_mode="zeros", align_corners=False)[0].permute(1, 2, 0)


def _clean(value):
    return torch.nan_to_num(value, nan=0., posinf=0., neginf=0.)


def _validate_packet(packet):
    if not isinstance(packet, Mapping):
        raise ValueError("A temporal packet must be a mapping.")
    albedo = packet.get("albedo")
    if not isinstance(albedo, torch.Tensor) or albedo.ndim != 3 or albedo.shape[-1] != 3:
        raise ValueError("albedo must be H x W x 3.")
    h, w = albedo.shape[:2]
    if min(h, w) < 1 or not albedo.is_floating_point():
        raise ValueError("Packets need nonempty floating-point images.")
    for key, channels in (("rgb", 3), ("albedo", 3), ("roughness", 1), ("metallic", 1),
                          ("normal", 3), ("depth", 1), ("opacity", 1), ("viewdirs", 3)):
        value = packet.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (h, w, channels):
            raise ValueError(f"{key} must have shape {(h, w, channels)}.")
        if value.device != albedo.device or value.dtype != albedo.dtype:
            raise ValueError("Packet images must share their floating-point device/dtype.")
    mask = packet.get("static_mask")
    if not isinstance(mask, torch.Tensor) or mask.shape != (h, w) or mask.dtype != torch.bool or mask.device != albedo.device:
        raise ValueError("static_mask must be a same-device HW boolean tensor.")
    for key, shape in (("camera_to_world", (4, 4)), ("intrinsics", (3, 3))):
        value = packet.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{key} must be a finite floating-point tensor of shape {shape}.")
    return h, w


def _surface_valid(packet):
    """Conservative geometry support, independent of any learned confidence."""
    depth, opacity = packet["depth"].detach(), packet["opacity"].detach()
    valid = packet["static_mask"].detach().clone()
    valid &= torch.isfinite(depth[..., 0]) & (depth[..., 0] > 0)
    valid &= torch.isfinite(opacity[..., 0]) & (opacity[..., 0] >= .95)
    for key in PAYLOAD:
        valid &= torch.isfinite(packet[key].detach()).all(-1)
    d = _clean(depth[..., 0])[None, None]
    padded = F.pad(d, (1, 1, 1, 1), mode="replicate")
    spread = F.max_pool2d(padded, 3, 1) + F.max_pool2d(-padded, 3, 1)
    valid &= spread[0, 0] <= .05 * d[0, 0]
    # One-pixel erosion prevents interpolation across object/mask boundaries.
    invalid = F.pad((~packet["static_mask"]).float()[None, None], (1,) * 4, value=1.)
    valid &= F.max_pool2d(invalid, 3, 1)[0, 0] == 0
    return valid[..., None]


def routing_guides(packet, ablation="none"):
    """Copy routing references only; never alter transported/shaded payloads."""
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown guide ablation: {ablation}")
    result = {key: packet[key].detach() for key in ("albedo", "normal", "roughness", "metallic")}
    if ablation.startswith("constant_"):
        key = ablation[len("constant_"):]
        result[key] = torch.zeros_like(result[key])
        if key == "normal":
            result[key][..., 2] = 1.
        else:
            result[key].fill_(.5)
    elif ablation.startswith("shuffle_"):
        key = ablation[len("shuffle_"):]
        frame_id = int(packet.get("frame_id", 0))
        shifts = tuple(0 if size == 1 else 1 + ((size // 3 + 17 * frame_id) % (size - 1))
                       for size in result[key].shape[:2])
        result[key] = torch.roll(result[key], shifts, (0, 1))
    return result


def project_history(current, previous, offset=None, *, normal_check=True, routing_normals=None):
    """Project a preceding packet into the current frame.

    ``offset`` is HWC2 in *previous-image pixels*. It retains gradients through
    payload sampling. Geometry, masks, depth residual, view delta, and cycle
    checks are detached and recomputed at the displaced sampling coordinates.
    The mask is never a learned loss weight. ``routing_normals`` optionally
    supplies two ablated routing maps without changing returned normal payload.
    Camera-Z disagreement must be <= .05 m + 2%; reverse projection must be
    within 1.5 current pixels; sign-invariant normal cosine must be >= .9.
    """
    h, w = _validate_packet(current)
    ph, pw = _validate_packet(previous)
    if current["albedo"].device != previous["albedo"].device or current["albedo"].dtype != previous["albedo"].dtype:
        raise ValueError("Current/history must have matching device and dtype.")
    device = current["albedo"].device
    dtype = current["albedo"].dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Projection requires float32 or float64 packets.")
    if offset is not None and (offset.shape != (h, w, 2) or offset.device != device or offset.dtype != dtype):
        raise ValueError("offset must be a same-device/dtype H x W x 2 tensor.")
    with torch.no_grad():
        yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype) + .5,
                                torch.arange(w, device=device, dtype=dtype) + .5, indexing="ij")
        pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1)
        c2w = current["camera_to_world"].detach().to(device=device, dtype=dtype)
        p2w = previous["camera_to_world"].detach().to(device=device, dtype=dtype)
        ck = current["intrinsics"].detach().to(device=device, dtype=dtype)
        pk = previous["intrinsics"].detach().to(device=device, dtype=dtype)
        try:
            cki, pki = torch.linalg.inv(ck), torch.linalg.inv(pk)
            c2p, p2c = torch.linalg.inv(p2w) @ c2w, torch.linalg.inv(c2w) @ p2w
        except RuntimeError as exc:
            raise ValueError("Cameras and intrinsics must be invertible.") from exc
        if not all(torch.isfinite(v).all() for v in (cki, pki, c2p, p2c)):
            raise ValueError("Camera/intrinsics inverses must be finite.")
        rays = pixels @ cki.T
        xyz = rays / rays[..., 2:3] * _clean(current["depth"].detach())
        pxyz = xyz @ c2p[:3, :3].T + c2p[:3, 3]
        puv = pxyz @ pk.T
        base_uv = puv[..., :2] / puv[..., 2:3]
        expected_depth = pxyz[..., 2:3]
    # Detaching base projection prevents renderer/camera gradients while
    # retaining a true gradient from sampled materials to predicted offsets.
    uv = base_uv if offset is None else base_uv + offset
    grid = torch.stack((2 * uv[..., 0] / pw - 1, 2 * uv[..., 1] / ph - 1), -1)
    grid = torch.nan_to_num(grid, nan=-2., posinf=-2., neginf=-2.)
    with torch.no_grad():
        uvd, gd = uv.detach(), grid.detach()
        pd = sample_image(_clean(previous["depth"].detach()), gd)
        depth_residual = (pd - expected_depth).abs()
        valid = (_surface_valid(current) & torch.isfinite(uvd).all(-1, keepdim=True)
                 & torch.isfinite(expected_depth) & (expected_depth > 0)
                 & (uvd[..., :1] >= .5) & (uvd[..., :1] <= pw - .5)
                 & (uvd[..., 1:2] >= .5) & (uvd[..., 1:2] <= ph - .5))
        footprint = sample_image(_surface_valid(previous).to(dtype), gd)
        valid &= footprint >= 1 - 1e-6
        valid &= depth_residual <= .05 + .02 * expected_depth.abs()
        ppixels = torch.cat((uvd, torch.ones_like(pd)), -1)
        prays = ppixels @ pki.T
        back_xyz = prays / prays[..., 2:3] * pd
        back_xyz = back_xyz @ p2c[:3, :3].T + p2c[:3, 3]
        back_uv = back_xyz @ ck.T
        back_uv = back_uv[..., :2] / back_uv[..., 2:3]
        cycle_error = torch.linalg.vector_norm(back_uv - pixels[..., :2], dim=-1, keepdim=True)
        valid &= torch.isfinite(cycle_error) & (back_xyz[..., 2:3] > 0) & (cycle_error <= 1.5)
        cn, pn = ((current["normal"], previous["normal"]) if routing_normals is None else routing_normals)
        pn = sample_image(_clean(pn.detach()), gd)
        cosine = (F.normalize(_clean(cn.detach()), dim=-1) * F.normalize(pn, dim=-1)).sum(-1, keepdim=True).abs()
        if normal_check:
            valid &= torch.isfinite(cn).all(-1, keepdim=True) & (cn.square().sum(-1, keepdim=True) > 1e-12)
            valid &= cosine >= .9
        pv = sample_image(_clean(previous["viewdirs"].detach()), gd)
        view_delta = 1 - (F.normalize(_clean(current["viewdirs"].detach()), dim=-1) * F.normalize(pv, dim=-1)).sum(-1, keepdim=True).clamp(-1, 1)
    warped = {key: sample_image(_clean(previous[key].detach()), grid) for key in PAYLOAD}
    # Transport irradiance-independent albedo in linear space: converting a
    # bilinearly sampled sRGB value afterwards would average in the wrong space.
    warped["albedo_linear"] = sample_image(srgb_to_linear(_clean(previous["albedo"].detach())), grid)
    return {"grid": grid, "valid": valid.detach(), "warped": warped,
            "displacement": _clean(uv.detach() - pixels[..., :2]),
            "depth_residual": _clean(depth_residual).detach(), "view_delta": _clean(view_delta).detach(),
            "normal_similarity": _clean(cosine).detach(), "cycle_error": _clean(cycle_error).detach(),
            "expected_depth": expected_depth.detach()}


def material_consistency(current, previous, roughness_weight=0.):
    """Robust linear-albedo consistency; detached history and hard support.

    Returns a scalar loss plus diagnostic maps/count. Only current albedo (and
    optional roughness) receives gradients; an empty support returns graph-safe
    zero. The Charbonnier penalty subtracts epsilon so equality is zero.
    """
    if not math.isfinite(roughness_weight) or roughness_weight < 0:
        raise ValueError("roughness_weight must be a finite nonnegative value.")
    projected = project_history(current, previous)
    mask = projected["valid"].to(current["albedo"].dtype)
    difference = srgb_to_linear(_clean(current["albedo"])) - projected["warped"]["albedo_linear"].detach()
    error = (difference.square() + 1e-6).sqrt().sub(.001).mean(-1, keepdim=True)
    if roughness_weight:
        delta = _clean(current["roughness"]) - projected["warped"]["roughness"].detach()
        error = error + roughness_weight * (delta.square() + 1e-6).sqrt().sub(.001)
    count = mask.sum()
    loss = (error * mask).sum() / count.clamp_min(1)
    return {"loss": loss, "valid": projected["valid"], "valid_count": count.detach(),
            "depth_residual": projected["depth_residual"], "displacement": projected["displacement"]}


class _Encoder(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
                                    nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
                                    nn.Conv2d(width, width, 3, padding=1), nn.SiLU())

    def forward(self, value):
        return self.layers(value)


class _LocalAttention(nn.Module):
    """One-query, local cross-attention over three projected 3 x 3 histories."""
    def __init__(self, width):
        super().__init__()
        self.query = nn.Conv2d(width, width, 1)
        self.key = nn.Conv2d(width, width, 1)
        self.value = nn.Conv2d(width, width, 1)
        self.output = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)
        self.ffn = nn.Sequential(nn.Conv2d(width, width * 2, 1), nn.SiLU(), nn.Conv2d(width * 2, width, 1))
        self.width = width

    def forward(self, current, histories, support, bias):
        b, c, h, w = current.shape
        query = self.query(current).flatten(2)
        keys = [self.key(current).flatten(2).unsqueeze(2)]
        values = [self.value(current).flatten(2).unsqueeze(2)]
        masks = [torch.ones((b, 1, h * w), dtype=torch.bool, device=current.device)]
        biases = [torch.zeros((b, 1, h * w), dtype=current.dtype, device=current.device)]
        for history, mask, prior in zip(histories, support, bias):
            keys.append(F.unfold(self.key(history), 3, padding=1).reshape(b, c, 9, h * w))
            values.append(F.unfold(self.value(history), 3, padding=1).reshape(b, c, 9, h * w))
            # Each local key must itself have a valid physical correspondence.
            masks.append(F.unfold(mask.float(), 3, padding=1) >= 1 - 1e-6)
            biases.append(F.unfold(prior, 3, padding=1))
        key, value = torch.cat(keys, 2), torch.cat(values, 2)
        valid, prior = torch.cat(masks, 1), torch.cat(biases, 1)
        logits = (query.unsqueeze(2) * key).sum(1) / math.sqrt(c) + prior
        weights = logits.masked_fill(~valid, float("-inf")).softmax(1)
        attended = (weights.unsqueeze(1) * value).sum(2).reshape(b, c, h, w)
        current = current + self.output(attended)
        return current + self.ffn(self.norm(current))


class EgoTemporalHead(nn.Module):
    """Quarter-resolution CNN or local Transformer with identical output heads.

    Up to three history packets (normally t-1,t-2,t-4) are accepted. Features are
    detached generated observations. ``guides='geometry'`` supplies RGB, depth,
    opacity and reprojection geometry; ``full`` adds A/N/R/M and view cosine.
    Both controls output albedo and share current-frame re-shading externally.
    Histories are never recursively refined: use frozen base-render packets.
    """
    def __init__(self, kind="cnn", guides="full", history=True, use_offset=True,
                 width=32, max_offset=2., max_residual=.1, guide_ablation="none"):
        super().__init__()
        if kind not in ("cnn", "transformer") or guides not in ("full", "geometry"):
            raise ValueError("kind must be cnn/transformer and guides full/geometry.")
        if guide_ablation not in ABLATIONS or (guides == "geometry" and guide_ablation != "none"):
            raise ValueError("Guide ablations require full guides and a supported ablation.")
        if not isinstance(width, int) or width < 4 or not math.isfinite(max_offset) or max_offset < 0 or not math.isfinite(max_residual) or max_residual < 0:
            raise ValueError("Invalid width, max_offset or max_residual.")
        self.kind, self.guides, self.history, self.use_offset = kind, guides, bool(history), bool(use_offset)
        self.max_offset, self.max_residual, self.guide_ablation = max_offset, max_residual, guide_ablation
        # RGB3 + depth1 + opacity1 + displacement2 + depth_residual1 +
        # view_delta1 + history_age1 + valid1 = 11; full adds A3/N3/R1/M1/nv1.
        self.input_channels = 20  # Zero material slots preserve control-model capacity.
        self.encoder = _Encoder(self.input_channels, width)
        self.fusion = nn.Sequential(nn.Conv2d(width * 4, width, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(width, width, 3, padding=1), nn.SiLU()) if kind == "cnn" else None
        self.attention = nn.ModuleList([_LocalAttention(width), _LocalAttention(width)]) if kind == "transformer" else nn.ModuleList()
        self.decoder = nn.Sequential(nn.Conv2d(width + self.input_channels, width, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.offset_head = nn.Conv2d(width, 6, 1)
        self.gate_head = nn.Conv2d(width, 4, 1)
        self.residual_head = nn.Conv2d(width, 3, 1)
        # Initialize near the current result, retaining a nonzero gate signal.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        with torch.no_grad():
            self.gate_head.bias[0] = 2.

    @staticmethod
    def _nchw(value):
        return value.permute(2, 0, 1)[None]

    def _features(self, packet, guides, projected=None, age=0):
        depth = packet["depth"].detach()
        h, w = depth.shape[:2]
        zero = torch.zeros_like(depth)
        if projected is None:
            rgb, material, opacity = packet["rgb"], guides, packet["opacity"]
            displacement, residual, view_delta, valid = zero.expand(h, w, 2), zero, zero, _surface_valid(packet).float()
        else:
            rgb, opacity = projected["warped"]["rgb"], projected["warped"]["opacity"]
            material = {key: sample_image(value, projected["grid"]) for key, value in guides.items()}
            displacement = projected["displacement"] / max(h, w)
            residual = projected["depth_residual"] / depth.detach().abs().clamp_min(.1)
            view_delta, valid = projected["view_delta"], projected["valid"].float()
        parts = [rgb, torch.log1p(depth.clamp_min(0)) / 10., opacity, displacement,
                 residual, view_delta, zero + age / 4., valid]
        if self.guides == "full":
            normal = F.normalize(material["normal"], dim=-1)
            view = F.normalize(packet["viewdirs"].detach(), dim=-1)
            nv = (normal * view).sum(-1, keepdim=True).abs()
            parts.extend((srgb_to_linear(material["albedo"]), normal, material["roughness"], material["metallic"], nv))
        else:
            parts.append(zero.expand(h, w, 9))
        return self._nchw(_clean(torch.cat(parts, -1)).detach())

    def forward(self, current, histories):
        h, w = _validate_packet(current)
        if len(histories) > 3:
            raise ValueError("At most three preceding history frames are supported.")
        histories = list(histories) if self.history else []
        current_guides = routing_guides(current, self.guide_ablation)
        feature = self._features(current, current_guides)
        encoded = self.encoder(feature)
        projected, encoded_history, support, priors = [], [], [], []
        history_guides = []
        low_shape = encoded.shape[-2:]
        for previous in histories:
            age = int(current.get("frame_id", 0)) - int(previous.get("frame_id", -1))
            if age <= 0:
                raise ValueError("History frames must strictly precede the current frame.")
            if current.get("stream_id") != previous.get("stream_id"):
                raise ValueError("History must come from the same camera/sequence stream.")
            guides = routing_guides(previous, self.guide_ablation)
            p = project_history(current, previous, normal_check=self.guides == "full",
                                routing_normals=(current_guides["normal"], guides["normal"]))
            projected.append(p)
            history_guides.append(guides)
            encoded_history.append(self.encoder(self._features(current, guides, p, age)))
            # Require the entire low-resolution cell to have geometry support.
            support.append(F.adaptive_avg_pool2d(self._nchw(p["valid"].float()), low_shape) >= 1 - 1e-6)
            bias = -p["depth_residual"] / current["depth"].detach().abs().clamp_min(.1)
            if self.guides == "full":
                wn = sample_image(guides["normal"], p["grid"])
                wr = sample_image(guides["roughness"], p["grid"])
                normal_delta = 1 - (F.normalize(current_guides["normal"], dim=-1) * F.normalize(wn, dim=-1)).sum(-1, keepdim=True).abs()
                bias = bias - 2 * normal_delta - 2 * (wr - current_guides["roughness"]).abs()
            priors.append(F.adaptive_avg_pool2d(self._nchw(_clean(bias).detach()), low_shape))
        while len(encoded_history) < 3:
            encoded_history.append(torch.zeros_like(encoded))
            support.append(torch.zeros_like(encoded[:, :1], dtype=torch.bool))
            priors.append(torch.zeros_like(encoded[:, :1]))
        if self.kind == "cnn":
            fused = self.fusion(torch.cat([encoded] + [f * m for f, m in zip(encoded_history, support)], 1))
        else:
            fused = encoded
            for attention in self.attention:
                fused = attention(fused, encoded_history, support, priors)
        decoded = self.decoder(torch.cat((F.interpolate(fused, (h, w), mode="bilinear", align_corners=False), feature), 1))
        offsets = self.offset_head(decoded).tanh()[0].permute(1, 2, 0).reshape(h, w, 3, 2) * self.max_offset
        if not self.use_offset:
            offsets = offsets * 0
        logits = self.gate_head(decoded)[0].permute(1, 2, 0)
        candidate_albedo = [srgb_to_linear(_clean(current["albedo"]).detach())]
        candidates_valid = [torch.ones_like(current["depth"], dtype=torch.bool)]
        for index, previous in enumerate(histories):
            refined = project_history(current, previous, offsets[..., index, :], normal_check=self.guides == "full",
                                      routing_normals=(current_guides["normal"], history_guides[index]["normal"]))
            candidate_albedo.append(refined["warped"]["albedo_linear"])
            candidates_valid.append(refined["valid"])
        while len(candidate_albedo) < 4:
            candidate_albedo.append(torch.zeros_like(candidate_albedo[0]))
            candidates_valid.append(torch.zeros_like(candidates_valid[0]))
        mask = torch.cat(candidates_valid, -1)
        gates = logits.masked_fill(~mask, float("-inf")).softmax(-1)
        blended = sum(value * gates[..., index:index+1] for index, value in enumerate(candidate_albedo))
        residual = self.residual_head(decoded).tanh()[0].permute(1, 2, 0) * self.max_residual
        # Dynamic/unsupported pixels never receive a material correction. On a
        # supported current surface with no valid past, only the current branch
        # and its learned residual remain; no invalid history can leak through.
        surface = _surface_valid(current)
        refined = linear_to_srgb((blended + residual).clamp(0, 1))
        albedo = torch.where(surface, refined, current["albedo"].detach())
        return {"albedo": albedo, "history_weight": 1 - gates[..., :1],
                "valid": mask[..., 1:].any(-1, keepdim=True), "offsets": offsets,
                "gate_weights": gates, "residual_linear": residual,
                "surface_valid": surface, "candidate_valid": mask[..., 1:]}
