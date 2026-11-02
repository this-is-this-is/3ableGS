"""Surface-aligned, direct material-value attention for Base v4.

Only generated packets enter this module. Image supervision and split selection
belong to the caller. Packet conventions match ``ego_temporal``: HWC maps,
camera-Z depth, world normals, sRGB albedo, and pixel centers at index + .5.
Routing and history are detached; the current albedo value remains differentiable.
"""

from collections.abc import Mapping, Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.ego_temporal import (
    _Encoder, _clean, _surface_valid, _validate_packet,
    linear_to_srgb, project_history, srgb_to_linear,
)


VARIANTS = ("B0", "H0", "C0", "C1", "A0", "A1", "A2")
DEFAULTS = {
    "variant": "A2", "width": 32, "chunk_size": 16384,
    "history_offsets": [1, 2, 4], "checkpoint_chunks": True,
    "start_step": 5000, "ramp_steps": 1000,
    "temporal_start": 6000, "temporal_ramp": 1000,
    "temporal_w": .05, "anchor_w": .01, "lr": 1e-4,
    "max_residual": .02, "max_blend": .5, "fixed_blend": .3,
    "normal_check": False,
    "depth_bias": 1., "normal_bias": 2., "roughness_bias": 2., "age_bias": .1,
}


def validate_attention_config(config=None):
    """Return an independent canonical configuration; reject silent typos."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("G-buffer attention config must be a mapping.")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown G-buffer attention settings: {sorted(unknown)}")
    result = {**DEFAULTS, **dict(config)}
    if result["variant"] not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}.")
    for name in ("width", "chunk_size", "ramp_steps", "temporal_ramp"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < (4 if name == "width" else 1):
            raise ValueError(f"{name} must be a positive integer" + (" >= 4." if name == "width" else "."))
    for name in ("start_step", "temporal_start"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    offsets = result["history_offsets"]
    if (not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)) or not 1 <= len(offsets) <= 3
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in offsets)
            or len(set(offsets)) != len(offsets)):
        raise ValueError("history_offsets needs one to three unique positive integers.")
    result["history_offsets"] = sorted(offsets)
    for name in ("checkpoint_chunks", "normal_check"):
        if not isinstance(result[name], bool):
            raise ValueError(f"{name} must be a boolean.")
    for name in ("temporal_w", "anchor_w", "lr", "max_residual", "max_blend", "fixed_blend",
                 "depth_bias", "normal_bias", "roughness_bias", "age_bias"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number.")
        result[name] = float(value)
    if result["lr"] <= 0 or max(result["max_blend"], result["fixed_blend"]) > 1:
        raise ValueError("lr must be positive and blend limits must be <= 1.")
    return result


def _ramp(step, start, duration):
    return max(0., min(1., (float(step) - start) / duration))


def attention_strength(config, step, training=True):
    config = validate_attention_config(config)
    if config["variant"] == "B0":
        return 0.
    return _ramp(step, config["start_step"], config["ramp_steps"]) if training else 1.


def temporal_weight(config, step):
    config = validate_attention_config(config)
    if config["variant"] in ("B0", "H0", "A1"):
        return 0.
    return config["temporal_w"] * _ramp(step, config["temporal_start"], config["temporal_ramp"])


def attention_material_loss(current_refined, previous_refined):
    """Fixed zero-offset geometry support, independent of attention acceptance.

    The input packets may replace albedo with a refined result. Normal is not a
    hard gate, so A0/A1/A2 share the same support. Previous outputs are targets.
    """
    projected = project_history(current_refined, previous_refined, normal_check=False)
    mask = projected["valid"].to(current_refined["albedo"].dtype)
    difference = srgb_to_linear(_clean(current_refined["albedo"])) - projected["warped"]["albedo_linear"].detach()
    error = (difference.square() + 1e-6).sqrt().sub(.001).mean(-1, keepdim=True)
    count = mask.sum()
    return {"loss": (error * mask).sum() / count.clamp_min(1),
            "valid": projected["valid"], "valid_count": count.detach()}


def _nchw(value):
    return value.permute(2, 0, 1).unsqueeze(0)


def _sample(value, uv, image_shape):
    """Sample N x K native-image coordinates in a possibly lower-res NCHW map."""
    h, w = image_shape
    grid = torch.stack((2 * uv[..., 0] / w - 1, 2 * uv[..., 1] / h - 1), -1)
    grid = torch.nan_to_num(grid, nan=-2., posinf=-2., neginf=-2.)
    return F.grid_sample(value, grid.unsqueeze(0), align_corners=False,
                         padding_mode="zeros")[0].permute(1, 2, 0)


class _DirectAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
        self.scale = math.sqrt(width)

    def forward(self, query, candidates, valid, prior):
        logits = (self.query(query)[:, None] * self.key(candidates)).sum(-1) / self.scale + prior
        weight = logits.masked_fill(~valid, float("-inf")).softmax(-1)
        feature = (weight[..., None] * self.value(candidates)).sum(1)
        query = query + self.output(feature)
        return query + self.ffn(self.norm(query)), weight


class GBufferAttentionHead(nn.Module):
    """Direct albedo attention with bounded residual and deterministic history.

    B0 is exact bypass, H0 fixed transport, C0 current-only, C1 temporal CNN,
    A0 geometry-only attention, A1/A2 full-guide attention (loss differs outside
    forward). ``strength=0`` is an exact identity, including unsupported pixels.
    """
    input_channels = 20

    def __init__(self, config=None):
        super().__init__()
        self.config = validate_attention_config(config)
        self.variant = self.config["variant"]
        self.width = width = self.config["width"]
        self.full_guides = self.variant != "A0"
        self.learned = self.variant not in ("B0", "H0")
        if self.learned:
            self.encoder = _Encoder(self.input_channels, width)
            if self.variant == "C1":
                # Match two attention blocks' parameter budget within rounding.
                target = 16 * width * width + 18 * width
                hidden = max(1, round((target - 29 * width - 28) / (5 * width + 1)))
                self.cnn_fusion = nn.Sequential(nn.Conv1d(4 * width, hidden, 1), nn.SiLU(),
                                                nn.Conv1d(hidden, width, 1))
                self.cnn_logits = nn.Linear(width, 28)
            else:
                self.attention = nn.ModuleList([_DirectAttention(width), _DirectAttention(width)])
            self.decoder = nn.Sequential(nn.Linear(width + self.input_channels, width), nn.SiLU(),
                                         nn.Linear(width, width), nn.SiLU())
            self.gate_head = nn.Linear(width, 1)
            self.residual_head = nn.Linear(width, 3)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)

    def _features(self, packet, age=0):
        # No dictionary iteration: arbitrary supervision keys cannot leak in.
        depth = _clean(packet["depth"].detach()).clamp_min(0)
        zero = torch.zeros_like(depth)
        h, w = depth.shape[:2]
        parts = [_clean(packet["rgb"].detach()), torch.log1p(depth) / 10.,
                 _clean(packet["opacity"].detach()), zero.expand(h, w, 2), zero, zero,
                 zero + age / 4., _surface_valid(packet).to(depth.dtype)]
        if self.full_guides:
            normal = F.normalize(_clean(packet["normal"].detach()), dim=-1)
            view = F.normalize(_clean(packet["viewdirs"].detach()), dim=-1)
            parts.extend((srgb_to_linear(_clean(packet["albedo"].detach())), normal,
                          _clean(packet["roughness"].detach()), _clean(packet["metallic"].detach()),
                          (normal * view).sum(-1, keepdim=True).abs()))
        else:
            parts.append(zero.expand(h, w, 9))
        return _nchw(torch.cat(parts, -1).detach())

    def _histories(self, current, histories):
        _validate_packet(current)
        if len(histories) > 3:
            raise ValueError("At most three history frames are supported.")
        seen, result = set(), []
        frame = current.get("frame_id")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError("Packets require integer frame_id.")
        for previous in histories:
            _validate_packet(previous)
            old = previous.get("frame_id")
            if isinstance(old, bool) or not isinstance(old, int) or old >= frame:
                raise ValueError("History must strictly precede the current frame.")
            if old in seen:
                raise ValueError("Duplicate history frame.")
            if frame - old not in self.config["history_offsets"]:
                raise ValueError("History age is not in history_offsets.")
            if current.get("stream_id") != previous.get("stream_id"):
                raise ValueError("History must use the same camera/sequence stream.")
            if (current["albedo"].dtype != previous["albedo"].dtype or
                    current["albedo"].device != previous["albedo"].device):
                raise ValueError("History and current device/dtype must match.")
            seen.add(old)
            result.append(previous)
        return sorted(result, key=lambda item: frame - item["frame_id"])

    def _prepare(self, current, histories):
        """Detached full-frame maps; candidate-sized tensors stay in chunks."""
        with torch.no_grad():
            device, dtype = current["albedo"].device, current["albedo"].dtype
            if dtype not in (torch.float32, torch.float64):
                raise ValueError("G-buffer geometry requires float32 or float64 packets.")
            ck = current["intrinsics"].detach().to(device=device, dtype=dtype)
            c2w = current["camera_to_world"].detach().to(device=device, dtype=dtype)
            try:
                cki, w2c = torch.linalg.inv(ck), torch.linalg.inv(c2w)
            except RuntimeError as exc:
                raise ValueError("Camera matrices must be invertible.") from exc
            if not torch.isfinite(cki).all() or not torch.isfinite(w2c).all():
                raise ValueError("Camera inverses must be finite.")
            result = []
            for previous in histories:
                pk = previous["intrinsics"].detach().to(device=device, dtype=dtype)
                p2w = previous["camera_to_world"].detach().to(device=device, dtype=dtype)
                try:
                    pki, c2p = torch.linalg.inv(pk), torch.linalg.inv(p2w) @ c2w
                except RuntimeError as exc:
                    raise ValueError("Camera matrices must be invertible.") from exc
                if not torch.isfinite(pki).all() or not torch.isfinite(c2p).all():
                    raise ValueError("Camera inverses must be finite.")
                maps = torch.cat((_clean(previous["depth"]), _surface_valid(previous).to(dtype),
                                  _clean(previous["normal"]), _clean(previous["roughness"]),
                                  srgb_to_linear(_clean(previous["albedo"]))), -1)
                result.append({"map": _nchw(maps.detach()), "shape": previous["albedo"].shape[:2],
                               "pk": pk, "pki": pki, "c2p": c2p, "p2c": w2c @ p2w,
                               "age": current["frame_id"] - previous["frame_id"]})
            return {"ck": ck, "cki": cki, "history": result,
                    "support": _surface_valid(current).reshape(-1),
                    "depth": _clean(current["depth"].detach()).reshape(-1, 1),
                    "normal": _clean(current["normal"].detach()).reshape(-1, 3),
                    "roughness": _clean(current["roughness"].detach()).reshape(-1, 1)}

    def _candidates(self, geometry, pixels, start, end):
        """Query-to-each-candidate depth/cycle validity, including native offsets."""
        with torch.no_grad():
            current_z = geometry["depth"][start:end]
            rays = pixels @ geometry["cki"].T
            xyz = rays / rays[:, 2:3] * current_z
            offsets = pixels.new_tensor([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)])
            candidate_data = []
            for previous in geometry["history"]:
                transform = previous["c2p"]
                pxyz = xyz @ transform[:3, :3].T + transform[:3, 3]
                projection = pxyz @ previous["pk"].T
                uv = projection[:, None, :2] / projection[:, None, 2:3] + offsets
                sample = _sample(previous["map"], uv, previous["shape"])
                sampled_z = sample[..., :1]
                expected_z = pxyz[:, None, 2:3]
                difference = (sampled_z - expected_z).abs()
                valid = geometry["support"][start:end, None].expand(-1, 9).clone()
                ph, pw = previous["shape"]
                valid &= torch.isfinite(uv).all(-1) & torch.isfinite(expected_z[..., 0]) & (expected_z[..., 0] > 0)
                valid &= (uv[..., 0] >= .5) & (uv[..., 0] <= pw - .5)
                valid &= (uv[..., 1] >= .5) & (uv[..., 1] <= ph - .5)
                valid &= sample[..., 1] >= 1 - 1e-6
                valid &= difference[..., 0] <= .05 + .02 * expected_z[..., 0].abs()
                ppixels = torch.cat((uv, torch.ones_like(sampled_z)), -1)
                prays = ppixels @ previous["pki"].T
                back_xyz = prays / prays[..., 2:3] * sampled_z
                transform = previous["p2c"]
                back_xyz = back_xyz @ transform[:3, :3].T + transform[:3, 3]
                back = back_xyz @ geometry["ck"].T
                back_uv = back[..., :2] / back[..., 2:3]
                cycle = torch.linalg.vector_norm(back_uv - pixels[:, None, :2], dim=-1)
                valid &= torch.isfinite(cycle) & (back_xyz[..., 2] > 0) & (cycle <= 1.5)
                normal = F.normalize(geometry["normal"][start:end], dim=-1)
                cosine = (normal[:, None] * F.normalize(sample[..., 2:5], dim=-1)).sum(-1).abs().clamp(0, 1)
                if self.config["normal_check"]:
                    valid &= cosine >= .9
                prior = -self.config["depth_bias"] * (difference[..., 0] / (.05 + .02 * expected_z[..., 0].abs()))
                if self.full_guides:
                    prior -= self.config["normal_bias"] * (1 - cosine)
                    prior -= self.config["roughness_bias"] * (sample[..., 5] - geometry["roughness"][start:end]).abs()
                prior -= self.config["age_bias"] * previous["age"] / 4.
                candidate_data.append((uv.detach(), valid.detach(), _clean(prior).detach(), sample[..., 6:9].detach()))
            return candidate_data

    def _chunk(self, current_encoded, current_features, current_linear, history_encoded,
               geometry, shape, start, end, strength):
        h, w = shape
        indices = torch.arange(start, end, device=current_linear.device)
        x, y = indices.remainder(w).to(current_linear.dtype) + .5, torch.div(indices, w, rounding_mode="floor").to(current_linear.dtype) + .5
        pixels = torch.stack((x, y, torch.ones_like(x)), -1)
        data = self._candidates(geometry, pixels, start, end)
        n = end - start
        zero = current_linear.new_zeros(n)
        valid = [torch.ones(n, dtype=torch.bool, device=current_linear.device)]
        priors, values = [zero], [current_linear]
        encoded = _sample(current_encoded, pixels[:, None, :2], shape)[:, 0] if self.learned else None
        features = [encoded] if self.learned else []
        by_age = {previous["age"]: index for index, previous in enumerate(geometry["history"])}
        for age in self.config["history_offsets"]:
            index = by_age.get(age)
            if index is None:
                # Preserve a missing t-1 slot instead of assigning t-2 to it.
                valid.extend([torch.zeros_like(valid[0])] * 9)
                priors.extend([zero] * 9)
                values.extend([torch.zeros_like(current_linear)] * 9)
                if self.learned:
                    features.extend([torch.zeros_like(encoded)] * 9)
                continue
            uv, accepted, prior, material = data[index]
            valid.extend(accepted.unbind(1))
            priors.extend(prior.unbind(1))
            values.extend(material.unbind(1))
            if self.learned:
                features.extend(_sample(history_encoded[index], uv, geometry["history"][index]["shape"]).unbind(1))
        # Three fixed history slots make CNN and attention controls comparable.
        while len(valid) < 28:
            valid.append(torch.zeros_like(valid[0]))
            priors.append(zero)
            values.append(torch.zeros_like(current_linear))
            if self.learned:
                features.append(torch.zeros_like(encoded))
        validity = torch.stack(valid, 1)
        value = torch.stack(values, 1)
        history_valid = validity[:, 1:].any(-1, keepdim=True)
        if self.variant == "H0":
            count = validity[:, 1:].sum(-1, keepdim=True).clamp_min(1)
            history_weights = validity[:, 1:].to(value.dtype) / count
            weight = torch.cat((1 - history_valid.to(value.dtype) * self.config["fixed_blend"],
                                history_weights * self.config["fixed_blend"]), 1)
            blended = (weight[..., None] * value).sum(1)
            residual = torch.zeros_like(current_linear)
            refined = current_linear + strength * (blended - current_linear)
            history_weight = strength * weight[:, 1:].sum(-1, keepdim=True)
        else:
            candidates = torch.stack(features, 1)
            prior = torch.stack(priors, 1)
            if self.variant == "C1":
                groups = [encoded]
                for slot in range(3):
                    mask = validity[:, 1 + slot * 9:1 + (slot + 1) * 9].to(value.dtype)
                    feature = candidates[:, 1 + slot * 9:1 + (slot + 1) * 9]
                    groups.append((feature * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1))
                fused = self.cnn_fusion(torch.cat(groups, -1).T.unsqueeze(0))[0].T
                weight = (self.cnn_logits(fused) + prior).masked_fill(~validity, float("-inf")).softmax(-1)
            else:
                fused = encoded
                for block in self.attention:
                    fused, weight = block(fused, candidates, validity, prior)
            inputs = _sample(current_features, pixels[:, None, :2], shape)[:, 0]
            decoded = self.decoder(torch.cat((fused, inputs), -1))
            beta = self.config["max_blend"] * self.gate_head(decoded).sigmoid()
            residual = self.config["max_residual"] * self.residual_head(decoded).tanh()
            attended = (weight[..., None] * value).sum(1)
            refined = current_linear + strength * (beta * (attended - current_linear) + residual)
            history_weight = strength * beta * (1 - weight[:, :1])
        entropy = -(weight * weight.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        return refined, history_weight, residual * strength, history_valid, entropy

    def forward(self, current, histories, strength=1.0):
        histories = self._histories(current, list(histories))
        if isinstance(strength, bool) or not isinstance(strength, (float, int)) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be a finite scalar in [0,1].")
        h, w = current["albedo"].shape[:2]
        surface = _surface_valid(current)
        zero = torch.zeros_like(current["depth"])
        if self.variant == "B0" or strength == 0:
            return {"albedo": current["albedo"], "history_weight": zero,
                    "residual_linear": torch.zeros_like(current["albedo"]),
                    "valid": torch.zeros_like(surface), "surface_valid": surface,
                    "attention_entropy": zero}
        if self.variant == "C0":
            histories = []
        geometry = self._prepare(current, histories)
        current_linear = srgb_to_linear(_clean(current["albedo"])).reshape(-1, 3)
        current_features = self._features(current) if self.learned else None
        encoded = self.encoder(current_features) if self.learned else None
        history_encoded = [self.encoder(self._features(previous, current["frame_id"] - previous["frame_id"]))
                           for previous in histories] if self.learned else []
        pieces = []
        for start in range(0, h * w, self.config["chunk_size"]):
            end = min(start + self.config["chunk_size"], h * w)
            # Bind ranges into each closure: backward runs after the loop ends.
            def run_chunk(cur_encoded, cur_features, cur_linear, *past_encoded, begin=start, finish=end):
                return self._chunk(cur_encoded, cur_features, cur_linear, past_encoded,
                                   geometry, (h, w), begin, finish, float(strength))
            args = (encoded, current_features, current_linear[start:end], *history_encoded)
            if self.learned and self.config["checkpoint_chunks"] and torch.is_grad_enabled():
                pieces.append(checkpoint(run_chunk, *args, use_reentrant=False, preserve_rng_state=False))
            else:
                pieces.append(run_chunk(*args))
        refined, history_weight, residual, valid, entropy = [torch.cat(items, 0) for items in zip(*pieces)]
        refined = linear_to_srgb(refined.reshape(h, w, 3).clamp(0, 1))
        return {"albedo": torch.where(surface, refined, current["albedo"]),
                "history_weight": history_weight.reshape(h, w, 1) * surface,
                "residual_linear": residual.reshape(h, w, 3) * surface,
                "valid": valid.reshape(h, w, 1), "surface_valid": surface,
                "attention_entropy": entropy.reshape(h, w, 1)}
