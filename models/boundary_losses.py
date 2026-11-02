"""Opt-in RGB boundary objectives and detached pseudo-label confidence.

All targets come from the current training image. LPIPS calls the frozen feature
network directly, never the stateful evaluation Metric.
"""
import math
import torch
import torch.nn.functional as F


DEFAULTS = dict(enabled=False, start_step=2000, ramp_steps=1000,
                rgb_w=0.05, gradient_w=0.02, perceptual_w=0.0,
                perceptual_final_w=0.0, perceptual_decay_start=0,
                perceptual_decay_steps=0,
                edge_threshold=0.05, radius=2, patch_size=128, num_patches=4,
                min_valid_fraction=0.99)
GBUFFER_DEFAULTS = dict(enabled=False, radius=2, edge_weight=0.25,
                        normal_angle_degrees=25.0, albedo_threshold=0.15)


def validate_boundary_config(config=None, *, gbuffer=False):
    defaults = GBUFFER_DEFAULTS if gbuffer else DEFAULTS
    cfg = {**defaults, **dict(config or {})}
    if set(cfg) - set(defaults):
        raise ValueError(f"unknown boundary options: {sorted(set(cfg) - set(defaults))}")
    if not isinstance(cfg['enabled'], bool):
        raise ValueError('boundary enabled must be boolean')
    for key, value in cfg.items():
        if key == 'enabled':
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'boundary {key} must be finite and nonnegative')
    for key in ('radius', 'start_step', 'ramp_steps', 'patch_size', 'num_patches',
                'perceptual_decay_start', 'perceptual_decay_steps'):
        if key in cfg and not isinstance(cfg[key], int):
            raise ValueError(f'boundary {key} must be an integer')
    if gbuffer:
        if cfg['edge_weight'] > 1 or not 0 < cfg['normal_angle_degrees'] <= 180:
            raise ValueError('invalid G-buffer boundary weight/angle')
    elif cfg['patch_size'] < 32 or cfg['num_patches'] < 1 or not 0 < cfg['min_valid_fraction'] <= 1 or cfg['edge_threshold'] <= 0:
        raise ValueError('invalid boundary patch size/count, edge threshold or valid fraction')
    return cfg


def sobel(image):
    """Signed x/y derivatives, HWC -> HW(C*2), including all RGB channels."""
    channels = image.shape[-1]
    kernels = image.new_tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]) / 8
    kernels = kernels[:, None].repeat(channels, 1, 1, 1)
    x = F.pad(image.permute(2, 0, 1)[None], (1, 1, 1, 1), mode='replicate')
    return F.conv2d(x, kernels, groups=channels)[0].permute(1, 2, 0)


def dilate(mask, radius):
    return F.max_pool2d(mask[None, None].float(), 2 * radius + 1, 1, radius)[0, 0]


def box_mean(image, height, width):
    integral = F.pad(image.cumsum(0).cumsum(1), (1, 0, 1, 0))
    return (integral[height:, width:] - integral[:-height, width:]
            - integral[height:, :-width] + integral[:-height, :-width]) / (height * width)


def boundary_rgb_losses(pred, target, valid, step, config, perceptual_net=None):
    cfg = validate_boundary_config(config)
    if not cfg['enabled'] or step < cfg['start_step']:
        return {}
    ramp = min(1., (step - cfg['start_step']) / max(cfg['ramp_steps'], 1)) if cfg['ramp_steps'] else 1.
    target, valid = target.detach(), valid.detach().float()
    # Exclude an invalid pixel's entire stencil; masking RGB before Sobel would
    # manufacture a strong edge on the ego-car mask itself.
    stencil = (dilate(1 - valid, 1) == 0).to(pred)
    gt_gradient = sobel(target)
    magnitude = gt_gradient.abs().amax(-1)
    # Convolution roundoff on a constant image must not manufacture edges.
    edge = torch.where(magnitude > 1e-6, magnitude / cfg['edge_threshold'], 0.).clamp(0, 1)
    edge = dilate(edge * stencil, cfg['radius']) * stencil
    denom = edge.sum().clamp_min(1)
    losses = {}
    if cfg['rgb_w']:
        losses['boundary_rgb_loss'] = ramp * cfg['rgb_w'] * ((pred - target).abs().mean(-1) * edge).sum() / denom
    if cfg['gradient_w']:
        losses['boundary_gradient_loss'] = ramp * cfg['gradient_w'] * ((sobel(pred) - gt_gradient).abs().mean(-1) * edge).sum() / denom
    perceptual_w = cfg['perceptual_w']
    if cfg['perceptual_decay_steps'] and step >= cfg['perceptual_decay_start']:
        decay = min(
            1.0,
            (step - cfg['perceptual_decay_start']) / cfg['perceptual_decay_steps'],
        )
        perceptual_w = (
            perceptual_w * (1.0 - decay) + cfg['perceptual_final_w'] * decay
        )
    if perceptual_w:
        if perceptual_net is None:
            raise ValueError('perceptual boundary loss requires a frozen LPIPS network')
        ph, pw = min(cfg['patch_size'], pred.shape[0]), min(cfg['patch_size'], pred.shape[1])
        value = pred.sum() * 0
        if min(ph, pw) >= 32:
            eligible = box_mean(valid, ph, pw) >= cfg['min_valid_fraction']
            scores = (box_mean(edge, ph, pw).clamp_min(0) + 0.05) * eligible
            if scores.sum() > 0:
                indices = torch.multinomial(scores.flatten(), cfg['num_patches'], replacement=True)
                crops_p, crops_t = [], []
                # Global torch RNG participates in the existing exact-resume state.
                for index in indices.tolist():
                    y, x = divmod(index, scores.shape[1])
                    t = target[y:y+ph, x:x+pw]
                    mask = valid[y:y+ph, x:x+pw, None]
                    crops_p.append((pred[y:y+ph, x:x+pw] * mask + t * (1-mask)).permute(2, 0, 1))
                    crops_t.append(t.permute(2, 0, 1))
                perceptual_net.eval()
                value = perceptual_net(torch.stack(crops_p).clamp(0, 1), torch.stack(crops_t).clamp(0, 1), normalize=True).mean()
        losses['boundary_perceptual_loss'] = ramp * perceptual_w * value
    return losses


@torch.no_grad()
def gbuffer_boundary_confidence(normal, albedo, semantic_masks, config):
    """Downweight target discontinuities without using learnable predictions.

    Inputs are already aligned to the native/cropped G-buffer loss frame.
    Sparse LiDAR holes and nonmetric DR depth are deliberately not depth edges.
    """
    cfg = validate_boundary_config(config, gbuffer=True)
    edge = albedo.new_zeros(albedo.shape[:2], dtype=torch.bool)
    n = F.normalize(normal, dim=-1) if normal is not None else None
    cosine = math.cos(math.radians(cfg['normal_angle_degrees']))
    for axis in (0, 1):
        first, second = [slice(None)] * 2, [slice(None)] * 2
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(first), tuple(second)
        discontinuity = (albedo[a] - albedo[b]).abs().amax(-1) > cfg['albedo_threshold']
        if n is not None:
            valid_n = (normal[a].norm(dim=-1) > 0.5) & (normal[b].norm(dim=-1) > 0.5)
            discontinuity |= valid_n & ((n[a] * n[b]).sum(-1) < cosine)
        for mask in semantic_masks:
            discontinuity |= (mask[a] - mask[b]).abs() > 0.5
        edge[a] |= discontinuity
        edge[b] |= discontinuity
    return (1 - (1 - cfg['edge_weight']) * dilate(edge, cfg['radius']))[..., None]
