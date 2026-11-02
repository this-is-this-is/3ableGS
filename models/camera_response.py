"""Shared, anchored camera response for the PBR display conversion.

The trainable curve remaps Reinhard-compressed radiance before the existing
sRGB transfer function.  It is shared across frames, weather and RGB channels;
there is no image embedding.  Fixed points (0, 0), (0.5, 0.5), (1, 1) restrict
exposure ambiguity, while positive interval widths guarantee monotonicity.
"""

import math
from typing import Dict

import torch
from torch import nn


class _StableReinhard(torch.autograd.Function):
    """Legacy forward with an analytic backward avoiding HDR cancellation."""

    @staticmethod
    def forward(ctx, linear):
        ctx.save_for_backward(linear)
        return linear / (1.0 + linear.clamp(min=0.0))

    @staticmethod
    def backward(ctx, grad_output):
        (linear,) = ctx.saved_tensors
        # Differentiating x/(1+x) literally subtracts two almost equal terms
        # for bright HDR values, producing tiny negative floating-point slopes.
        # The analytic derivative is positive and uses no such subtraction.
        slope = (1.0 + linear.clamp(min=0.0)).reciprocal().square()
        return grad_output * slope


class _ReducedKnotInterpolation(torch.autograd.Function):
    """Piecewise-linear interpolation with contention-free knot gradients.

    Autograd implements ``offsets[index]`` backward as a scatter-add.  A
    full-resolution RGB image scatters millions of values into only a handful
    of knots, which serializes the CUDA kernel on atomic additions.  Reducing
    each knot's two neighbouring interval contributions avoids that hot spot
    while preserving the exact forward calculation.
    """

    @staticmethod
    def forward(ctx, x, offsets, knots):
        index = torch.searchsorted(knots, x.contiguous(), right=True)
        index = (index - 1).clamp(0, len(knots) - 2)
        fraction = (x - knots[index]) / (knots[index + 1] - knots[index])
        ctx.save_for_backward(index, fraction, offsets, knots)
        return x + offsets[index] + fraction * (offsets[index + 1] - offsets[index])

    @staticmethod
    def backward(ctx, grad_output):
        index, fraction, offsets, knots = ctx.saved_tensors
        grad_x = grad_offsets = None
        if ctx.needs_input_grad[0]:
            slope = (offsets[index + 1] - offsets[index]) / (
                knots[index + 1] - knots[index]
            )
            grad_x = grad_output * (1.0 + slope)
        if ctx.needs_input_grad[1]:
            # The number of knots is small (nine in Base v3).  Dense reductions
            # are substantially faster than contended atomic scatter-adds.
            zero = grad_output.new_zeros(())
            grad_offsets = torch.stack([
                torch.where(
                    index == knot, grad_output * (1.0 - fraction), zero
                ).sum()
                + torch.where(
                    index + 1 == knot, grad_output * fraction, zero
                ).sum()
                for knot in range(len(offsets))
            ])
        return grad_x, grad_offsets, None


class MonotonicCameraResponse(nn.Module):
    """A piecewise-linear monotonic remap, initialized to exact legacy display.

    ``num_knots`` includes both endpoints and must be odd so that the middle
    knot anchors compressed intensity 0.5 (linear radiance 1). ``min_slope``
    is a strictly positive floor on each remapping interval's slope.  The
    floor also keeps extreme, finite logits numerically well conditioned.

    Parameters follow the trainer's miscellaneous-model constructor contract;
    ``n`` is accepted for compatibility and creates no per-frame parameters.
    """

    def __init__(
        self,
        class_name: str = "CameraResponse",
        n: int = 0,
        num_knots: int = 9,
        min_slope: float = 0.05,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        super().__init__()
        if isinstance(num_knots, bool) or int(num_knots) != num_knots:
            raise ValueError("num_knots must be an odd integer >= 5")
        num_knots = int(num_knots)
        if num_knots < 5 or num_knots % 2 != 1:
            raise ValueError("num_knots must be an odd integer >= 5")
        min_slope = float(min_slope)
        if not math.isfinite(min_slope) or not 0.0 < min_slope < 1.0:
            raise ValueError("min_slope must be finite and strictly between 0 and 1")
        self.class_prefix = class_name + "#"
        self.num_knots = num_knots
        self.min_slope = min_slope
        self.register_buffer("knot_x", torch.linspace(0.0, 1.0, num_knots, device=device))
        # linspace can round the midpoint away from exactly 0.5 for some
        # non-power-of-two interval counts (e.g. 14 intervals in float32).
        self.knot_x[(num_knots - 1) // 2] = 0.5
        self.interval_logits = nn.Parameter(
            torch.zeros(2, (num_knots - 1) // 2, device=device)
        )

    @property
    def device(self):
        return self.interval_logits.device

    def knot_offsets(self) -> torch.Tensor:
        """Offsets from identity, exactly zero at initialization and anchors."""
        probabilities = torch.softmax(self.interval_logits, dim=-1)
        uniform = torch.full_like(probabilities, 1.0 / probabilities.shape[-1])
        # Subtract BEFORE cumulative summation: zero logits then give exactly
        # zero offsets, including with knot counts that are not powers of two.
        offsets = (probabilities - uniform).cumsum(dim=-1)
        offsets = offsets[:, :-1] * (0.5 * (1.0 - self.min_slope))
        zero = self.interval_logits.new_zeros(1)
        return torch.cat((zero, offsets[0], zero, offsets[1], zero))

    def knot_values(self) -> torch.Tensor:
        return self.knot_x + self.knot_offsets()

    def remap(self, compressed: torch.Tensor) -> torch.Tensor:
        """Remap intensities in [0, 1]; saturate values outside that domain."""
        x = compressed.clamp(0.0, 1.0)
        knots = self.knot_x.to(dtype=x.dtype)
        offsets = self.knot_offsets().to(dtype=x.dtype)
        # Adding an interpolated residual, rather than interpolating y itself,
        # preserves the exact legacy floating-point output at initialization.
        return _ReducedKnotInterpolation.apply(x, offsets, knots)

    def forward(self, linear: torch.Tensor) -> torch.Tensor:
        """Map linear radiance to display sRGB, before the existing Affine."""
        # Keep these operations identical to models.pbr.to_display. Avoid
        # importing that module so the response and its tests remain CPU-only
        # and do not require the rasterizer/nvdiffrast import dependencies.
        compressed = _StableReinhard.apply(linear)
        x = self.remap(compressed).clamp(min=0.0)
        return torch.where(
            x <= 0.0031308,
            x * 12.92,
            1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055,
        )

    def get_param_groups(self) -> Dict[str, list]:
        return {self.class_prefix + "all": list(self.parameters())}

    def compute_reg_loss(self, reg_cfg=None) -> Dict[str, torch.Tensor]:
        """Optional identity and adjacent-slope penalties, each using ``w``."""
        if reg_cfg is None:
            return {}
        losses = {}
        offsets = self.knot_offsets()
        values = self.knot_x + offsets
        slopes = (values[1:] - values[:-1]) / (self.knot_x[1:] - self.knot_x[:-1])
        terms = {
            "identity": offsets.square().mean(),
            "smoothness": (slopes[1:] - slopes[:-1]).square().mean(),
        }
        for name, penalty in terms.items():
            cfg = reg_cfg.get(name)
            if cfg is not None:
                weight = float(cfg.get("w", 0.0))
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError(f"camera response {name}.w must be finite and nonnegative")
                losses[f"camera_response_{name}"] = penalty * weight
        return losses
