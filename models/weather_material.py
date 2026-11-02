"""Small residual decoder for weather-dependent PBR materials.

The decoder predicts *logit-space* changes rather than complete materials.
Keeping the residual in logit space makes the identity initialization exact
while guaranteeing that the activated albedo and roughness stay in range.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class WeatherMaterialDecoder(nn.Module):
    """Predict bounded albedo/roughness logit residuals per Gaussian.

    The input is ``feature | base albedo | base metallic | base roughness |
    upness``.  Consequently its width is ``feature_dim + 6`` (38 for the
    32-D features used by the full PBR model).  ``upness`` is expected to be a
    sign-invariant scalar computed from the Gaussian's world-space surface
    axis, not from a camera-facing normal.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        albedo_delta_cap: float = 4.0,
        roughness_delta_cap: float = 5.0,
        num_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        feature_dim = int(feature_dim)
        albedo_delta_cap = float(albedo_delta_cap)
        roughness_delta_cap = float(roughness_delta_cap)
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        hidden_dim = feature_dim if hidden_dim is None else int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        num_hidden_layers = int(num_hidden_layers)
        if num_hidden_layers <= 0:
            raise ValueError(
                "num_hidden_layers must be positive, got "
                f"{num_hidden_layers}"
            )
        if albedo_delta_cap <= 0 or roughness_delta_cap <= 0:
            raise ValueError(
                "weather-material logit caps must be positive, got "
                f"albedo={albedo_delta_cap}, roughness={roughness_delta_cap}"
            )

        self.feature_dim = feature_dim
        self.input_dim = self.feature_dim + 6
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.albedo_delta_cap = float(albedo_delta_cap)
        self.roughness_delta_cap = float(roughness_delta_cap)

        layers = []
        input_dim = self.input_dim
        for _ in range(self.num_hidden_layers):
            layers.extend((
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            ))
            input_dim = self.hidden_dim
        layers.append(nn.Linear(input_dim, 4))
        self.network = nn.Sequential(*layers)

        # A newly attached weather branch must reproduce the base material
        # exactly before its first update.  Zeroing only the output layer keeps
        # the hidden layer conventionally initialized and lets the output layer
        # start learning immediately.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _check_last_dim(name: str, value: torch.Tensor, expected: int) -> None:
        if value.shape[-1] != expected:
            raise ValueError(
                f"{name} must end in {expected} channels, got {tuple(value.shape)}"
            )

    def forward(
        self,
        features: torch.Tensor,
        base_albedo: torch.Tensor,
        base_metallic: torch.Tensor,
        base_roughness: torch.Tensor,
        upness: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._check_last_dim("features", features, self.feature_dim)
        self._check_last_dim("base_albedo", base_albedo, 3)
        self._check_last_dim("base_metallic", base_metallic, 1)
        self._check_last_dim("base_roughness", base_roughness, 1)
        self._check_last_dim("upness", upness, 1)

        leading_shape = features.shape[:-1]
        for name, value in (
            ("base_albedo", base_albedo),
            ("base_metallic", base_metallic),
            ("base_roughness", base_roughness),
            ("upness", upness),
        ):
            if value.shape[:-1] != leading_shape:
                raise ValueError(
                    f"{name} leading shape {tuple(value.shape[:-1])} does not "
                    f"match features {tuple(leading_shape)}"
                )

        raw_delta = self.network(torch.cat(
            [features, base_albedo, base_metallic, base_roughness, upness],
            dim=-1,
        ))
        return {
            "delta_albedo_logit": (
                self.albedo_delta_cap * torch.tanh(raw_delta[..., 0:3])
            ),
            "delta_roughness_logit": (
                self.roughness_delta_cap * torch.tanh(raw_delta[..., 3:4])
            ),
        }
