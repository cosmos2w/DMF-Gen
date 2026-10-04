"""Source distributions sampled at every query coordinate for rectified flow.

Example::

    prior = RFFGaussianPrior(coord_dim=3, n_features=256, lengthscale=0.15)
    x0 = prior(query_coords, n_channels=5)  # [batch, queries, fields]

The IID source is independent across points. The RFF source is smooth because
one random feature weight vector is shared across all points of a draw.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class IIDGaussianPrior(nn.Module):
    """Draw independent standard-normal values for each point and field."""

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(
            bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype
        )


class RFFGaussianPrior(nn.Module):
    """Approximate a smooth Gaussian field with random Fourier features."""

    def __init__(
        self, coord_dim: int = 3, n_features: int = 256, lengthscale: float = 0.15
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer(
            "omega", torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6)
        )
        self.register_buffer("phase", 2 * math.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        # New weights per batch item/field give independent draws on the same
        # fixed coordinate feature map, so every point in a draw stays coupled.
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(
            bsz, n_channels, n_feat, device=coords.device, dtype=coords.dtype
        )
        return torch.einsum("bnf,bcf->bnc", phi, weights)
