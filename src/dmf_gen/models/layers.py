"""Attention, coordinate encoding, and indexed-gather layers used by GL-RBF.

Example::

    block = CrossAttentionBlock(dim=64, num_heads=4)
    updated_queries = block(query_tokens, sensor_tokens, kv_padding_mask=padded_sensors)

The padding mask is True for sensor slots that must not contribute to attention.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, act=nn.GELU) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(dim, hidden_dim), act()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class FourierPositionalEncoding(nn.Module):
    """Sine-cosine frequency encoding for spatial coordinates."""

    def __init__(self, coord_dim: int, num_bands: int = 32, max_freq: float = 64.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_bands = num_bands
        self.out_dim = coord_dim * num_bands * 2
        freqs = torch.linspace(1.0, max_freq / 2.0, num_bands)
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords[..., : self.coord_dim] * 2.0 - 1.0
        x = coords.unsqueeze(-1) * self.freqs * math.pi
        enc = torch.cat([x.sin(), x.cos()], dim=-1)
        return enc.reshape(*coords.shape[:-1], self.out_dim)

# Neighbor indices have shape [batch, queries, K]; the helpers preserve that
# leading layout when selecting scalar or vector sensor features.
def batched_gather_2d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather from x with shape [B, M] using idx with shape [B, N, K].
    Returns shape [B, N, K].
    """
    bsz = x.shape[0]
    batch_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1).expand_as(idx)
    return x[batch_idx, idx]


def batched_gather_3d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather from x with shape [B, M, C] using idx with shape [B, N, K].
    Returns shape [B, N, K, C].
    """
    bsz = x.shape[0]
    batch_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1).expand_as(idx)
    return x[batch_idx, idx]

class FeedForward(nn.Module):
    """
    Standard Transformer feed-forward block used after attention.
    """
    def __init__(self, dim: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block with residual connection and FFN.

    q  : [B, Tq, D]
    kv : [B, Tk, D]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, ff_mult=ff_mult, dropout=mlp_dropout)
        # Execution counters are plain Python state and never alter checkpoints.
        self.kv_projection_calls = 0

    def reset_execution_counters(self) -> None:
        self.kv_projection_calls = 0

    def prepare_kv(
        self,
        kv: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Normalize/project condition-static K/V once without detaching it."""
        kv_in = self.norm_kv(kv)
        dim = self.attn.embed_dim
        kv_proj = F.linear(
            kv_in,
            self.attn.in_proj_weight[dim:],
            None if self.attn.in_proj_bias is None else self.attn.in_proj_bias[dim:],
        )
        key, value = kv_proj.chunk(2, dim=-1)
        batch_size, source_length, _ = key.shape
        head_dim = dim // self.attn.num_heads
        key = key.view(batch_size, source_length, self.attn.num_heads, head_dim)
        value = value.view(batch_size, source_length, self.attn.num_heads, head_dim)
        attn_mask = None
        if kv_padding_mask is not None:
            attn_mask = torch.zeros(
                (batch_size, 1, 1, source_length),
                dtype=key.dtype,
                device=key.device,
            ).masked_fill(kv_padding_mask[:, None, None, :].bool(), float("-inf"))
            # Match MultiheadAttention's per-head padding-mask layout in SDPA.
            attn_mask = attn_mask.expand(
                batch_size, self.attn.num_heads, 1, source_length,
            ).contiguous()
        self.kv_projection_calls += 1
        return {
            "key": key.transpose(1, 2),
            "value": value.transpose(1, 2),
            "attn_mask": attn_mask,
        }

    def forward_prepared(
        self,
        q: torch.Tensor,
        prepared_kv: Mapping[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """Run the original residual/FFN block using preprojected sensor K/V."""
        q_in = self.norm_q(q)
        dim = self.attn.embed_dim
        q_proj = F.linear(
            q_in,
            self.attn.in_proj_weight[:dim],
            None if self.attn.in_proj_bias is None else self.attn.in_proj_bias[:dim],
        )
        batch_size, target_length, _ = q_proj.shape
        head_dim = dim // self.attn.num_heads
        q_proj = q_proj.view(
            batch_size, target_length, self.attn.num_heads, head_dim,
        ).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q_proj,
            prepared_kv["key"],
            prepared_kv["value"],
            attn_mask=prepared_kv["attn_mask"],
            dropout_p=self.attn.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, target_length, dim,
        )
        attn_out = self.attn.out_proj(attn_out)
        x = q + attn_out
        return x + self.ff(self.norm_ff(x))

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Normalize queries and keys/values independently.
        q_in = self.norm_q(q)
        kv_in = self.norm_kv(kv)
        self.kv_projection_calls += 1

        # key_padding_mask: True means "ignore this token".
        attn_out, _ = self.attn(
            q_in,
            kv_in,
            kv_in,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )

        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class CompactLatentReadout(nn.Module):
    """Project query coordinates against latent memory with reusable latent K/V."""

    def __init__(
        self,
        query_in_dim: int,
        latent_dim: int,
        query_dim: int,
        rank: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1 or rank % num_heads != 0:
            raise ValueError(
                f"cq_readout_rank must be positive and divisible by cq_readout_heads; "
                f"got rank={rank}, heads={num_heads}."
            )
        if query_dim < 1 or query_dim % num_heads != 0:
            raise ValueError(
                f"cq_query_dim must be positive and divisible by cq_readout_heads; "
                f"got query_dim={query_dim}, heads={num_heads}."
            )
        self.rank = int(rank)
        self.num_heads = int(num_heads)
        self.head_rank = self.rank // self.num_heads
        self.query_dim = int(query_dim)
        self.head_value_dim = self.query_dim // self.num_heads
        self.attn_dropout = float(attn_dropout)

        self.query_norm = nn.LayerNorm(query_in_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.q_proj = nn.Linear(query_in_dim, rank, bias=False)
        self.k_proj = nn.Linear(latent_dim, rank, bias=False)
        self.v_proj = nn.Linear(latent_dim, query_dim, bias=False)
        self.out_norm = nn.LayerNorm(query_dim)

    def project_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project condition-static latent memory once for reuse across chunks/NFEs."""
        bsz, n_latents, _ = latents.shape
        normalized = self.latent_norm(latents)
        keys = self.k_proj(normalized).view(
            bsz, n_latents, self.num_heads, self.head_rank,
        ).transpose(1, 2)
        values = self.v_proj(normalized).view(
            bsz, n_latents, self.num_heads, self.head_value_dim,
        ).transpose(1, 2)
        return keys, values

    def forward(
        self,
        query_features: torch.Tensor,
        *,
        latents: Optional[torch.Tensor] = None,
        projected_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if projected_kv is None:
            if latents is None:
                raise ValueError("CompactLatentReadout requires latents or projected_kv.")
            projected_kv = self.project_latents(latents)
        keys, values = projected_kv
        bsz, n_query, _ = query_features.shape
        queries = self.q_proj(self.query_norm(query_features)).view(
            bsz, n_query, self.num_heads, self.head_rank,
        ).transpose(1, 2)
        logits = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_rank)
        weights = torch.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.attn_dropout, training=self.training)
        readout = torch.matmul(weights, values).transpose(1, 2).reshape(
            bsz, n_query, self.query_dim,
        )
        return self.out_norm(readout)


class SelfAttentionBlock(nn.Module):
    """
    Standard latent self-attention block with residual connection and FFN.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, ff_mult=ff_mult, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.norm_attn(x)
        attn_out, _ = self.attn(x_in, x_in, x_in, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x
