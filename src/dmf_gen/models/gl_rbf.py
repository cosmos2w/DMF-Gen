"""Global/local RBF velocity backbone for marked sparse measurements.

Example::

    flow = build_pointcloud_model({"model_name": "GL_rbf", "coord_dim": 3}, n_fields=5)
    velocity = flow.model(t, x_t, query_coords, sensor_coords, sensor_values, sensor_mask, field_ids)

Shapes are ``[B,Q,D]`` for query coordinates, ``[B,M,D]`` for sensor coordinates, ``[B,M,1]`` for scalar observations, and ``[B,Q,C]`` for the returned velocity. Use the factory to select the base or enhanced checkpoint profile.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pykeops.torch import LazyTensor
except ImportError:  # KeOps is optional; the Torch backend remains supported.
    LazyTensor = None

from ..cache.geometry import (
    cache_tensors,
    validate_persistent_topk_geometry_cache,
)
from .layers import (
    CrossAttentionBlock,
    FourierPositionalEncoding,
    SelfAttentionBlock,
    batched_gather_2d,
    batched_gather_3d,
    make_mlp,
)


class ConditionalPointHybridLocalGlobalRBF(nn.Module):
    """Map sparse scalar sensor tokens and query state to a full-field velocity.

    The learned latents first read the marked sensor set; sensor tokens then
    read back from those latents. Each query combines its current flow state,
    a global latent summary or readout, and an RBF-weighted local sensor
    neighborhood. The factory enables enhanced features without changing this
    class's base interface or production checkpoint parameter names.
    """

    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        latent_dim: int = 256,
        num_latents: int = 64,
        num_heads: int = 8,
        num_latent_blocks: int = 3,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        rbf_sigma: float = 0.05,
        summary_type: str = "cls",   # ["cls", "mean"]

        gather_mode: str = "rbf",    # ["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"]
        gather_topk: int = 32,
        gather_query_chunk_size: Optional[int] = None,
        learnable_rbf_sigma: bool = False,
        neighbor_backend: str = "torch",      # ["auto", "torch", "keops"]

        sensor_local_topk: int = 8,
        sensor_local_dropout: float = 0.0,
        use_fourier_pe: bool = False,
        fourier_pe_num_bands: int = 32,
        fourier_pe_max_freq: float = 64.0,
        enhanced_backbone: bool = False,
        sensor_coord_encoding: str = "raw",
        latent_sensor_reinject: bool = False,
        latent_reinject_every: int = 1,
        condition_attention_execution: str = "legacy_mha",
        sensor_attention_padding_mode: str = "full",
        sensor_attention_buckets: Sequence[int] = (256, 320, 384),
        query_latent_readout: bool = False,
        query_readout_type: str = "point",
        query_readout_scale_init: float = 0.0,
        enhanced_head_norm: bool = False,
        glres_scale_init: float = 0.0,
    ) -> None:
        super().__init__()

        if summary_type not in ["cls", "mean"]:
            raise ValueError(f"summary_type must be 'cls' or 'mean', got {summary_type}")
        if sensor_coord_encoding not in ["raw", "fourier"]:
            raise ValueError(
                f"sensor_coord_encoding must be one of ['raw', 'fourier'], got {sensor_coord_encoding}"
            )
        if query_readout_type not in ["point", "coord"]:
            raise ValueError(
                f"query_readout_type must be one of ['point', 'coord'], got {query_readout_type}"
            )
        if latent_reinject_every < 1:
            raise ValueError(f"latent_reinject_every must be >= 1, got {latent_reinject_every}")
        if condition_attention_execution not in ("legacy_mha", "cached_kv"):
            raise ValueError(
                "condition_attention_execution must be 'legacy_mha' or 'cached_kv'."
            )
        if sensor_attention_padding_mode not in ("full", "static_buckets"):
            raise ValueError(
                "sensor_attention_padding_mode must be 'full' or 'static_buckets'."
            )
        normalized_buckets = tuple(sorted({int(value) for value in sensor_attention_buckets}))
        if not normalized_buckets or normalized_buckets[0] < 1:
            raise ValueError("sensor_attention_buckets must contain positive lengths.")

        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.summary_type = summary_type
        self.use_fourier_pe = use_fourier_pe
        self.pos_enc = FourierPositionalEncoding(
            coord_dim, num_bands=fourier_pe_num_bands, max_freq=fourier_pe_max_freq
        ) if use_fourier_pe else None
        self.coord_feat_dim = self.pos_enc.out_dim if self.pos_enc is not None else coord_dim
        self.enhanced_backbone = bool(enhanced_backbone)
        self.sensor_coord_encoding = sensor_coord_encoding
        self.latent_sensor_reinject = bool(latent_sensor_reinject)
        self.latent_reinject_every = int(latent_reinject_every)
        self.condition_attention_execution = str(condition_attention_execution)
        self.sensor_attention_padding_mode = str(sensor_attention_padding_mode)
        self.sensor_attention_buckets = normalized_buckets
        self.query_latent_readout_enabled = bool(query_latent_readout)
        self.query_readout_type = query_readout_type
        self.enhanced_head_norm = bool(enhanced_head_norm)

        gather_modes = ["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"]
        if gather_mode not in gather_modes:
            raise ValueError(
                f"gather_mode must be one of {gather_modes}, got {gather_mode}"
            )
        if neighbor_backend not in ["auto", "torch", "keops"]:
            raise ValueError(
                f"neighbor_backend must be one of ['auto', 'torch', 'keops'], got {neighbor_backend}"
            )
        self.gather_mode = gather_mode
        self.gather_topk = int(gather_topk)
        self.gather_query_chunk_size = gather_query_chunk_size
        self.learnable_rbf_sigma = learnable_rbf_sigma
        self.neighbor_backend = neighbor_backend

        # Only build the heavy query-side gate when the gate mode is actually selected.
        if self.gather_mode == "topk_rbf_gate":
            self.query_to_cond = nn.Linear(hidden_dim, cond_dim, bias=False)

            # Scalar query-neighbor reweighting.
            gate_in_dim = cond_dim + cond_dim + coord_dim + 1
            self.gather_gate = nn.Sequential(
                nn.Linear(gate_in_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )

        if self.gather_topk < 1:
            raise ValueError(f"gather_topk must be >= 1, got {self.gather_topk}")
        # A learned log scale keeps the RBF width positive when enabled.
        if learnable_rbf_sigma:
            self.log_rbf_sigma = nn.Parameter(torch.log(torch.tensor(float(rbf_sigma))))

        self.sensor_local_topk = int(sensor_local_topk)
        self.sensor_local_dropout_p = float(sensor_local_dropout)

        if self.sensor_local_topk < 1:
            raise ValueError(f"sensor_local_topk must be >= 1, got {self.sensor_local_topk}")

        # -------------------------
        # Point/query branch
        # -------------------------
        # Query point token from [coords, x_t, t]
        self.point_encoder = make_mlp(
            in_dim=self.coord_feat_dim + n_fields + 1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=3,
        )

        # -------------------------
        # Sparse sensor branch
        # -------------------------
        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        # Initial sparse sensor token from [obs_coords, obs_value, field_embed]
        sensor_coord_dim = (
            self.coord_feat_dim
            if sensor_coord_encoding == "fourier" and self.pos_enc is not None
            else coord_dim
        )
        self.sensor_in_proj = make_mlp(
            in_dim=sensor_coord_dim + 1 + field_embed_dim,
            hidden_dim=latent_dim,
            out_dim=latent_dim,
            depth=3,
        )

        # Project the refined sensor tokens to the local conditioning width
        # used by the RBF gather.
        self.sensor_out_proj = make_mlp(
            in_dim=latent_dim,
            hidden_dim=cond_dim,
            out_dim=cond_dim,
            depth=2,
        )

        # --------------------------------------------------
        # Optional sensor-side local refinement block Used only in gather_mode == "topk_rbf_ptlocal"
        # This is intentionally placed AFTER sensor_out_proj so it works on cond_dim features,
        # which keeps memory and compute lower than refining in latent_dim.
        # --------------------------------------------------
        if self.gather_mode == "topk_rbf_ptlocal":
            self.sensor_local_q = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_k = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_v = nn.Linear(cond_dim, cond_dim, bias=False)
            # Relative position encoding: [dx, dy, dz, ||d||]
            self.sensor_local_pos = make_mlp(
                in_dim=coord_dim + 1,
                hidden_dim=cond_dim,
                out_dim=cond_dim,
                depth=2,
            )
            # Lightweight Point-Transformer-style scalar attention over local neighbors.
            self.sensor_local_attn = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )
            self.sensor_local_out = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_dropout = nn.Dropout(sensor_local_dropout)
            self.sensor_local_norm = nn.LayerNorm(cond_dim)

        # Optional query-to-latent readout can be used by enhanced GL_rbf for any gather mode.
        # The legacy topk_rbf_glres path reuses these same modules to preserve old behavior.
        self.use_query_latent_readout = self.query_latent_readout_enabled or self.gather_mode == "topk_rbf_glres"
        if self.use_query_latent_readout:
            if self.query_readout_type == "coord":
                self.query_decoder_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
                self.query_readout_in = nn.Linear(self.coord_feat_dim + hidden_dim, latent_dim, bias=False)
            else:
                self.query_decoder_token = None
                self.query_readout_in = nn.Linear(hidden_dim, latent_dim, bias=False)
            self.query_latent_readout = CrossAttentionBlock(
                dim=latent_dim,
                num_heads=max(1, min(num_heads, 4)),
                ff_mult=max(1, ff_mult // 2),
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            self.query_readout_out = nn.Linear(latent_dim, hidden_dim, bias=False)
            self.query_readout_scale = nn.Parameter(torch.tensor(float(query_readout_scale_init)))

        if self.gather_mode == "topk_rbf_glres":
            # Coarse scaffold is summary-driven and pointwise, so it avoids [B, N, K, C] tensors.
            self.coarse_film = nn.Linear(hidden_dim, 2 * hidden_dim)
            self.coarse_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(hidden_dim, n_fields),
            )
            self.coarse_scale = nn.Parameter(torch.tensor(float(glres_scale_init)))

            # Sensor importance is computed once per refined sensor token, then gathered as scalars.
            self.sensor_importance = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )
            self.sensor_importance_scale = nn.Parameter(torch.tensor(float(glres_scale_init)))

        # -------------------------
        # Latent global processor
        # -------------------------
        self.latents = nn.Parameter(
            torch.randn(num_latents, latent_dim) / math.sqrt(latent_dim)
        )

        # Latents attend to sparse sensor tokens
        self.input_cross_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Process latents in latent space
        self.latent_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=latent_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(num_latent_blocks)
        ])

        # Refine local sensor tokens using the processed global latent memory.
        self.sensor_back_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Separate projection for the latent summary used as a global feature
        self.summary_proj = make_mlp(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=2,
        )

        # -------------------------
        # Final velocity head
        # -------------------------
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, n_fields),
        )
        head_in_dim = hidden_dim + hidden_dim + cond_dim
        self.head_in_norm = nn.LayerNorm(head_in_dim) if enhanced_head_norm else nn.Identity()

    def _build_sensor_tokens(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build sparse sensor tokens from:
          - sensor coordinates
          - observed scalar value
          - field identity embedding
        """
        safe_field_ids = obs_field_ids.clamp_min(0)
        field_feat = self.field_embed(safe_field_ids)                 # [B, M, E]
        field_feat = field_feat * obs_mask.unsqueeze(-1)             # zero padded rows

        # Enhanced mode uses the same Fourier coordinate representation for sensors and queries.
        # This mirrors Senseiver-style spatial tokenization while preserving GL_rbf's local gather.
        if self.sensor_coord_encoding == "fourier" and self.pos_enc is not None:
            sensor_coord_feat = self.pos_enc(obs_coords)
        else:
            sensor_coord_feat = obs_coords

        sensor_in = torch.cat([sensor_coord_feat, obs_values, field_feat], dim=-1)
        sensor_tokens = self.sensor_in_proj(sensor_in)               # [B, M, D]
        sensor_tokens = sensor_tokens * obs_mask.unsqueeze(-1)
        return sensor_tokens

    def _encode_latents(
        self,
        sensor_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Let the learned latent array absorb and process the sparse sensor set.
        """
        bsz = sensor_tokens.shape[0]

        # Expand learned latents across the batch
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)      # [B, L, D]

        # key_padding_mask: True means "ignore this token"
        sensor_padding_mask = ~obs_mask.bool()

        bucket_groups = self._sensor_attention_bucket_groups(obs_mask)
        if bucket_groups is not None:
            prepared_groups = []
            for batch_indices, bucket_length in bucket_groups:
                tokens = sensor_tokens.index_select(0, batch_indices)[:, :bucket_length]
                padding = sensor_padding_mask.index_select(0, batch_indices)[:, :bucket_length]
                prepared = (
                    self.input_cross_attn.prepare_kv(tokens, padding)
                    if self.condition_attention_execution == "cached_kv"
                    else None
                )
                prepared_groups.append((batch_indices, tokens, padding, prepared))

            def attend(current_latents: torch.Tensor) -> torch.Tensor:
                output = torch.zeros_like(current_latents)
                for batch_indices, tokens, padding, prepared in prepared_groups:
                    group_latents = current_latents.index_select(0, batch_indices)
                    if prepared is None:
                        group_output = self.input_cross_attn(
                            q=group_latents, kv=tokens, kv_padding_mask=padding,
                        )
                    else:
                        group_output = self.input_cross_attn.forward_prepared(
                            group_latents, prepared,
                        )
                    output = output.index_copy(0, batch_indices, group_output)
                return output
        elif self.condition_attention_execution == "cached_kv":
            prepared = self.input_cross_attn.prepare_kv(
                sensor_tokens, sensor_padding_mask,
            )

            def attend(current_latents: torch.Tensor) -> torch.Tensor:
                return self.input_cross_attn.forward_prepared(current_latents, prepared)
        else:
            def attend(current_latents: torch.Tensor) -> torch.Tensor:
                return self.input_cross_attn(
                    q=current_latents,
                    kv=sensor_tokens,
                    kv_padding_mask=sensor_padding_mask,
                )

        # Latents attend to sparse sensor tokens
        latents = attend(latents)

        # Process in latent space, optionally re-reading sparse sensors between blocks.
        for i, block in enumerate(self.latent_blocks):
            if (
                self.latent_sensor_reinject
                and i > 0
                and i % self.latent_reinject_every == 0
            ):
                # Senseiver-style re-injection: latents re-read the sparse measurements.
                # Cost scales with L*M, not N*M, so this preserves query-side efficiency.
                latents = attend(latents)
            latents = block(latents)

        return latents

    def _sensor_attention_bucket_groups(
        self,
        obs_mask: torch.Tensor,
    ) -> Optional[list[tuple[torch.Tensor, int]]]:
        """Return stable batch groups, or None for the exact full-padding path."""
        if self.sensor_attention_padding_mode != "static_buckets":
            return None
        valid = obs_mask.bool()
        counts = valid.sum(dim=1)
        positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
        if not torch.equal(valid, positions < counts.unsqueeze(1)):
            return None
        assigned = []
        max_length = int(valid.shape[1])
        for count in counts.tolist():
            bucket = next(
                (size for size in self.sensor_attention_buckets if size >= int(count)),
                max_length,
            )
            assigned.append(min(bucket, max_length))
        bucket_tensor = torch.tensor(assigned, device=valid.device)
        return [
            (torch.nonzero(bucket_tensor == bucket, as_tuple=False).flatten(), int(bucket))
            for bucket in sorted(set(assigned))
        ]

    def _refine_sensor_tokens(
        self,
        sensor_tokens: torch.Tensor,
        latents: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run sensor-back-attention without evaluating known padded tail slots."""
        bucket_groups = self._sensor_attention_bucket_groups(obs_mask)
        if bucket_groups is None:
            return self.sensor_back_attn(
                q=sensor_tokens, kv=latents, kv_padding_mask=None,
            )
        refined = torch.zeros_like(sensor_tokens)
        max_length = sensor_tokens.shape[1]
        for batch_indices, bucket_length in bucket_groups:
            group_refined = self.sensor_back_attn(
                q=sensor_tokens.index_select(0, batch_indices)[:, :bucket_length],
                kv=latents.index_select(0, batch_indices),
                kv_padding_mask=None,
            )
            if bucket_length < max_length:
                group_refined = F.pad(group_refined, (0, 0, 0, max_length - bucket_length))
            refined = refined.index_copy(0, batch_indices, group_refined)
        return refined

    def _extract_global_summary(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Convert the latent array into one global summary vector.

        If summary_type == 'cls', the last latent slot is treated as the summary token.
        If summary_type == 'mean', use the mean of all latent slots.
        """
        if self.summary_type == "cls":
            summary = latents[:, -1]         # [B, D]
        else:
            summary = latents.mean(dim=1)    # [B, D]

        return self.summary_proj(summary)    # [B, H]

    def _build_query_readout_tokens(
        self,
        point_feat: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build query tokens for latent readout from point features or coordinate decoder tokens.
        """
        if self.query_readout_type == "coord":
            bsz, n_query, _ = coords.shape
            coord_feat = self.pos_enc(coords) if self.pos_enc is not None else coords
            dq = self.query_decoder_token.view(1, 1, -1).expand(bsz, n_query, -1)
            return self.query_readout_in(torch.cat([coord_feat, dq], dim=-1))
        return self.query_readout_in(point_feat)

    def _readout_query_global_chunked(
        self,
        point_feat: torch.Tensor,
        coords: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query-to-latent readout in chunks. This is O(B * N * L), avoiding query-sensor
        [B, N, K, C] feature materialization.
        """
        n_query = point_feat.shape[1]
        chunk_size = self.gather_query_chunk_size
        if chunk_size is None and n_query > 4096:
            chunk_size = 4096

        if chunk_size is None or n_query <= chunk_size:
            q = self._build_query_readout_tokens(point_feat, coords)
            readout = self.query_latent_readout(q=q, kv=latents, kv_padding_mask=None)
            return self.query_readout_out(readout)

        outputs = []
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            q = self._build_query_readout_tokens(point_feat[:, start:end], coords[:, start:end])
            readout = self.query_latent_readout(q=q, kv=latents, kv_padding_mask=None)
            outputs.append(self.query_readout_out(readout))
        return torch.cat(outputs, dim=1)

    def _predict_global_coarse(
        self,
        point_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.coarse_film(global_feat).chunk(2, dim=-1)
        coarse_feat = (
            point_feat * (1.0 + torch.tanh(gamma).unsqueeze(1))
            + beta.unsqueeze(1)
        )
        return self.coarse_head(coarse_feat)

    def _compute_sensor_importance_bias(
        self,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        bias = self.sensor_importance(refined_sensor_feat).squeeze(-1)
        return bias * obs_mask.to(dtype=bias.dtype)

    def _use_keops(self) -> bool:
        """
        Decide whether to use KeOps.

        - rbf mode can benefit a lot from KeOps soft reductions
        - topk modes can use KeOps KNN search
        """
        if self.neighbor_backend == "torch":
            return False

        if self.neighbor_backend == "keops":
            if LazyTensor is None:
                raise ImportError(
                    "neighbor_backend='keops' was requested, but pykeops is not installed."
                )
            return True

        # auto
        return LazyTensor is not None

    def _aggregate_rbf_keops(
        self,
        query_coords: torch.Tensor,         # [B, N, D]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
    ) -> torch.Tensor:
        """
        Full RBF gather using KeOps sumsoftmaxweight, without building the dense [B, N, M] matrix.
        """
        sigma = torch.exp(self.log_rbf_sigma).clamp_min(1e-6) if self.learnable_rbf_sigma else self.rbf_sigma
        gamma = 1.0 / (2 * sigma ** 2 + 1e-12)

        # --- Force contiguous memory for KeOps ---
        query_coords = query_coords.contiguous()
        obs_coords = obs_coords.contiguous()
        refined_sensor_feat = refined_sensor_feat.contiguous()
        # -----------------------------------------

        # KeOps symbolic tensors
        x_i = LazyTensor(query_coords[:, :, None, :])                 # [B, N, 1, D]
        y_j = LazyTensor(obs_coords[:, None, :, :])                   # [B, 1, M, D]
        v_j = LazyTensor(refined_sensor_feat[:, None, :, :])          # [B, 1, M, Cc]

        # Scalar logits: -gamma * ||x_i - y_j||^2
        sqdist_ij = ((x_i - y_j) ** 2).sum(-1)                        # [B, N, M, 1]
        logits_ij = -gamma * sqdist_ij

        # Mask invalid sensor slots by adding a large negative number
        mask_j = LazyTensor(obs_mask[:, None, :, None].to(query_coords.dtype).contiguous())   # [B, 1, M, 1]
        logits_ij = logits_ij + (mask_j - 1.0) * 1e6

        # Softmax-weighted sum over the sensor axis.
        # With one batch dimension, the j-axis is dim=2.
        local_cond = logits_ij.sumsoftmaxweight(v_j, dim=2)           # [B, N, Cc]
        return local_cond

    def _knn_search_keops(
        self,
        query_coords: torch.Tensor,         # [B, N, D]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
        k: int,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Top-k neighbor search using KeOps Kmin_argKmin.
        """

        # --- Force contiguous memory for KeOps ---
        query_coords = query_coords.contiguous()
        obs_coords = obs_coords.contiguous()
        # -----------------------------------------

        x_i = LazyTensor(query_coords[:, :, None, :])                 # [B, N, 1, D]
        y_j = LazyTensor(obs_coords[:, None, :, :])                   # [B, 1, M, D]

        sqdist_ij = ((x_i - y_j) ** 2).sum(-1)                        # [B, N, M, 1]

        # Mask invalid sensor slots
        mask_j = LazyTensor(obs_mask[:, None, :, None].to(query_coords.dtype).contiguous())
        sqdist_ij = sqdist_ij + (1.0 - mask_j) * 1e6

        # With one batch dimension, the j-axis is dim=2.
        topk_d2, topk_idx = sqdist_ij.Kmin_argKmin(K=k, dim=2)

        # KeOps can return indices in a non-long dtype; convert explicitly.
        topk_idx = topk_idx.long()

        topk_valid = batched_gather_2d(obs_mask, topk_idx).bool()
        if not return_features:
            return topk_d2, topk_idx, None, None, topk_valid

        topk_sensor_feat = batched_gather_3d(refined_sensor_feat, topk_idx)
        topk_sensor_coords = batched_gather_3d(obs_coords, topk_idx)
        return topk_d2, topk_idx, topk_sensor_feat, topk_sensor_coords, topk_valid

    def _knn_search_torch(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        k: int,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Fallback KNN search using torch.cdist + torch.topk.
        """
        d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        topk_d2, topk_idx = torch.topk(d2, k=k, dim=-1, largest=False)

        topk_valid = batched_gather_2d(obs_mask, topk_idx).bool()
        if not return_features:
            return topk_d2, topk_idx, None, None, topk_valid

        topk_sensor_feat = batched_gather_3d(refined_sensor_feat, topk_idx)
        topk_sensor_coords = batched_gather_3d(obs_coords, topk_idx)
        return topk_d2, topk_idx, topk_sensor_feat, topk_sensor_coords, topk_valid

    def _get_topk_neighbors(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        k: int,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Return distances, sensor slots, and validity through the chosen backend."""
        if self._use_keops():
            return self._knn_search_keops(
                query_coords=query_coords,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                k=k,
                return_features=return_features,
            )

        return self._knn_search_torch(
            query_coords=query_coords,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
            k=k,
            return_features=return_features,
        )

    def _sensor_local_refine(
        self,
        sensor_coords: torch.Tensor,      # [B, M, D]
        sensor_feat: torch.Tensor,        # [B, M, Cc]
        obs_mask: torch.Tensor,           # [B, M]
    ) -> torch.Tensor:
        """
        Point-Transformer-style local refinement on the sensor graph.

        - This operates on M sensors, not N query points, so its memory cost is much
          smaller than query-side gating.
        - It gives each refined sensor token awareness of its local sensor neighborhood
          before the final query-side top-k RBF gather.

        Implementation notes:
        - Uses the existing neighbor backend (torch / keops) through _get_topk_neighbors.
        - Uses K+1 neighbors and drops the first one, which is usually the sensor itself.
        """
        # Search one extra neighbor so we can discard self-neighbor.
        k_search = min(self.sensor_local_topk + 1, sensor_coords.shape[1])

        nbr_d2, _, nbr_feat, nbr_coords, nbr_valid = self._get_topk_neighbors(
            query_coords=sensor_coords,
            obs_coords=sensor_coords,
            refined_sensor_feat=sensor_feat,
            obs_mask=obs_mask,
            k=k_search,
        )

        # Drop the first neighbor slot, which is typically the point itself.
        if k_search > 1:
            nbr_d2 = nbr_d2[:, :, 1:]
            nbr_feat = nbr_feat[:, :, 1:]
            nbr_coords = nbr_coords[:, :, 1:]
            nbr_valid = nbr_valid[:, :, 1:]

        # If there was only one valid sensor total, keep the feature unchanged.
        if nbr_feat.shape[2] == 0:
            return sensor_feat

        q = self.sensor_local_q(sensor_feat).unsqueeze(2)   # [B, M, 1, Cc]
        k = self.sensor_local_k(nbr_feat)                   # [B, M, Ks, Cc]
        v = self.sensor_local_v(nbr_feat)                   # [B, M, Ks, Cc]

        rel = sensor_coords.unsqueeze(2) - nbr_coords       # [B, M, Ks, D]
        rel_dist = torch.sqrt(nbr_d2.clamp_min(0.0)).unsqueeze(-1)  # [B, M, Ks, 1]
        pos = self.sensor_local_pos(torch.cat([rel, rel_dist], dim=-1))  # [B, M, Ks, Cc]

        # Lightweight Point-Transformer-style attention:
        # attention is driven by query-key difference plus relative position.
        attn_logits = self.sensor_local_attn(torch.tanh(q - k + pos)).squeeze(-1)  # [B, M, Ks]
        attn_logits = attn_logits.masked_fill(~nbr_valid, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)

        update = torch.sum(attn.unsqueeze(-1) * (v + pos), dim=2)       # [B, M, Cc]
        out = self.sensor_local_norm(sensor_feat + self.sensor_local_dropout(self.sensor_local_out(update)))

        # Keep padded sensor rows zeroed out.
        out = out * obs_mask.unsqueeze(-1)
        return out

    def _aggregate_chunk(
        self,
        query_coords: torch.Tensor,         # [B, Nc, D]
        query_feat: torch.Tensor,           # [B, Nc, H]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
        sensor_importance_bias: Optional[torch.Tensor] = None,  # [B, M]
    ) -> torch.Tensor:
        """RBF-average local sensor features for one bounded query chunk."""
        sigma = torch.exp(self.log_rbf_sigma).clamp_min(1e-6) if self.learnable_rbf_sigma else self.rbf_sigma

        # --------------------------------------------------
        # Default: full RBF gather
        # --------------------------------------------------
        if self.gather_mode == "rbf":
            if self._use_keops():
                return self._aggregate_rbf_keops(
                    query_coords=query_coords,
                    obs_coords=obs_coords,
                    refined_sensor_feat=refined_sensor_feat,
                    obs_mask=obs_mask,
                )

            d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
            large = torch.full_like(d2, 1e6)
            d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

            logits = -d2 / (2 * sigma ** 2 + 1e-12)
            weights = torch.softmax(logits, dim=-1)
            return torch.einsum("bnm,bmd->bnd", weights, refined_sensor_feat)

        # --------------------------------------------------
        # top-k modes
        # --------------------------------------------------
        k = min(self.gather_topk, obs_coords.shape[1])

        topk_d2, topk_idx, topk_sensor_feat, topk_sensor_coords, topk_valid = self._get_topk_neighbors(
            query_coords=query_coords,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
            k=k,
        )

        logits = -topk_d2 / (2 * sigma ** 2 + 1e-12)

        if self.gather_mode == "topk_rbf_gate":
            query_cond = self.query_to_cond(query_feat)                    # [B, Nc, Cc]
            query_cond = query_cond.unsqueeze(2).expand(-1, -1, k, -1)    # [B, Nc, k, Cc]

            rel = query_coords.unsqueeze(2) - topk_sensor_coords           # [B, Nc, k, D]
            rel_dist = torch.sqrt(topk_d2.clamp_min(0.0)).unsqueeze(-1)    # [B, Nc, k, 1]

            gate_in = torch.cat([query_cond, topk_sensor_feat, rel, rel_dist], dim=-1)
            gate_logits = self.gather_gate(gate_in).squeeze(-1)            # [B, Nc, k]

            logits = logits + gate_logits

        if sensor_importance_bias is not None:
            topk_sensor_bias = batched_gather_2d(sensor_importance_bias, topk_idx)
            logits = logits + self.sensor_importance_scale * topk_sensor_bias

        logits = logits.masked_fill(~topk_valid, -1e9)
        weights = torch.softmax(logits, dim=-1)
        local_cond = torch.sum(weights.unsqueeze(-1) * topk_sensor_feat, dim=2)
        return local_cond

    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        query_feat: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        sensor_importance_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gather the globally enriched local sensor features back to query points.

        Policy:
          - rbf: with KeOps, chunking can usually be disabled
          - topk_rbf: with KeOps, chunking can usually be disabled
          - topk_rbf_gate: still keep optional chunking because gate tensors are [B, N, K, U]
        """
        n_query = query_coords.shape[1]

        if self.gather_mode == "topk_rbf_gate":
            # Gate mode still benefits from chunking because it builds [B, N, K, ...] tensors.
            chunk_size = self.gather_query_chunk_size if self.gather_query_chunk_size is not None else 2048
        else:
            # rbf / topk_rbf / topk_rbf_ptlocal all keep the cheaper gather path.
            chunk_size = self.gather_query_chunk_size

        if chunk_size is None or n_query <= chunk_size:
            return self._aggregate_chunk(
                query_coords=query_coords,
                query_feat=query_feat,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                sensor_importance_bias=sensor_importance_bias,
            )

        outputs = []
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)

            local_chunk = self._aggregate_chunk(
                query_coords=query_coords[:, start:end],
                query_feat=query_feat[:, start:end],
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                sensor_importance_bias=sensor_importance_bias,
            )
            outputs.append(local_chunk)

        return torch.cat(outputs, dim=1)

    def prepare_condition_context(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode the sensor set once so query chunks and ODE steps can reuse it."""
        sensor_tokens = self._build_sensor_tokens(
            obs_coords, obs_values, obs_mask, obs_field_ids,
        )
        latents = self._encode_latents(sensor_tokens, obs_mask)
        global_feat = self._extract_global_summary(latents)
        refined = self._refine_sensor_tokens(sensor_tokens, latents, obs_mask)
        refined = refined * obs_mask.unsqueeze(-1)
        refined = self.sensor_out_proj(refined) * obs_mask.unsqueeze(-1)
        if self.gather_mode == "topk_rbf_ptlocal":
            refined = self._sensor_local_refine(obs_coords, refined, obs_mask)
        context = {
            "obs_coords": obs_coords,
            "obs_mask": obs_mask,
            "latents": latents,
            "global_feat": global_feat,
            "refined_sensor_feat": refined,
        }
        if self.gather_mode == "topk_rbf_glres":
            context["sensor_importance_bias"] = self._compute_sensor_importance_bias(
                refined, obs_mask,
            )
        return context

    def _aggregate_topk_from_geometry(
        self,
        topk_d2: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_valid: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
        topk_sensor_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gather_mode not in ("topk_rbf", "topk_rbf_glres"):
            raise ValueError("Geometry caching supports topk_rbf and topk_rbf_glres.")
        sensor_feat = topk_sensor_feat
        if sensor_feat is None:
            sensor_feat = batched_gather_3d(
                condition_context["refined_sensor_feat"], topk_idx,
            )
        sigma = (
            torch.exp(self.log_rbf_sigma).clamp_min(1e-6)
            if self.learnable_rbf_sigma else self.rbf_sigma
        )
        logits = -topk_d2 / (2 * sigma ** 2 + 1e-12)
        importance = condition_context.get("sensor_importance_bias")
        if importance is not None:
            logits = logits + self.sensor_importance_scale * batched_gather_2d(
                importance, topk_idx,
            )
        weights = torch.softmax(logits.masked_fill(~topk_valid, -1e9), dim=-1)
        return torch.sum(weights.unsqueeze(-1) * sensor_feat, dim=2)

    def prepare_query_context(
        self,
        coords: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
        cache_level: str = "none",
        chunk_size: Optional[int] = None,
        precomputed_geometry: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Optionally cache neighbor geometry or inference-only query features."""
        if cache_level not in ("none", "geometry", "static_features"):
            raise ValueError("Unknown reconstruction cache level.")
        chunk_size = max(1, int(chunk_size or self.gather_query_chunk_size or 8192))
        context: Dict[str, Any] = {
            "cache_level": cache_level,
            "n_query": int(coords.shape[1]),
            "chunk_size": chunk_size,
        }
        if cache_level == "none":
            if precomputed_geometry is not None:
                raise ValueError("Persistent geometry requires geometry or static_features cache level.")
            return context
        if self.gather_mode not in ("topk_rbf", "topk_rbf_glres"):
            raise ValueError("Query caching supports topk_rbf and topk_rbf_glres.")
        persistent_topk = None
        if precomputed_geometry is not None:
            validate_persistent_topk_geometry_cache(
                precomputed_geometry, self, coords=coords,
                obs_coords=condition_context["obs_coords"],
                obs_mask=condition_context["obs_mask"],
            )
            persistent_topk = cache_tensors(precomputed_geometry)
        if self.pos_enc is None:
            context["coord_feat"] = coords
        else:
            coord_feat = coords.new_empty(coords.shape[0], coords.shape[1], self.coord_feat_dim)
            for start in range(0, coords.shape[1], chunk_size):
                end = min(start + chunk_size, coords.shape[1])
                coord_feat[:, start:end] = self.pos_enc(coords[:, start:end])
            context["coord_feat"] = coord_feat
        if cache_level == "geometry":
            if persistent_topk is not None:
                topk_d2, topk_idx, topk_valid = persistent_topk
                context.update(topk_d2=topk_d2, topk_idx=topk_idx, topk_valid=topk_valid)
                return context
            k = min(self.gather_topk, condition_context["obs_coords"].shape[1])
            topk_d2 = coords.new_empty(coords.shape[0], coords.shape[1], k)
            topk_idx = torch.empty(
                coords.shape[0], coords.shape[1], k,
                dtype=torch.long, device=coords.device,
            )
            topk_valid = torch.empty(
                coords.shape[0], coords.shape[1], k,
                dtype=torch.bool, device=coords.device,
            )
            for start in range(0, coords.shape[1], chunk_size):
                end = min(start + chunk_size, coords.shape[1])
                d2, idx, _, _, valid = self._get_topk_neighbors(
                    coords[:, start:end],
                    condition_context["obs_coords"],
                    condition_context["refined_sensor_feat"],
                    condition_context["obs_mask"],
                    k,
                )
                topk_d2[:, start:end] = d2
                topk_idx[:, start:end] = idx
                topk_valid[:, start:end] = valid
            context.update(
                topk_d2=topk_d2,
                topk_idx=topk_idx,
                topk_valid=topk_valid,
            )
            return context
        if self.training and torch.is_grad_enabled():
            raise ValueError("static_features caching is inference-only.")
        if self.use_query_latent_readout and self.query_readout_type != "coord":
            raise ValueError("static_features requires coordinate query readout.")
        local_cache = coords.new_empty(
            coords.shape[0], coords.shape[1],
            condition_context["refined_sensor_feat"].shape[-1],
        )
        query_global_cache = (
            coords.new_empty(coords.shape[0], coords.shape[1], condition_context["global_feat"].shape[-1])
            if self.use_query_latent_readout else None
        )
        for start in range(0, coords.shape[1], chunk_size):
            end = min(start + chunk_size, coords.shape[1])
            coords_c = coords[:, start:end]
            empty_feat = coords_c.new_empty(coords_c.shape[0], coords_c.shape[1], 0)
            if persistent_topk is None:
                local_cache[:, start:end] = self._aggregate_chunk(
                    coords_c,
                    empty_feat,
                    condition_context["obs_coords"],
                    condition_context["refined_sensor_feat"],
                    condition_context["obs_mask"],
                    condition_context.get("sensor_importance_bias"),
                )
            else:
                topk_d2, topk_idx, topk_valid = persistent_topk
                local_cache[:, start:end] = self._aggregate_topk_from_geometry(
                    topk_d2[:, start:end],
                    topk_idx[:, start:end],
                    topk_valid[:, start:end],
                    condition_context,
                )
            if self.use_query_latent_readout:
                q = self._build_query_readout_tokens(empty_feat, coords_c)
                readout = self.query_latent_readout(
                    q=q, kv=condition_context["latents"], kv_padding_mask=None,
                )
                query_global_cache[:, start:end] = self.query_readout_out(readout)
        context["local_cond"] = local_cache
        if query_global_cache is not None:
            context["query_global"] = query_global_cache
        return context

    def forward_query_chunk(
        self,
        t: torch.Tensor,
        x_t_chunk: torch.Tensor,
        coords_chunk: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
        query_context: Optional[Mapping[str, Any]] = None,
        query_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        """Predict one query chunk from reusable condition and query context."""
        bsz, n_pts, _ = x_t_chunk.shape
        query_slice = query_slice or slice(0, n_pts)
        cached_coord = None if query_context is None else query_context.get("coord_feat")
        coord_feat = (
            cached_coord[:, query_slice]
            if cached_coord is not None
            else (self.pos_enc(coords_chunk) if self.pos_enc is not None else coords_chunk)
        )
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        point_feat = self.point_encoder(torch.cat([coord_feat, x_t_chunk, t_feat], dim=-1))
        global_feat = condition_context["global_feat"]
        if self.use_query_latent_readout:
            cached_global = None if query_context is None else query_context.get("query_global")
            if cached_global is None:
                q = self._build_query_readout_tokens(point_feat, coords_chunk)
                readout = self.query_latent_readout(
                    q=q, kv=condition_context["latents"], kv_padding_mask=None,
                )
                query_global = self.query_readout_out(readout)
            else:
                query_global = cached_global[:, query_slice]
            global_for_head = global_feat.unsqueeze(1) + self.query_readout_scale * query_global
        else:
            global_for_head = global_feat.unsqueeze(1).expand(bsz, n_pts, -1)
        cached_local = None if query_context is None else query_context.get("local_cond")
        if cached_local is not None:
            local_cond = cached_local[:, query_slice]
        elif query_context is not None and "topk_idx" in query_context:
            local_cond = self._aggregate_topk_from_geometry(
                query_context["topk_d2"][:, query_slice],
                query_context["topk_idx"][:, query_slice],
                query_context["topk_valid"][:, query_slice],
                condition_context,
            )
        else:
            local_cond = self.aggregate_sparse_obs(
                coords_chunk,
                point_feat,
                condition_context["obs_coords"],
                condition_context["refined_sensor_feat"],
                condition_context["obs_mask"],
                condition_context.get("sensor_importance_bias"),
            )
        head_in = torch.cat([point_feat, global_for_head, local_cond], dim=-1)
        residual = self.head(self.head_in_norm(head_in))
        if self.gather_mode == "topk_rbf_glres":
            return self.coarse_scale * self._predict_global_coarse(point_feat, global_feat) + residual
        return residual

    @staticmethod
    def context_nbytes(context: Mapping[str, Any]) -> int:
        seen: set[tuple[int, int]] = set()
        total = 0

        def visit(value: Any) -> None:
            nonlocal total
            if torch.is_tensor(value):
                storage = value.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
            elif isinstance(value, Mapping):
                for child in value.values():
                    visit(child)

        visit(context)
        return total

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a ``[batch, queries, fields]`` velocity from marked sensors."""
        bsz, n_pts, _ = x_t.shape

        # -------------------------
        # Query-point features
        # -------------------------
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        coord_feat = self.pos_enc(coords) if self.pos_enc is not None else coords
        point_feat = self.point_encoder(torch.cat([coord_feat, x_t, t_feat], dim=-1))  # [B, N, H]

        # -------------------------
        # Local sensor tokens
        # -------------------------
        sensor_tokens = self._build_sensor_tokens(
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )  # [B, M, D]

        # -------------------------
        # Global latent processing
        # -------------------------
        latents = self._encode_latents(sensor_tokens=sensor_tokens, obs_mask=obs_mask)  # [B, L, D]

        # Broadcast the global summary to each query. Enhanced mode also reads
        # the latent memory separately for each query before local/global fusion.
        global_feat = self._extract_global_summary(latents)                 # [B, H]
        if self.use_query_latent_readout:
            query_global = self._readout_query_global_chunked(point_feat, coords, latents)  # [B, N, H]
            global_for_head = global_feat.unsqueeze(1) + self.query_readout_scale * query_global
        else:
            global_for_head = global_feat.unsqueeze(1).expand(bsz, n_pts, -1)

        # -------------------------
        # Sensor tokens read back from the global latent memory.
        # -------------------------
        refined_sensor_tokens = self._refine_sensor_tokens(
            sensor_tokens, latents, obs_mask,
        )  # [B, M, D]

        # Zero out padded sensor rows again after attention
        refined_sensor_tokens = refined_sensor_tokens * obs_mask.unsqueeze(-1)

        # Project refined sensor tokens to the local conditioning width
        refined_sensor_feat = self.sensor_out_proj(refined_sensor_tokens)   # [B, M, cond_dim]
        refined_sensor_feat = refined_sensor_feat * obs_mask.unsqueeze(-1)

        if self.gather_mode == "topk_rbf_glres":
            sensor_importance_bias = self._compute_sensor_importance_bias(
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
            )

            local_cond = self.aggregate_sparse_obs(
                query_coords=coords,
                query_feat=point_feat,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                sensor_importance_bias=sensor_importance_bias,
            )  # [B, N, cond_dim]

            coarse_pred = self.coarse_scale * self._predict_global_coarse(point_feat, global_feat)

            head_in = torch.cat([point_feat, global_for_head, local_cond], dim=-1)
            residual = self.head(self.head_in_norm(head_in))
            return coarse_pred + residual

        # Optional sensor-side local graph refinement.
        if self.gather_mode == "topk_rbf_ptlocal":
            refined_sensor_feat = self._sensor_local_refine(
                sensor_coords=obs_coords,
                sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,)

        # -------------------------
        # Gather back to queries
        # -------------------------
        local_cond = self.aggregate_sparse_obs(
            query_coords=coords,
            query_feat=point_feat,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
        )  # [B, N, cond_dim]

        # -------------------------
        # Final velocity prediction
        # -------------------------
        head_in = torch.cat([point_feat, global_for_head, local_cond], dim=-1)
        out = self.head(self.head_in_norm(head_in))
        return out
