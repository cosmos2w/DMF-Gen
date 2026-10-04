"""Optional compact-query head over the enhanced GL-RBF condition encoder.

Example::

    flow = build_pointcloud_model({"model_name": "GL_rbf_ENH_CQ", "coord_dim": 3}, n_fields=5)
    velocity = flow.model(t, x_t, query_coords, sensor_coords, sensor_values, sensor_mask, field_ids)

This variant keeps the enhanced sensor and latent pathway while changing query readout and fusion. It is an optional architecture example, not a substitute for the manuscript combustion checkpoints.
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
from .gl_rbf import ConditionalPointHybridLocalGlobalRBF
from .layers import (
    CompactLatentReadout,
    CrossAttentionBlock,
    batched_gather_2d,
    batched_gather_3d,
    make_mlp,
)


class ConditionalPointHybridLocalGlobalRBFCQ(ConditionalPointHybridLocalGlobalRBF):
    """Use compact query tokens, latent readout, and configurable additive fusion.

    ``additive`` projects global and local evidence to the query width before
    summing; ``structured_concat`` keeps those components separate at the head.
    Both modes share the enhanced sensor encoder inherited from GL-RBF.
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
        summary_type: str = "cls",
        gather_mode: str = "topk_rbf_glres",
        gather_topk: int = 32,
        gather_query_chunk_size: Optional[int] = None,
        learnable_rbf_sigma: bool = False,
        neighbor_backend: str = "torch",
        sensor_local_topk: int = 8,
        sensor_local_dropout: float = 0.0,
        use_fourier_pe: bool = False,
        fourier_pe_num_bands: int = 32,
        fourier_pe_max_freq: float = 64.0,
        sensor_coord_encoding: str = "fourier",
        latent_sensor_reinject: bool = True,
        latent_reinject_every: int = 1,
        condition_attention_execution: str = "legacy_mha",
        sensor_attention_padding_mode: str = "full",
        sensor_attention_buckets: Sequence[int] = (256, 320, 384),
        glres_scale_init: float = 1.0e-2,
        cq_query_dim: int = 128,
        cq_readout_mode: str = "lowrank",
        cq_readout_rank: int = 64,
        cq_readout_heads: int = 4,
        cq_global_scale_init: float = 1.0,
        cq_local_scale_init: float = 1.0,
        cq_readout_scale_init: float = 1.0e-2,
        cq_fusion_mode: str = "additive",
        cq_time_conditioning: str = "scalar_concat",
        cq_time_embed_dim: int = 128,
        cq_time_max_period: float = 10000.0,
        cq_time_film_zero_init: bool = True,
        cq_measurement_support_mode: str = "none",
        cq_measurement_support_normalize: bool = True,
    ) -> None:
        if cq_readout_mode not in ("full", "lowrank"):
            raise ValueError(
                f"cq_readout_mode must be one of ['full', 'lowrank'], got {cq_readout_mode!r}."
            )
        if cq_fusion_mode not in ("additive", "structured_concat"):
            raise ValueError(
                "cq_fusion_mode must be one of ['additive', 'structured_concat'], "
                f"got {cq_fusion_mode!r}."
            )
        if cq_readout_heads < 1:
            raise ValueError(f"cq_readout_heads must be positive, got {cq_readout_heads}.")
        if cq_query_dim < 1 or cq_query_dim % cq_readout_heads != 0:
            raise ValueError(
                "cq_query_dim must be positive and divisible by cq_readout_heads; "
                f"got query_dim={cq_query_dim}, heads={cq_readout_heads}."
            )
        if cq_readout_rank < 1 or cq_readout_rank % cq_readout_heads != 0:
            raise ValueError(
                "cq_readout_rank must be positive and divisible by cq_readout_heads; "
                f"got rank={cq_readout_rank}, heads={cq_readout_heads}."
            )
        if cq_time_conditioning not in ("scalar_concat", "sinusoidal_film"):
            raise ValueError(
                "cq_time_conditioning must be 'scalar_concat' or 'sinusoidal_film'."
            )
        if cq_time_embed_dim < 2:
            raise ValueError("cq_time_embed_dim must be at least 2.")
        if cq_time_max_period <= 0:
            raise ValueError("cq_time_max_period must be positive.")
        if cq_measurement_support_mode not in ("none", "rbf_value_support"):
            raise ValueError(
                "cq_measurement_support_mode must be 'none' or 'rbf_value_support'."
            )
        if cq_measurement_support_mode == "rbf_value_support" and gather_mode not in (
            "topk_rbf", "topk_rbf_glres"
        ):
            raise ValueError("CQ measurement support requires topk_rbf or topk_rbf_glres gather.")

        # Build the enhanced sensor/latent/local core first, then replace only
        # its query modules. This preserves the shared checkpoint layout.
        super().__init__(
            n_fields=n_fields,
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            field_embed_dim=field_embed_dim,
            latent_dim=latent_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            num_latent_blocks=num_latent_blocks,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            rbf_sigma=rbf_sigma,
            summary_type=summary_type,
            gather_mode=gather_mode,
            gather_topk=gather_topk,
            gather_query_chunk_size=gather_query_chunk_size,
            learnable_rbf_sigma=learnable_rbf_sigma,
            neighbor_backend=neighbor_backend,
            sensor_local_topk=sensor_local_topk,
            sensor_local_dropout=sensor_local_dropout,
            use_fourier_pe=use_fourier_pe,
            fourier_pe_num_bands=fourier_pe_num_bands,
            fourier_pe_max_freq=fourier_pe_max_freq,
            enhanced_backbone=True,
            sensor_coord_encoding=sensor_coord_encoding,
            latent_sensor_reinject=latent_sensor_reinject,
            latent_reinject_every=latent_reinject_every,
            condition_attention_execution=condition_attention_execution,
            sensor_attention_padding_mode=sensor_attention_padding_mode,
            sensor_attention_buckets=sensor_attention_buckets,
            query_latent_readout=True,
            query_readout_type="coord",
            query_readout_scale_init=cq_readout_scale_init,
            enhanced_head_norm=True,
            glres_scale_init=glres_scale_init,
        )

        self.cq_query_dim = int(cq_query_dim)
        self.cq_readout_mode = str(cq_readout_mode)
        self.cq_fusion_mode = str(cq_fusion_mode)
        self.cq_readout_rank = int(cq_readout_rank)
        self.cq_readout_heads = int(max(1, min(num_heads, 4)) if cq_readout_mode == "full" else cq_readout_heads)
        self.cq_time_conditioning = str(cq_time_conditioning)
        self.cq_time_embed_dim = int(cq_time_embed_dim)
        self.cq_time_max_period = float(cq_time_max_period)
        self.cq_time_film_zero_init = bool(cq_time_film_zero_init)
        self.cq_measurement_support_mode = str(cq_measurement_support_mode)
        self.cq_measurement_support_normalize = bool(cq_measurement_support_normalize)
        self.cq_timestep_film_enabled = self.cq_time_conditioning == "sinusoidal_film"
        self.cq_measurement_support_enabled = self.cq_measurement_support_mode == "rbf_value_support"
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)

        for name in (
            "point_encoder", "query_decoder_token", "query_readout_in",
            "query_latent_readout", "query_readout_out", "query_readout_scale",
            "head", "head_in_norm", "coarse_film", "coarse_head", "coarse_scale",
        ):
            if hasattr(self, name):
                delattr(self, name)

        if self.gather_mode == "topk_rbf_gate":
            self.query_to_cond = nn.Linear(self.cq_query_dim, cond_dim, bias=False)

        self.cq_point_encoder = make_mlp(
            in_dim=self.coord_feat_dim + n_fields + 1,
            hidden_dim=self.cq_query_dim,
            out_dim=self.cq_query_dim,
            depth=3,
        )
        self.cq_global_proj = nn.Linear(hidden_dim, self.cq_query_dim)
        if self.cq_timestep_film_enabled:
            self.cq_timestep_mlp = nn.Sequential(
                nn.Linear(self.cq_time_embed_dim, self.cq_time_embed_dim),
                nn.SiLU(),
                nn.Linear(self.cq_time_embed_dim, self.cq_time_embed_dim),
                nn.SiLU(),
            )
            self.cq_timestep_film = nn.Linear(
                self.cq_time_embed_dim, 2 * self.cq_query_dim,
            )
            if self.cq_time_film_zero_init:
                nn.init.zeros_(self.cq_timestep_film.weight)
                nn.init.zeros_(self.cq_timestep_film.bias)
        raw_feature_dim = 2 * self.n_fields if self.cq_measurement_support_enabled else 0
        if self.cq_measurement_support_enabled and self.cq_measurement_support_normalize:
            self.cq_measurement_support_norm = nn.LayerNorm(raw_feature_dim)
        if self.cq_fusion_mode == "additive":
            # Keep this module set, ordering, and all shapes unchanged so CQ
            # checkpoints created before cq_fusion_mode continue to strict-load.
            self.cq_local_proj = (
                nn.Identity()
                if cond_dim == self.cq_query_dim
                else nn.Linear(cond_dim, self.cq_query_dim)
            )
            self.cq_global_scale = nn.Parameter(torch.tensor(float(cq_global_scale_init)))
            self.cq_local_scale = nn.Parameter(torch.tensor(float(cq_local_scale_init)))
            self.cq_readout_scale = nn.Parameter(torch.tensor(float(cq_readout_scale_init)))
            fusion_dim = self.cq_query_dim + raw_feature_dim
            self.cq_fusion_norm = nn.LayerNorm(fusion_dim)
            self.cq_head = nn.Sequential(
                nn.Linear(fusion_dim, self.cq_query_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(self.cq_query_dim, self.cq_query_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(self.cq_query_dim, n_fields),
            )
        else:
            fusion_dim = 2 * self.cq_query_dim + self.cond_dim + raw_feature_dim
            self.cq_readout_scale = nn.Parameter(torch.tensor(float(cq_readout_scale_init)))
            self.cq_fusion_norm = nn.LayerNorm(fusion_dim)
            self.cq_head = nn.Sequential(
                nn.Linear(fusion_dim, self.cq_query_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(self.cq_query_dim, self.cq_query_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(self.cq_query_dim, n_fields),
            )
        if self.gather_mode == "topk_rbf_glres":
            self.cq_coarse_film = nn.Linear(self.cq_query_dim, 2 * self.cq_query_dim)
            self.cq_coarse_head = nn.Sequential(
                nn.LayerNorm(self.cq_query_dim),
                nn.Linear(self.cq_query_dim, self.cq_query_dim),
                nn.GELU(),
                nn.Linear(self.cq_query_dim, n_fields),
            )
            self.cq_coarse_scale = nn.Parameter(torch.tensor(float(glres_scale_init)))

        # Construct the only variant-specific module last so CQ-Full and CQ-LR
        # receive identical seed-controlled initialization for every shared CQ module.
        if self.cq_readout_mode == "full":
            self.cq_query_decoder_token = nn.Parameter(
                torch.randn(1, self.cq_query_dim) * 0.02
            )
            self.cq_readout_in = nn.Linear(
                self.coord_feat_dim + self.cq_query_dim, latent_dim, bias=False,
            )
            self.cq_latent_readout = CrossAttentionBlock(
                dim=latent_dim,
                num_heads=max(1, min(num_heads, 4)),
                ff_mult=max(1, ff_mult // 2),
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            self.cq_readout_out = nn.Linear(latent_dim, self.cq_query_dim, bias=False)
        else:
            self.cq_latent_readout = CompactLatentReadout(
                query_in_dim=self.coord_feat_dim,
                latent_dim=latent_dim,
                query_dim=self.cq_query_dim,
                rank=self.cq_readout_rank,
                num_heads=self.cq_readout_heads,
                attn_dropout=attn_dropout,
            )

    def _cq_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Return a deterministic sinusoidal embedding without cacheable time state."""
        t = t.reshape(-1).to(dtype=self.cq_global_proj.weight.dtype)
        half_dim = self.cq_time_embed_dim // 2
        exponent = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        frequencies = torch.exp(
            -math.log(self.cq_time_max_period) * exponent / max(half_dim - 1, 1)
        )
        angles = t.unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.cq_time_embed_dim:
            embedding = F.pad(embedding, (0, self.cq_time_embed_dim - embedding.shape[-1]))
        return embedding

    def _cq_apply_timestep_film(
        self,
        point_q: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cq_timestep_film_enabled:
            return point_q
        embedding = self.cq_timestep_mlp(self._cq_timestep_embedding(t))
        scale, shift = self.cq_timestep_film(embedding).chunk(2, dim=-1)
        # Standard residual FiLM. The zero-initialized projection makes this an
        # exact identity at initialization without normalizing every query token.
        return point_q * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _cq_measurement_support_from_geometry(
        self,
        topk_d2: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_valid: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build per-field raw measurement and soft support from Top-K geometry."""
        if not self.cq_measurement_support_enabled:
            raise RuntimeError("CQ measurement/support shortcut is disabled.")
        values = batched_gather_3d(condition_context["raw_obs_values"], topk_idx).squeeze(-1)
        field_ids = batched_gather_2d(condition_context["raw_obs_field_ids"], topk_idx)
        sigma = (
            torch.exp(self.log_rbf_sigma).clamp_min(1e-6)
            if self.learnable_rbf_sigma else self.rbf_sigma
        )
        logits = -topk_d2 / (2 * sigma ** 2 + 1e-12)
        weights = torch.softmax(logits.masked_fill(~topk_valid, -1e9), dim=-1)
        weights = weights * topk_valid.to(dtype=weights.dtype)
        # Padded sensors carry field ID -1. Their weights are zero, but scatter_add
        # still validates every index, so redirect those inactive slots to field 0.
        field_ids = field_ids.clamp_min(0)
        # Accumulate directly into the configured field slots. This is equivalent to
        # the one-hot [B,Q,K,F] formulation but avoids materializing that large
        # tensor and remains differentiable with respect to the RBF weights.
        output_shape = (*weights.shape[:2], self.n_fields)
        support = weights.new_zeros(output_shape).scatter_add(2, field_ids, weights)
        numerator = weights.new_zeros(output_shape).scatter_add(
            2, field_ids, weights * values,
        )
        measurement = numerator / support.clamp_min(1e-6)
        measurement = torch.where(support > 0, measurement, torch.zeros_like(measurement))
        return torch.cat([measurement, support], dim=-1)

    def _cq_uncached_local_and_raw(
        self,
        query_coords: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run exactly one Top-K search and reuse it for learned and explicit features."""
        k = min(self.gather_topk, condition_context["obs_coords"].shape[1])
        topk_d2, topk_idx, _, _, topk_valid = self._get_topk_neighbors(
            query_coords,
            condition_context["obs_coords"],
            condition_context["refined_sensor_feat"],
            condition_context["obs_mask"],
            k,
            return_features=False,
        )
        raw_features = self._cq_measurement_support_from_geometry(
            topk_d2, topk_idx, topk_valid, condition_context,
        )
        topk_sensor_feat = batched_gather_3d(
            condition_context["refined_sensor_feat"], topk_idx,
        )
        local_cond = self._aggregate_topk_from_geometry(
            topk_d2, topk_idx, topk_valid, condition_context, topk_sensor_feat,
        )
        return local_cond, raw_features

    def _cq_readout(
        self,
        coord_feat: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Read the same latent memory with full or low-rank query attention."""
        if self.cq_readout_mode == "full":
            bsz, n_query, _ = coord_feat.shape
            token = self.cq_query_decoder_token.view(1, 1, -1).expand(
                bsz, n_query, -1,
            )
            query = self.cq_readout_in(torch.cat([coord_feat, token], dim=-1))
            readout = self.cq_latent_readout(
                q=query, kv=condition_context["latents"], kv_padding_mask=None,
            )
            return self.cq_readout_out(readout)
        return self.cq_latent_readout(
            coord_feat,
            projected_kv=(
                condition_context["cq_latent_k"],
                condition_context["cq_latent_v"],
            ),
        )

    def _cq_readout_chunked(
        self,
        coord_feat: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        n_query = int(coord_feat.shape[1])
        chunk_size = self.gather_query_chunk_size
        if chunk_size is None and n_query > 4096:
            chunk_size = 4096
        if chunk_size is None or n_query <= chunk_size:
            return self._cq_readout(coord_feat, condition_context)
        return torch.cat([
            self._cq_readout(coord_feat[:, start:start + chunk_size], condition_context)
            for start in range(0, n_query, chunk_size)
        ], dim=1)

    def _predict_cq_coarse(
        self,
        point_q: torch.Tensor,
        global_q: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.cq_coarse_film(global_q).chunk(2, dim=-1)
        coarse_feat = (
            point_q * (1.0 + torch.tanh(gamma).unsqueeze(1))
            + beta.unsqueeze(1)
        )
        return self.cq_coarse_head(coarse_feat)

    def prepare_condition_context(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extend the shared sensor context with CQ global and latent projections."""
        context = super().prepare_condition_context(
            obs_coords, obs_values, obs_mask, obs_field_ids,
        )
        context["global_q"] = self.cq_global_proj(context["global_feat"])
        if self.cq_measurement_support_enabled:
            context["raw_obs_values"] = obs_values
            context["raw_obs_field_ids"] = obs_field_ids
        if self.cq_readout_mode == "lowrank":
            keys, values = self.cq_latent_readout.project_latents(context["latents"])
            context["cq_latent_k"] = keys
            context["cq_latent_v"] = values
        return context

    def prepare_query_context(
        self,
        coords: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
        cache_level: str = "none",
        chunk_size: Optional[int] = None,
        precomputed_geometry: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Cache reusable query geometry and optional raw sensor support."""
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

        coord_feat = coords.new_empty(
            coords.shape[0], coords.shape[1], self.coord_feat_dim,
        )
        for start in range(0, coords.shape[1], chunk_size):
            end = min(start + chunk_size, coords.shape[1])
            coord_feat[:, start:end] = (
                self.pos_enc(coords[:, start:end])
                if self.pos_enc is not None else coords[:, start:end]
            )
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
            context.update(topk_d2=topk_d2, topk_idx=topk_idx, topk_valid=topk_valid)
            return context

        if self.training and torch.is_grad_enabled():
            raise ValueError("static_features caching is inference-only.")
        local_cache = coords.new_empty(
            coords.shape[0], coords.shape[1],
            condition_context["refined_sensor_feat"].shape[-1],
        )
        readout_cache = coords.new_empty(
            coords.shape[0], coords.shape[1], self.cq_query_dim,
        )
        raw_cache = (
            coords.new_empty(coords.shape[0], coords.shape[1], 2 * self.n_fields)
            if self.cq_measurement_support_enabled else None
        )
        for start in range(0, coords.shape[1], chunk_size):
            end = min(start + chunk_size, coords.shape[1])
            coords_c = coords[:, start:end]
            empty_feat = coords_c.new_empty(coords_c.shape[0], coords_c.shape[1], 0)
            if persistent_topk is None:
                if self.cq_measurement_support_enabled:
                    local_chunk, raw_chunk = self._cq_uncached_local_and_raw(
                        coords_c, condition_context,
                    )
                    local_cache[:, start:end] = local_chunk
                    raw_cache[:, start:end] = raw_chunk
                else:
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
                if raw_cache is not None:
                    raw_cache[:, start:end] = self._cq_measurement_support_from_geometry(
                        topk_d2[:, start:end],
                        topk_idx[:, start:end],
                        topk_valid[:, start:end],
                        condition_context,
                    )
            readout_cache[:, start:end] = self._cq_readout(
                coord_feat[:, start:end], condition_context,
            )
        context["local_cond"] = local_cache
        context["query_global"] = readout_cache
        if raw_cache is not None:
            context["raw_measurement_support"] = raw_cache
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
        """Fuse dynamic query state with cached global, local, and CQ evidence."""
        bsz, n_pts, _ = x_t_chunk.shape
        query_slice = query_slice or slice(0, n_pts)
        cached_coord = None if query_context is None else query_context.get("coord_feat")
        coord_feat = (
            cached_coord[:, query_slice]
            if cached_coord is not None
            else (self.pos_enc(coords_chunk) if self.pos_enc is not None else coords_chunk)
        )
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        point_q = self.cq_point_encoder(
            torch.cat([coord_feat, x_t_chunk, t_feat], dim=-1),
        )
        point_q = self._cq_apply_timestep_film(point_q, t)

        cached_readout = None if query_context is None else query_context.get("query_global")
        query_global_q = (
            cached_readout[:, query_slice]
            if cached_readout is not None
            else self._cq_readout_chunked(coord_feat, condition_context)
        )
        cached_local = None if query_context is None else query_context.get("local_cond")
        raw_features = None
        cached_raw = None if query_context is None else query_context.get("raw_measurement_support")
        if cached_local is not None:
            local_cond = cached_local[:, query_slice]
            if cached_raw is not None:
                raw_features = cached_raw[:, query_slice]
        elif query_context is not None and "topk_idx" in query_context:
            local_cond = self._aggregate_topk_from_geometry(
                query_context["topk_d2"][:, query_slice],
                query_context["topk_idx"][:, query_slice],
                query_context["topk_valid"][:, query_slice],
                condition_context,
            )
            if self.cq_measurement_support_enabled:
                raw_features = self._cq_measurement_support_from_geometry(
                    query_context["topk_d2"][:, query_slice],
                    query_context["topk_idx"][:, query_slice],
                    query_context["topk_valid"][:, query_slice],
                    condition_context,
                )
        elif self.cq_measurement_support_enabled:
            local_cond, raw_features = self._cq_uncached_local_and_raw(
                coords_chunk, condition_context,
            )
        else:
            local_cond = self.aggregate_sparse_obs(
                coords_chunk,
                point_q,
                condition_context["obs_coords"],
                condition_context["refined_sensor_feat"],
                condition_context["obs_mask"],
                condition_context.get("sensor_importance_bias"),
            )
        global_q = condition_context["global_q"]
        if self.cq_fusion_mode == "structured_concat":
            global_for_head = (
                global_q.unsqueeze(1)
                + self.cq_readout_scale * query_global_q
            )
            head_input = torch.cat([point_q, global_for_head, local_cond], dim=-1)
        else:
            head_input = (
                point_q
                + self.cq_global_scale * global_q.unsqueeze(1)
                + self.cq_local_scale * self.cq_local_proj(local_cond)
                + self.cq_readout_scale * query_global_q
            )
        if self.cq_measurement_support_enabled:
            if raw_features is None:
                raise RuntimeError("CQ measurement/support features were not constructed.")
            if self.cq_measurement_support_normalize:
                raw_features = self.cq_measurement_support_norm(raw_features)
            head_input = torch.cat([head_input, raw_features], dim=-1)
        residual = self.cq_head(self.cq_fusion_norm(head_input))
        if self.gather_mode == "topk_rbf_glres":
            return self.cq_coarse_scale * self._predict_cq_coarse(point_q, global_q) + residual
        return residual

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
        """Predict a full-field velocity after one sensor-context encoding."""
        condition_context = self.prepare_condition_context(
            obs_coords, obs_values, obs_mask, obs_field_ids,
        )
        return self.forward_query_chunk(t, x_t, coords, condition_context)

    def model_summary(self) -> Dict[str, Any]:
        query_prefixes = ("cq_", "query_to_cond", "gather_gate")
        query_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(query_prefixes)
        )
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        point_input_dim = self.coord_feat_dim + self.n_fields + 1
        raw_feature_dim = 2 * self.n_fields if self.cq_measurement_support_enabled else 0
        fusion_dim = (
            self.cq_query_dim + raw_feature_dim
            if self.cq_fusion_mode == "additive"
            else 2 * self.cq_query_dim + self.cond_dim + raw_feature_dim
        )
        if self.cq_fusion_mode == "additive":
            local_projection_macs = (
                0 if self.cond_dim == self.cq_query_dim
                else self.cond_dim * self.cq_query_dim
            )
            head_and_coarse_macs = (
                3 * self.cq_query_dim ** 2
                + 2 * self.cq_query_dim * self.n_fields
                + raw_feature_dim * self.cq_query_dim
            )
        else:
            local_projection_macs = 0
            head_and_coarse_macs = (
                fusion_dim * self.cq_query_dim
                + 2 * self.cq_query_dim ** 2
                + 2 * self.cq_query_dim * self.n_fields
            )
        common_linear_macs = (
            point_input_dim * self.cq_query_dim
            + 2 * self.cq_query_dim ** 2
            + local_projection_macs
            + head_and_coarse_macs
        )
        if self.cq_readout_mode == "full":
            readout_linear_macs = (
                (self.coord_feat_dim + self.cq_query_dim) * self.latent_dim
                + 2 * self.latent_dim ** 2
                + 2 * self.latent_dim * (
                    self.cq_latent_readout.ff.net[0].out_features
                )
                + self.latent_dim * self.cq_query_dim
            )
            attention_macs = 2 * self.num_latents * self.latent_dim
        else:
            readout_linear_macs = self.coord_feat_dim * self.cq_readout_rank
            attention_macs = self.num_latents * (
                self.cq_readout_rank + self.cq_query_dim
            )
        return {
            "backbone": "GL_rbf_ENH_CQ",
            "total_parameters": total_parameters,
            "condition_core_parameters": total_parameters - query_parameters,
            "query_decoder_parameters": query_parameters,
            "query_dim": self.cq_query_dim,
            "latent_dim": self.latent_dim,
            "cond_dim": self.cond_dim,
            "readout_mode": self.cq_readout_mode,
            "fusion_mode": self.cq_fusion_mode,
            "time_conditioning": self.cq_time_conditioning,
            "time_embed_dim": (
                self.cq_time_embed_dim if self.cq_timestep_film_enabled else None
            ),
            "measurement_support_mode": self.cq_measurement_support_mode,
            "measurement_support_normalize": self.cq_measurement_support_normalize,
            "measurement_support_width": raw_feature_dim,
            "condition_attention_execution": self.condition_attention_execution,
            "sensor_attention_padding_mode": self.sensor_attention_padding_mode,
            "sensor_attention_buckets": list(self.sensor_attention_buckets),
            "readout_rank": (self.cq_readout_rank if self.cq_readout_mode == "lowrank" else None),
            "readout_heads": self.cq_readout_heads,
            "point_state_width": self.cq_query_dim,
            "global_width": self.cq_query_dim,
            "local_width": (
                self.cq_query_dim
                if self.cq_fusion_mode == "additive" else self.cond_dim
            ),
            "legacy_concat_width": 2 * self.hidden_dim + self.cond_dim,
            "cq_fused_width": fusion_dim,
            "theoretical_query_linear_macs_per_query_excluding_attention": (
                common_linear_macs + readout_linear_macs
            ),
            "theoretical_query_attention_macs_per_query": attention_macs,
            "mac_estimate_note": (
                "Linear/attention multiply-accumulates only; excludes activations, "
                "normalization, RBF gather, and condition-static projections."
            ),
        }
