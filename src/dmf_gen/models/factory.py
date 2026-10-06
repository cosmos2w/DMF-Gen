"""Build a GL-RBF velocity backbone together with its rectified-flow source.

Example::

    model = build_pointcloud_model({"model_name": "GL_rbf_ENH", "coord_dim": 3}, n_fields=5)
    loss, metrics = model.training_loss(x1, coords, obs_coords, obs_values, obs_mask, field_ids)

The YAML profiles set the remaining architecture dimensions explicitly; passing only a name uses historical construction defaults, not a paper training recipe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..priors import IIDGaussianPrior, RFFGaussianPrior
from .gl_rbf import ConditionalPointHybridLocalGlobalRBF
from .gl_rbf_cq import ConditionalPointHybridLocalGlobalRBFCQ
from .pointcloud_ffm import PointCloudFFM

_PUBLIC_TO_INTERNAL = {
    "GL_rbf": "GL_rbf",
    "GL_rbf_ENH": "GL_rbf_ENH",
    "GL_rbf_ENH_CQ": "GL_rbf_ENH_CQ",
    # Historical names used by the production configs.
    "GL_rbf_CQ": "GL_rbf_ENH_CQ",
    "GL_rbf_CQ-fast": "GL_rbf_ENH_CQ",
}
_INTERNAL_NAMES = frozenset(_PUBLIC_TO_INTERNAL.values())


def _get(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _model_identity(config: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve public and legacy backbone names without a config dependency."""
    public = config.get("model_name")
    internal = config.get("backbone")
    if public is None and internal is None:
        # Preserve the production factory default. Release configs should name
        # their chosen backbone explicitly.
        return "GL_rbf_ENH", "GL_rbf_ENH"
    if public is None:
        internal = str(internal)
        if internal not in _INTERNAL_NAMES:
            raise ValueError(f"Unknown backbone {internal!r}.")
        public = "GL_rbf_CQ" if internal == "GL_rbf_ENH_CQ" else internal
        return str(public), internal

    public = str(public)
    if public not in _PUBLIC_TO_INTERNAL:
        raise ValueError(
            f"Unknown model name {public!r}; expected one of "
            f"{sorted(_PUBLIC_TO_INTERNAL)}."
        )
    expected = _PUBLIC_TO_INTERNAL[public]
    if internal is not None and str(internal) != expected:
        raise ValueError(
            f"Conflicting model identifiers: model_name={public!r} maps to "
            f"backbone={expected!r}, but backbone={internal!r} was supplied."
        )
    return public, expected


def build_pointcloud_model(
    model_name_or_config: str | Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    *,
    n_fields: int = 5,
    device: torch.device | str = "cpu",
    prior_override: nn.Module | None = None,
) -> PointCloudFFM:
    """Build ``GL_rbf``, ``GL_rbf_ENH`` or ``GL_rbf_ENH_CQ``.

    Pass a mapping as the first argument for the release API, or pass a model
    name and a separate mapping for compatibility with older call sites. Model
    dimensions and construction defaults match the production builder so its
    checkpoints load with strict state-dictionary checks.
    """
    if isinstance(model_name_or_config, Mapping):
        if config is not None:
            raise TypeError("Pass either a config mapping or (model_name, config), not both.")
        resolved = dict(model_name_or_config)
    else:
        resolved = dict(config or {})
        resolved["model_name"] = str(model_name_or_config)

    if isinstance(resolved.get("ablation"), Mapping) and bool(
        resolved["ablation"].get("enabled", False)
    ):
        raise ValueError("Ablation wrappers are not part of the released model package.")

    _, backbone_name = _model_identity(resolved)
    coord_dim = int(_get(resolved, "coord_dim", 3))
    if coord_dim < 1:
        raise ValueError("coord_dim must be positive.")
    is_cq = backbone_name == "GL_rbf_ENH_CQ"
    enhanced = backbone_name in {"GL_rbf_ENH", "GL_rbf_ENH_CQ"}

    # The base and enhanced profiles share the same Python class. These flags
    # determine token encoding and latent readout without renaming checkpoint keys.

    sensor_coord_encoding = _get(
        resolved, "sensor_coord_encoding", "fourier" if enhanced else "raw"
    )
    latent_sensor_reinject = bool(_get(resolved, "latent_sensor_reinject", enhanced))
    glres_scale_init = float(
        _get(resolved, "glres_scale_init", 1.0e-2 if enhanced else 0.0)
    )

    prior_name = str(_get(resolved, "prior", "rff"))
    if prior_name not in {"iid", "rff"}:
        raise ValueError(f"Unknown prior {prior_name!r}; expected 'iid' or 'rff'.")
    prior = prior_override
    if prior is None:
        prior = (
            IIDGaussianPrior()
            if prior_name == "iid"
            else RFFGaussianPrior(
                coord_dim=coord_dim,
                n_features=int(_get(resolved, "rff_features", 256)),
                lengthscale=float(_get(resolved, "rff_lengthscale", 0.15)),
            )
        )

    # Keep construction defaults aligned with the production checkpoint schema.
    common = {
        "n_fields": n_fields,
        "coord_dim": coord_dim,
        "hidden_dim": int(_get(resolved, "hidden_dim", 256)),
        "cond_dim": int(_get(resolved, "cond_dim", 128)),
        "field_embed_dim": int(_get(resolved, "field_embed_dim", 64)),
        "latent_dim": int(_get(resolved, "latent_dim", 256)),
        "num_latents": int(_get(resolved, "num_latents", 128)),
        "num_heads": int(_get(resolved, "num_heads", 8)),
        "num_latent_blocks": int(_get(resolved, "num_latent_blocks", 4)),
        "ff_mult": int(_get(resolved, "ff_mult", 4)),
        "attn_dropout": float(_get(resolved, "attn_dropout", 0.0)),
        "mlp_dropout": float(_get(resolved, "mlp_dropout", 0.0)),
        "rbf_sigma": float(_get(resolved, "rbf_sigma", 0.05)),
        "summary_type": str(_get(resolved, "summary_type", "cls")),
        "gather_mode": str(_get(resolved, "gather_mode", "rbf")),
        "gather_topk": int(_get(resolved, "gather_topk", 32)),
        "gather_query_chunk_size": _get(resolved, "gather_query_chunk_size", None),
        "learnable_rbf_sigma": bool(_get(resolved, "learnable_rbf_sigma", False)),
        "neighbor_backend": str(_get(resolved, "neighbor_backend", "torch")),
        "sensor_local_topk": int(_get(resolved, "sensor_local_topk", 8)),
        "sensor_local_dropout": float(_get(resolved, "sensor_local_dropout", 0.0)),
        "use_fourier_pe": bool(_get(resolved, "USE_FOURIER_PE", False)),
        "fourier_pe_num_bands": int(_get(resolved, "fourier_pe_num_bands", 32)),
        "fourier_pe_max_freq": float(_get(resolved, "fourier_pe_max_freq", 64.0)),
        "sensor_coord_encoding": str(sensor_coord_encoding),
        "latent_sensor_reinject": latent_sensor_reinject,
        "latent_reinject_every": int(_get(resolved, "latent_reinject_every", 1)),
        "condition_attention_execution": str(
            _get(resolved, "condition_attention_execution", "legacy_mha")
        ),
        "sensor_attention_padding_mode": str(
            _get(resolved, "sensor_attention_padding_mode", "full")
        ),
        "sensor_attention_buckets": tuple(
            int(value)
            for value in _get(resolved, "sensor_attention_buckets", [256, 320, 384])
        ),
        "glres_scale_init": glres_scale_init,
    }

    # CQ replaces the query head while retaining the conditioned sensor core.
    if is_cq:
        backbone = ConditionalPointHybridLocalGlobalRBFCQ(
            **common,
            cq_query_dim=int(_get(resolved, "cq_query_dim", 128)),
            cq_readout_mode=str(_get(resolved, "cq_readout_mode", "lowrank")),
            cq_fusion_mode=str(_get(resolved, "cq_fusion_mode", "additive")),
            cq_readout_rank=int(_get(resolved, "cq_readout_rank", 64)),
            cq_readout_heads=int(_get(resolved, "cq_readout_heads", 4)),
            cq_global_scale_init=float(_get(resolved, "cq_global_scale_init", 1.0)),
            cq_local_scale_init=float(_get(resolved, "cq_local_scale_init", 1.0)),
            cq_readout_scale_init=float(
                _get(resolved, "cq_readout_scale_init", 1.0e-2)
            ),
            cq_time_conditioning=str(
                _get(resolved, "cq_time_conditioning", "scalar_concat")
            ),
            cq_time_embed_dim=int(_get(resolved, "cq_time_embed_dim", 128)),
            cq_time_max_period=float(_get(resolved, "cq_time_max_period", 10000.0)),
            cq_time_film_zero_init=bool(_get(resolved, "cq_time_film_zero_init", True)),
            cq_measurement_support_mode=str(
                _get(resolved, "cq_measurement_support_mode", "none")
            ),
            cq_measurement_support_normalize=bool(
                _get(resolved, "cq_measurement_support_normalize", True)
            ),
        )
    else:
        query_latent_readout = bool(_get(resolved, "query_latent_readout", enhanced))
        backbone = ConditionalPointHybridLocalGlobalRBF(
            **common,
            enhanced_backbone=enhanced,
            query_latent_readout=query_latent_readout,
            query_readout_type=str(
                _get(
                    resolved,
                    "query_readout_type",
                    "coord" if enhanced or query_latent_readout else "point",
                )
            ),
            query_readout_scale_init=float(
                _get(resolved, "query_readout_scale_init", 1.0e-2 if enhanced else 0.0)
            ),
            enhanced_head_norm=bool(_get(resolved, "enhanced_head_norm", enhanced)),
        )

    return PointCloudFFM(
        backbone,
        prior,
        sigma_min=float(_get(resolved, "sigma_min", 1.0e-4)),
    ).to(torch.device(device))


__all__ = ["build_pointcloud_model"]
