"""Train and sample a conditional rectified flow on query-aligned fields.

Example::

    flow = build_pointcloud_model({"model_name": "GL_rbf", "coord_dim": 3}, n_fields=5)
    loss, metrics = flow.training_loss(x1, coords, obs_coords, obs_values, obs_mask, field_ids)
    prediction = flow.sample(coords, obs_coords, obs_values, obs_mask, field_ids,
                             n_steps=2, clamp_indices=observed_query_slots)

The backbone predicts velocity; this wrapper owns the source draw, straight-line loss, ODE integration, and measured-value consistency.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pykeops.torch import LazyTensor
except ImportError:  # KeOps is optional; the Torch backend remains supported.
    LazyTensor = None

from ..cache.geometry import (
    build_persistent_topk_geometry_cache,
)
from ..observation import (
    apply_endpoint_observation_consistency,
    build_pointwise_observation_maps,
    build_smooth_observation_maps,
    normalize_obs_consistency_mode,
    scatter_observed_values,
)


class PointCloudFFM(nn.Module):
    """Implement the paper's straight-line, one-step rectified-flow objective.

    For source ``x0`` and normalized target ``x1``, draw uniform ``t``, form
    ``x_t = (1-t)x0 + t x1``, and regress velocity against ``x1-x0``. Sampling
    integrates that learned velocity from ``t=0`` to ``t=1``.
    """
    def __init__(self, model: nn.Module, prior: nn.Module, sigma_min: float = 1e-4):
        super().__init__()
        self.model = model
        self.prior = prior

        # Historical checkpoint/config compatibility; sigma_min is not used
        # by the straight-line rectified-flow objective.
        self.sigma_min = sigma_min

    def sample_source(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Draw a source sample x0 from the chosen prior on the query coordinates.
        This is the pi_0 endpoint in rectified flow.
        """
        return self.prior(coords, self.model.n_fields)

    @torch.no_grad()
    def prepare_reconstruction_geometry_cache(
        self,
        *,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_mask: torch.Tensor,
        chunk_size: int = 8192,
    ) -> Any:
        """Build reusable Top-K geometry without condition-dependent features."""
        return build_persistent_topk_geometry_cache(
            self.model, coords=coords, obs_coords=obs_coords,
            obs_mask=obs_mask, chunk_size=chunk_size,
        )

    def simulate(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Straight-line interpolation between source x0 and target x1.

        x_t = (1 - t) * x0 + t * x1
        """
        alpha = t.view(-1, 1, 1)
        return (1.0 - alpha) * x0 + alpha * x1

    def target_vector_field(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        1-RF target velocity is the constant straight-line displacement.

        v*(x_t, t) = x1 - x0
        """
        return x1 - x0

    def training_loss(
        self,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Regress the conditioned velocity against one sampled straight bridge."""
        # Sample x0 from the source prior for the current query coordinates.
        x0 = self.sample_source(coords)

        # Uniform time for standard 1-RF training.
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)

        # Straight interpolation and constant target velocity.
        x_t = self.simulate(t, x0, x1)
        target = self.target_vector_field(x0, x1)

        # Predict the velocity under sparse conditioning.
        pred = self.model(t, x_t, coords, obs_coords, obs_values, obs_mask, obs_field_ids)

        # Standard supervised regression loss used in 1-RF.
        loss = F.mse_loss(pred, target)

        return loss, {
            "loss": float(loss.detach().cpu()),
            "target_rms": float(target.pow(2).mean().sqrt().detach().cpu()),
        }

    def prepare_training_bridge(
        self,
        x1: torch.Tensor,
        coords: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Sample one coherent RF stochastic bridge for all effective queries."""
        x0 = self.sample_source(coords)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        return {
            "x0": x0,
            "t": t,
            "x_t": self.simulate(t, x0, x1),
            "target": self.target_vector_field(x0, x1),
        }

    def training_loss_microbatched(
        self,
        *,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
        query_microbatch_size: int,
        backward: bool = False,
        reuse_condition_context: bool = True,
        synchronize_timing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Split queries while reusing one sampled bridge and one sensor context."""
        del obs_indices  # Point-cloud GL-RBF uses coordinates/field IDs directly.
        n_query = int(coords.shape[1])
        chunk_size = max(1, int(query_microbatch_size))
        if chunk_size >= n_query:
            loss, metrics = self.training_loss(
                x1=x1, coords=coords, obs_coords=obs_coords,
                obs_values=obs_values, obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
            )
            if backward:
                loss.backward()
            metrics.update({
                "rf_bridge_ms": 0.0,
                "condition_context_ms": 0.0,
                "query_chunk_forward_ms": 0.0,
                "query_chunk_backward_ms": 0.0,
                "query_microbatches": 1.0,
            })
            return loss.detach() if backward else loss, metrics

        def sync() -> None:
            if synchronize_timing and x1.device.type == "cuda":
                torch.cuda.synchronize(x1.device)

        sync()
        start = time.perf_counter()
        bridge = self.prepare_training_bridge(x1, coords)
        sync()
        bridge_ms = (time.perf_counter() - start) * 1000.0

        condition_context = None
        sync()
        start = time.perf_counter()
        if reuse_condition_context:
            if not hasattr(self.model, "prepare_condition_context"):
                raise ValueError("Condition-context reuse is unavailable for this backbone.")
            condition_context = self.model.prepare_condition_context(
                obs_coords, obs_values, obs_mask, obs_field_ids,
            )
        sync()
        condition_ms = (time.perf_counter() - start) * 1000.0

        total_elements = int(bridge["target"].numel())
        total_loss = x1.new_zeros(())
        forward_ms = 0.0
        backward_ms = 0.0
        chunks = 0
        for start_index in range(0, n_query, chunk_size):
            end_index = min(start_index + chunk_size, n_query)
            query_slice = slice(start_index, end_index)
            sync()
            start = time.perf_counter()
            if condition_context is not None:
                pred = self.model.forward_query_chunk(
                    t=bridge["t"],
                    x_t_chunk=bridge["x_t"][:, query_slice],
                    coords_chunk=coords[:, query_slice],
                    condition_context=condition_context,
                )
            else:
                pred = self.model(
                    bridge["t"],
                    bridge["x_t"][:, query_slice],
                    coords[:, query_slice],
                    obs_coords,
                    obs_values,
                    obs_mask,
                    obs_field_ids,
                )
            chunk_loss = F.mse_loss(
                pred, bridge["target"][:, query_slice], reduction="sum",
            ) / total_elements
            sync()
            forward_ms += (time.perf_counter() - start) * 1000.0
            if backward:
                sync()
                start = time.perf_counter()
                chunk_loss.backward(
                    retain_graph=condition_context is not None and end_index < n_query,
                )
                sync()
                backward_ms += (time.perf_counter() - start) * 1000.0
                total_loss = total_loss + chunk_loss.detach()
            else:
                total_loss = total_loss + chunk_loss
            chunks += 1
            del pred, chunk_loss

        target = bridge["target"]
        metrics = {
            "loss": float(total_loss.detach().cpu()),
            "target_rms": float(target.pow(2).mean().sqrt().detach().cpu()),
            "rf_bridge_ms": bridge_ms,
            "condition_context_ms": condition_ms,
            "query_chunk_forward_ms": forward_ms,
            "query_chunk_backward_ms": backward_ms,
            "query_microbatches": float(chunks),
        }
        return total_loss, metrics

    def _sample_cached_streamed(
        self,
        *,
        x: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        clamp_indices: Optional[torch.Tensor],
        ts: torch.Tensor,
        ode_solver: str,
        obs_consistency_mode: str,
        obs_consistency_strength: float,
        obs_consistency_schedule_power: float,
        obs_consistency_final_clamp: bool,
        value_map: Optional[torch.Tensor],
        mask_map: Optional[torch.Tensor],
        reconstruction_query_chunk_size: int,
        reconstruction_cache_level: str,
        reconstruction_geometry_cache: Optional[Any],
    ) -> torch.Tensor:
        """Integrate in query chunks while reusing sensor and geometry state."""
        if not hasattr(self.model, "prepare_condition_context"):
            raise ValueError(
                "cached_streamed reconstruction requires a backbone with the context API."
            )
        chunk_size = max(1, int(reconstruction_query_chunk_size))
        profile = bool(getattr(self, "_reconstruction_profile_enabled", False))

        def profile_sync() -> None:
            if profile and coords.device.type == "cuda":
                torch.cuda.synchronize(coords.device)

        profile_sync()
        condition_start = time.perf_counter()
        condition_context = self.model.prepare_condition_context(
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )
        profile_sync()
        condition_seconds = time.perf_counter() - condition_start
        query_start = time.perf_counter()
        query_context = self.model.prepare_query_context(
            coords=coords,
            condition_context=condition_context,
            cache_level=reconstruction_cache_level,
            chunk_size=chunk_size,
            precomputed_geometry=reconstruction_geometry_cache,
        )
        self._last_reconstruction_condition_bytes = self.model.context_nbytes(condition_context)
        self._last_reconstruction_cache_bytes = self.model.context_nbytes(query_context)
        profile_sync()
        self._last_reconstruction_condition_seconds = condition_seconds
        self._last_reconstruction_query_seconds = time.perf_counter() - query_start
        ode_start = time.perf_counter()
        bsz = coords.shape[0]
        for i in range(len(ts) - 1):
            t0 = ts[i].expand(bsz)
            dt = ts[i + 1] - ts[i]
            for start in range(0, coords.shape[1], chunk_size):
                end = min(start + chunk_size, coords.shape[1])
                query_slice = slice(start, end)
                x_chunk = x[:, query_slice]
                coords_chunk = coords[:, query_slice]
                v0 = self.model.forward_query_chunk(
                    t=t0,
                    x_t_chunk=x_chunk,
                    coords_chunk=coords_chunk,
                    condition_context=condition_context,
                    query_context=query_context,
                    query_slice=query_slice,
                )
                if obs_consistency_mode in ("endpoint", "endpoint_smooth"):
                    v0 = apply_endpoint_observation_consistency(
                        x_t=x_chunk,
                        v=v0,
                        t=t0,
                        value_map=value_map[:, query_slice],
                        mask_map=mask_map[:, query_slice],
                        strength=obs_consistency_strength,
                        schedule_power=obs_consistency_schedule_power,
                    )
                if ode_solver == "heun":
                    x_euler = x_chunk + dt * v0
                    t1 = ts[i + 1].expand(bsz)
                    v1 = self.model.forward_query_chunk(
                        t=t1,
                        x_t_chunk=x_euler,
                        coords_chunk=coords_chunk,
                        condition_context=condition_context,
                        query_context=query_context,
                        query_slice=query_slice,
                    )
                    if (
                        obs_consistency_mode in ("endpoint", "endpoint_smooth")
                        and float(ts[i + 1].item()) < 1.0
                    ):
                        v1 = apply_endpoint_observation_consistency(
                            x_t=x_euler,
                            v=v1,
                            t=t1,
                            value_map=value_map[:, query_slice],
                            mask_map=mask_map[:, query_slice],
                            strength=obs_consistency_strength,
                            schedule_power=obs_consistency_schedule_power,
                        )
                    x[:, query_slice] = x_chunk + 0.5 * dt * (v0 + v1)
                else:
                    x[:, query_slice] = x_chunk + dt * v0

            if obs_consistency_mode == "default_hard" and clamp_indices is not None:
                x = scatter_observed_values(
                    x=x,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_indices=clamp_indices,
                    obs_field_ids=obs_field_ids,
                    strength=1.0,
                )

        if (
            obs_consistency_final_clamp
            and obs_consistency_mode != "none"
            and clamp_indices is not None
        ):
            x = scatter_observed_values(
                x=x,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                strength=1.0,
            )
        profile_sync()
        self._last_reconstruction_ode_seconds = time.perf_counter() - ode_start
        return x

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        n_steps: int = 8,
        clamp_indices: Optional[torch.Tensor] = None,
        ode_solver: str = "euler",
        obs_consistency_mode: str = "default_hard",
        obs_consistency_strength: float = 1.0,
        obs_consistency_sigma: float = 0.05,
        obs_consistency_schedule_power: float = 2.0,
        obs_consistency_final_clamp: bool = True,
        obs_consistency_chunk_size: int = 8192,
        reconstruction_execution_mode: str = "legacy_full",
        reconstruction_query_chunk_size: int = 8192,
        reconstruction_cache_level: str = "static_features",
        reconstruction_geometry_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        """Integrate from a source draw with Euler or Heun and optional sensor guidance.

        ``default_hard`` replaces measured query slots after each step.
        ``endpoint`` and ``endpoint_smooth`` guide the predicted clean endpoint.
        ``cached_streamed`` keeps large-grid query work in bounded chunks.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if reconstruction_execution_mode not in ("legacy_full", "cached_streamed"):
            raise ValueError(
                "reconstruction_execution_mode must be 'legacy_full' or 'cached_streamed'."
            )
        if reconstruction_geometry_cache is not None and reconstruction_execution_mode != "cached_streamed":
            raise ValueError(
                "reconstruction_geometry_cache requires cached_streamed execution."
            )

        bsz = coords.shape[0]
        x = self.sample_source(coords)
        obs_consistency_mode = normalize_obs_consistency_mode(obs_consistency_mode)
        if obs_consistency_mode != "none" and clamp_indices is None:
            if obs_consistency_mode in ("default_hard", "endpoint"):
                raise ValueError(
                    f"obs_consistency_mode={obs_consistency_mode!r} requires clamp_indices."
                )

        # Guidance maps are built once; hard consistency uses the query-slot
        # indices directly and needs no dense map.
        value_map = None
        mask_map = None
        if obs_consistency_mode == "endpoint":
            value_map, mask_map = build_pointwise_observation_maps(
                coords=coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
            )
        elif obs_consistency_mode == "endpoint_smooth":
            value_map, mask_map = build_smooth_observation_maps(
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
                sigma=obs_consistency_sigma,
                chunk_size=obs_consistency_chunk_size,
            )

        ts = torch.linspace(
            0.0, 1.0, n_steps + 1, device=coords.device, dtype=coords.dtype
        )

        if reconstruction_execution_mode == "cached_streamed":
            return self._sample_cached_streamed(
                x=x,
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                clamp_indices=clamp_indices,
                ts=ts,
                ode_solver=ode_solver,
                obs_consistency_mode=obs_consistency_mode,
                obs_consistency_strength=obs_consistency_strength,
                obs_consistency_schedule_power=obs_consistency_schedule_power,
                obs_consistency_final_clamp=obs_consistency_final_clamp,
                value_map=value_map,
                mask_map=mask_map,
                reconstruction_query_chunk_size=reconstruction_query_chunk_size,
                reconstruction_cache_level=reconstruction_cache_level,
                reconstruction_geometry_cache=reconstruction_geometry_cache,
            )
        self._last_reconstruction_condition_bytes = 0
        self._last_reconstruction_cache_bytes = 0

        for i in range(n_steps):
            t0 = ts[i].expand(bsz)
            dt = ts[i + 1] - ts[i]

            # Velocity at the current state.
            v0 = self.model(t0, x, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
            if obs_consistency_mode in ("endpoint", "endpoint_smooth"):
                # RF clean-endpoint observation masking: guide x1_hat, then
                # convert the consistent endpoint back to a velocity.
                v0 = apply_endpoint_observation_consistency(
                    x_t=x,
                    v=v0,
                    t=t0,
                    value_map=value_map,
                    mask_map=mask_map,
                    strength=obs_consistency_strength,
                    schedule_power=obs_consistency_schedule_power,
                )

            if ode_solver == "heun":
                # Optional predictor-corrector step.
                x_euler = x + dt * v0
                t1 = ts[i + 1].expand(bsz)
                v1 = self.model(t1, x_euler, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
                if obs_consistency_mode in ("endpoint", "endpoint_smooth") and float(ts[i + 1].item()) < 1.0:
                    v1 = apply_endpoint_observation_consistency(
                        x_t=x_euler,
                        v=v1,
                        t=t1,
                        value_map=value_map,
                        mask_map=mask_map,
                        strength=obs_consistency_strength,
                        schedule_power=obs_consistency_schedule_power,
                    )
                x = x + 0.5 * dt * (v0 + v1)
            else:
                # Default 1-RF benchmark solver.
                x = x + dt * v0

            # Hard consistency copies trusted measurements into query slots
            # after each integration step.
            if obs_consistency_mode == "default_hard" and clamp_indices is not None:
                x = scatter_observed_values(
                    x=x,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_indices=clamp_indices,
                    obs_field_ids=obs_field_ids,
                    strength=1.0,
                )

        if obs_consistency_final_clamp and obs_consistency_mode != "none" and clamp_indices is not None:
            x = scatter_observed_values(
                x=x,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                strength=1.0,
            )

        return x
