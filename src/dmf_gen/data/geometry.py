"""Represent changing-geometry cases on one enclosing query grid.

Example::

    grid = EnclosingGrid(shape=(41, 41), x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    geometry = SupportGeometry(grid, support=material, sensor_candidates=band)
    sample = SupportMaskAdapter(stats).adapt(stress_sample, geometry)

Every case shares the enclosing grid; its own support (material or fluid) is
a reconstruction target channel, and sensors come only from its prescribed
candidate points. Values outside the support carry a fixed fill value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch

from .contracts import FieldSample, NormalizationStats


@runtime_checkable
class GeometryAdapter(Protocol):
    """Map a field sample onto a task-specific geometry."""

    def adapt(self, sample: FieldSample, geometry: object) -> FieldSample:
        """Return a sample represented on the requested geometry."""
        ...


class UnsupportedGeometryAdapter:
    """Adapter that reports geometry-changing reconstruction as unsupported."""

    def adapt(self, sample: FieldSample, geometry: object) -> FieldSample:
        del sample, geometry
        raise NotImplementedError(
            "geometry-changing reconstruction is not implemented"
        )


@dataclass(frozen=True)
class EnclosingGrid:
    """Sample-invariant uniform grid that encloses every case's support.

    Points are ordered row-major with ``x`` fastest, so ``values.reshape(shape)``
    gives ``[y, x]`` images. Model coordinates are ``(x, y, 0)`` on [0, 1]; the
    zero third axis matches the three-coordinate checkpoints and source prior.
    """

    shape: tuple[int, int]
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    def __post_init__(self) -> None:
        shape = tuple(int(size) for size in self.shape)
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError("grid shape must be (ny, nx) with at least two points per axis")
        for name in ("x_range", "y_range"):
            low, high = (float(value) for value in getattr(self, name))
            if not np.isfinite([low, high]).all() or high <= low:
                raise ValueError(f"{name} must be a finite increasing pair")
            object.__setattr__(self, name, (low, high))
        object.__setattr__(self, "shape", shape)

    @property
    def n_points(self) -> int:
        return self.shape[0] * self.shape[1]

    def _points(self, x_range: tuple[float, float], y_range: tuple[float, float]) -> np.ndarray:
        ny, nx = self.shape
        # A (y, x) meshgrid of float64 axes cast to float32 reproduces exactly
        # the coordinates the paper checkpoints were trained on.
        yy, xx = np.meshgrid(np.linspace(*y_range, ny), np.linspace(*x_range, nx), indexing="ij")
        planar = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        return np.concatenate([planar, np.zeros((planar.shape[0], 1), np.float32)], axis=-1)

    def normalized_coordinates(self) -> np.ndarray:
        return self._points((0.0, 1.0), (0.0, 1.0))

    def physical_coordinates(self) -> np.ndarray:
        return self._points(self.x_range, self.y_range)


@dataclass(frozen=True)
class SupportGeometry:
    """One case's physical support and permitted sensor points on an enclosing grid."""

    grid: EnclosingGrid
    support: np.ndarray
    sensor_candidates: np.ndarray
    support_name: str = "support"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=bool).reshape(-1)
        candidates = np.asarray(self.sensor_candidates, dtype=bool).reshape(-1)
        if support.size != self.grid.n_points or candidates.size != self.grid.n_points:
            raise ValueError("support and sensor_candidates must cover every grid point")
        if not support.any():
            raise ValueError("the physical support must contain at least one point")
        if not candidates.any():
            raise ValueError("sensor_candidates must contain at least one point")
        if np.any(candidates & ~support):
            raise ValueError("sensor candidates must lie on the physical support")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "sensor_candidates", candidates)


class SupportMaskAdapter:
    """Add a case's support mask to its physical channels on the enclosing grid.

    ``stats`` holds training-split statistics for the full target order: the
    physical channels plus ``mask_field``. Outside the support each physical
    channel is set to its fill value (zero unless given; positive for a
    log-transformed channel); no field metric scores those points. The mask is
    one on the support and zero elsewhere, and sensors are limited to the
    geometry's candidates.
    """

    def __init__(
        self,
        stats: NormalizationStats,
        *,
        mask_field: str = "mask",
        fill_values: Mapping[str, float] | None = None,
    ) -> None:
        if mask_field not in stats.field_names:
            raise ValueError(f"statistics must include the support field {mask_field!r}")
        self.stats = stats
        self.mask_field = mask_field
        self.physical_fields = tuple(name for name in stats.field_names if name != mask_field)
        fills = dict(fill_values or {})
        unknown = sorted(set(fills) - set(self.physical_fields))
        if unknown:
            raise ValueError(f"fill values given for unknown field(s) {unknown}")
        self.fill_values = {name: float(fills.get(name, 0.0)) for name in self.physical_fields}
        transforms = stats.transforms or ("identity",) * len(stats.field_names)
        for name, transform in zip(stats.field_names, transforms, strict=True):
            if transform == "log" and name != mask_field and self.fill_values[name] <= 0:
                raise ValueError(f"log-transformed field {name!r} needs a positive fill value")
        self._physical_stats = stats.subset(self.physical_fields)
        self._grid_coordinates: dict[EnclosingGrid, tuple[torch.Tensor, torch.Tensor]] = {}

    def _coordinates(self, grid: EnclosingGrid) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized and physical grid coordinates, shared by every case on ``grid``."""
        if grid not in self._grid_coordinates:
            self._grid_coordinates[grid] = (
                torch.from_numpy(grid.normalized_coordinates()),
                torch.from_numpy(grid.physical_coordinates()),
            )
        return self._grid_coordinates[grid]

    def adapt(self, sample: FieldSample, geometry: object) -> FieldSample:
        if not isinstance(geometry, SupportGeometry):
            raise TypeError("SupportMaskAdapter requires a SupportGeometry")
        if sample.field_names != self.physical_fields:
            raise ValueError(
                f"expected physical fields {self.physical_fields}, got {sample.field_names}"
            )
        expected = self._physical_stats
        if (
            not torch.equal(sample.stats.mean, expected.mean)
            or not torch.equal(sample.stats.std, expected.std)
            or sample.stats.transforms != expected.transforms
        ):
            raise ValueError("sample statistics differ from the adapter's training statistics")
        grid = geometry.grid
        if sample.values.shape[0] != grid.n_points:
            raise ValueError("sample points must be the enclosing grid points")
        coordinates, physical_coordinates = self._coordinates(grid)
        if sample.coordinates.shape != coordinates.shape or not torch.equal(
            sample.coordinates, coordinates
        ):
            raise ValueError("sample coordinates must be the enclosing grid coordinates")

        support = torch.from_numpy(geometry.support)
        physical = sample.values.clone()
        for column, name in enumerate(self.physical_fields):
            physical[~support, column] = self.fill_values[name]
        columns = []
        for name in self.stats.field_names:
            if name == self.mask_field:
                columns.append(support.to(physical.dtype))
            else:
                columns.append(physical[:, self.physical_fields.index(name)])
        metadata = {
            **dict(sample.metadata),
            **dict(geometry.metadata),
            "support_name": geometry.support_name,
            "support_points": int(geometry.support.sum()),
            "sensor_candidates": int(geometry.sensor_candidates.sum()),
        }
        return FieldSample(
            sample_id=sample.sample_id,
            values=torch.stack(columns, dim=-1),
            coordinates=coordinates,
            coordinates_raw=physical_coordinates,
            field_names=self.stats.field_names,
            stats=self.stats,
            resolution=sample.resolution,
            logical_shape=grid.shape,
            case_id=sample.case_id,
            frame_id=sample.frame_id,
            time=sample.time,
            metadata=metadata,
            support_field=self.mask_field,
            sensor_mask=torch.from_numpy(geometry.sensor_candidates.copy()),
        )


def dilate_four_connected(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary dilation of a 2-D mask by a plus-shaped element, zero outside the grid."""
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 2:
        raise ValueError("dilation expects a 2-D mask")
    for _ in range(int(iterations)):
        grown = result.copy()
        grown[1:, :] |= result[:-1, :]
        grown[:-1, :] |= result[1:, :]
        grown[:, 1:] |= result[:, :-1]
        grown[:, :-1] |= result[:, 1:]
        result = grown
    return result


def support_boundary_band(support: np.ndarray, shape: Sequence[int], cells: int) -> np.ndarray:
    """Support points within ``cells`` four-connected grid steps of the non-support."""
    if int(cells) < 1:
        raise ValueError("the sensor band must be at least one cell wide")
    inside = np.asarray(support, dtype=bool).reshape(tuple(shape))
    return (inside & dilate_four_connected(~inside, int(cells))).reshape(-1)


def ellipse_ring(
    grid: EnclosingGrid,
    center: Sequence[float],
    semi_axes: Sequence[float],
    halfwidth: float,
) -> np.ndarray:
    """Grid points whose normalized elliptic radius lies within ``1 +/- halfwidth``.

    Center and semi-axes are in model coordinates. Evaluated in float32, as in
    the paper pipeline, so points on the ring's edge are classified identically.
    """
    if min(float(value) for value in semi_axes) <= 0 or not 0 < float(halfwidth) < 1:
        raise ValueError("ellipse semi-axes must be positive and 0 < halfwidth < 1")
    xy = grid.normalized_coordinates()[:, :2]
    dx = (xy[:, 0] - np.float32(center[0])) / np.float32(semi_axes[0])
    dy = (xy[:, 1] - np.float32(center[1])) / np.float32(semi_axes[1])
    radius = np.sqrt(dx**2 + dy**2)
    return (radius >= np.float32(1.0 - float(halfwidth))) & (
        radius <= np.float32(1.0 + float(halfwidth))
    )


__all__ = [
    "EnclosingGrid",
    "GeometryAdapter",
    "SupportGeometry",
    "SupportMaskAdapter",
    "UnsupportedGeometryAdapter",
    "dilate_four_connected",
    "ellipse_ring",
    "support_boundary_band",
]
