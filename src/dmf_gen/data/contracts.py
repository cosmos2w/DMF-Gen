"""Tensor contracts shared by the combustion, CFD, and changing-geometry adapters.

Example::

    sample = dataset[0]  # FieldSample with all physical target channels
    batch = make_observation_batch([sample], observed_channels=["T"],
                                   sensors_per_channel=256, query_count=None)

The batch keeps scalar sensors marked by field ID, while every query target
contains the complete field vector. Variable sensor counts use ``obs_mask``.
Geometry samples also name a hidden support-mask target and restrict sensors
to their prescribed candidate points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import torch


def _tensor(value: torch.Tensor | np.ndarray | Sequence[float], *, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device="cpu").contiguous()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


FIELD_TRANSFORMS = ("identity", "log")


@dataclass(frozen=True)
class NormalizationStats:
    """Per-field population statistics used by both training and inference.

    ``transforms`` optionally lists one transform per field, applied before
    standardization: ``identity`` or ``log`` for strictly positive fields such
    as airfoil density and pressure. ``inverse`` applies ``exp`` after the
    affine inverse, so those fields stay positive for any model output.
    ``None`` means no transform was recorded and every field is affine.
    """

    mean: torch.Tensor
    std: torch.Tensor
    field_names: tuple[str, ...]
    transforms: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        mean = _tensor(self.mean, name="mean").reshape(-1)
        std = _tensor(self.std, name="std").reshape(-1)
        names = tuple(str(name) for name in self.field_names)
        if not names or len(names) != mean.numel() or mean.shape != std.shape:
            raise ValueError("mean, std, and field_names must have the same nonzero length")
        if len(set(names)) != len(names):
            raise ValueError("field_names must be unique")
        if torch.any(std <= 0):
            raise ValueError("std values must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "field_names", names)
        if self.transforms is not None:
            transforms = tuple(str(value) for value in self.transforms)
            if len(transforms) != len(names):
                raise ValueError("transforms must contain one entry per field")
            unknown = sorted(set(transforms) - set(FIELD_TRANSFORMS))
            if unknown:
                raise ValueError(f"unknown field transform(s) {unknown}; use {FIELD_TRANSFORMS}")
            object.__setattr__(self, "transforms", transforms)

    @property
    def log_channels(self) -> tuple[int, ...]:
        """Channel indices that are standardized in log space."""
        if self.transforms is None:
            return ()
        return tuple(index for index, name in enumerate(self.transforms) if name == "log")

    def _check_channels(self, values: torch.Tensor) -> None:
        if values.shape[-1] != len(self.field_names):
            raise ValueError(
                f"expected {len(self.field_names)} field channels, got {values.shape[-1]}"
            )

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        self._check_channels(values)
        columns = list(self.log_channels)
        if columns:
            if torch.any(values[..., columns] <= 0):
                raise ValueError("log-transformed fields must be strictly positive")
            values = values.clone()
            values[..., columns] = torch.log(values[..., columns])
        return (values - self.mean.to(values)) / self.std.to(values)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        self._check_channels(values)
        result = values * self.std.to(values) + self.mean.to(values)
        columns = list(self.log_channels)
        if columns:
            result[..., columns] = torch.exp(result[..., columns])
        return result

    def inverse_entries(self, values: torch.Tensor, field_ids: torch.Tensor) -> torch.Tensor:
        """Invert scalars shaped like ``field_ids``, each from its own field."""
        if values.shape != field_ids.shape:
            raise ValueError("values and field_ids must have the same shape")
        ids = field_ids.long().clamp(min=0).to(self.mean.device)
        result = values * self.std[ids].to(values) + self.mean[ids].to(values)
        columns = list(self.log_channels)
        if columns:
            is_log = torch.zeros(len(self.field_names), dtype=torch.bool)
            is_log[columns] = True
            result = torch.where(is_log[ids].to(result.device), torch.exp(result), result)
        return result

    def subset(self, field_names: Sequence[str]) -> "NormalizationStats":
        """Statistics for selected fields, in the requested order."""
        indices = [self.field_names.index(str(name)) for name in field_names]
        return NormalizationStats(
            self.mean[indices],
            self.std[indices],
            tuple(self.field_names[index] for index in indices),
            None if self.transforms is None else tuple(self.transforms[index] for index in indices),
        )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        field_names: Sequence[str] | None = None,
    ) -> "NormalizationStats":
        names = data.get("field_names", field_names)
        if names is None:
            raise ValueError("field_names are required when loading unnamed statistics")
        transforms = data.get("transforms")
        return cls(
            data["mean"],
            data["std"],
            tuple(str(name) for name in names),
            None if transforms is None else tuple(str(value) for value in transforms),
        )


@dataclass
class FieldSample:
    """One native-resolution field snapshot.

    ``values`` are in physical units. ``coordinates`` are normalized to [0, 1]
    per coordinate dimension; ``coordinates_raw`` preserve the source coordinates.
    Changing-geometry samples name ``support_field``, a target channel equal to
    one where the physical field exists and zero elsewhere, and may restrict
    sensors to the points marked in ``sensor_mask``.
    """

    sample_id: str
    values: torch.Tensor
    coordinates: torch.Tensor
    coordinates_raw: torch.Tensor
    field_names: tuple[str, ...]
    stats: NormalizationStats
    resolution: str | None = None
    logical_shape: tuple[int, ...] | None = None
    case_id: int | None = None
    frame_id: int | None = None
    time: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    support_field: str | None = None
    sensor_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.values = _tensor(self.values, name="values")
        self.coordinates = _tensor(self.coordinates, name="coordinates")
        self.coordinates_raw = _tensor(self.coordinates_raw, name="coordinates_raw")
        self.field_names = tuple(str(name) for name in self.field_names)
        if self.values.ndim != 2:
            raise ValueError("values must have shape [points, fields]")
        if self.coordinates.ndim != 2 or self.coordinates_raw.ndim != 2:
            raise ValueError("coordinates must have shape [points, dimensions]")
        if self.values.shape[0] != self.coordinates.shape[0]:
            raise ValueError("values and coordinates must have the same number of points")
        if self.coordinates_raw.shape != self.coordinates.shape:
            raise ValueError("coordinates and coordinates_raw must have matching shapes")
        if self.values.shape[1] != len(self.field_names):
            raise ValueError("values channel count must match field_names")
        if self.stats.field_names != self.field_names:
            raise ValueError("sample and normalization statistics field_names must match")
        if self.logical_shape is not None:
            self.logical_shape = tuple(int(size) for size in self.logical_shape)
            if not self.logical_shape or int(np.prod(self.logical_shape)) != self.values.shape[0]:
                raise ValueError("logical_shape must describe all native sample points")
        if self.support_field is not None:
            self.support_field = str(self.support_field)
            if self.support_field not in self.field_names:
                raise ValueError(f"support_field {self.support_field!r} is not a sample field")
            support = self.values[:, self.field_names.index(self.support_field)]
            if not torch.all((support == 0) | (support == 1)):
                raise ValueError("the support field must contain only zeros and ones")
        if self.sensor_mask is not None:
            self.sensor_mask = torch.as_tensor(self.sensor_mask, dtype=torch.bool, device="cpu")
            if self.sensor_mask.shape != (self.values.shape[0],):
                raise ValueError("sensor_mask must have shape [points]")
            if not torch.any(self.sensor_mask):
                raise ValueError("sensor_mask must allow at least one sensor point")

    @property
    def support(self) -> torch.Tensor | None:
        """Boolean physical support, or ``None`` for fixed-domain samples."""
        if self.support_field is None:
            return None
        return self.values[:, self.field_names.index(self.support_field)] > 0.5

    @property
    def normalized_values(self) -> torch.Tensor:
        return self.stats.normalize(self.values)

    def to_model_space(self) -> torch.Tensor:
        return self.normalized_values

    def to_physical(self, values: torch.Tensor | None = None) -> torch.Tensor:
        """Invert normalization for this sample or a model output with all fields."""
        return self.stats.inverse(self.normalized_values if values is None else values)


@dataclass
class ObservationBatch:
    """Padded marked sensors and query-aligned full-field targets.

    ``obs_indices`` is present only when every active sensor belongs to the
    query set; exact hard replacement requires that condition.
    """

    obs_coords: torch.Tensor
    obs_values: torch.Tensor
    obs_field_ids: torch.Tensor
    obs_mask: torch.Tensor
    query_coords: torch.Tensor
    target_fields: torch.Tensor
    query_mask: torch.Tensor
    query_indices: torch.Tensor
    obs_indices: torch.Tensor | None
    field_names: tuple[str, ...]
    stats: NormalizationStats
    sample_ids: tuple[str, ...]
    resolution: str | None
    logical_shape: tuple[int, ...] | None

    def __post_init__(self) -> None:
        for name in ("obs_coords", "obs_values", "query_coords", "target_fields"):
            setattr(self, name, _tensor(getattr(self, name), name=name))
        self.obs_field_ids = torch.as_tensor(self.obs_field_ids, dtype=torch.long, device="cpu")
        self.query_indices = torch.as_tensor(self.query_indices, dtype=torch.long, device="cpu")
        self.obs_mask = torch.as_tensor(self.obs_mask, dtype=torch.bool, device="cpu")
        self.query_mask = torch.as_tensor(self.query_mask, dtype=torch.bool, device="cpu")
        if self.obs_indices is not None:
            self.obs_indices = torch.as_tensor(self.obs_indices, dtype=torch.long, device="cpu")
        self.field_names = tuple(self.field_names)

        if self.obs_coords.ndim != 3 or self.query_coords.ndim != 3:
            raise ValueError(
                "observation and query coordinates must be [batch, points, dimensions]"
            )
        batch, n_obs, coord_dim = self.obs_coords.shape
        if self.obs_values.shape != (batch, n_obs, 1):
            raise ValueError("obs_values must have shape [batch, observations, 1]")
        if self.obs_field_ids.shape != (batch, n_obs) or self.obs_mask.shape != (batch, n_obs):
            raise ValueError("observation IDs and mask must match [batch, observations]")
        if self.query_coords.shape[0] != batch or self.query_coords.shape[2] != coord_dim:
            raise ValueError(
                "query coordinates must match observation batch and coordinate dimensions"
            )
        if self.target_fields.shape[:2] != self.query_coords.shape[:2]:
            raise ValueError("target_fields must align with query coordinates")
        if self.target_fields.shape[2] != len(self.field_names):
            raise ValueError("target field channel count must match field_names")
        if self.query_mask.shape != self.query_coords.shape[:2]:
            raise ValueError("query_mask must match [batch, queries]")
        if self.query_indices.shape != self.query_coords.shape[:2]:
            raise ValueError("query_indices must match [batch, queries]")
        if self.obs_indices is not None and self.obs_indices.shape != (batch, n_obs):
            raise ValueError("obs_indices must match [batch, observations]")
        if self.stats.field_names != self.field_names:
            raise ValueError("batch and normalization statistics field_names must match")
        if len(self.sample_ids) != batch:
            raise ValueError("sample_ids must contain one ID per batch item")
        if torch.any(self.obs_field_ids[self.obs_mask] < 0) or torch.any(
            self.obs_field_ids[self.obs_mask] >= len(self.field_names)
        ):
            raise ValueError("active observation field IDs must be valid channels")
        if torch.any(self.obs_field_ids[~self.obs_mask] != -1):
            raise ValueError("padded observation field IDs must be -1")
        if self.obs_indices is not None:
            valid = self.obs_indices[self.obs_mask]
            if torch.any(valid < 0) or torch.any(valid >= self.query_coords.shape[1]):
                raise ValueError("active obs_indices must point to a query slot")
            if torch.any(self.obs_indices[~self.obs_mask] != -1):
                raise ValueError("padded obs_indices must be -1")

    @property
    def normalized_targets(self) -> torch.Tensor:
        return self.target_fields

    def to_physical(self, values: torch.Tensor | None = None) -> torch.Tensor:
        """Invert normalization for all reconstructed field channels."""
        return self.stats.inverse(self.target_fields if values is None else values)

    def observations_to_physical(self) -> torch.Tensor:
        """Invert each scalar observation using its own observed field's stats."""
        values = self.stats.inverse_entries(self.obs_values[..., 0], self.obs_field_ids)
        return values.unsqueeze(-1).masked_fill(~self.obs_mask.unsqueeze(-1), 0.0)


def group_samples_by_resolution(
    samples: Sequence[FieldSample],
) -> dict[str | None, list[FieldSample]]:
    """Group samples without reading additional data or padding native grids."""
    grouped: dict[str | None, list[FieldSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.resolution, []).append(sample)
    return grouped


def _channel_ids(
    field_names: tuple[str, ...], observed_channels: Sequence[str | int] | str | int
) -> list[int]:
    requested = (
        [observed_channels]
        if isinstance(observed_channels, (str, int))
        else list(observed_channels)
    )
    if not requested:
        raise ValueError("observed_channels must not be empty")
    ids: list[int] = []
    for item in requested:
        if isinstance(item, str):
            if item not in field_names:
                raise ValueError(f"unknown observed field {item!r}; choose from {field_names}")
            index = field_names.index(item)
        else:
            index = int(item)
            if index < 0 or index >= len(field_names):
                raise ValueError(f"observed channel index {index} is out of range")
        if index in ids:
            raise ValueError("observed_channels must not contain duplicates")
        ids.append(index)
    return ids


SensorCountRange = Mapping[str, int]
SensorCountValue = int | SensorCountRange
SensorCountSpec = (
    SensorCountValue
    | Sequence[SensorCountValue]
    | Mapping[str | int, SensorCountValue]
)


def _count_range(value: SensorCountValue) -> tuple[int, int]:
    if isinstance(value, Mapping):
        if set(value) != {"min", "max"}:
            raise ValueError("sensor count ranges must have exactly 'min' and 'max' keys")
        lower, upper = value["min"], value["max"]
    else:
        lower = upper = value
    if isinstance(lower, bool) or isinstance(upper, bool):
        raise ValueError("sensor counts must be integers")
    try:
        minimum, maximum = int(lower), int(upper)
    except (TypeError, ValueError) as error:
        raise ValueError("sensor counts must be integers") from error
    if minimum != lower or maximum != upper:
        raise ValueError("sensor counts must be integers")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("sensor count ranges require 1 <= min <= max")
    return minimum, maximum


def _sensor_ranges(
    field_names: tuple[str, ...],
    channel_ids: list[int],
    sensors_per_channel: SensorCountSpec,
) -> list[tuple[int, int]]:
    if isinstance(sensors_per_channel, Mapping):
        if set(sensors_per_channel) == {"min", "max"}:
            return [_count_range(sensors_per_channel)] * len(channel_ids)  # type: ignore[arg-type]
        ranges = []
        for channel_id in channel_ids:
            name = field_names[channel_id]
            if name in sensors_per_channel:
                value = sensors_per_channel[name]
            elif channel_id in sensors_per_channel:
                value = sensors_per_channel[channel_id]
            else:
                raise ValueError(f"missing sensor count for field {name!r}")
            ranges.append(_count_range(value))  # type: ignore[arg-type]
        return ranges
    if isinstance(sensors_per_channel, int):
        return [_count_range(sensors_per_channel)] * len(channel_ids)
    ranges = [_count_range(value) for value in sensors_per_channel]
    if len(ranges) != len(channel_ids):
        raise ValueError("sensor count sequence must align with observed_channels")
    return ranges


def make_observation_batch(
    samples: Sequence[FieldSample],
    observed_channels: Sequence[str | int] | str | int,
    sensors_per_channel: SensorCountSpec,
    query_count: int | None = None,
    seed: int = 42,
) -> ObservationBatch:
    """Sample independent sensors per field and aligned query targets.

    Only samples with equal native resolution, point count, coordinate dimension,
    field order, and normalization statistics may share a batch. Use
    :func:`group_samples_by_resolution` before batching mixed-resolution samples.
    Sensor counts can be fixed integers or inclusive ``{"min": ..., "max": ...}``
    ranges, globally or per channel. Counts and sensor locations are sampled
    independently per sample and channel; shorter rows are padded with a false
    ``obs_mask`` and field ID -1. Different fields may observe the same point.
    A sample's ``sensor_mask`` restricts its sensors to the marked points, and
    its ``support_field`` is a reconstruction target that may not be observed.
    ``obs_indices`` is returned only if every active sensor is also present among
    the sampled query points; padded entries are -1.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("samples must contain at least one FieldSample")
    first = samples[0]
    compatibility = (
        first.resolution,
        first.values.shape[0],
        first.coordinates.shape[1],
        first.logical_shape,
        first.field_names,
        first.support_field,
    )
    for sample in samples:
        current = (
            sample.resolution,
            sample.values.shape[0],
            sample.coordinates.shape[1],
            sample.logical_shape,
            sample.field_names,
            sample.support_field,
        )
        if current != compatibility:
            raise ValueError("samples must have the same native resolution and tensor contract")
        if (
            not torch.equal(sample.stats.mean, first.stats.mean)
            or not torch.equal(sample.stats.std, first.stats.std)
            or sample.stats.transforms != first.stats.transforms
        ):
            raise ValueError("all samples in a batch must share the same normalization statistics")

    channel_ids = _channel_ids(first.field_names, observed_channels)
    support_field = first.support_field
    if support_field is not None and first.field_names.index(support_field) in channel_ids:
        raise ValueError(
            f"{first.support_field!r} is the hidden support target and cannot be observed"
        )
    count_ranges = _sensor_ranges(first.field_names, channel_ids, sensors_per_channel)
    n_points = first.values.shape[0]
    n_queries = n_points if query_count is None else int(query_count)
    if n_queries <= 0 or n_queries > n_points:
        raise ValueError(f"query_count must be in [1, {n_points}]")
    if any(maximum > n_points for _, maximum in count_ranges):
        raise ValueError(f"sensor counts cannot exceed the {n_points} native points")
    # Samples with prescribed sensor points draw only from them. Requesting more
    # sensors than a sample has candidates raises an error.
    candidate_pools = [
        None if sample.sensor_mask is None else np.flatnonzero(sample.sensor_mask.numpy())
        for sample in samples
    ]
    largest_request = max(maximum for _, maximum in count_ranges)
    for sample, pool in zip(samples, candidate_pools, strict=True):
        if pool is not None and largest_request > pool.size:
            raise ValueError(
                f"{sample.sample_id} has {pool.size} sensor candidates; "
                f"{largest_request} sensors were requested"
            )

    rng = np.random.default_rng(seed)
    # Query points and sensors are sampled independently. The complete target
    # vector is retained at each query even when only one field is observed.
    query_indices = np.stack(
        [np.sort(rng.choice(n_points, size=n_queries, replace=False)) for _ in samples]
    )
    # Draw a separate sensor count for every sample and observed channel, then
    # pad only to this batch's maximum number of marked observations.
    counts_by_sample: list[list[int]] = []
    for _ in samples:
        counts = []
        for minimum, maximum in count_ranges:
            count = minimum if minimum == maximum else int(rng.integers(minimum, maximum + 1))
            counts.append(count)
        counts_by_sample.append(counts)

    max_observations = max(sum(counts) for counts in counts_by_sample)
    obs_indices = np.full((len(samples), max_observations), -1, dtype=np.int64)
    obs_field_ids = np.full((len(samples), max_observations), -1, dtype=np.int64)
    obs_mask = np.zeros((len(samples), max_observations), dtype=bool)
    for batch_id, counts in enumerate(counts_by_sample):
        offset = 0
        pool = candidate_pools[batch_id]
        for channel_id, count in zip(channel_ids, counts, strict=True):
            population = n_points if pool is None else pool
            selection = np.sort(rng.choice(population, size=count, replace=False))
            obs_indices[batch_id, offset : offset + count] = selection
            obs_field_ids[batch_id, offset : offset + count] = channel_id
            obs_mask[batch_id, offset : offset + count] = True
            offset += count

    # Hard sensor replacement is legal only if all active sensor points occur
    # in the selected query set. Subset queries often do not satisfy this.
    q_to_slot = [
        {int(source_index): slot for slot, source_index in enumerate(row)}
        for row in query_indices
    ]
    mapped = np.full((len(samples), max_observations), -1, dtype=np.int64)
    all_observations_are_queries = True
    for batch_id in range(len(samples)):
        n_active = sum(counts_by_sample[batch_id])
        for obs_slot, source_index in enumerate(obs_indices[batch_id, :n_active]):
            slot = q_to_slot[batch_id].get(int(source_index))
            if slot is None:
                all_observations_are_queries = False
            else:
                mapped[batch_id, obs_slot] = slot

    obs_coords = torch.zeros(
        (len(samples), max_observations, first.coordinates.shape[1]), dtype=torch.float32
    )
    obs_values = torch.zeros((len(samples), max_observations, 1), dtype=torch.float32)
    query_coords, target_fields = [], []
    for batch_id, sample in enumerate(samples):
        n_active = sum(counts_by_sample[batch_id])
        selected_obs = obs_indices[batch_id, :n_active]
        selected_fields = obs_field_ids[batch_id, :n_active]
        values = sample.normalized_values
        obs_coords[batch_id, :n_active] = sample.coordinates[selected_obs]
        obs_values[batch_id, :n_active, 0] = values[selected_obs, selected_fields]
        selected_queries = query_indices[batch_id]
        query_coords.append(sample.coordinates[selected_queries])
        target_fields.append(values[selected_queries])

    return ObservationBatch(
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_field_ids=torch.from_numpy(obs_field_ids),
        obs_mask=torch.from_numpy(obs_mask),
        query_coords=torch.stack(query_coords),
        target_fields=torch.stack(target_fields),
        query_mask=torch.ones((len(samples), n_queries), dtype=torch.bool),
        query_indices=torch.from_numpy(query_indices),
        obs_indices=torch.from_numpy(mapped) if all_observations_are_queries else None,
        field_names=first.field_names,
        stats=first.stats,
        sample_ids=tuple(sample.sample_id for sample in samples),
        resolution=first.resolution,
        logical_shape=first.logical_shape,
    )
