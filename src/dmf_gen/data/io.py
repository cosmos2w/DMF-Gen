"""Load field statistics, normalize coordinates, and split cases for lazy dataset adapters.

Example::

    stats = load_stats("dataset/combustion.stats.pt",
                       field_names=("CH4", "CO", "T", "U_1", "p"))
    model_coords = normalize_coordinates(raw_coords)  # one [0, 1] range per axis

Statistics must match the requested field order. Constant channels use unit
scale so normalization remains invertible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .contracts import NormalizationStats


def decode_field_names(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        names = tuple(
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in value  # type: ignore[union-attr]
        )
    return names or None


def normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Normalize each coordinate axis to [0, 1], mapping constant axes to 0."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape [points, dimensions]")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates contain non-finite values")
    lower = coordinates.min(axis=0, keepdims=True)
    span = coordinates.max(axis=0, keepdims=True) - lower
    safe_span = np.where(span > 0, span, 1.0)
    normalized = (coordinates - lower) / safe_span
    normalized[:, span.reshape(-1) == 0] = 0.0
    return normalized.astype(np.float32, copy=False)


def load_stats(
    source: NormalizationStats | Mapping[str, object] | str | Path,
    *,
    field_names: Sequence[str],
) -> NormalizationStats:
    """Load stats from a contract, mapping, `.npz`, or tensor-only `.pt` file."""
    if isinstance(source, NormalizationStats):
        stats = source
    elif isinstance(source, Mapping):
        stats = NormalizationStats.from_mapping(source, field_names=field_names)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"statistics file does not exist: {path}")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                data: dict[str, object] = {key: archive[key] for key in archive.files}
        elif path.suffix in {".pt", ".pth"}:
            data = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(data, Mapping):
                raise ValueError("PyTorch statistics file must contain a mapping")
        else:
            raise ValueError("statistics files must use .npz, .pt, or .pth")
        stats = NormalizationStats.from_mapping(data, field_names=field_names)
    if stats.field_names != tuple(field_names):
        raise ValueError(
            f"statistics fields {stats.field_names} do not match selected fields "
            f"{tuple(field_names)}"
        )
    return stats


def stats_from_moments(
    total: np.ndarray,
    sum_squares: np.ndarray,
    count: int,
    field_names: Sequence[str],
    transforms: Sequence[str] | None = None,
) -> NormalizationStats:
    """Population moments of (already transformed) training values."""
    if count <= 0:
        raise ValueError("cannot fit statistics on an empty training split")
    mean = total / count
    variance = np.maximum(sum_squares / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    # Constant channels remain invertible: subtracting their constant mean and
    # dividing by one maps them to zero without producing NaNs.
    std[std <= np.finfo(np.float64).eps] = 1.0
    return NormalizationStats(
        mean=torch.as_tensor(mean, dtype=torch.float32),
        std=torch.as_tensor(std, dtype=torch.float32),
        field_names=tuple(str(name) for name in field_names),
        transforms=None if transforms is None else tuple(transforms),
    )


def resolve_stats(
    *,
    explicit: NormalizationStats | Mapping[str, object] | str | Path | None,
    stats_path: str | Path | None,
    field_names: Sequence[str],
    compute,
    transforms: Sequence[str] | None = None,
) -> NormalizationStats:
    """Load or fit statistics that use the adapter's ``transforms``.

    Statistics recorded without transforms (mean and std only) adopt them;
    statistics recorded with different transforms are rejected.
    """
    if explicit is not None and stats_path is not None:
        raise ValueError("pass either stats or stats_path, not both")
    source = explicit if explicit is not None else stats_path
    stats = compute() if source is None else load_stats(source, field_names=field_names)
    if transforms is None:
        return stats
    required = tuple(str(value) for value in transforms)
    if stats.transforms is None:
        return NormalizationStats(stats.mean, stats.std, stats.field_names, required)
    if stats.transforms != required:
        raise ValueError(
            f"statistics transforms {stats.transforms} do not match the adapter's {required}"
        )
    return stats


def seeded_case_split(
    n_cases: int, train_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted train/test case IDs from one seeded shuffle of all cases.

    The first ``int(n_cases * train_fraction)`` shuffled cases train; the rest
    are held out. This is the rule behind the paper's 80/20, seed-42 splits.
    """
    if n_cases < 2:
        raise ValueError("a case split requires at least two cases")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    order = np.arange(n_cases, dtype=np.int64)
    np.random.default_rng(int(seed)).shuffle(order)
    n_train = min(max(int(n_cases * float(train_fraction)), 1), n_cases - 1)
    return np.sort(order[:n_train]), np.sort(order[n_train:])
