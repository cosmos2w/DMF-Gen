"""Read processed PDEBench CFD at native L, M, or H query resolution.

Example::

    dataset = PDEBenchCFDDataset("dataset/pdebench/Processed", split="train",
                                 selected_fields=("Vx",),
                                 resolution_fractions={"L": 1, "M": 1, "H": 1},
                                 stats_path="dataset/cfd_hml.stats.pt")
    sample = dataset[0]  # FieldSample; use indices_by_resolution for batching

Resolution allocation is by training case; frames of one case stay at one
resolution. The test split uses its configured evaluation resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import FieldSample, NormalizationStats
from .io import decode_field_names, normalize_coordinates, resolve_stats, stats_from_moments

PDEBENCH_CFD_FIELDS = ("Vx", "Vy", "density", "pressure")
_RESOLUTIONS = ("L", "M", "H")


@dataclass(frozen=True)
class PDEBenchSampleRecord:
    """Cheap index metadata that can be grouped without reading HDF5 fields."""

    dataset_index: int
    case_id: int
    frame_id: int
    resolution: str


def _normalized_resolution_fractions(
    fractions: Mapping[str, float],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name, fraction in fractions.items():
        key = str(name).upper()
        if key not in _RESOLUTIONS:
            raise ValueError(f"unknown resolution {name!r}; use L, M, and/or H")
        value = float(fraction)
        if not np.isfinite(value) or value < 0:
            raise ValueError("resolution fractions must be finite and nonnegative")
        normalized[key] = value
    if not normalized or sum(normalized.values()) <= 0:
        raise ValueError("resolution_fractions must contain positive fractions")
    total = sum(normalized.values())
    return {key: value / total for key, value in normalized.items() if value > 0}


def _assign_case_resolutions(
    case_ids: np.ndarray,
    fractions: Mapping[str, float],
    seed: int,
) -> dict[int, str]:
    """Assign exact largest-remainder counts after a seeded case permutation."""
    levels = list(fractions)
    quotas = np.asarray([fractions[level] * len(case_ids) for level in levels])
    counts = np.floor(quotas).astype(int)
    remainder = len(case_ids) - int(counts.sum())
    order = np.argsort(-(quotas - counts), kind="stable")
    counts[order[:remainder]] += 1
    order_ids = np.random.default_rng(seed).permutation(case_ids)
    result: dict[int, str] = {}
    start = 0
    for level, count in zip(levels, counts, strict=True):
        for case_id in order_ids[start : start + int(count)]:
            result[int(case_id)] = level
        start += int(count)
    return result


class PDEBenchCFDDataset(Dataset[FieldSample]):
    """Case-split snapshots from processed ``CFD_{L,M,H}_res.h5`` files.

    Cases are split contiguously (first ``train_fraction`` for training, the
    remainder for testing) to match the released CFD processing workflow and
    avoid train/test leakage across frames of one case. When supplied,
    ``resolution_fractions`` deterministically assigns training cases to native
    L/M/H levels. Test cases use ``eval_resolution``. Normalization is fitted on
    training cases at H resolution and then shared across every resolution.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        train_fraction: float = 0.9,
        resolution_by_case: Mapping[int, str] | None = None,
        eval_resolution: str = "H",
        frame_indices: Sequence[int] | None = None,
        stats: NormalizationStats | Mapping[str, object] | str | Path | None = None,
        stats_path: str | Path | None = None,
        stats_frame_indices: Sequence[int] | None = None,
        stats_case_chunk: int = 8,
        selected_fields: Sequence[str] | None = None,
        resolution_fractions: Mapping[str, float] | None = None,
        resolution_seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.processed_dir = self.root if self.root.name == "Processed" else self.root / "Processed"
        self._check_split(split, train_fraction)
        if stats_case_chunk <= 0:
            raise ValueError("stats_case_chunk must be positive")
        if stats is not None and stats_path is not None:
            raise ValueError("pass either stats or stats_path, not both")
        eval_resolution = str(eval_resolution).upper()
        if eval_resolution not in _RESOLUTIONS:
            raise ValueError("eval_resolution must be L, M, or H")
        if resolution_by_case is not None and resolution_fractions is not None:
            raise ValueError("pass either resolution_by_case or resolution_fractions, not both")

        self.split = split
        self.train_fraction = float(train_fraction)
        self.eval_resolution = eval_resolution
        self.stats_case_chunk = int(stats_case_chunk)
        self._handle_by_resolution: dict[str, h5py.File] = {}
        self._stats: NormalizationStats | None = None
        self._stats_source = stats
        self._stats_path = stats_path

        self._paths = {
            level: self.processed_dir / f"CFD_{level}_res.h5" for level in _RESOLUTIONS
        }
        missing = [str(path) for path in self._paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing processed PDEBench file(s): " + ", ".join(missing))

        self._raw_coordinates: dict[str, np.ndarray] = {}
        self.coordinates: dict[str, np.ndarray] = {}
        self.times: dict[str, np.ndarray] = {}
        self._logical_shapes: dict[str, tuple[int, ...]] = {}
        metadata: dict[str, tuple[int, int, int, tuple[str, ...]]] = {}
        for level, path in self._paths.items():
            with h5py.File(path, "r") as handle:
                if "fields" not in handle or "coordinates" not in handle:
                    raise ValueError(f"{path} must contain fields and coordinates")
                fields = handle["fields"]
                if fields.ndim != 6 or fields.shape[3:5] != (1, 1):
                    raise ValueError(
                        f"expected {path.name} fields with shape "
                        "[cases, frames, points, 1, 1, channels]"
                    )
                attr_names = decode_field_names(fields.attrs.get("selected_fields"))
                source_names = attr_names or PDEBENCH_CFD_FIELDS
                n_cases = int(fields.shape[0])
                n_frames = int(fields.shape[1])
                n_points = int(fields.shape[2])
                n_fields = int(fields.shape[5])
                if len(source_names) != n_fields:
                    raise ValueError(
                        f"field name metadata does not match {path.name} channel count"
                    )
                coordinates = np.asarray(handle["coordinates"], dtype=np.float32).reshape(
                    n_points, -1
                )
                times = (
                    np.asarray(handle["time"], dtype=np.float64)
                    if "time" in handle
                    else np.arange(n_frames, dtype=np.float64)
                )
                if coordinates.shape[0] != n_points or times.shape != (n_frames,):
                    raise ValueError(f"coordinates/time dimensions do not match {path.name}")
                self._raw_coordinates[level] = coordinates
                self.coordinates[level] = normalize_coordinates(coordinates)
                self.times[level] = times
                metadata[level] = (n_cases, n_frames, n_points, source_names)

        metadata_values = list(metadata.values())
        if len({item[0] for item in metadata_values}) != 1 or len(
            {item[1] for item in metadata_values}
        ) != 1:
            raise ValueError("PDEBench L/M/H files must have the same case and frame counts")
        n_cases, self.n_frames = metadata_values[0][0], metadata_values[0][1]
        if len({item[3] for item in metadata_values}) != 1:
            raise ValueError("PDEBench L/M/H files must use the same field order")
        self.source_field_names = metadata_values[0][3]
        self.field_names = (
            tuple(selected_fields) if selected_fields is not None else self.source_field_names
        )
        if not self.field_names or len(set(self.field_names)) != len(self.field_names):
            raise ValueError("selected_fields must be nonempty and contain no duplicates")
        unknown = [name for name in self.field_names if name not in self.source_field_names]
        if unknown:
            raise ValueError(
                f"unknown selected field(s) {unknown}; available: {self.source_field_names}"
            )
        # Select Vx by verified HDF5 field name rather than an older raw-index
        # comment that mislabeled processed channel zero as density.
        self._field_indices = np.asarray(
            [self.source_field_names.index(name) for name in self.field_names], dtype=np.int64
        )
        self.n_cases = int(n_cases)
        for level, (_, _, points, _) in metadata.items():
            side = int(round(np.sqrt(points)))
            self._logical_shapes[level] = (side, side) if side * side == points else (points,)

        if self.n_cases < 2:
            raise ValueError("PDEBench train/test split requires at least two cases")
        n_train = int(np.floor(self.train_fraction * self.n_cases))
        n_train = min(max(n_train, 1), self.n_cases - 1)
        # Keep all frames of a case on one side of the split. Resolution labels
        # are assigned only within the training cases below.
        self.train_case_ids = np.arange(n_train, dtype=np.int64)
        self.test_case_ids = np.arange(n_train, self.n_cases, dtype=np.int64)
        self.case_ids = self.train_case_ids if split == "train" else self.test_case_ids

        self.frame_ids = self._validate_frame_indices(frame_indices, default_all=True)
        self.stats_frame_ids = np.sort(
            self._validate_frame_indices(stats_frame_indices, default_all=True)
        )

        if resolution_by_case is not None:
            case_levels = {
                int(case): str(level).upper() for case, level in resolution_by_case.items()
            }
            invalid_cases = sorted(set(case_levels) - set(self.train_case_ids.tolist()))
            if invalid_cases:
                raise ValueError(
                    "resolution_by_case contains non-training case IDs: "
                    f"{invalid_cases[:5]}"
                )
            invalid_levels = sorted(set(case_levels.values()) - set(_RESOLUTIONS))
            if invalid_levels:
                raise ValueError(f"unknown resolution level(s): {invalid_levels}")
            self.resolution_by_case = case_levels
        elif resolution_fractions is not None:
            fractions = _normalized_resolution_fractions(resolution_fractions)
            self.resolution_by_case = _assign_case_resolutions(
                self.train_case_ids, fractions, seed=int(resolution_seed)
            )
        else:
            self.resolution_by_case = {}

        self.sample_records: tuple[PDEBenchSampleRecord, ...] = self._make_records()
        grouped: dict[str, list[int]] = {level: [] for level in _RESOLUTIONS}
        for record in self.sample_records:
            grouped[record.resolution].append(record.dataset_index)
        self.indices_by_resolution = grouped

    @staticmethod
    def _check_split(split: str, train_fraction: float) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if not 0.0 < float(train_fraction) < 1.0:
            raise ValueError("train_fraction must be strictly between 0 and 1")

    def _validate_frame_indices(
        self, indices: Sequence[int] | None, *, default_all: bool
    ) -> np.ndarray:
        if indices is None:
            if default_all:
                return np.arange(self.n_frames, dtype=np.int64)
            return np.empty(0, dtype=np.int64)
        result = np.asarray(indices, dtype=np.int64).reshape(-1)
        if result.size == 0 or np.any(result < 0) or np.any(result >= self.n_frames):
            raise ValueError(f"frame indices must be nonempty and in [0, {self.n_frames})")
        if len(np.unique(result)) != len(result):
            raise ValueError("frame indices must not contain duplicates")
        return result

    def _make_records(self) -> tuple[PDEBenchSampleRecord, ...]:
        records = []
        for case_id in self.case_ids:
            case_id = int(case_id)
            if self.split == "train":
                level = self.resolution_by_case.get(case_id, "H")
            else:
                level = self.eval_resolution
            for frame_id in self.frame_ids:
                records.append(
                    PDEBenchSampleRecord(
                        dataset_index=len(records),
                        case_id=case_id,
                        frame_id=int(frame_id),
                        resolution=level,
                    )
                )
        return tuple(records)

    def __len__(self) -> int:
        return len(self.sample_records)

    def _file(self, resolution: str) -> h5py.File:
        handle = self._handle_by_resolution.get(resolution)
        if handle is None or not handle.id.valid:
            handle = h5py.File(self._paths[resolution], "r")
            self._handle_by_resolution[resolution] = handle
        return handle

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handle_by_resolution"] = {}
        return state

    def close(self) -> None:
        handles = getattr(self, "_handle_by_resolution", {})
        for handle in handles.values():
            handle.close()
        handles.clear()

    def __del__(self) -> None:
        self.close()

    @property
    def stats(self) -> NormalizationStats:
        if self._stats is None:
            self._stats = resolve_stats(
                explicit=self._stats_source,
                stats_path=self._stats_path,
                field_names=self.field_names,
                compute=self._fit_train_stats,
            )
        return self._stats

    def _fit_train_stats(self) -> NormalizationStats:
        """Fit shared field statistics from H-resolution training cases only."""
        total = np.zeros(len(self.field_names), dtype=np.float64)
        sum_squares = np.zeros_like(total)
        point_count = 0
        with h5py.File(self._paths["H"], "r") as handle:
            fields = handle["fields"]
            for start in range(0, len(self.train_case_ids), self.stats_case_chunk):
                stop = min(start + self.stats_case_chunk, len(self.train_case_ids))
                case_start = int(self.train_case_ids[start])
                case_stop = int(self.train_case_ids[stop - 1]) + 1
                block = np.asarray(
                    fields[case_start:case_stop, self.stats_frame_ids, :, 0, 0, :],
                    dtype=np.float64,
                )[..., self._field_indices]
                total += block.sum(axis=(0, 1, 2), dtype=np.float64)
                sum_squares += np.square(block).sum(axis=(0, 1, 2), dtype=np.float64)
                point_count += block.shape[0] * block.shape[1] * block.shape[2]
        return stats_from_moments(total, sum_squares, point_count, self.field_names)

    def __getitem__(self, index: int) -> FieldSample:
        """Read one case/frame at its assigned native grid resolution."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record = self.sample_records[index]
        values = np.asarray(
            self._file(record.resolution)["fields"][record.case_id, record.frame_id, :, 0, 0, :],
            dtype=np.float32,
        )[:, self._field_indices]
        return FieldSample(
            sample_id=f"pdebench:case:{record.case_id}:frame:{record.frame_id}:{record.resolution}",
            values=torch.from_numpy(np.ascontiguousarray(values)),
            coordinates=torch.from_numpy(self.coordinates[record.resolution]),
            coordinates_raw=torch.from_numpy(self._raw_coordinates[record.resolution]),
            field_names=self.field_names,
            stats=self.stats,
            resolution=record.resolution,
            logical_shape=self._logical_shapes[record.resolution],
            case_id=record.case_id,
            frame_id=record.frame_id,
            time=float(self.times[record.resolution][record.frame_id]),
            metadata={
                "dataset": "pdebench_cfd",
                "case_id": record.case_id,
                "frame_id": record.frame_id,
                "resolution": record.resolution,
            },
        )
