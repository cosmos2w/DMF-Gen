"""Read the paper's five-field combustion HDF5 without copying its payload.

Example::

    dataset = CombustionH5Dataset("dataset/combustion.h5", split="train",
                                  stats_path="dataset/combustion.stats.pt")
    state = dataset[0]  # FieldSample with CH4, CO, T, U_1, p targets

Training and test frames share the field order and training normalization.
The HDF5 handle opens lazily in each worker process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import FieldSample, NormalizationStats
from .io import decode_field_names, normalize_coordinates, resolve_stats, stats_from_moments

PAPER_COMBUSTION_FIELDS = ("CH4", "CO", "T", "U_1", "p")


class CombustionH5Dataset(Dataset[FieldSample]):
    """Frame-level reader for ``Merged_CH4COTU1P.h5``.

    The file remains on disk and is opened lazily in each worker process. The
    deterministic split is over time frames; statistics are fitted on the
    training frame IDs only and reused unchanged for test frames.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        train_fraction: float = 0.9,
        split_seed: int = 42,
        stats: NormalizationStats | Mapping[str, object] | str | Path | None = None,
        stats_path: str | Path | None = None,
        stats_frame_chunk: int = 16,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"combustion HDF5 file does not exist: {self.path}")
        self._check_split(split, train_fraction)
        if stats_frame_chunk <= 0:
            raise ValueError("stats_frame_chunk must be positive")
        if stats is not None and stats_path is not None:
            raise ValueError("pass either stats or stats_path, not both")
        self.split = split
        self.split_seed = int(split_seed)
        self.stats_frame_chunk = int(stats_frame_chunk)
        self._handle: h5py.File | None = None
        self._stats: NormalizationStats | None = None

        with h5py.File(self.path, "r") as handle:
            if "fields" not in handle or "coordinates" not in handle:
                raise ValueError("combustion file must contain fields and coordinates datasets")
            field_data = handle["fields"]
            if field_data.ndim != 6 or field_data.shape[0] != 1 or field_data.shape[3:5] != (1, 1):
                raise ValueError(
                    "expected combustion fields with shape [1, frames, points, 1, 1, channels]"
                )
            n_frames = field_data.shape[1]
            n_points = field_data.shape[2]
            n_fields = field_data.shape[5]
            attr_names = decode_field_names(field_data.attrs.get("selected_fields"))
            self.field_names = attr_names or PAPER_COMBUSTION_FIELDS
            if self.field_names != PAPER_COMBUSTION_FIELDS or n_fields != len(
                PAPER_COMBUSTION_FIELDS
            ):
                raise ValueError(
                    "paper combustion adapter requires fields in order CH4,CO,T,U_1,p; "
                    f"found names={self.field_names}, channels={n_fields}"
                )
            coordinates = np.asarray(handle["coordinates"], dtype=np.float32)
            self.times = np.asarray(handle["time"], dtype=np.float64) if "time" in handle else None

        self.coordinates_raw = coordinates.reshape(n_points, -1)
        if self.coordinates_raw.shape[0] != n_points:
            raise ValueError("combustion coordinate count does not match field point count")
        self.coordinates = normalize_coordinates(self.coordinates_raw)
        self.logical_shape = (n_points,)
        self.n_frames = int(n_frames)
        if self.times is not None and self.times.shape != (self.n_frames,):
            raise ValueError("combustion time vector must match the frame count")
        if self.n_frames < 2:
            raise ValueError("combustion train/test split requires at least two frames")

        # Partition frame IDs once; sensor draws later happen independently in
        # make_observation_batch and never determine this held-out split.
        permutation = np.random.default_rng(self.split_seed).permutation(self.n_frames)
        train_count = int(np.floor(float(train_fraction) * self.n_frames))
        train_count = min(max(train_count, 1), self.n_frames - 1)
        self.train_frame_ids = np.sort(permutation[:train_count]).astype(np.int64)
        self.test_frame_ids = np.sort(permutation[train_count:]).astype(np.int64)
        self.frame_ids = self.train_frame_ids if split == "train" else self.test_frame_ids
        self._stats_source = stats
        self._stats_path = stats_path

    @staticmethod
    def _check_split(split: str, train_fraction: float) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if not 0.0 < float(train_fraction) < 1.0:
            raise ValueError("train_fraction must be strictly between 0 and 1")

    def __len__(self) -> int:
        return int(self.frame_ids.size)

    def _file(self) -> h5py.File:
        if self._handle is None or not self._handle.id.valid:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None

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
        """Stream population moments over training frames only."""
        total = np.zeros(len(self.field_names), dtype=np.float64)
        sum_squares = np.zeros_like(total)
        point_count = 0
        with h5py.File(self.path, "r") as handle:
            fields = handle["fields"]
            for start in range(0, len(self.train_frame_ids), self.stats_frame_chunk):
                frame_ids = self.train_frame_ids[start : start + self.stats_frame_chunk]
                block = np.asarray(fields[0, frame_ids, :, 0, 0, :], dtype=np.float64)
                total += block.sum(axis=(0, 1), dtype=np.float64)
                sum_squares += np.square(block).sum(axis=(0, 1), dtype=np.float64)
                point_count += block.shape[0] * block.shape[1]
        return stats_from_moments(total, sum_squares, point_count, self.field_names)

    def __getitem__(self, index: int) -> FieldSample:
        """Read one frame and attach shared coordinates and statistics."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        frame_id = int(self.frame_ids[index])
        values = np.asarray(self._file()["fields"][0, frame_id, :, 0, 0, :], dtype=np.float32)
        time = float(self.times[frame_id]) if self.times is not None else None
        return FieldSample(
            sample_id=f"combustion:frame:{frame_id}",
            values=torch.from_numpy(values),
            coordinates=torch.from_numpy(self.coordinates),
            coordinates_raw=torch.from_numpy(self.coordinates_raw),
            field_names=self.field_names,
            stats=self.stats,
            resolution=None,
            logical_shape=self.logical_shape,
            frame_id=frame_id,
            time=time,
            metadata={"dataset": "paper_combustion", "frame_id": frame_id},
        )
