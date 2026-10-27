"""Read transonic NACA airfoil flows on their 101 x 101 enclosing grid.

Example::

    dataset = AirfoilDataset("dataset/airfoil", split="test",
                             stats_path="dataset/airfoil.stats.pt")
    case = dataset[0]  # FieldSample with rho, u, v, p and the fluid mask

Each case is one Geo-FNO NACA flow resampled from its body-fitted grid. Static
pressure sensors sit on a fixed ellipse around the airfoil; the airfoil shape
is reconstructed through ``mask``. Mach is derived from the predicted state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..metrics import (
    airfoil_drag_coefficients,
    airfoil_mach,
    geometry_errors,
    support_relative_l2,
)
from .contracts import FieldSample, NormalizationStats
from .geometry import (
    EnclosingGrid,
    SupportGeometry,
    SupportMaskAdapter,
    ellipse_ring,
)
from .io import resolve_stats, seeded_case_split, stats_from_moments

AIRFOIL_FIELDS = ("rho", "u", "v", "p", "mask")
FLOW_FIELDS = AIRFOIL_FIELDS[:4]
AIRFOIL_SOURCE_CHANNELS = ("rho", "u", "v", "p", "Ma")
AIRFOIL_TRANSFORMS = ("log", "identity", "identity", "log", "identity")
AIRFOIL_FILES = {
    "fields": "NACA_Q_interp.npy",
    "mask": "NACA_mask_interp.npy",
    "x": "NACA_X_interp.npy",
    "y": "NACA_Y_interp.npy",
}
# Inside the airfoil, where the flow is undefined, density and pressure take the
# freestream-normalized value 1 (zero in log space) and the velocity is zero;
# no metric scores these points.
AIRFOIL_FILL_VALUES = {"rho": 1.0, "u": 0.0, "v": 0.0, "p": 1.0}
# Training cases whose stored channels are checked against the Mach identity.
CHANNEL_CHECK_CASES = 4


class AirfoilDataset(Dataset[FieldSample]):
    """Case-split airfoil flows: density, velocity, static pressure, fluid mask.

    The processed source stores ``[cases, 5, 101, 101]`` channels in the order
    ``rho, u, v, p, Ma`` (freestream-normalized, zero inside the airfoil) and a
    boolean fluid mask. The loader checks that order with the identity
    ``Ma = |(u, v)| / sqrt(1.4 p / rho)`` and drops the stored Mach channel.
    Density and pressure are standardized in log space, so reconstructions
    stay positive. Statistics are moments of the transformed channels over all
    points of the training cases, unless a statistics file is given.
    """

    support_field = "mask"

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        train_fraction: float = 0.8,
        split_seed: int = 42,
        stats: NormalizationStats | Mapping[str, object] | str | Path | None = None,
        stats_path: str | Path | None = None,
        ellipse_center: Sequence[float] = (0.5, 0.5),
        ellipse_semi_axes: Sequence[float] = (0.30, 0.12),
        ellipse_ring_halfwidth: float = 0.08,
        stats_case_chunk: int = 64,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if stats is not None and stats_path is not None:
            raise ValueError("pass either stats or stats_path, not both")
        if stats_case_chunk <= 0:
            raise ValueError("stats_case_chunk must be positive")
        root = Path(root)
        folder = root / "naca_interp_5f" if (root / "naca_interp_5f").is_dir() else root
        self._paths = {name: folder / filename for name, filename in AIRFOIL_FILES.items()}
        missing = [str(path) for path in self._paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing airfoil file(s): " + ", ".join(missing))

        fields = np.load(self._paths["fields"], mmap_mode="r")
        n_channels = len(AIRFOIL_SOURCE_CHANNELS)
        if fields.ndim != 4 or fields.shape[1] != n_channels:
            raise ValueError(
                f"expected [cases, 5, ny, nx] airfoil channels (rho, u, v, p, Ma), "
                f"got {fields.shape}"
            )
        self._fields: np.ndarray | None = fields
        n_cases, _, ny, nx = fields.shape
        fluid = np.load(self._paths["mask"])
        if fluid.shape != (n_cases, ny, nx) or fluid.dtype != np.bool_:
            raise ValueError("the airfoil mask must be a boolean [cases, ny, nx] array")
        self._fluid = fluid.reshape(n_cases, -1)
        self.grid = self._read_grid(n_cases, ny, nx)

        self.root = root
        self.split = split
        self.field_names = AIRFOIL_FIELDS
        self.logical_shape = self.grid.shape
        self.n_cases = int(n_cases)
        self.stats_case_chunk = int(stats_case_chunk)
        self.train_case_ids, self.test_case_ids = seeded_case_split(
            self.n_cases, train_fraction, split_seed
        )
        self.case_ids = self.train_case_ids if split == "train" else self.test_case_ids
        self._check_source_channels(self.train_case_ids[:CHANNEL_CHECK_CASES])

        # One sample-invariant ring for every case: the elliptic band, kept to
        # points that are fluid in every training case, so sensor positions
        # carry no information about an individual airfoil.
        ring = ellipse_ring(self.grid, ellipse_center, ellipse_semi_axes, ellipse_ring_halfwidth)
        self.sensor_ring = ring & self._fluid[self.train_case_ids].all(axis=0)
        if not self.sensor_ring.any():
            raise ValueError("the elliptic sensor ring contains no always-fluid points")
        self.ellipse = {
            "center": [float(value) for value in ellipse_center],
            "semi_axes": [float(value) for value in ellipse_semi_axes],
            "ring_halfwidth": float(ellipse_ring_halfwidth),
        }
        self._coordinates = torch.from_numpy(self.grid.normalized_coordinates())
        self._coordinates_raw = torch.from_numpy(self.grid.physical_coordinates())
        self._stats_source = stats
        self._stats_path = stats_path
        self._stats: NormalizationStats | None = None
        self._adapter: SupportMaskAdapter | None = None

    def _read_grid(self, n_cases: int, ny: int, nx: int) -> EnclosingGrid:
        """Read the physical grid; check it is uniform and equal in the first and last case."""
        axes = []
        for name in ("x", "y"):
            values = np.load(self._paths[name], mmap_mode="r")
            if values.shape != (n_cases, ny, nx):
                raise ValueError(f"airfoil {name} coordinates must be [cases, ny, nx]")
            first, last = np.asarray(values[0], np.float64), np.asarray(values[-1], np.float64)
            if not np.array_equal(first, last):
                raise ValueError("airfoil grid coordinates must be identical for every case")
            axes.append(first)
        x, y = axes
        grid = EnclosingGrid(
            shape=(ny, nx),
            x_range=(float(x.min()), float(x.max())),
            y_range=(float(y.min()), float(y.max())),
        )
        expected = grid.physical_coordinates()[:, :2]
        stored = np.stack([x.ravel(), y.ravel()], axis=-1)
        if not np.allclose(stored, expected, rtol=0.0, atol=1e-5):
            raise ValueError("airfoil coordinates must form a uniform row-major x-y grid")
        return grid

    def _array(self) -> np.ndarray:
        if self._fields is None:
            self._fields = np.load(self._paths["fields"], mmap_mode="r")
        return self._fields

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_fields"] = None
        return state

    def _check_source_channels(self, case_ids: np.ndarray) -> None:
        """Verify channel order from the Mach identity and the zero-filled airfoil."""
        for case_id in case_ids:
            values = np.asarray(self._array()[int(case_id)], np.float64)
            values = values.reshape(len(AIRFOIL_SOURCE_CHANNELS), -1)
            fluid = self._fluid[int(case_id)]
            if np.any(values[:, ~fluid] != 0):
                raise ValueError("airfoil source channels must be zero inside the airfoil")
            rho, u, v, p, mach = values[:, fluid]
            if np.any(rho <= 0) or np.any(p <= 0):
                raise ValueError("airfoil density and pressure must be positive in the fluid")
            residual = np.abs(airfoil_mach(rho, u, v, p) - mach)
            if float(np.percentile(residual, 99)) > 1e-2:
                raise ValueError(
                    "airfoil channels do not satisfy Ma = |(u, v)| / sqrt(1.4 p / rho); "
                    "expected source order rho, u, v, p, Ma"
                )

    def __len__(self) -> int:
        return int(self.case_ids.size)

    @property
    def stats(self) -> NormalizationStats:
        if self._stats is None:
            self._stats = resolve_stats(
                explicit=self._stats_source,
                stats_path=self._stats_path,
                field_names=self.field_names,
                compute=self._fit_train_stats,
                transforms=AIRFOIL_TRANSFORMS,
            )
        return self._stats

    def _fit_train_stats(self) -> NormalizationStats:
        """Stream transformed moments over every point of the training cases."""
        total = np.zeros(len(self.field_names))
        squares = np.zeros_like(total)
        count = 0
        for start in range(0, self.train_case_ids.size, self.stats_case_chunk):
            case_ids = self.train_case_ids[start : start + self.stats_case_chunk]
            source = np.asarray(self._array()[case_ids], np.float64)
            source = source.reshape(len(case_ids), len(AIRFOIL_SOURCE_CHANNELS), -1)
            fluid = self._fluid[case_ids]
            columns = []
            for channel, name in enumerate(FLOW_FIELDS):
                values = np.where(fluid, source[:, channel], AIRFOIL_FILL_VALUES[name])
                columns.append(np.log(values) if AIRFOIL_TRANSFORMS[channel] == "log" else values)
            block = np.stack([*columns, fluid.astype(np.float64)], axis=-1)
            total += block.sum(axis=(0, 1))
            squares += np.square(block).sum(axis=(0, 1))
            count += block.shape[0] * block.shape[1]
        return stats_from_moments(total, squares, count, self.field_names, AIRFOIL_TRANSFORMS)

    def __getitem__(self, index: int) -> FieldSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        case_id = int(self.case_ids[index])
        if self._adapter is None:
            self._adapter = SupportMaskAdapter(
                self.stats, mask_field=self.support_field, fill_values=AIRFOIL_FILL_VALUES
            )
        n_flow = len(FLOW_FIELDS)
        source = np.asarray(self._array()[case_id, :n_flow], np.float32).reshape(n_flow, -1).T
        flow = FieldSample(
            sample_id=f"airfoil:case:{case_id}",
            values=torch.from_numpy(np.ascontiguousarray(source)),
            coordinates=self._coordinates,
            coordinates_raw=self._coordinates_raw,
            field_names=FLOW_FIELDS,
            stats=self.stats.subset(FLOW_FIELDS),
            logical_shape=self.grid.shape,
            case_id=case_id,
            metadata={"dataset": "airfoil", "case_id": case_id, "split": self.split},
        )
        geometry = SupportGeometry(
            self.grid,
            support=self._fluid[case_id],
            sensor_candidates=self.sensor_ring,
            support_name="fluid",
            metadata={"sensor_rule": "fixed elliptic ring", "ellipse": self.ellipse},
        )
        return self._adapter.adapt(flow, geometry)

    def case_metrics(
        self,
        sample: FieldSample,
        target: np.ndarray,
        prediction: np.ndarray,
        query_indices: np.ndarray,
    ) -> dict[str, float]:
        """Derived Mach error, airfoil-shape errors, and drag coefficients for one case.

        Mach is computed from density, velocity, and pressure for reference and
        reconstruction alike. Drag needs every grid point in grid order and is
        NaN otherwise.
        """
        channels = {name: self.field_names.index(name) for name in self.field_names}
        support = np.asarray(target[:, channels["mask"]]) > 0.5

        def mach(values: np.ndarray) -> np.ndarray:
            return airfoil_mach(*(values[:, channels[name]] for name in FLOW_FIELDS))

        result = {
            "Ma_relative_l2": support_relative_l2(mach(target), mach(prediction), support),
            **geometry_errors(support, prediction[:, channels["mask"]]),
            "drag_coefficient_true": float("nan"),
            "drag_coefficient_pred": float("nan"),
        }
        if np.array_equal(query_indices, np.arange(self.grid.n_points)):
            drag = airfoil_drag_coefficients(
                target,
                prediction,
                sample.coordinates[:, :2].numpy(),
                self.grid.shape,
                (self.grid.x_range, self.grid.y_range),
                channels,
            )
            if drag is not None:
                result["drag_coefficient_true"], result["drag_coefficient_pred"] = drag
        return result


__all__ = ["AIRFOIL_FIELDS", "AIRFOIL_SOURCE_CHANNELS", "AirfoilDataset"]
