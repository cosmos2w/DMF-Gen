"""Read hyperelastic unit cells with a void on their 41 x 41 enclosing grid.

Example::

    dataset = ElasticityDataset("dataset/elasticity", split="test",
                                stats_path="dataset/elasticity.stats.pt")
    case = dataset[0]  # FieldSample with sigma and the material mask

Each case is one Geo-FNO elasticity unit cell. Stress sensors lie in the
material next to the void; the void shape is reconstructed through ``mask``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from ..metrics import geometry_errors, stress_concentration
from .contracts import FieldSample, NormalizationStats
from .geometry import (
    EnclosingGrid,
    SupportGeometry,
    SupportMaskAdapter,
    support_boundary_band,
)
from .io import resolve_stats, seeded_case_split, stats_from_moments

ELASTICITY_FIELDS = ("sigma", "mask")
ELASTICITY_FILES = {
    "sigma": "Random_UnitCell_sigma_10_interp.npy",
    "mask": "Random_UnitCell_mask_10_interp.npy",
}


class ElasticityDataset(Dataset[FieldSample]):
    """Case-split unit cells: von Mises stress ``sigma`` and material ``mask``.

    The source arrays are ``[41, 41, cases]`` on the unit square, rows along
    ``y``. ``mask`` is one in the material and zero in the void, where the
    stored stress is zero. Sensors observe ``sigma`` at material points within
    ``sensor_band_cells`` four-connected grid steps of the void. Statistics are
    population moments of both channels over all points of the training cases,
    unless a statistics file is given.
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
        sensor_band_cells: int = 3,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if stats is not None and stats_path is not None:
            raise ValueError("pass either stats or stats_path, not both")
        root = Path(root)
        folder = root / "Interp" if (root / "Interp").is_dir() else root
        paths = {name: folder / filename for name, filename in ELASTICITY_FILES.items()}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing elasticity file(s): " + ", ".join(missing))
        stress = np.load(paths["sigma"], mmap_mode="r")
        mask = np.load(paths["mask"], mmap_mode="r")
        if stress.ndim != 3 or stress.shape != mask.shape:
            raise ValueError(
                f"expected matching [ny, nx, cases] stress and mask arrays, got "
                f"{stress.shape} and {mask.shape}"
            )
        ny, nx, n_cases = stress.shape
        # Cases are stored last; keep one [cases, points] copy of each channel.
        self._stress = np.ascontiguousarray(np.asarray(stress).reshape(-1, n_cases).T, np.float32)
        mask_values = np.asarray(mask).reshape(-1, n_cases).T
        if not np.isin(mask_values, (0, 1)).all():
            raise ValueError("the elasticity mask must contain only 0 (void) and 1 (material)")
        self._material = mask_values.astype(bool)
        if not np.isfinite(self._stress).all():
            raise ValueError("elasticity stress contains non-finite values")
        if np.any(self._stress[~self._material] != 0):
            raise ValueError("elasticity void points must store zero stress")

        self.root = root
        self.split = split
        self.field_names = ELASTICITY_FIELDS
        self.grid = EnclosingGrid(shape=(ny, nx), x_range=(0.0, 1.0), y_range=(0.0, 1.0))
        self.logical_shape = self.grid.shape
        self.sensor_band_cells = int(sensor_band_cells)
        self.n_cases = int(n_cases)
        self.train_case_ids, self.test_case_ids = seeded_case_split(
            self.n_cases, train_fraction, split_seed
        )
        self.case_ids = self.train_case_ids if split == "train" else self.test_case_ids
        self._coordinates = torch.from_numpy(self.grid.normalized_coordinates())
        self._coordinates_raw = torch.from_numpy(self.grid.physical_coordinates())
        self._stats_source = stats
        self._stats_path = stats_path
        self._stats: NormalizationStats | None = None
        self._adapter: SupportMaskAdapter | None = None

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
            )
        return self._stats

    def _fit_train_stats(self) -> NormalizationStats:
        stress = self._stress[self.train_case_ids].astype(np.float64)
        material = self._material[self.train_case_ids].astype(np.float64)
        total = np.asarray([stress.sum(), material.sum()])
        squares = np.asarray([np.square(stress).sum(), np.square(material).sum()])
        return stats_from_moments(total, squares, stress.size, self.field_names)

    def sensor_candidates(self, case_id: int) -> np.ndarray:
        """Material points within the configured band around the case's void."""
        return support_boundary_band(
            self._material[case_id], self.grid.shape, self.sensor_band_cells
        )

    def __getitem__(self, index: int) -> FieldSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        case_id = int(self.case_ids[index])
        if self._adapter is None:
            self._adapter = SupportMaskAdapter(self.stats, mask_field=self.support_field)
        stress = FieldSample(
            sample_id=f"elasticity:case:{case_id}",
            values=torch.from_numpy(self._stress[case_id, :, None].copy()),
            coordinates=self._coordinates,
            coordinates_raw=self._coordinates_raw,
            field_names=("sigma",),
            stats=self.stats.subset(("sigma",)),
            logical_shape=self.grid.shape,
            case_id=case_id,
            metadata={"dataset": "elasticity", "case_id": case_id, "split": self.split},
        )
        geometry = SupportGeometry(
            self.grid,
            support=self._material[case_id],
            sensor_candidates=self.sensor_candidates(case_id),
            support_name="material",
            metadata={
                "sensor_rule": f"material points within {self.sensor_band_cells} grid steps "
                "of the void"
            },
        )
        return self._adapter.adapt(stress, geometry)

    def case_metrics(
        self,
        sample: FieldSample,
        target: np.ndarray,
        prediction: np.ndarray,
        query_indices: np.ndarray,
    ) -> dict[str, float]:
        """Void-shape errors and the stress-concentration factor for one case."""
        del sample, query_indices
        sigma, mask = (self.field_names.index(name) for name in ("sigma", "mask"))
        support = np.asarray(target[:, mask]) > 0.5
        kt_true, peak_true = stress_concentration(target[:, sigma], support)
        kt_pred, peak_pred = stress_concentration(prediction[:, sigma], support)
        return {
            **geometry_errors(support, prediction[:, mask]),
            "stress_concentration_true": kt_true,
            "stress_concentration_pred": kt_pred,
            "peak_stress_true": peak_true,
            "peak_stress_pred": peak_pred,
        }


__all__ = ["ELASTICITY_FIELDS", "ElasticityDataset"]
