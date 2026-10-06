"""Lazy task adapters and one marked-observation contract for DMF-Gen.

Example::

    dataset = CombustionH5Dataset("dataset/combustion.h5", split="train",
                                  stats_path="dataset/combustion.stats.pt")
    batch = make_observation_batch([dataset[0]], ["T"], 256)

Changing-geometry adapters return the same ``FieldSample`` with a support-mask
target channel and per-case sensor candidates.
"""

from .airfoil import AIRFOIL_FIELDS, AirfoilDataset
from .combustion import PAPER_COMBUSTION_FIELDS, CombustionH5Dataset
from .contracts import (
    FieldSample,
    NormalizationStats,
    ObservationBatch,
    group_samples_by_resolution,
    make_observation_batch,
)
from .elasticity import ELASTICITY_FIELDS, ElasticityDataset
from .geometry import (
    EnclosingGrid,
    GeometryAdapter,
    SupportGeometry,
    SupportMaskAdapter,
    UnsupportedGeometryAdapter,
)
from .pdebench import PDEBENCH_CFD_FIELDS, PDEBenchCFDDataset, PDEBenchSampleRecord

__all__ = [
    "AIRFOIL_FIELDS",
    "AirfoilDataset",
    "CombustionH5Dataset",
    "ELASTICITY_FIELDS",
    "ElasticityDataset",
    "EnclosingGrid",
    "FieldSample",
    "GeometryAdapter",
    "NormalizationStats",
    "ObservationBatch",
    "PAPER_COMBUSTION_FIELDS",
    "PDEBENCH_CFD_FIELDS",
    "PDEBenchCFDDataset",
    "PDEBenchSampleRecord",
    "SupportGeometry",
    "SupportMaskAdapter",
    "UnsupportedGeometryAdapter",
    "group_samples_by_resolution",
    "make_observation_batch",
]
