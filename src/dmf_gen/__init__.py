"""Public model API for direct measurement-to-field generation.

Example::

    flow = build_pointcloud_model({"model_name": "GL_rbf_ENH", "coord_dim": 3}, n_fields=5)

Dataset and marked-observation contracts are exported from ``dmf_gen.data``.
"""

from .models import (
    GL_rbf,
    GL_rbf_CQ,
    GL_rbf_ENH,
    GL_rbf_ENH_CQ,
    PointCloudFFM,
    build_pointcloud_model,
)
from .priors import IIDGaussianPrior, RFFGaussianPrior

__all__ = [
    "GL_rbf",
    "GL_rbf_CQ",
    "GL_rbf_ENH",
    "GL_rbf_ENH_CQ",
    "IIDGaussianPrior",
    "PointCloudFFM",
    "RFFGaussianPrior",
    "build_pointcloud_model",
]
