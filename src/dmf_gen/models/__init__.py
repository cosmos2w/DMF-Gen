"""Choose a paper-correlated GL-RBF profile through the public factory.

Example::

    flow = build_pointcloud_model({"model_name": "GL_rbf", "coord_dim": 3}, n_fields=5)

``GL_rbf`` and ``GL_rbf_ENH`` share one configurable backbone class; the
factory sets their different feature and readout defaults.
"""

from .factory import build_pointcloud_model
from .gl_rbf import ConditionalPointHybridLocalGlobalRBF
from .gl_rbf_cq import ConditionalPointHybridLocalGlobalRBFCQ
from .pointcloud_ffm import PointCloudFFM

GL_rbf = ConditionalPointHybridLocalGlobalRBF
GL_rbf_ENH = ConditionalPointHybridLocalGlobalRBF
GL_rbf_CQ = ConditionalPointHybridLocalGlobalRBFCQ
GL_rbf_ENH_CQ = ConditionalPointHybridLocalGlobalRBFCQ

__all__ = [
    "ConditionalPointHybridLocalGlobalRBF",
    "ConditionalPointHybridLocalGlobalRBFCQ",
    "GL_rbf",
    "GL_rbf_CQ",
    "GL_rbf_ENH",
    "GL_rbf_ENH_CQ",
    "PointCloudFFM",
    "build_pointcloud_model",
]
