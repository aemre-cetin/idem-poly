"""
idempotent-poly: Orthogonal Polynomial Tensors & Zero-Backprop Algebraic Solvers
================================================================================
Author: Dr. A. Emre ÇETİN
"""

from .chebyshev import ChebyshevTensorLayer, ChebyshevPolyFFN
from .algebraic import AlgebraicIdempotentSolver
from .surgeon import (
    PolySurgeon,
    AutoPolySurgeon,
    ModelArchitectureDetector,
    ArchitectureType,
    ResidualPolyMLP,
)
from .polyformer import IdemPolyAttention, PolyFormerBlock
from .idemformer_engine import (
    IdemFormerEngine,
    SubspaceKVCompactor,
    TarskiFixpointVerifier,
    TropicalAttentionEvaluator,
)
from .ssm_bridge import (
    make_hippo_chebyshev_matrix,
    HurwitzStabilityProjection,
    convert_attention_head_to_ssm,
)
from .adaptive_surgeon import (
    RayleighRitzAdaptiveSurgeon,
    BordaSurgeryFusion,
    RankChebyshevTensorLayer,
)
from .young_poly import YoungFactorizedPolyFFN

__version__ = "0.1.3"
__author__ = "Dr. A. Emre ÇETİN"

__all__ = [
    "ChebyshevTensorLayer",
    "ChebyshevPolyFFN",
    "AlgebraicIdempotentSolver",
    "PolySurgeon",
    "AutoPolySurgeon",
    "ModelArchitectureDetector",
    "ArchitectureType",
    "ResidualPolyMLP",
    "IdemPolyAttention",
    "PolyFormerBlock",
    "IdemFormerEngine",
    "SubspaceKVCompactor",
    "TarskiFixpointVerifier",
    "TropicalAttentionEvaluator",
    "make_hippo_chebyshev_matrix",
    "HurwitzStabilityProjection",
    "convert_attention_head_to_ssm",
    "RayleighRitzAdaptiveSurgeon",
    "BordaSurgeryFusion",
    "RankChebyshevTensorLayer",
    "YoungFactorizedPolyFFN",
]
