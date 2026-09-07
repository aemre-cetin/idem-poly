"""
idempotent-poly: Orthogonal Polynomial Tensors & Zero-Backprop Algebraic Solvers
================================================================================
Author: Dr. A. Emre ÇETİN
"""

from .chebyshev import ChebyshevTensorLayer, ChebyshevPolyFFN
from .algebraic import AlgebraicIdempotentSolver
from .surgeon import PolySurgeon, ResidualPolyMLP
from .polyformer import IdemPolyAttention, PolyFormerBlock

__version__ = "0.1.1"
__author__ = "Dr. A. Emre ÇETİN"

__all__ = [
    "ChebyshevTensorLayer",
    "ChebyshevPolyFFN",
    "AlgebraicIdempotentSolver",
    "PolySurgeon",
    "ResidualPolyMLP",
    "IdemPolyAttention",
    "PolyFormerBlock",
]

