"""
Zero-Backprop Algebraic Idempotent Solver
==========================================
"""

from typing import Optional
import torch
import torch.nn as nn
from .chebyshev import ChebyshevTensorLayer


class AlgebraicIdempotentSolver(nn.Module):
    """
    Backprop olmadan, SVD ve Ridge alt-uzay projeksiyonu ile
    kapalı formda analitik çözücü.
    """

    def __init__(self, in_dim: int, out_dim: int, poly_degree: int = 3, l2_reg: float = 1e-4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.poly_degree = poly_degree
        self.l2_reg = l2_reg
        self.feature_dim = in_dim * (poly_degree + 1)

        self.register_buffer("projector_basis", torch.zeros(self.feature_dim, self.feature_dim))
        self.register_buffer("weight_matrix", torch.zeros(self.feature_dim, out_dim))
        self.register_buffer("bias", torch.zeros(out_dim))
        self.is_fitted = False

    def compute_chebyshev_basis(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.tanh(x)
        t0 = torch.ones_like(x_norm)
        if self.poly_degree == 0:
            return t0
        t1 = x_norm
        basis = [t0, t1]
        for _ in range(1, self.poly_degree):
            t_next = 2.0 * x_norm * basis[-1] - basis[-2]
            basis.append(t_next)
        stacked = torch.stack(basis[: self.poly_degree + 1], dim=-1)
        return stacked.reshape(x.size(0), -1)

    def fit(self, X: torch.Tensor, Y: torch.Tensor, rank: Optional[int] = None) -> float:
        import time
        start_time = time.time()
        device = X.device

        Phi = self.compute_chebyshev_basis(X)
        Y_mean = Y.mean(dim=0, keepdim=True)
        Y_centered = Y - Y_mean
        self.bias.copy_(Y_mean.squeeze(0))

        U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)
        V = Vh.mH

        if rank is None or rank > V.size(1):
            cum_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            k = torch.searchsorted(cum_energy, 0.99).item() + 1
            rank = max(min(k, V.size(1)), 1)

        V_r = V[:, :rank]
        Pi = V_r @ V_r.mH
        self.projector_basis.copy_(Pi)

        reg_I = self.l2_reg * torch.eye(self.feature_dim, device=device)
        A = Phi.T @ Phi + reg_I
        B = Phi.T @ Y_centered
        W = torch.linalg.solve(A, B)

        W_idempotent = Pi @ W
        self.weight_matrix.copy_(W_idempotent)
        self.is_fitted = True
        return time.time() - start_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmedi. Önce fit() çağrılmalıdır.")
        Phi = self.compute_chebyshev_basis(x)
        Phi_proj = Phi @ self.projector_basis
        return Phi_proj @ self.weight_matrix + self.bias

