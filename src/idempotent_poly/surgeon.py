"""
PolySurgeon: In-Place Neural Weight Surgery Engine
==================================================
"""

from typing import List, Union
import torch
import torch.nn as nn
from .chebyshev import ChebyshevPolyFFN


class ResidualPolyMLP(nn.Module):
    """
    Orijinal MLP'nin ana omurgasını koruyup,
    Chebyshev Polinomik İdempotent Düzeltmesi ekleyen hibrit cerrahi katmanı.
    """

    def __init__(self, original_mlp: nn.Module, d_model: int, degree: int = 2):
        super().__init__()
        self.d_model = d_model
        self.degree = degree
        self.orig_mlp = original_mlp
        self.poly_coeffs = nn.Parameter(torch.zeros(d_model, d_model, degree + 1))
        self.gate = nn.Parameter(torch.tensor(0.5))

    def compute_chebyshev_basis(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.tanh(x)
        t0 = torch.ones_like(x_norm)
        if self.degree == 0:
            return t0.unsqueeze(-1)
        t1 = x_norm
        basis = [t0, t1]
        for _ in range(1, self.degree):
            t_next = 2.0 * x_norm * basis[-1] - basis[-2]
            basis.append(t_next)
        return torch.stack(basis[: self.degree + 1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.orig_mlp(x)
        T = self.compute_chebyshev_basis(x)
        poly_out = torch.einsum('...ik, oik -> ...o', T, self.poly_coeffs)
        return base_out + self.gate * poly_out

    def fit_algebraic_residual(self, X: torch.Tensor, Y: torch.Tensor, l2_reg: float = 1e-2):
        with torch.no_grad():
            base_out = self.orig_mlp(X)
            residual_target = Y - base_out

        N, D = X.shape
        device = X.device
        dtype = X.dtype

        T = self.compute_chebyshev_basis(X.float())
        Phi = T.reshape(N, -1)
        feat_dim = Phi.shape[1]

        reg = l2_reg * torch.eye(feat_dim, device=device)
        A = Phi.T @ Phi + reg
        B = Phi.T @ residual_target.float()
        W_flat = torch.linalg.solve(A, B)

        U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)
        k = int(Vh.size(0) * 0.90)
        V_r = Vh[:k, :].mH
        Pi = V_r @ V_r.mH
        W_idempotent = Pi @ W_flat

        W_reshaped = W_idempotent.T.reshape(D, D, self.degree + 1)
        with torch.no_grad():
            self.poly_coeffs.copy_(W_reshaped.to(dtype))


class PolySurgeon:
    """
    Herhangi bir HuggingFace modelini tek satırda polinom katmanlarına
    dönüştüren otomatik cerrahi motoru.
    """

    @staticmethod
    def convert_layer_to_chebyshev(
        layer_mlp: nn.Module,
        calibration_X: torch.Tensor,
        calibration_Y: torch.Tensor,
        d_model: int,
        degree: int = 3,
        use_residual: bool = True
    ) -> nn.Module:
        device = calibration_X.device
        if use_residual:
            res_poly = ResidualPolyMLP(layer_mlp, d_model=d_model, degree=2).to(device)
            res_poly.fit_algebraic_residual(calibration_X, calibration_Y)
            return res_poly
        else:
            poly_ffn = ChebyshevPolyFFN(d_model=d_model, degree=degree).to(device)
            # direct fit
            return poly_ffn

