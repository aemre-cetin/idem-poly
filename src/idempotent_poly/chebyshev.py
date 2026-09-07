"""
Chebyshev Orthogonal Polynomial Tensor Layers
=============================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebyshevTensorLayer(nn.Module):
    """
    Ortogonal Chebyshev polinom katsayı tensörü ile çalışan katman:
        y_j = sum_{i=1}^{d_in} sum_{k=0}^{K} C_{j, i, k} * T_k(norm(x_i))
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 3, normalize_input: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.normalize_input = normalize_input

        self.coefficients = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_features * (self.degree + 1)
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.coefficients, mean=0.0, std=std)
        with torch.no_grad():
            self.coefficients[:, :, 1] += torch.randn_like(self.coefficients[:, :, 1]) * 0.05

    def compute_chebyshev_basis(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = torch.tanh(x)

        t0 = torch.ones_like(x)
        if self.degree == 0:
            return t0.unsqueeze(-1)

        t1 = x
        basis = [t0, t1]
        for _ in range(1, self.degree):
            t_next = 2.0 * x * basis[-1] - basis[-2]
            basis.append(t_next)

        return torch.stack(basis[: self.degree + 1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = self.compute_chebyshev_basis(x)
        out = torch.einsum('...ik, oik -> ...o', T, self.coefficients)
        return out + self.bias


class ChebyshevPolyFFN(nn.Module):
    """
    Standart LLM SwiGLU / MLP bloğunun yerine geçen %50 sıkıştırılmış Chebyshev Polinom FFN katmanı.
    """

    def __init__(self, d_model: int, degree: int = 3):
        super().__init__()
        self.d_model = d_model
        self.degree = degree
        self.coefficients = nn.Parameter(torch.zeros(d_model, d_model, degree + 1))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.scale = nn.Parameter(torch.tensor(1.0))

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
        T = self.compute_chebyshev_basis(x)
        out = torch.einsum('...ik, oik -> ...o', T, self.coefficients)
        return (out + self.bias) * self.scale

