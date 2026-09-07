"""
PolyFormer: Idempotent Polynomial Transformer Block
===================================================
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .chebyshev import ChebyshevTensorLayer


class IdemPolyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, poly_degree: int = 2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.poly_degree = poly_degree

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.kernel_weights = nn.Parameter(torch.ones(poly_degree + 1) / (poly_degree + 1))

    def chebyshev_kernel(self, S: torch.Tensor) -> torch.Tensor:
        S_clamped = torch.tanh(S)
        t0 = torch.ones_like(S_clamped)
        if self.poly_degree == 0:
            return t0
        t1 = S_clamped
        basis = [t0, t1]
        for _ in range(1, self.poly_degree):
            t_next = 2.0 * S_clamped * basis[-1] - basis[-2]
            basis.append(t_next)
        stacked = torch.stack(basis[: self.poly_degree + 1], dim=-1)
        w = F.softmax(self.kernel_weights, dim=0)
        return torch.einsum('bhijk, k -> bhij', stacked, w)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        S = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            S = S.masked_fill(mask == 0, -1e4)

        K_mat = self.chebyshev_kernel(S)
        P = F.softmax(K_mat, dim=-1)

        out = torch.matmul(P, v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class PolyFormerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, poly_degree: int = 3):
        super().__init__()
        self.d_model = d_model
        self.attn = IdemPolyAttention(d_model=d_model, n_heads=n_heads, poly_degree=2)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = ChebyshevTensorLayer(in_features=d_model, out_features=d_model, degree=poly_degree)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_adaptive(self, x: torch.Tensor, max_iters: int = 5, tol: float = 0.01) -> Tuple[torch.Tensor, int]:
        curr = x
        steps = 0
        for step in range(1, max_iters + 1):
            next_state = self.forward(curr)
            diff = torch.norm(next_state - curr, p=2, dim=-1).mean().item()
            curr = next_state
            steps = step
            if diff < tol:
                break
        return curr, steps

