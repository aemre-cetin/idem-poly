import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import torch
import torch.nn as nn
from idempotent_poly.adaptive_surgeon import (
    RankChebyshevTensorLayer,
    RayleighRitzAdaptiveSurgeon,
    BordaSurgeryFusion,
)
from idempotent_poly.chebyshev import ChebyshevPolyFFN


def test_rank_chebyshev_layer_forward():
    layer = RankChebyshevTensorLayer(in_features=16, out_features=16, degree=3)
    x = torch.randn(4, 10, 16)
    out = layer(x)
    assert out.shape == (4, 10, 16)

    # Test outlier immunity (magnitude spike)
    x_outlier = x.clone()
    x_outlier[0, 0, 0] = 1000.0 # extreme activation spike
    out_outlier = layer(x_outlier)
    assert not torch.isnan(out_outlier).any()
    assert not torch.isinf(out_outlier).any()


def test_rayleigh_ritz_spectrum_analysis():
    # Construct synthetic activations with low rank structure
    torch.manual_seed(42)
    B, D = 64, 32
    # 4 dominant directions + low noise
    true_latent = torch.randn(B, 4)
    proj = torch.randn(4, D)
    X = true_latent @ proj + 0.01 * torch.randn(B, D)

    analysis = RayleighRitzAdaptiveSurgeon.analyze_layer_spectrum(X, energy_threshold=0.95)
    assert "optimal_rank" in analysis
    assert "recommended_degree" in analysis
    assert analysis["optimal_rank"] >= 4
    assert analysis["optimal_rank"] < D
    assert analysis["recommended_degree"] in [2, 3, 4]


def test_adaptive_chebyshev_ffn_fit():
    class DummyMLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 2)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(dim * 2, dim)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    dim = 16
    mlp = DummyMLP(dim)
    calib_X = torch.randn(50, dim)
    calib_Y = mlp(calib_X)

    poly_ffn, meta = RayleighRitzAdaptiveSurgeon.fit_adaptive_chebyshev_ffn(
        mlp, calib_X, calib_Y, d_model=dim
    )
    assert isinstance(poly_ffn, ChebyshevPolyFFN)
    assert "chosen_degree" in meta
    assert "chosen_rank" in meta

    test_x = torch.randn(8, dim)
    test_out = poly_ffn(test_x)
    assert test_out.shape == (8, dim)


def test_borda_surgery_fusion():
    # 3 domain weight tensors (e.g. Math, Code, NLP)
    torch.manual_seed(123)
    shape = (16, 16, 4)
    w_math = torch.randn(shape)
    w_code = torch.randn(shape)
    w_nlp = torch.randn(shape)

    fused = BordaSurgeryFusion.fuse_domain_coefficients(
        [w_math, w_code, w_nlp],
        domain_weights=[0.4, 0.3, 0.3]
    )
    assert fused.shape == shape
    assert not torch.isnan(fused).any()

    # Verify single domain identity
    single_fused = BordaSurgeryFusion.fuse_domain_coefficients([w_math])
    assert torch.allclose(single_fused, w_math)


def test_young_factorized_poly_ffn():
    from idempotent_poly import YoungFactorizedPolyFFN

    d_model = 128
    tile_size = 32 # 4 tiles
    ffn = YoungFactorizedPolyFFN(d_model=d_model, tile_size=tile_size, degree=3)

    # Monolithic FFN would have 128 x 128 x 4 = 65,536 params
    # Young factorized FFN has 4 x (32 x 32 x 4) = 16,384 params (75% savings)
    mono_params = d_model * d_model * 4
    actual_params = ffn.tile_coefficients.numel()
    assert actual_params == mono_params // 4

    # Test forward pass
    x = torch.randn(2, 8, d_model)
    out = ffn(x)
    assert out.shape == (2, 8, d_model)
    assert not torch.isnan(out).any()
