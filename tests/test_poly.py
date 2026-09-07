import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import torch.nn as nn
from idempotent_poly import (
    ChebyshevTensorLayer,
    ChebyshevPolyFFN,
    AlgebraicIdempotentSolver,
    PolySurgeon,
    ResidualPolyMLP,
    IdemPolyAttention,
    PolyFormerBlock,
)

def test_chebyshev_basis():
    layer = ChebyshevTensorLayer(in_features=4, out_features=4, degree=2)
    x = torch.tensor([[0.5, -0.5, 0.0, 1.0]])
    basis = layer.compute_chebyshev_basis(x)
    assert basis.shape == (1, 4, 3)
    assert torch.allclose(basis[..., 0], torch.ones_like(x))

def test_algebraic_idempotency():
    solver = AlgebraicIdempotentSolver(in_dim=16, out_dim=16, poly_degree=2)
    X = torch.randn(64, 16)
    Y = torch.randn(64, 16)
    dt = solver.fit(X, Y)
    Pi = solver.projector_basis
    rel_err = torch.norm(Pi @ Pi - Pi) / torch.norm(Pi)
    assert rel_err.item() < 1e-4

def test_poly_ffn_forward():
    ffn = ChebyshevPolyFFN(d_model=32, degree=2)
    x = torch.randn(4, 8, 32)
    out = ffn(x)
    assert out.shape == (4, 8, 32)

def test_poly_surgeon():
    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 64)
            self.fc2 = nn.Linear(64, 16)
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    mlp = DummyMLP()
    calib_X = torch.randn(30, 16)
    calib_Y = mlp(calib_X)
    converted = PolySurgeon.convert_layer_to_chebyshev(
        mlp, calib_X, calib_Y, d_model=16, degree=2, use_residual=True
    )
    x_test = torch.randn(5, 16)
    out = converted(x_test)
    assert out.shape == (5, 16)

def test_polyformer_block():
    block = PolyFormerBlock(d_model=32, n_heads=4, poly_degree=2)
    x = torch.randn(2, 10, 32)
    out = block(x)
    assert out.shape == (2, 10, 32)

if __name__ == '__main__':
    test_chebyshev_basis()
    print('PASS: test_chebyshev_basis')
    test_algebraic_idempotency()
    print('PASS: test_algebraic_idempotency')
    test_poly_ffn_forward()
    print('PASS: test_poly_ffn_forward')
    test_poly_surgeon()
    print('PASS: test_poly_surgeon')
    test_polyformer_block()
    print('PASS: test_polyformer_block')
    print('ALL 5 UNIT TESTS PASSED!')
