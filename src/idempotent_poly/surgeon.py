"""
PolySurgeon: In-Place Neural Weight Surgery Engine & Auto Architecture Converter
================================================================================
Automatic model architecture detection (Transformer, SSM/Mamba, Hybrid)
and in-situ zero-backpropagation idempotent polynomial / state-space surgery.

Protected under U.S. Patent Application Nos. 64/148,668, 64/149,520, 64/149,540.
Author: Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)
"""

from enum import Enum
from typing import List, Union, Dict, Any, Optional
import time
import torch
import torch.nn as nn
from .chebyshev import ChebyshevPolyFFN
from .ssm_bridge import HurwitzStabilityProjection, convert_attention_head_to_ssm


class ArchitectureType(str, Enum):
    TRANSFORMER = "transformer"
    SSM_MAMBA = "ssm_mamba"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class ModelArchitectureDetector:
    """
    Inspects model layers and modules to automatically detect the underlying architecture:
    - Transformer (Llama, Mistral, Qwen, DeepSeek, Gemma, SmolLM, etc.)
    - SSM / Mamba (Mamba, S4, S6, Falcon-Mamba, etc.)
    - Hybrid (Jamba, Zamba, etc.)
    """
    @staticmethod
    def detect(model: nn.Module) -> Dict[str, Any]:
        has_attention = False
        has_ssm = False
        attn_count = 0
        ssm_count = 0
        mlp_count = 0
        layer_count = 0

        # Look for layers in common attributes
        layers = None
        if hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
            layers = model.backbone.layers

        if layers is not None:
            layer_count = len(layers)
            for l in layers:
                if hasattr(l, "self_attn") or hasattr(l, "attn") or hasattr(l, "attention"):
                    has_attention = True
                    attn_count += 1
                if hasattr(l, "mixer") or hasattr(l, "ssm") or hasattr(l, "conv1d") or hasattr(l, "dt_proj"):
                    has_ssm = True
                    ssm_count += 1
                if hasattr(l, "mlp") or hasattr(l, "ffn") or hasattr(l, "feed_forward"):
                    mlp_count += 1
        else:
            # Fallback: scan all named modules
            for name, mod in model.named_modules():
                name_l = name.lower()
                if "self_attn" in name_l or "attention" in name_l:
                    has_attention = True
                    attn_count += 1
                if "mixer" in name_l or "ssm" in name_l or "mamba" in name_l or "conv1d" in name_l:
                    has_ssm = True
                    ssm_count += 1
                if "mlp" in name_l or "ffn" in name_l:
                    mlp_count += 1

        if has_attention and has_ssm:
            arch = ArchitectureType.HYBRID
        elif has_ssm:
            arch = ArchitectureType.SSM_MAMBA
        elif has_attention or mlp_count > 0:
            arch = ArchitectureType.TRANSFORMER
        else:
            arch = ArchitectureType.UNKNOWN

        return {
            "architecture": arch,
            "layer_count": layer_count,
            "has_attention": has_attention,
            "has_ssm": has_ssm,
            "attention_modules": attn_count,
            "ssm_modules": ssm_count,
            "mlp_modules": mlp_count,
        }


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
    dönüştüren temel cerrahi motoru.
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
            return poly_ffn


class AutoPolySurgeon:
    """
    Unified Automated Surgery Engine.
    Detects model architecture (Transformer, SSM/Mamba, Hybrid) automatically
    and executes appropriate in-situ polynomial and state-space transformations.
    """
    @classmethod
    def detect_architecture(cls, model: nn.Module) -> Dict[str, Any]:
        return ModelArchitectureDetector.detect(model)

    @classmethod
    def transform(
        cls,
        model: nn.Module,
        target_mode: str = "auto",
        degree: int = 3,
        convert_attention_to_ssm: bool = False,
        hurwitz_margin: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Executes automatic in-situ surgery based on detected architecture.
        
        Args:
            model: PyTorch model (Transformer, Mamba, or Hybrid)
            target_mode: "auto", "transformer", "ssm_mamba", or "hybrid"
            degree: Chebyshev polynomial degree
            convert_attention_to_ssm: If True and model is Transformer, maps attention heads to SSM states
            hurwitz_margin: Minimum dissipative margin for Hurwitz projection
        """
        t0 = time.perf_counter()
        meta = cls.detect_architecture(model)
        arch = meta["architecture"]
        
        if target_mode != "auto":
            arch = ArchitectureType(target_mode)

        layers_transformed = 0
        ssm_stabilized = 0
        attention_converted = 0

        layers = None
        if hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
            layers = model.backbone.layers

        # 1. TRANSFORMER SURGERY
        if arch in (ArchitectureType.TRANSFORMER, ArchitectureType.HYBRID):
            hidden_size = getattr(getattr(model, "config", None), "hidden_size", 256)
            if layers is not None:
                for layer in layers:
                    # MLP -> Chebyshev PolyFFN
                    if hasattr(layer, "mlp") and not isinstance(layer.mlp, ChebyshevPolyFFN):
                        layer.mlp = ChebyshevPolyFFN(d_model=hidden_size, degree=degree)
                        layers_transformed += 1
                    # Attention -> SSM Head Surgery (if requested)
                    if convert_attention_to_ssm and hasattr(layer, "self_attn"):
                        attn = layer.self_attn
                        if hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
                            ssm_res = convert_attention_head_to_ssm(
                                attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, d_state=16
                            )
                            # Register converted state matrices as non-trainable buffers on layer
                            layer.register_buffer("ssm_A_state", ssm_res["A"])
                            layer.register_buffer("ssm_B_proj", ssm_res["B_proj"])
                            layer.register_buffer("ssm_C_proj", ssm_res["C_proj"])
                            attention_converted += 1

        # 2. SSM / MAMBA SURGERY
        if arch in (ArchitectureType.SSM_MAMBA, ArchitectureType.HYBRID):
            hurwitz = HurwitzStabilityProjection(margin=hurwitz_margin)
            # Scan for state matrices (A, A_log, or mixer.A_log)
            for name, param in model.named_parameters():
                if "a_log" in name.lower() or name.endswith(".A"):
                    with torch.no_grad():
                        # If raw A: project directly; if A_log: project exp(A)
                        if "a_log" in name.lower():
                            A_eff = -torch.exp(param.data)
                            A_proj = hurwitz.project(A_eff)
                            if A_proj.dim() >= 2 and A_proj.shape[-1] == A_proj.shape[-2]:
                                pos_diag = (-A_proj.diag()).clamp(min=1e-3)
                                param.data.copy_(torch.log(pos_diag).repeat(param.shape[0], 1) if param.dim() > 1 else torch.log(pos_diag))
                            else:
                                pos_diag = (-A_proj).clamp(min=1e-3)
                                param.data.copy_(torch.log(pos_diag))
                        else:
                            param.data.copy_(hurwitz.project(param.data))
                    ssm_stabilized += 1

        elapsed = time.perf_counter() - t0
        return {
            "detected_architecture": arch.value,
            "layers_transformed": layers_transformed,
            "ssm_matrices_stabilized": ssm_stabilized,
            "attention_heads_converted_to_ssm": attention_converted,
            "elapsed_ms": elapsed * 1000.0,
            "status": "SUCCESS",
        }
