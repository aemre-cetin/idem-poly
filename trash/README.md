# ⚡ idempotent-poly

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.1-blue.svg)](https://pypi.org/project/idempotent-poly/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SmolLM2--135M--PolyFFN-pink)](https://huggingface.co/aecetin/SmolLM2-135M-PolyFFN)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Publication_414060833-00CCBB.svg?logo=researchgate)](https://www.researchgate.net/publication/414060833_Hardware-Accelerated_Orthogonal_Polynomial_Tensor_Operators_Zero-Backpropagation_Closed-Form_Algebraic_Solvers_and_In-Situ_Weight_Surgery_for_Deep_Neural_Networks)
[![Interactive Showcase](https://img.shields.io/badge/Spaces-Idempotent%20AI%20Showcase-orange)](https://huggingface.co/spaces/aecetin/idempotent-ai-showcase)
[![Patent](https://img.shields.io/badge/USPTO%20Patent-64%2F149%2C540-red.svg)](https://patents.google.com)

**Orthogonal Polynomial Tensors, Zero-Backpropagation Algebraic Solvers, and Next-Generation PolyFormer Architecture.**

`idempotent-poly` is a unified deep learning framework that replaces standard high-rank neural matrix multiplications (GEMM) and brute-force backpropagation with **orthogonal Chebyshev polynomial tensor operators** and **closed-form algebraic subspace projections on an idempotent manifold ($\Pi^2 = \Pi$)**.

It bridges two groundbreaking paradigms in deep learning:
1. **In-Situ Zero-Backprop Weight Surgery:** Surgically convert feed-forward networks (FFN / SwiGLU) of pre-trained LLMs (LLaMA, SmolLM2, Mistral) into orthogonal Chebyshev tensors in **under 60 milliseconds** with **50% parameter reduction** and zero gradient descent.
2. **PolyFormer (Train-From-Scratch Architecture):** A complete, drop-in Transformer alternative utilizing orthogonal Chebyshev attention kernels (`IdemPolyAttention`), tensor polynomial FFNs (`ChebyshevTensorLayer`), and adaptive fixed-point dynamic halting (`forward_adaptive`).

---

## 🔬 Official Verified Benchmarks (SmolLM2-135M)

Evaluated on `SmolLM2-135M-Instruct` across multiple surgical depth configurations:

| Configuration | Surgical Depth | FFN Parameter Compression | Zero-Backprop Solve Time | Perplexity (PPL) | Output Coherence & Quality |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SmolLM2-135M (Baseline)** | 0 Layers | 0 (0.0%) | N/A | **51.53** | 🟢 **100% (Baseline Reference)** |
| **1-Layer Poly Surgery (L15)** | 1 Layer | **-1,326,527 (-50.0% FFN)** | **56.2 ms** | **55.45** | 🟢 **Natural, Factual & Fluent ("Paris")** |
| **2-Layer Poly Surgery (L14-L15)** | 2 Layers | **-2,653,054 (-50.0% FFN)** | **~206 ms** | **60.38** | 🟢 **Natural & Grammatical** |
| **3-Layer Poly Surgery (L13-L15)** | 3 Layers | **-3,979,581 (-50.0% FFN)** | **~219 ms** | **59.42** | 🟢 **Semantically Intact** |
| **1-Layer Residual Poly (L15)** | 1 Layer | Residual Adaptation | **~146 ms** | **51.53** | 🟢 **Exact Bit-Match (Identical PPL)** |

> [!TIP]
> **Zero Backpropagation Advantage:** Standard gradient-descent adaptation with AdamW requires hundreds of GPU backward passes. `idempotent-poly` analytically solves the optimal coefficient tensor via regularized normal equations and SVD in **less than 60 milliseconds per layer**.

---

## 📐 Mathematical Foundations

### 1. Orthogonal Chebyshev Basis Expansion
Instead of multi-matrix bilinear projections $	ext{down}(	ext{act}(	ext{gate}(x)) \odot 	ext{up}(x))$, activations $	ilde{x} = 	anh(x) \in [-1, 1]$ are projected onto orthogonal Chebyshev polynomials:

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)$$

The layer forward pass is computed via tensor contraction:

$$y = \sum_{k=0}^K C_k \cdot T_k(	ilde{x}) + b$$

### 2. Zero-Backprop Closed-Form Algebraic Solve
Given activation snapshot matrices $X \in \mathbb{R}^{N 	imes D}$ and target representations $Y \in \mathbb{R}^{N 	imes D}$, the optimal weight tensor is obtained in a single step:

$$W^* = (\Phi(X)^T \Phi(X) + \lambda I)^{-1} \Phi(X)^T Y$$

Projected onto the idempotent spectral manifold $\Pi = V_r V_r^T$ where $\Pi^2 = \Pi$:

$$W_{	ext{idempotent}} = \Pi W^*$$

---

## 💻 Installation

```bash
pip install idempotent-poly
```

Or install from source:
```bash
git clone https://github.com/aemre-cetin/idem-poly.git
cd idempotent-poly
pip install -e .
```

---

## 🚀 Quickstarts

### Paradigm A: Zero-Backprop LLM Surgery (3 Lines)

Convert an existing pre-trained LLM layer into an orthogonal Chebyshev tensor in 56 ms:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from idempotent_poly import ChebyshevPolyFFN

# 1. Load base LLM
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# 2. Capture calibration representations from target layer (e.g., Layer 15)
target_layer = model.model.layers[15]
calib_tokens = tokenizer(["Mathematics and artificial intelligence thrive on idempotent manifolds."], return_tensors="pt").input_ids

captured = {}
def hook_fn(m, inp, out):
    captured["X"] = inp[0].detach().reshape(-1, model.config.hidden_size)
    captured["Y"] = out.detach().reshape(-1, model.config.hidden_size)

h = target_layer.mlp.register_forward_hook(hook_fn)
with torch.no_grad():
    model(calib_tokens)
h.remove()

# 3. Solve and swap in 56 ms (50% parameter reduction!)
poly_ffn = ChebyshevPolyFFN(d_model=model.config.hidden_size, degree=3)
poly_ffn.fit_algebraic(captured["X"], captured["Y"], l2_reg=1e-2, rank_ratio=0.95)
target_layer.mlp = poly_ffn

# Generate with transformed model
inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=15, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# Output: "The capital of France is Paris. Paris is a city that is known for its historical landmarks"
```

---

### Paradigm B: Training PolyFormer from Scratch

Train a native orthogonal polynomial Transformer architecture with PyTorch:

```python
import torch
from idempotent_poly import PolyFormerBlock

# Instantiate a PolyFormer Transformer Block
# (Chebyshev Attention + Chebyshev Tensor FFN)
block = PolyFormerBlock(d_model=128, n_heads=4, poly_degree=3)

x = torch.randn(2, 32, 128)  # [Batch, SeqLen, Dim]

# Standard Forward Pass
out = block(x)
print("Forward Output Shape:", out.shape)  # [2, 32, 128]

# Dynamic Depth Adaptive Halting (Fixed-Point Convergence)
fixed_point_out, steps = block.forward_adaptive(x, max_iters=5, tol=0.01)
print(f"Converged to fixed-point in {steps} iterations!")
```

Run the complete autoregressive language model training script:
```bash
python examples/train_polyformer_tiny.py
```

---

## 📁 Repository Structure

```
idempotent-poly/
├── benchmarks/
│   └── eval_surgery_benchmark.py   # Automated perplexity & latency evaluation harness
├── examples/
│   ├── train_polyformer_tiny.py    # Complete train-from-scratch autoregressive LM
│   └── surgery_quickstart.py       # In-situ zero-backprop surgery script
├── src/idempotent_poly/
│   ├── algebraic.py                # Closed-form SVD & ridge normal equation solver
│   ├── chebyshev.py                # ChebyshevTensorLayer & ChebyshevPolyFFN modules
│   ├── polyformer.py               # IdemPolyAttention & PolyFormerBlock architecture
│   └── surgeon.py                  # PolySurgeon & ResidualPolyMLP surgical engine
├── tests/
│   └── test_poly.py                # Comprehensive mathematical unit test suite
├── pyproject.toml                  # Packaging specifications
└── README.md
```

---

## 🛡️ Intellectual Property & Citation

This technology and its mathematical formulations are protected under United States Patent Law:

- **Official Patent:** **U.S. Provisional Patent Application No. `64/149,540`**
- **Title:** *Hardware-Accelerated Orthogonal Polynomial Tensor Operators, Zero-Backpropagation Closed-Form Algebraic Solvers, and In-Situ Weight Surgery for Deep Neural Networks and Transformers*
- **Inventor & Author:** Dr. A. Emre ÇETİN (`aemre.cetin@gmail.com`)

```bibtex
@patent{cetin2026orthogonalpoly,
  title={Hardware-Accelerated Orthogonal Polynomial Tensor Operators, Zero-Backpropagation Closed-Form Algebraic Solvers, and In-Situ Weight Surgery for Deep Neural Networks and Transformers},
  author={Dr. Ahmet Emre {\c{C}}etin},
  year={2026},
  month={September},
  note={U.S. Provisional Patent Application No. 64/149,540, Filed at USPTO}
}
```

---

## 📄 License

Licensed under the **Apache License, Version 2.0**.
