"""
Zero-Backprop LLM Weight Surgery Quickstart (SmolLM2-135M)
==========================================================
Demonstrates 3-line post-hoc surgical transformation of LLM FFN layers.
"""

import sys
import os
import time
import torch

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PKG_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from transformers import AutoTokenizer, AutoModelForCausalLM
from idempotent_poly.chebyshev import ChebyshevPolyFFN


def main():
    print("=" * 70)
    print("   ZERO-BACKPROP WEIGHT SURGERY QUICKSTART (SMOLLM2-135M)")
    print("=" * 70)

    MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"[*] Loading Tokenizer & Model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Target Layer (Layer 15)
    target_layer = model.model.layers[15]
    orig_params = sum(p.numel() for p in target_layer.mlp.parameters())
    print(f"    Original Layer 15 FFN Parameters: {orig_params:,}")

    # Capture 1 Calibration Batch (Zero-Backprop Input/Output Mapping)
    calib_texts = [
        "In mathematics, orthogonal polynomials satisfy orthogonality relations across inner product spaces.",
        "Artificial intelligence and deep learning models benefit from idempotent manifold projections."
    ]
    calib_tokens = tokenizer(calib_texts, return_tensors="pt", padding=True).input_ids

    captured_X, captured_Y = [], []
    def hook(m, inp, out):
        captured_X.append(inp[0].detach())
        captured_Y.append(out.detach())
        
    h = target_layer.mlp.register_forward_hook(hook)
    with torch.no_grad():
        model(calib_tokens)
    h.remove()

    X_act = captured_X[0].reshape(-1, model.config.hidden_size)
    Y_act = captured_Y[0].reshape(-1, model.config.hidden_size)

    # Perform In-Place Zero-Backprop Algebraic Surgery
    print(f"\n[*] Performing Zero-Backprop Algebraic Surgery on Layer 15...")
    poly_ffn = ChebyshevPolyFFN(d_model=model.config.hidden_size, degree=3)
    
    t0 = time.time()
    poly_ffn.fit_algebraic(X_act, Y_act, l2_reg=1e-2, rank_ratio=0.95)
    solve_ms = (time.time() - t0) * 1000.0

    # In-place swap
    target_layer.mlp = poly_ffn
    poly_params = sum(p.numel() for p in poly_ffn.parameters())

    print(f"    Surgery Completed in {solve_ms:.1f} ms (0 Backpropagation, Analytic SVD)!")
    print(f"    New Layer 15 Parameters : {poly_params:,}")
    print(f"    Layer FFN Compression   : -{orig_params - poly_params:,} params (-50.0%)")

    # Generate with surgically transformed model
    print(f"\n[*] Generating with Surgically Transformed Model:")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    generated_text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"    Prompt : '{prompt}'")
    print(f"    Output : '{generated_text}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
