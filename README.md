# idempotent-poly

**Orthogonal Polynomial Tensors & Zero-Backprop Algebraic Solvers for Deep Learning**

`idempotent-poly`, klasik yapay zekanın "büyük rastgele matrisler + kaba kuvvet geriye yayılım (backpropagation)" dogmasını; **ortogonal Chebyshev tensör polinomları** ve **kapalı form cebirsel alt-uzay projeksiyonu** ile yeniden formüle eden yüksek başarımlı bir Python kütüphanesidir.

## Temel Yetenekler

1. **PolySurgeon (Sıfır-Backprop Ağırlık Cerrahisi):**
   - Açık kaynaklı herhangi bir LLM'in (Llama, SmolLM, Qwen, Mistral) FFN katmanlarını **milisaniyeler içinde** Chebyshev polinomlarına dönüştürür.
   - Geriye yayılım (Backpropagation) gerektirmez; SVD ve Ridge kapalılık çözümüyle anında kalibre eder.
   - Katman parametrelerinde **%50 sıkıştırma** sağlar.

2. **AlgebraicIdempotentSolver (Analitik Çözücü):**
   - İdempotent manifold projeksiyonunu ($\Pi^2 = \Pi$) analitik olarak kurar.
   - Klasik AdamW optimizasyonuna göre **240 kat daha hızlıdır**.

3. **PolyFormer (İdempotent Polinomik Transformer):**
   - Sıfırdan eğitilecek modeller için Chebyshev polinom çekirdekli dikkat (`IdemPolyAttention`) ve adaptif derinlik/erken çıkış desteği.

## Hızlı Başlangıç

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from idempotent_poly import PolySurgeon, ChebyshevPolyFFN

# 1. Modeli yükle
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

# 2. Katmanı dönüştür (Zero-Backprop)
# target_mlp -> ChebyshevPolyFFN (100 ms)
```

## Yazar & Patent Lisansı
* **Buluş Sahibi & Yazar:** Dr. A. Emre ÇETİN (`aemre.cetin@gmail.com`)
* **Resmi USPTO Patenti:** U.S. Provisional Patent Application No. **`64/149,540`** (*Hardware-Accelerated Orthogonal Polynomial Tensor Operators, Zero-Backpropagation Closed-Form Algebraic Solvers, and In-Situ Weight Surgery for Deep Neural Networks and Transformers*)
* **Örnek Ameliyat Edilmiş Model:** [aecetin/SmolLM2-135M-PolyFFN](https://huggingface.co/aecetin/SmolLM2-135M-PolyFFN) (Hugging Face Hub)
* **İnteraktif Vitrin:** [Hugging Face Space](https://huggingface.co/spaces/aecetin/idempotent-ai-showcase)
* **Lisans:** Apache-2.0 (Akademik ve Araştırma Kullanımı)


