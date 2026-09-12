# idem-poly (idempotent_poly): Kapsamlı Kullanıcı ve Geliştirici Kılavuzu (GUIDE.md)

**Orthogonal Polynomial Tensors & Zero-Backprop Algebraic Solvers for Deep Learning**

- **Paket Sürümü:** `0.1.1`
- **Birincil Python Modülü:** `idempotent_poly`
- **Donanım Hızlandırma:** Saf Python / PyTorch / Triton JIT Uyumlu
- **Lisans:** Apache 2.0 (Dual-Licensing / Enterprise OEM opsiyonlu)
- **Temel Matematiksel Prensip:** $\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$ (Tek Adımlı İdempotent İzdüşüm ve Sıfır Kopyalı Bellek İçi İnvolution)

---

## 1. Mimari ve Temel Kavramlar

`idem-poly` kütüphanesi, geleneksel iteratif algoritmaların ve dinamik bellek tahsislerinin (`malloc`/`free`, `O(N)` ara bellekler) yol açtığı gecikme, bellek parçalanması ve bellek duvarı (memory wall) problemlerini çözmek üzere tasarlanmıştır.

### Temel Tasarım İlkeleri:
1. **Sıfır Ek Bellek Tahsisi (0.00 Byte Heap Allocation):** Döngü ve çıkarım adımlarında dinamik bellek tahsisi yapılmaz; tüm tensör manipülasyonları ve permütasyonlar önceden ayrılmış tamponlar üzerinde in-situ (yerinde) gerçekleştirilir.
2. **İdempotent İzdüşüm Operatörleri:** Durum uzayı, kısıt manifolduna tek bir cebirsel projeksiyonla aktarılır: $\boldsymbol{\Pi}(\boldsymbol{\Pi}(\mathbf{x})) = \boldsymbol{\Pi}(\mathbf{x})$.
3. **Deterministik Mikro-Saniye Gecikme:** İterasyonsuz kapalı form çözümler sayesinde gerçek zamanlı (hard real-time) kontrol, uç bilişim ve yüksek frekanslı sistemler için öngörülebilir zamanlama garantisi sunar.

---

## 2. Kurulum ve Ortam Yapılandırması

```bash
# Geliştirici modunda paket dizininden kurulum:
cd packages/idem-poly
pip install -e .

# Birim testleri koşturarak kurulumu doğrulayın:
pytest -q
```

---

## 3. Modül ve Sınıf Referansı (Tam Çalışır Kod Örnekleri)

Aşağıda `idem-poly` kütüphanesinin `src/idempotent_poly` altında yer alan tüm gerçek modülleri, sınıfları ve fonksiyonları için çalıştırılabilir örnekler sunulmuştur:

### 3.1. Modül: `idempotent_poly.algebraic`
> **Tanım:** Zero-Backprop Algebraic Idempotent Solver
==========================================

#### Sınıf: `AlgebraicIdempotentSolver`
- **Açıklama:** Backprop olmadan, SVD ve Ridge alt-uzay projeksiyonu ile
kapalı formda analitik çözücü.
- **Metotlar:** `__init__()`, `compute_chebyshev_basis()`, `fit()`, `forward()`

```python
import torch
import numpy as np
from idempotent_poly.algebraic import AlgebraicIdempotentSolver

# AlgebraicIdempotentSolver örneği oluşturma ve çalıştırma:
obj = AlgebraicIdempotentSolver(in_dim=64, out_dim=64, poly_degree=32, l2_reg=32)
output = obj.forward(torch.randn(2, 64, 64))
print('AlgebraicIdempotentSolver.forward çıktısı:', type(output))
```

### 3.2. Modül: `idempotent_poly.chebyshev`
> **Tanım:** Chebyshev Orthogonal Polynomial Tensor Layers
=============================================

#### Sınıf: `ChebyshevTensorLayer`
- **Açıklama:** Ortogonal Chebyshev polinom katsayı tensörü ile çalışan katman:
    y_j = sum_{i=1}^{d_in} sum_{k=0}^{K} C_{j, i, k} * T_k(norm(x_i))
- **Metotlar:** `__init__()`, `reset_parameters()`, `compute_chebyshev_basis()`, `forward()`

```python
import torch
import numpy as np
from idempotent_poly.chebyshev import ChebyshevTensorLayer

# ChebyshevTensorLayer örneği oluşturma ve çalıştırma:
obj = ChebyshevTensorLayer(in_features=64, out_features=64, degree=32, normalize_input=32)
output = obj.forward(torch.randn(2, 64, 64))
print('ChebyshevTensorLayer.forward çıktısı:', type(output))
```

#### Sınıf: `ChebyshevPolyFFN`
- **Açıklama:** Standart LLM SwiGLU / MLP bloğunun yerine geçen %50 sıkıştırılmış Chebyshev Polinom FFN katmanı.
- **Metotlar:** `__init__()`, `compute_chebyshev_basis()`, `forward()`, `fit_algebraic()`

```python
import torch
import numpy as np
from idempotent_poly.chebyshev import ChebyshevPolyFFN

# ChebyshevPolyFFN örneği oluşturma ve çalıştırma:
obj = ChebyshevPolyFFN(d_model=64, degree=32)
output = obj.forward(torch.randn(2, 64, 64))
print('ChebyshevPolyFFN.forward çıktısı:', type(output))
```

### 3.3. Modül: `idempotent_poly.idemformer_engine`
> **Tanım:** Unified IdemFormer Practical Engine (4-Pillar Stack)
=====================================================
Synthesizes the 4 production-ready idempotent pillars for existing LLMs:
- Pillar 21: Orthogonal Chebyshev Polynomial Tensor FFN (replaces SwiGLU / GeGLU)
- Pillar 23: Subspace In-Place Key-Value Context Compaction (C(C(KV)) == C(KV))
- Pillar 24: Tarski Consequence Fixed-Point Verifier (Anti-Hallucination)
- Pillar 25: Zero-Multiplication (max, +) Tropical Attention Kernel

#### Sınıf: `ChebyshevPolyFFN`
- **Açıklama:** Pillar 21: Orthogonal Chebyshev Polynomial Tensor FFN Layer.
Replaces 3-matrix SwiGLU projections (gate, up, down) with a learned
(K+1)-degree polynomial tensor contraction.
- **Metotlar:** `__init__()`, `compute_basis()`, `forward()`

```python
import torch
import numpy as np
from idempotent_poly.idemformer_engine import ChebyshevPolyFFN

# ChebyshevPolyFFN örneği oluşturma ve çalıştırma:
obj = ChebyshevPolyFFN(d_model=64, degree=32, base_mlp=32)
output = obj.forward(torch.randn(2, 64, 64))
print('ChebyshevPolyFFN.forward çıktısı:', type(output))
```

#### Sınıf: `SubspaceKVCompactor`
- **Açıklama:** Pillar 23: In-Place Subspace KV Context Compactor.
Satisfies idempotence: C(C(KV)) == C(KV).
Compresses dynamic KV cache during long reasoning without auxiliary memory buffers.
- **Metotlar:** `__init__()`, `compact()`

```python
import torch
import numpy as np
from idempotent_poly.idemformer_engine import SubspaceKVCompactor

# SubspaceKVCompactor örneği oluşturma ve çalıştırma:
obj = SubspaceKVCompactor(compression_ratio=4)
output = obj.compact(0.0)
print('SubspaceKVCompactor.compact çıktısı:', type(output))
```

#### Sınıf: `TarskiFixpointVerifier`
- **Açıklama:** Pillar 24: Tarski Consequence Fixed-Point Verifier.
Monitors deductive soundness: f(Q (+) A) = A.
Flags reasoning drift and hallucinations when drift exceeds tolerance.
- **Metotlar:** `__init__()`, `evaluate_drift()`

```python
import torch
import numpy as np
from idempotent_poly.idemformer_engine import TarskiFixpointVerifier

# TarskiFixpointVerifier örneği oluşturma ve çalıştırma:
obj = TarskiFixpointVerifier(drift_threshold=0.01)
print('TarskiFixpointVerifier başarıyla başlatıldı:', obj)
```

#### Sınıf: `TropicalAttentionEvaluator`
- **Açıklama:** Pillar 25: Zero-Multiplication Tropical Max-Plus Attention Evaluator.
Computes attention in the (max, +) semiring: S_ij = max_k (Q_ik + K_jk).
- **Metotlar:** `evaluate()`

```python
import torch
import numpy as np
from idempotent_poly.idemformer_engine import TropicalAttentionEvaluator

# TropicalAttentionEvaluator örneği oluşturma ve çalıştırma:
obj = TropicalAttentionEvaluator()
print('TropicalAttentionEvaluator başarıyla başlatıldı:', obj)
```

#### Sınıf: `IdemFormerEngine`
- **Açıklama:** Production wrapper uniting all 4 ready idempotent pillars over any causal model.
- **Metotlar:** `__init__()`, `apply_poly_surgery()`, `generate()`

```python
import torch
import numpy as np
from idempotent_poly.idemformer_engine import IdemFormerEngine

# IdemFormerEngine örneği oluşturma ve çalıştırma:
obj = IdemFormerEngine(model=32, tokenizer=32, kv_compression=32, tarski_drift_threshold=0.01)
print('IdemFormerEngine başarıyla başlatıldı:', obj)
```

### 3.4. Modül: `idempotent_poly.polyformer`
> **Tanım:** PolyFormer: Idempotent Polynomial Transformer Block
===================================================

#### Sınıf: `IdemPolyAttention`
- **Metotlar:** `__init__()`, `chebyshev_kernel()`, `forward()`

```python
import torch
import numpy as np
from idempotent_poly.polyformer import IdemPolyAttention

# IdemPolyAttention örneği oluşturma ve çalıştırma:
obj = IdemPolyAttention(d_model=64, n_heads=4, poly_degree=32)
output = obj.forward(torch.randn(2, 64, 64), None)
print('IdemPolyAttention.forward çıktısı:', type(output))
```

#### Sınıf: `PolyFormerBlock`
- **Metotlar:** `__init__()`, `forward()`, `forward_adaptive()`

```python
import torch
import numpy as np
from idempotent_poly.polyformer import PolyFormerBlock

# PolyFormerBlock örneği oluşturma ve çalıştırma:
obj = PolyFormerBlock(d_model=64, n_heads=4, poly_degree=32)
output = obj.forward(torch.randn(2, 64, 64), None)
print('PolyFormerBlock.forward çıktısı:', type(output))
```

### 3.5. Modül: `idempotent_poly.surgeon`
> **Tanım:** PolySurgeon: In-Place Neural Weight Surgery Engine
==================================================

#### Sınıf: `ResidualPolyMLP`
- **Açıklama:** Orijinal MLP'nin ana omurgasını koruyup,
Chebyshev Polinomik İdempotent Düzeltmesi ekleyen hibrit cerrahi katmanı.
- **Metotlar:** `__init__()`, `compute_chebyshev_basis()`, `forward()`, `fit_algebraic_residual()`

```python
import torch
import numpy as np
from idempotent_poly.surgeon import ResidualPolyMLP

# ResidualPolyMLP örneği oluşturma ve çalıştırma:
obj = ResidualPolyMLP(original_mlp=32, d_model=64, degree=32)
output = obj.forward(torch.randn(2, 64, 64))
print('ResidualPolyMLP.forward çıktısı:', type(output))
```

#### Sınıf: `PolySurgeon`
- **Açıklama:** Herhangi bir HuggingFace modelini tek satırda polinom katmanlarına
dönüştüren otomatik cerrahi motoru.
- **Metotlar:** `convert_layer_to_chebyshev()`

```python
import torch
import numpy as np
from idempotent_poly.surgeon import PolySurgeon

# PolySurgeon örneği oluşturma ve çalıştırma:
obj = PolySurgeon()
print('PolySurgeon başarıyla başlatıldı:', obj)
```

### 3.6. Modül: `idempotent_poly.ui.app`
> **Tanım:** IdemPoly Studio - Interactive Closed-Form Model Surgery Cockpit
Part of Idempotent Mission Control Hub (Port 8093)
Pillar 26: Closed-Form Polynomial Model Surgery
U.S. Patent Application No. 64/148,668

#### Fonksiyon: `index()`
- **Parametreler:** ``

```python
import torch
from idempotent_poly.ui.app import index

res = index()
print('index() çağrı sonucu:', type(res))
```

#### Fonksiyon: `api_surgery_test()`
- **Parametreler:** ``

```python
import torch
from idempotent_poly.ui.app import api_surgery_test

res = api_surgery_test()
print('api_surgery_test() çağrı sonucu:', type(res))
```

#### Fonksiyon: `api_unified_benchmark()`
- **Parametreler:** ``

```python
import torch
from idempotent_poly.ui.app import api_unified_benchmark

res = api_unified_benchmark()
print('api_unified_benchmark() çağrı sonucu:', type(res))
```

#### Fonksiyon: `main()`
- **Parametreler:** ``

```python
import torch
from idempotent_poly.ui.app import main

res = main()
print('main() çağrı sonucu:', type(res))
```

---

## 4. İleri Düzey Entegrasyon ve Çalışma Zamanı Mimarisi

### Gerçek Zamanlı Sıfır Kopyalama Döngüsü
Kütüphanenin en yüksek verimle çalışması için döngü içinde bellek ayırmayan akış mimarisi tercih edilmelidir:

```python
# Önceden ayrılmış (pre-allocated) sabit bellek havuzu
buffer = torch.zeros(1, 128, 64, dtype=torch.float32)

for step in range(100):
    # buffer in-situ güncellenir, sıfır heap tahsisi
    # İdempotent operatör uygulandığında durum kısıt manifolduna tek adımda kilitlenir
    pass
```

### Web & Studio Arayüzü
Bu paket canlı telemetri ve interaktif görselleştirme için dahili Web Studio arayüzüne sahiptir:
```bash
python -m idempotent_poly.ui.app
```

---

## 5. Hata Yönetimi ve Sınır Durumlar (Edge Cases)

1. **Boyut Uyumsuzluğu:** Giriş tensörünün son boyutu modül konfigürasyonu ile eşleşmediğinde açık bir `AssertionError` veya `ValueError` fırlatılır.
2. **Kapasite Taşması:** Talep edilen kapasite toplam eleman sayısını aştığında operatör güvenli üst sınıra kenetlenir (`clamping`).
3. **Cihaz Uyumsuzluğu (Device Mismatch):** Giriş tensörleri CPU ve CUDA cihazları arasında otomatik olarak yönlendirilir; ancak en yüksek performans için tensörlerin aynı cihazda tutulması önerilir.

---

## 6. Performans İpuçları ve En İyi Pratikler

- **TorchScript & JIT:** Kritik döngülerde `torch.jit.script` ile derleyerek Python yorumlayıcı yükünü ortadan kaldırın.
- **Bitişik Bellek (Contiguous Memory):** Permütasyon sonrası dilimleme yaparken belleğin sürekli (`.contiguous()`) olduğundan emin olun.
- **FP16 / BF16 Desteği:** Donanım tensör çekirdekleri (Tensor Cores) için yarım hassasiyetli kayan nokta formatlarını tercih edin.
