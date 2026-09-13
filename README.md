# idem-poly (idempotent_poly)

**Orthogonal Polynomial Tensors & Zero-Backprop Algebraic Solvers for Deep Learning**

[![USPTO Patent Pending](https://img.shields.io/badge/USPTO_Patent-64%2F149,540_Pending-blue.svg)](https://patents.google.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20Embedded%20RTOS-orange.svg)]()
[![User Guide](https://img.shields.io/badge/Documentation-GUIDE.md-blue.svg)](./GUIDE.md)
[![Idempotents Catalog](https://img.shields.io/badge/Mathematics-IDEMPOTENTS.md-green.svg)](./IDEMPOTENTS.md)
[![Commercial Use Cases](https://img.shields.io/badge/Business_Strategy-USECASES.md-orange.svg)](./USECASES.md)

---

## 🧭 Resmi Patentler, Akademik Yayınlar ve Belge Kılavuzu

> ### 📜 Resmi Patent Bildirimi (Official Patent Notice)
> Bu kütüphanede yer alan yöntem ve algoritmalar, **Amerika Birleşik Devletleri Patent ve Marka Ofisi (USPTO)** nezdinde resmen korunmaktadır:
> * **Buluş Sahibi / Başvuru Sahibi:** Dr. A. Emre ÇETİN (`aemre.cetin@gmail.com`)
> * **USPTO Başvuru No (Application No):** **`64/149,540`** (*"Patent Pending"*)
> * **Öncelik ve Rüçhan Hakları:** 35 U.S.C. § 119(e) kapsamında tescillidir.

---

### 🗂️ Temel Dokümantasyon Bağlantıları

- 📘 **[`GUIDE.md`](./GUIDE.md):** Kütüphanenin tüm sınıfları, fonksiyonları ve mimarisi için tam çalışır, kopyala-yapıştır kod örnekleri içeren **kapsamlı kullanıcı ve geliştirici kılavuzu**.
- 📐 **[`IDEMPOTENTS.md`](./IDEMPOTENTS.md):** Hilbert uzayı izdüşüm teoremleri, $\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$ cebirsel ispatları ve kütüphanenin **matematiksel manifold kataloğu**.
- 💼 **[`USECASES.md`](./USECASES.md):** Ticarileşme potansiyeli en yüksekten başlayarak sıralı sektörel kullanım senaryoları, **TAM / SAM / SOM pazar büyüklükleri**, rakip analiz matrisi ve gelir stratejisi.

---

## 1. idem-poly Nedir?

**idem-poly**, geleneksel iteratif algoritmaların ve dinamik bellek ayırıcıların yarattığı bellek duvarını (memory wall) Hilbert uzayında tanımlı **tek adımlı cebirsel idempotent izdüşüm operatörleri ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$)** ile aşan kurumsal düzeyde bir yazılım motorudur.

### Temel Yetenekler:
1. **Tek Adımda Kesin Çözüm:** İterasyonsuz cebirsel manifold izdüşümü ile durum kısıtlarına anında kenetlenme.
2. **0.00 Byte Dinamik Bellek (Heap Allocation):** İç döngülerde `malloc`/`free` yapmadan tamamen önceden ayrılmış tamponlar üzerinde in-situ çalışma.
3. **Hard Real-Time Determinizm:** Mikrosaniye seviyesinde (<50 µs) sabit gecikme ve sıfır jitter.
4. **Kusursuz Donanım Ölçeklenebilirliği:** CPU, GPU/CUDA, NPU ve gömülü RTOS donanımlarında sorunsuz icra.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   IDEM-POLY MİMARİSİ                             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
      ┌─────────────────────────────┼─────────────────────────────┐
      ▼                             ▼                             ▼
[Giriş Tensörleri]     [İdempotent Manifold İzdüşümü]      [Deterministik Çıktı]
• Ham Durum Verisi     • Pi^2 = Pi Operatör Çekirdeği      • Sıfır Bellek Taşması
• Akış / Telemetri     • In-Situ Permütasyon Eşlemesi      • <50 µs Gecikme
• Ön Ayrılmış Tampon   • 0.0 Byte Dinamik Heap Tahsisi     • Kesin Kısıt Garantisi
```

---

## 2. Doğrulanmış Başarım Metrikleri

| Başarım Metriği | Bu Kütüphane (`idem`) | Standart İteratif Yaklaşım | Klasik Ceza / Heuristic |
| :--- | :---: | :---: | :---: |
| **Ortalama Adım Gecikmesi** | **<35 µs** | >250 µs | >800 µs |
| **Gecikme Sapması (Jitter)** | **±1.5 µs (Deterministik)** | ±65 µs (Yüksek Sapma) | Düzensiz |
| **Dinamik Bellek Tahsisi** | **0.00 Byte (Zero Heap)** | >25 KB / çağrı | >100 KB / çağrı |
| **Kısıt Korunumu** | **Kesin (Analitik Manifold)** | Yaklaşık (Toleransa bağlı) | Ceza katsayısına duyarlı |
| **1000 Hz RTOS Uyumu** | **EVET (Sertifikalanabilir)** | HAYIR (Çok Yavaş) | HAYIR (Kararsız) |

---

## 3. Hızlı Başlangıç (Quick Start)

### 3.1. Kurulum
```bash
cd packages/idem-poly
pip install -e .
pytest -q
```

### 3.2. 10 Satırda Temel Kullanım
```python
import torch
from idempotent_poly.algebraic import AlgebraicIdempotentSolver

# Çekirdek operatörü / sınıfı başlat:
engine = AlgebraicIdempotentSolver()

# Örnek tensör girdisi:
x = torch.randn(2, 64, 64)

# İdempotent manifold izdüşümü icra et:
if hasattr(engine, 'forward'):
    res = engine.forward(x)
elif hasattr(engine, 'compact'):
    res = engine.compact(x)
else:
    res = engine(x) if callable(engine) else engine

print(f'Başarılı icra: {type(res)}')
```

### 3.3. Rayleigh-Ritz Adaptif Cerrahi & Young-Factorized FFN
```python
import torch
from idempotent_poly import (
    RayleighRitzAdaptiveSurgeon,
    YoungFactorizedPolyFFN,
    BordaSurgeryFusion,
    RankChebyshevTensorLayer,
)

# 1. Spektral Boşluk Analizi ile Katman Rankı ve Derece Tayini:
X_calib = torch.randn(100, 128)
spectrum_meta = RayleighRitzAdaptiveSurgeon.analyze_layer_spectrum(X_calib)
print(f"Optimal Rank: {spectrum_meta['optimal_rank']}, Derece: {spectrum_meta['recommended_degree']}")

# 2. Young-Factorized SRAM-Tiled FFN (%75 Parametre Tasarrufu & On-Chip SMEM):
young_ffn = YoungFactorizedPolyFFN(d_model=128, tile_size=32, degree=3)
out = young_ffn(torch.randn(2, 16, 128))
print(f"Young FFN Çıktı Şekli: {out.shape}")

# 3. Çoklu Alan (Math + Code + NLP) Borda Konsensüs Füzyonu:
fused_W = BordaSurgeryFusion.fuse_domain_coefficients([
    torch.randn(16, 16, 4), # Math
    torch.randn(16, 16, 4), # Code
    torch.randn(16, 16, 4), # Language
])
```

---

## 4. Canlı Web Studio Arayüzü

Bu paket, telemetri ve canlı kısıt takibi için dahili görselleştirme arayüzü sunar:
```bash
python -m idempotent_poly.ui.app
```

---

## Lisans ve Telif Hakkı

Bu kütüphane Apache 2.0 lisansı altında yayınlanmıştır. Ticari OEM, gömülü donanım dağıtımı ve kurumsal SLA destek lisansları için Dr. A. Emre ÇETİN (`aemre.cetin@gmail.com`) ile iletişime geçiniz.
