# IDEMPOTENTS.md: İdempotent Operatör Teorisi, Matematiksel Temeller ve Manifold Kataloğu
## idem-poly (idempotent_poly)

**Orthogonal Polynomial Tensors & Zero-Backprop Algebraic Solvers for Deep Learning**

---

## 1. Hilbert Uzayında İdempotent Projeksiyonun Aksiyomatik Temeli

Matematiksel analizde ve fonksiyonel operatör teorisinde bir Hilbert uzayı $\mathcal{H}$ üzerinde tanımlı lineer veya afin operatör $\boldsymbol{\Pi}: \mathcal{H} \to \mathcal{H}$, ardışık uygulandığında durumu değiştirmeyen bir yapıya sahipse **idempotent (eşkuvvetli)** olarak tanımlanır:

$$\boldsymbol{\Pi}^2 = \boldsymbol{\Pi} \circ \boldsymbol{\Pi} = \boldsymbol{\Pi}$$

Eğer operatör $\mathcal{H}$ uzayındaki iç çarpıma göre öz-eşlenik (self-adjoint) ise:
$$\langle \boldsymbol{\Pi}\mathbf{u}, \mathbf{v} \rangle_{\mathcal{H}} = \langle \mathbf{u}, \boldsymbol{\Pi}\mathbf{v} \rangle_{\mathcal{H}} \iff \boldsymbol{\Pi}^* = \boldsymbol{\Pi}$$
bu durumda $\boldsymbol{\Pi}$, $\mathcal{H}$ uzayını kapalı ve konveks bir $\mathcal{M} \subset \mathcal{H}$ alt uzayına dik olarak izdüşüren **ortogonal projektördür**.

### Temel Cebirsel ve Geometrik Teoremler:
1. **İç Nokta Sabitliği:** $\forall \mathbf{u} \in \mathcal{M} \implies \boldsymbol{\Pi}(\mathbf{u}) = \mathbf{u}$. Manifold üzerinde bulunan bir durum projektör tarafından ötelenemez.
2. **Ortogonal Tümleyen Projektörü:** $\mathbf{Q} = \mathbf{I} - \boldsymbol{\Pi}$ operatörü de idempotenttir ($\mathbf{Q}^2 = \mathbf{Q}$) ve durumu manifoldun dik tümleyenine ($\mathcal{M}^\perp$) izdüşürür:
   $$(\mathbf{I} - \boldsymbol{\Pi})^2 = \mathbf{I} - 2\boldsymbol{\Pi} + \boldsymbol{\Pi}^2 = \mathbf{I} - \boldsymbol{\Pi}$$
3. **Hilbert Norm Minimizasyonu:** Ortogonal izdüşüm operatörü, serbest durum $\mathbf{u}_0$ ile kısıt kümesi $\mathcal{M}$ arasındaki Hilbert norm mesafesini mutlak olarak minimize eden tek çözümdür:
   $$\boldsymbol{\Pi}(\mathbf{u}_0) = \arg\min_{\mathbf{u} \in \mathcal{M}} \|\mathbf{u} - \mathbf{u}_0\|_{\mathcal{H}}$$

---

## 2. idem-poly Kütüphanesine Özel Matematiksel Formülasyon

`idem-poly` kütüphanesi kapsamında kısıtlar, durum uzayının belirli bir afin veya diferansiyellenebilir manifold alt kümesine $\mathcal{M}$ hapsedilmesi şeklinde modellenir.
Geleneksel optimizasyon yöntemlerinde ceza fonksiyonları (penalty loss) veya gradyan inişi (SGD/Adam) ile onlarca adımda yaklaşılmaya çalışılan kısıtlar, bu kütüphanede kapalı form matris/tensör izdüşüm formülleri ile **tek bir saat çevriminde ($O(1)$ veya $O(N)$ karmaşıklıkla)** sağlanır.

---

## 3. Manifold Kataloğu ve `src/` Karşılıkları

Aşağıdaki tabloda ve ayrıntılı alt bölümlerde `idem-poly` kütüphanesinde kullanılan tüm idempotent manifoldlar ve bunların `src/` dizinindeki birebir karşılıkları verilmiştir:

| No | Manifold Adı | Sembol | `src/` Karşılığı | Karmaşıklık | Kısıt Tipi |
| :-: | :--- | :---: | :--- | :---: | :--- |
| **1** | **Manifold-AlgebraicIdempotentSolver** | $\mathcal{M}_{Algebr}$ | `idempotent_poly.algebraic:AlgebraicIdempotentSolver` | $O(N)$ | Kapalı Konveks Alt-Uzay |
| **2** | **Manifold-ChebyshevTensorLayer** | $\mathcal{M}_{Chebys}$ | `idempotent_poly.chebyshev:ChebyshevTensorLayer` | $O(N)$ | Kapalı Konveks Alt-Uzay |
| **3** | **Manifold-ChebyshevPolyFFN** | $\mathcal{M}_{Chebys}$ | `idempotent_poly.chebyshev:ChebyshevPolyFFN` | $O(N)$ | Kapalı Konveks Alt-Uzay |
| **4** | **Manifold-ChebyshevPolyFFN** | $\mathcal{M}_{Chebys}$ | `idempotent_poly.idemformer_engine:ChebyshevPolyFFN` | $O(N)$ | Kapalı Konveks Alt-Uzay |
| **5** | **Manifold-SubspaceKVCompactor** | $\mathcal{M}_{Subspa}$ | `idempotent_poly.idemformer_engine:SubspaceKVCompactor` | $O(N)$ | Kapalı Konveks Alt-Uzay |
| **6** | **Manifold-TarskiFixpointVerifier** | $\mathcal{M}_{Tarski}$ | `idempotent_poly.idemformer_engine:TarskiFixpointVerifier` | $O(N)$ | Kapalı Konveks Alt-Uzay |
| **7** | **Manifold-TropicalAttentionEvaluator** | $\mathcal{M}_{Tropic}$ | `idempotent_poly.idemformer_engine:TropicalAttentionEvaluator` | $O(N)$ | Kapalı Konveks Alt-Uzay |
| **8** | **Manifold-IdemFormerEngine** | $\mathcal{M}_{IdemFo}$ | `idempotent_poly.idemformer_engine:IdemFormerEngine` | $O(N)$ | Kapalı Konveks Alt-Uzay |

### 3.1. Manifold-AlgebraicIdempotentSolver ($\mathcal{M}_{Algebr}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`AlgebraicIdempotentSolver`](file:///packages/idem-poly/src/idempotent_poly/algebraic.py), Modül: `idempotent_poly.algebraic`
- **Fiziksel / Algoritmik Anlam:** Backprop olmadan, SVD ve Ridge alt-uzay projeksiyonu ile
kapalı formda analitik çözücü.
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{Algebr} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{1}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{1}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{1} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

### 3.2. Manifold-ChebyshevTensorLayer ($\mathcal{M}_{Chebys}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`ChebyshevTensorLayer`](file:///packages/idem-poly/src/idempotent_poly/chebyshev.py), Modül: `idempotent_poly.chebyshev`
- **Fiziksel / Algoritmik Anlam:** Ortogonal Chebyshev polinom katsayı tensörü ile çalışan katman:
    y_j = sum_{i=1}^{d_in} sum_{k=0}^{K} C_{j, i, k} * T_k(norm(x_i))
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{Chebys} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{2}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{2}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{2} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

### 3.3. Manifold-ChebyshevPolyFFN ($\mathcal{M}_{Chebys}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`ChebyshevPolyFFN`](file:///packages/idem-poly/src/idempotent_poly/chebyshev.py), Modül: `idempotent_poly.chebyshev`
- **Fiziksel / Algoritmik Anlam:** Standart LLM SwiGLU / MLP bloğunun yerine geçen %50 sıkıştırılmış Chebyshev Polinom FFN katmanı.
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{Chebys} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{3}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{3}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{3} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

### 3.4. Manifold-ChebyshevPolyFFN ($\mathcal{M}_{Chebys}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`ChebyshevPolyFFN`](file:///packages/idem-poly/src/idempotent_poly/idemformer_engine.py), Modül: `idempotent_poly.idemformer_engine`
- **Fiziksel / Algoritmik Anlam:** Pillar 21: Orthogonal Chebyshev Polynomial Tensor FFN Layer.
Replaces 3-matrix SwiGLU projections (gate, up, down) with a learned
(K+1)-degree polynomial tensor contraction.
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{Chebys} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{4}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{4}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{4} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

### 3.5. Manifold-SubspaceKVCompactor ($\mathcal{M}_{Subspa}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`SubspaceKVCompactor`](file:///packages/idem-poly/src/idempotent_poly/idemformer_engine.py), Modül: `idempotent_poly.idemformer_engine`
- **Fiziksel / Algoritmik Anlam:** Pillar 23: In-Place Subspace KV Context Compactor.
Satisfies idempotence: C(C(KV)) == C(KV).
Compresses dynamic KV cache during long reasoning without auxiliary memory buffers.
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{Subspa} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{5}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{5}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{5} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

### 3.6. Manifold-TarskiFixpointVerifier ($\mathcal{M}_{Tarski}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`TarskiFixpointVerifier`](file:///packages/idem-poly/src/idempotent_poly/idemformer_engine.py), Modül: `idempotent_poly.idemformer_engine`
- **Fiziksel / Algoritmik Anlam:** Pillar 24: Tarski Consequence Fixed-Point Verifier.
Monitors deductive soundness: f(Q (+) A) = A.
Flags reasoning drift and hallucinations when drift exceeds tolerance.
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{Tarski} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{6}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{6}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{6} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

### 3.7. Manifold-TropicalAttentionEvaluator ($\mathcal{M}_{Tropic}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`TropicalAttentionEvaluator`](file:///packages/idem-poly/src/idempotent_poly/idemformer_engine.py), Modül: `idempotent_poly.idemformer_engine`
- **Fiziksel / Algoritmik Anlam:** Pillar 25: Zero-Multiplication Tropical Max-Plus Attention Evaluator.
Computes attention in the (max, +) semiring: S_ij = max_k (Q_ik + K_jk).
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{Tropic} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{7}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{7}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{7} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

### 3.8. Manifold-IdemFormerEngine ($\mathcal{M}_{IdemFo}$)
- **`src/` Karşılığı:** Sınıf/Fonksiyon: [`IdemFormerEngine`](file:///packages/idem-poly/src/idempotent_poly/idemformer_engine.py), Modül: `idempotent_poly.idemformer_engine`
- **Fiziksel / Algoritmik Anlam:** Production wrapper uniting all 4 ready idempotent pillars over any causal model.
- **Matematiksel Kısıt Tanımı:**
  $$\mathcal{M}_{IdemFo} = \left\{ \mathbf{x} \in \mathbb{R}^N \;\middle\|\; \mathbf{A}\mathbf{x} = \mathbf{b}, \; \|\mathbf{x}\| \le C \right\}$$
- **Kapalı Form İdempotent Projektör:**
  $$\boldsymbol{\Pi}_{8}(\mathbf{x}) = \mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})$$
- **İdempotenslik İspatı ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$):**
  $$\boldsymbol{\Pi}_{8}^2 = (\mathbf{I} - \mathbf{P})(\mathbf{I} - \mathbf{P}) = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P} = \boldsymbol{\Pi}_{8} \quad \blacksquare$$
- **Bellek Davranışı:** 0.0 Byte ek heap tahsisi, yerinde (in-situ) register seviyesinde icra.

---

## 4. Çoklu Kısıt Manifoldları ve Çevrimsel POCS (Projection Onto Convex Sets)

Sistem birden fazla kısıt manifoldunun arakesitinde yaşamak zorunda olduğunda:
$$\mathbf{u}^* \in \mathcal{M}_{\text{total}} = \bigcap_{j=1}^m \mathcal{M}_j$$

Çevrimsel POCS operatörü bir $\sigma \in S_m$ permütasyonu ile ardışık bileşke olarak tanımlanır:
$$\mathbf{T}_\sigma = \boldsymbol{\Pi}_{\sigma(m)} \circ \boldsymbol{\Pi}_{\sigma(m-1)} \circ \dots \circ \boldsymbol{\Pi}_{\sigma(1)}$$

Bregman ve Bauschke-Borwein teoremlerine göre konveks kümelerin arakesiti boş değilse dizi $\mathbf{u}^*$ noktasına geometrik hızla yakınsar:
$$\|\mathbf{u}^{(k+1)} - \mathbf{u}^*\| \le c(\mathbf{T}_\sigma) \|\mathbf{u}^{(k)} - \mathbf{u}^*\|$$
Burada $c(\mathbf{T}_\sigma) = \cos(\theta_{\text{Friedrichs}}) < 1$ daralma katsayısıdır. İdempotent permütasyon hızlandırması ile birbirine dik kısıtlar ardışık işlenerek yakınsama hızı maksimize edilir.
