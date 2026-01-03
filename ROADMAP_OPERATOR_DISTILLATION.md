# Roadmap: Operator Distillation from Transformer

> **Status:** January 2026
> **Goal:** Extract symbolic operator/kernel from trained neural network

---

## 1. Confirmed Findings

### 1.1 Models That Work

| Model | NLL | Err@100 | Status |
|-------|-----|---------|--------|
| Baseline SpacingMDN | -0.069 | 0.22 | OK |
| Memory Q3 (spacings) | -0.689 | 2.41 | NLL good, rollout bad |
| **Flash (residuals)** | **-0.712** | **0.26** | **BEST** |

### 1.2 Kernel Extraction (PySR)

**Damped exponential kernel (R²=0.99):**
```
μ(d) = (0.127·d + 0.062) × exp(-1.16·√d) + 0.0017
```
- **Linear term (0.127·d):** Level repulsion
- **Stretched exp (√d):** Spectral rigidity
- **Offset (0.0017):** Background

### 1.3 π-Conservation Law Discovery

**Best symbolic formula from distillation:**
```
              π
     s_n ≈ ─────────────────
           s₋₁ + s₋₂ + s₋₃

Constant: 3.1055553 ≈ π (1.1% error!)
```

**Physical interpretation:**
- Sum of 3 consecutive spacings → constant ≈ π
- Small spacings → next large (repulsion)
- Large spacings → next small (compression)
- **Consistent with GUE level repulsion!**

### 1.4 Progressive Masking

| Mask % | Active % | NLL | Status |
|--------|----------|-----|--------|
| 0% | 100% | 0.188 | Baseline |
| **40%** | **60%** | **0.161** | **BEST (-14.5%)** |
| 50% | 50% | 0.236 | Degraded |

**Key insight:** 60% of weights contain 100%+ of knowledge. Model is over-parameterized.

### 1.5 Attention Head Ablation

```
Critical heads (removing them hurts):
  L0.H3: +106% NLL (PRIMARY)
  L0.H2: +64%  NLL (PRIMARY)
  L0.H5: +32%  NLL (PRIMARY)

Noise heads (removing them HELPS):
  L1-L2: -20% to -52% NLL
```

**Core operator lives in Layer 0, Heads 2,3,5!**

### 1.6 RMT Memory Effect

- Elephant (memory): +63% SFF improvement
- Placebo (noise): -18% degradation
- Ratio: 1.98x
- **Memory mechanism is real, not artifact**

### 1.7 SFF Level Repulsion

- Real zeros plateau: 0.49
- Poisson plateau: 1.04
- **Suppression: 53%** (GUE signature)

### 1.8 2π Spike Validation

- Jitter test: 458% retention
- **Real physics, not binning artifact**

---

## 2. Debunked Claims

| Claim | Test | Result |
|-------|------|--------|
| Memory Bank → ln(primes) | Null test | p=0.60, random |
| Hermitian emergence | ||H-H^T|| test | score=0.994, non-Hermitian |
| Toeplitz → GUE | Sine kernel test | Even sine kernel gives Poisson |
| Memory helps rollout | Eval | Err@100: 2.41 vs 0.22 (FIXED with Flash!) |

---

## 3. Current State (January 2026)

### 3.1 Best Model: Flash

```
Checkpoint: out/mdn_memory_q3_flash/best.pt
Data: data/continuous_residuals (s - 1.0)
NLL: -0.712 (best)
Err@100: 0.26 (solved rollout!)
```

**What made Flash work:**
1. Residuals (centered at 0)
2. entropy_reg = 0.01 (2x higher)
3. RoPE positional encoding
4. seq_len = 512 (2x longer)

### 3.2 Data Infrastructure

```
data/continuous_clean/     # 200M zeros, spacings (mean=1)
data/continuous_residuals/ # 200M zeros, residuals (mean=0)
```

---

## 4. Roadmap to Operator Distillation

### Phase 1: Validate Flash Architecture [NEXT]

```
[ ] Masking analysis on Flash model
    → Find minimal core (expect 40-60%)
[ ] Attention ablation on Flash model
    → Identify critical heads
[ ] Compare with E4 results
```

**Script:**
```bash
PYTHONPATH=flash:$PYTHONPATH python scripts/masking_analysis.py \
  --ckpt out/mdn_memory_q3_flash/best.pt
```

### Phase 2: Symbolic Distillation from Flash

```
[ ] Extract MDN predictions (π, μ, σ)
[ ] Run PySR with extended operators
[ ] Test sinc, Bessel, and π-based formulas
```

**Expected formula (from prior work):**
```
              π
     s_n ≈ ─────────────────
           s₋₁ + s₋₂ + s₋₃
```

Or more complex:
```
s_n = 1 + Σ a_k * sinc(π * k * (s_{n-1} - μ))
```

### Phase 3: Toeplitz Kernel Construction

```
[ ] Extract attention logits A[i,j]
[ ] Fit K(d) = A(|i-j|) as function of distance
[ ] Compare with GUE sine kernel: sinc²(πd)
```

**Known limitation:** Toeplitz → Poisson, not GUE. Need breaking translational invariance.

### Phase 4: Operator Compression

```
[ ] Prune to critical heads (L0.H2, H3, H5)
[ ] Reduce to ~1.2M params (25% of original)
[ ] Retrain minimal model
[ ] Verify symbolic formula still holds
```

### Phase 5: Mathematical Formalization

```
[ ] Connect π-conservation to GUE pair correlation
[ ] Derive spacing predictor from R₂(s) = 1 - sinc²(πs)
[ ] Verify connection to Montgomery-Odlyzko
```

---

## 5. Key Questions to Answer

1. **Does Flash model have same structure as E4?**
   - Same critical heads?
   - Same compression ratio?

2. **Is π-conservation universal?**
   - Test on different heights (γ ranges)
   - Test on other L-functions

3. **Can we derive sinc kernel from attention?**
   - Attention logits → pair correlation?
   - Memory slots → global normalization?

4. **Why does Toeplitz fail for GUE?**
   - Need non-translation-invariant corrections?
   - Memory as symmetry breaker?

---

## 6. Files & Scripts

### Models
```
out/mdn_memory_q3_flash/best.pt  # BEST (NLL=-0.712, Err@100=0.26)
out/mdn_memory_q3_runpod/best.pt # Good NLL, bad rollout
out/mdn_clean_baseline/best.pt   # Baseline
```

### Analysis Scripts
```
scripts/masking_analysis.py      # Weight masking
scripts/attention_ablation.py    # Head importance
scripts/symbolic_distillation.py # PySR extraction
```

### Evaluation
```
eval_mdn.py                      # Standard eval
flash/eval_mdn_flash.py          # Flash eval (residuals)
ablation_memory_slots.py         # Slot ablation
```

---

## 7. Success Criteria

### Minimum Viable Result
- [ ] Symbolic formula with R² > 0.95
- [ ] π appears in formula with < 5% error
- [ ] Formula explains > 50% variance

### Stretch Goals
- [ ] sinc² kernel emerges from attention
- [ ] Operator construction produces GUE statistics
- [ ] Connection to Hilbert-Pólya conjecture

---

## 8. References

1. **Masters et al.** (2024) — Progressive masking distillation
2. **PySR** — Symbolic regression (Cranmer et al.)
3. **Montgomery** (1973) — Pair correlation of zeros
4. **Odlyzko** (1987) — Distribution of spacings

---

*Last updated: 2026-01-03*
*Project: Neural Telescope for Riemann Hypothesis*
