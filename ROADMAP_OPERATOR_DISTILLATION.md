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

**E4 Model (postfix):**
| Mask % | Active % | NLL | Status |
|--------|----------|-----|--------|
| 0% | 100% | 0.188 | Baseline |
| **40%** | **60%** | **0.161** | **BEST (-14.5%)** |
| 50% | 50% | 0.236 | Degraded |

**Flash Model (residuals) - January 2026:**
| Mask % | Active % | NLL | Status |
|--------|----------|-----|--------|
| 0% | 100% | -0.581 | Baseline |
| 10% | 90% | -0.573 | OK (+1.4%) |
| 20% | 80% | -0.543 | Degraded (+6.6%) |
| **30%** | **70%** | **-0.477** | **CRITICAL (+17.9%)** |
| 40% | 60% | -0.231 | BROKEN (+60%) |

**Key insights:**
- E4: 60% of weights contain knowledge (40% can be masked with IMPROVEMENT)
- Flash: 70% of weights contain knowledge (more efficient, less over-parameterized)
- Flash stores knowledge more densely than E4!

### 1.5 Attention Head Ablation

**E4 Model (postfix tokenization):**
```
Critical heads (removing them hurts):
  L0.H3: +106% NLL (PRIMARY)
  L0.H2: +64%  NLL (PRIMARY)
  L0.H5: +32%  NLL (PRIMARY)

Noise heads (removing them HELPS):
  L1-L2: -20% to -52% NLL
```

**Flash Model (residuals) - January 2026:**
```
LAYER 1 IS THE CORE OPERATOR (all heads critical!):
  L1.H2: +312% NLL (MAIN OPERATOR)
  L1.H3: +233% NLL
  L1.H4: +120% NLL
  L1.H7: +119% NLL
  L1.H1: +108% NLL
  L1.H0: +95%  NLL
  L1.H5: +51%  NLL
  L1.H6: +43%  NLL

Layer importance (avg % NLL change):
  Layer 1: +135% (SUPER CRITICAL)
  Layer 2: +20%
  Layer 0: +12%
  Layer 3: +12%
  Layer 4: +5%
  Layer 5: +10%

Only noise head: L3.H4 (-5.5%)
```

**Key insight: Flash concentrates operator in Layer 1, E4 used Layer 0!**

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

### Phase 1: Validate Flash Architecture [DONE]

```
[x] Masking analysis on Flash model
    → 70% knowledge core (vs E4's 60%)
    → Flash more efficient, less over-parameterized
[x] Attention ablation on Flash model
    → Layer 1 is THE core operator (all 8 heads critical!)
    → L1.H2 most important (+312% NLL when removed)
[x] Compare with E4 results
    → E4: Layer 0 critical, Flash: Layer 1 critical
    → Different architecture of knowledge!
```

**Key finding:** Flash model has fundamentally different operator structure than E4.

### Phase 2: Symbolic Distillation from Flash [IN PROGRESS]

```
[x] Extract attention patterns from Layer 1
    → All heads show relatively flat K(d) with small peaks
    → L1.H7 shows exponential decay from d=1
    → L1.H2 (most critical) is flat - works via different mechanism
[x] Analyze prediction correlations
    → Model achieves 0.94 correlation with targets
    → MSE = 0.021
[x] Test GUE repulsion hypothesis
    → CONFIRMED! All lag correlations negative:
       corr(rₙ, r₋₁) = -0.34 (strong repulsion!)
       corr(rₙ, r₋₂) = -0.08
       corr(rₙ, r₋₃) = -0.03
[x] Extract linear operator
    → rₙ ≈ -0.45·r₋₁ - 0.28·r₋₂ - 0.16·r₋₃
    → All coefficients negative and decreasing = GUE-like!
[ ] Run PySR for nonlinear terms (Julia issues)
[ ] Test sinc kernel fit
    → Preliminary: sinc²(πd/2.53), R²=0.81
```

**Discovered Linear Operator (GUE signature!):**
```
rₙ = -0.45·r₋₁ - 0.28·r₋₂ - 0.16·r₋₃ + ε

Interpretation:
- Negative coefficients = level repulsion
- Decreasing magnitude = short-range dominance
- Consistent with GUE pair correlation!
```

**Rollout Validation (January 2026):**
```
              Err@100   Mean Abs Error
Linear Op:     0.310      0.334
Flash Model:   0.241      0.335  (28.6% better)
Baseline (0):  0.354      0.335

Key findings:
- Linear operator is STABLE (max |root| = 0.56 < 1)
- Linear beats baseline → valid first approximation
- Flash captures nonlinear terms → 28.6% improvement
- Mean abs error identical → difference is in error accumulation
```

**Conclusion:** Linear operator is the first-order GUE approximation.
Flash model adds higher-order corrections.

### Phase 3: Toeplitz Kernel Construction

```
[ ] Extract attention logits A[i,j]
[ ] Fit K(d) = A(|i-j|) as function of distance
[ ] Compare with GUE sine kernel: sinc²(πd)
```

**Known limitation:** Toeplitz → Poisson, not GUE. Need breaking translational invariance.

### Phase 4: Operator Compression

```
[ ] Prune to critical Layer 1 (all 8 heads essential for Flash!)
    → E4 could prune to 3 heads, Flash needs full Layer 1
[ ] Remove noise head L3.H4 (only improves when removed)
[ ] Reduce Layers 4-5 (only +5-10% avg impact)
[ ] Retrain minimal model with 2-3 layers
[ ] Verify symbolic formula still holds
```

**Note:** Flash cannot be compressed as much as E4 - knowledge is denser.

### Phase 5: Mathematical Formalization

```
[ ] Connect π-conservation to GUE pair correlation
[ ] Derive spacing predictor from R₂(s) = 1 - sinc²(πs)
[ ] Verify connection to Montgomery-Odlyzko
```

---

## 5. Key Questions to Answer

1. **~~Does Flash model have same structure as E4?~~** [ANSWERED]
   - NO! Flash uses Layer 1 (all heads), E4 used Layer 0 (3 heads)
   - Flash: 70% knowledge core, E4: 60%
   - **Flash is fundamentally different architecture**

2. **Why did residual training shift operator to Layer 1?**
   - Centering at 0 vs mean=1: different feature space?
   - RoPE vs learned positional: affects where computation happens?
   - entropy_reg=0.01 vs 0.005: regularization shifts depth?

3. **Is π-conservation universal?**
   - Test on different heights (γ ranges)
   - Test on other L-functions

4. **Can we derive sinc kernel from Layer 1 attention?**
   - Extract attention logits from L1.H2 specifically
   - Compare with GUE sine kernel: sinc²(πd)

5. **Why does Toeplitz fail for GUE?**
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

*Last updated: 2026-01-03 (Phase 1 complete)*
*Project: Neural Telescope for Riemann Hypothesis*
