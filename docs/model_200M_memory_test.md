# Memory Map Q3: Testing Global Invariants for γ Prediction

**Date:** 2025-12-30
**Status:** Stage 0 - Proof of Concept

---

> **WARNING (2026-01-02):** Error accumulation baselines in this document are INCORRECT!
>
> The original eval used `context_len=10` but models were trained on `seq_len=256`.
> After fixing to `context_len=128`:
> - Err@100: 1.53 → **0.22** (7x better)
> - Err@200: 3.15 → N/A (seq too short for 128+200)
>
> See `SESSION_SUMMARY_2026_01_02.md` for corrected values.

---

## Overview

Testing whether **Memory Bank** (8 learnable global tokens) reduces **bias** and **drift** in MDN rollout predictions.

**Core Hypothesis:** Transformer attention alone cannot learn global invariants from local context (256 spacings ≈ 0.01% of 200M). Memory Bank provides explicit "global registers" that can store these invariants.

---

## 1. Data: Source, Preparation, Quality

### 1.1 Source

**Riemann zeta zeros** from Dave Platt's high-precision computations.

| Property | Value |
|----------|-------|
| Source | LMFDB / Platt .dat files |
| Total zeros | 200,000,000 |
| γ range | [14.13, 81,702,129.86] |
| Raw files | 101 .dat files (Platt binary format) |
| Location (raw) | `data/raw/*.dat` |

### 1.2 Preparation Pipeline

**Script:** `data/rebuild_from_dat.py`

**Steps:**
1. Read Platt binary .dat files (8-byte header + 32-byte block headers + 13 bytes/zero)
2. Extract γ values (high precision)
3. Compute **unfolded spacings**:
   ```
   u(γ) = (γ / 2π) * log(γ / 2πe)
   s_n = u(γ_{n+1}) - u(γ_n)
   ```
4. Split into sequences (seq_len=256)
5. Train/val split: last 10% = val (by index, not random)
6. Save as PyTorch tensors

**Output location:** `data/continuous_clean/`

```
data/continuous_clean/
├── train.pt      # 720 MB, 703,125 sequences
├── val.pt        # 80 MB, 78,124 sequences
└── meta.pt       # Metadata
```

**meta.pt contents:**
```python
{
    'n_zeros': 200_000_000,
    'gamma_min': 14.13,
    'gamma_max': 81_702_129.86,
    'spacing_mean': 1.0000001,  # PERFECT
    'spacing_std': 0.4133,      # close to GUE (0.42)
    'seq_len': 256,
    'n_train': 703_125,
    'n_val': 78_124,
    'source_files': 101
}
```

### 1.3 Quality Verification

**Date:** 2025-12-30

#### Test 1: Stationarity Check (10 chunks)

**Purpose:** Verify mean/std don't drift across height.

**Code:** Inline Python (see session log)

**Results (TRAIN):**

| Chunk | Mean | Std | Max | >3.0 |
|-------|------|-----|-----|------|
| 00 | 1.000000 | 0.4108 | 3.469 | 0.0018% |
| 01 | 1.000000 | 0.4125 | 3.563 | 0.0025% |
| ... | 1.000000 | ~0.413 | ~3.5 | ~0.003% |
| 09 | 1.000000 | 0.4142 | 3.541 | 0.0028% |

**Verdict:** ✅ **No drift.** Mean exactly 1.0 in all chunks.

#### Test 2: Train vs Val Comparison

| Dataset | Mean | Std |
|---------|------|-----|
| Train | 1.000000 | 0.4133 |
| Val (tail) | 1.000000 | 0.4143 |
| **Δ** | 0.000000 | 0.001 |

**Verdict:** ✅ **Identical distributions.** Val is valid held-out tail.

#### Test 3: Distribution vs Wigner Surmise (GUE)

**GUE Wigner surmise:**
```
P(s) = (32/π²) * s² * exp(-4s²/π)
```

**Results:**
- KL divergence: 0.000474 (nearly zero)
- L1 distance: 0.0246

**Plot:** `reports/spacing_vs_wigner.png`

**Verdict:** ✅ **Data matches GUE almost perfectly.**

#### Test 4: Autocorrelation

| Lag | Correlation |
|-----|-------------|
| 1 | -0.354 (repulsion) |
| 2 | -0.076 |
| 3 | -0.038 |
| 5 | -0.020 |
| 10 | +0.010 |

**Verdict:** ✅ **Expected GUE-like repulsion at lag-1, fast decay.**

#### Test 5: Data Corruption Check

**Issue found (before rebuild):** 24 gaps with ~112K missing zeros in old data.

**After rebuild from .dat files:** 0 gaps, 0 anomalies.

```
max spacing: 3.633
count > 10: 0
```

**Verdict:** ✅ **Clean data after rebuild.**

---

## 2. Completed Experiments & Results

### 2.1 Baseline MDN Training

**Date:** 2025-12-30
**Script:** `train_mdn.py`
**Output:** `out/mdn_clean_baseline/`

**Command:**
```bash
python train_mdn.py \
  --data-dir data/continuous_clean \
  --out-dir out/mdn_clean_baseline \
  --n-layer 6 --n-head 8 --n-embd 256 \
  --n-components 8 \
  --batch-size 256 --lr 3e-4 \
  --warmup-steps 1500 --entropy-reg 0.005 \
  --max-steps 20000
```

**Training curve:**

| Step | NLL | MAE | H |
|------|-----|-----|---|
| 1000 | 0.383 | 0.290 | 1.99 |
| 5000 | 0.207 | 0.244 | 2.06 |
| 10000 | 0.090 | 0.218 | 2.05 |
| 15000 | -0.018 | 0.196 | 2.05 |
| 19000 | -0.069 | 0.186 | 2.05 |
| 20000 | -0.060 | 0.188 | 2.05 |

**Best checkpoint:** step 19000, NLL=-0.0690

**Checkpoints saved:**
```
out/mdn_clean_baseline/
├── best.pt         # step 19000
├── final.pt        # step 20000
├── ckpt_5000.pt
├── ckpt_10000.pt
├── ckpt_15000.pt
└── ckpt_20000.pt
```

### 2.2 Baseline MDN Evaluation

**Date:** 2025-12-30 (Updated 2026-01-02 with `context_len=128`)
**Script:** `eval_mdn.py`
**Checkpoint:** `out/mdn_clean_baseline/best.pt`

**Command:**
```bash
python eval_mdn.py \
  --ckpt out/mdn_clean_baseline/best.pt \
  --data-dir data/continuous_clean \
  --n-pit 20000 --n-crps 2000
```

**Results:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| PIT mean | 0.481 | 0.5 | ⚠️ slight bias |
| PIT std | 0.279 | 0.289 | ✅ good |
| CRPS | 0.129 | ↓ | ✅ |

**Rollout Error Accumulation (Corrected context_len=128):**

| Horizon | Mean Err | Sample Err |
|---------|----------|------------|
| h=10 | 0.18 | ? |
| h=25 | 0.20 | ? |
| h=50 | 0.23 | ? |
| h=100 | 0.22 | ? |

**Key Finding:** Error accumulation is much more stable with proper context length. Baseline error @ h=100 is ~0.22.

**Interpretation:** Model is probabilistically correct and relatively stable for long horizons when given sufficient context.

**Plots saved:**
```
reports/mdn/
├── pit_histogram.png
└── error_accumulation.png
```

### 2.3 σ Diagnostics (No Collapse)

**Purpose:** Verify model doesn't overfit via σ-collapse.

**Results:**
- Entropy H ≈ 2.05 (max = log(8) = 2.08)
- All K=8 components used
- σ not hitting floor

**Verdict:** ✅ **Healthy mixture, no collapse.**

---

## 3. Known Issues Fixed

### 3.1 eval_mdn.py Bugs (Fixed 2025-12-30)

**Bug 1:** Error accumulation used wrong index range
```python
# OLD (WRONG):
true_cum = seq[0, :horizon].cumsum(0)[-1].item()

# NEW (CORRECT):
true_cum = seq[0, 10:10+horizon].sum().item()
```

**Bug 2:** CRPS used unaligned targets
```python
# OLD (WRONG):
y = subset[b + 1, 0].item()  # next batch item!

# NEW (CORRECT):
y = targets[b, t].item()  # aligned target from same sequence
```

### 3.2 Data Corruption (Fixed 2025-12-30)

**Issue:** Old zeros_100M.txt had 24 gaps with ~112K missing zeros at γ=xxx999 boundaries.

**Fix:** Rebuilt from original Platt .dat files using `data/rebuild_from_dat.py`.

### 3.3 LR Multiplier Bug (Fixed 2025-12-30)

**Issue:** Memory LR identified by index `i==1` (fragile).

**Fix:** Added `is_memory` flag to param groups.

---

## Memory Map Q3: 8 Invariant Slots

Each slot maps to a Q3 operator invariant:

| Slot | Q3 Name | Lens Metaphor | ML Role |
|------|---------|---------------|---------|
| **M0** | SIGN | Polarity of correction | arch - prime direction |
| **M1** | NORM | Scale calibration | mean=1 globally |
| **M2** | TORUS | Translation invariance | only distances matter |
| **M3** | SYMBOL | Kernel/PSF shape | global fingerprint |
| **M4** | FLOOR | Uncertainty floor | anti-σ-collapse |
| **M5** | TOEPLITZ | Discretization stability | seq_len robustness |
| **M6** | PRIME-CAP | Limit global correction | memory doesn't dominate |
| **M7** | GOAL | Stability margin | rollout health |

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Memory Bank: [M0, M1, ..., M7]         │
│  (8 learnable vectors, n_embd each)     │
│                    ↓                    │
│  Concat: [M0..M7, s1..sT]               │
│                    ↓                    │
│  Transformer (6L/8H/256E)               │
│  - Memory tokens participate in attn    │
│  - Causal mask for sequence tokens      │
│                    ↓                    │
│  MDN Head (only on s1..sT)              │
│  - K=8 Gaussian components              │
│  - Outputs: π, μ, σ                     │
└─────────────────────────────────────────┘
```

---

## Regularizations (Anti-Collapse)

| Regularizer | Purpose | Implementation |
|-------------|---------|----------------|
| **diversity_loss** | Slots stay orthogonal | `L = ((GramMatrix - I)^2).mean()` |
| **memory_cap** | No slot dominates | `norm(slot) ≤ cap` |
| **slot_dropout** | Each slot independently useful | 10% dropout per batch |
| **memory_lr_mult** | Slow learning prevents early collapse | LR × 0.1 for memory |

---

## Staged Experiment Plan

### Stage 0: Memory as Class (CURRENT)

**Question:** Does having 8 global memory slots reduce bias/drift?

**Setup:**
- 8 **unnamed** learnable slots (permutation symmetric)
- No role enforcement
- diversity + cap + dropout regularization

**Success Criteria:**
- PIT mean: 0.483 → closer to 0.5
- Err@100: 1.53 → decrease by 10-20%
- Err@200: 3.15 → decrease

**If Stage 0 fails:** Memory Bank concept doesn't help → reconsider approach.

---

### Stage 1: Role Identification (Post-hoc Probe)

**Question:** Can we identify which slot learned which invariant?

**Method:**
```python
for slot_idx in range(8):
    response = measure_slot_response(model, slot_idx, test_scenarios)
    # test_scenarios:
    #   - "high variance input" → M4 FLOOR candidate
    #   - "rescaled input" → M1 NORM candidate
    #   - "long rollout" → M7 GOAL candidate
```

**Success Criteria:**
- Clear specialization visible in slot responses
- Slots interpretable as Q3 invariants

---

### Stage 2: Role Enforcement (Auxiliary Losses)

**Question:** Can we force specific slots to specific roles?

**Method:**
```python
# M1 (NORM): stable under rescaling
loss_m1 = stability_under_rescaling(slot[1])

# M4 (FLOOR): correlates with uncertainty
loss_m4 = -correlation(slot[4], sigma.mean())

# M7 (GOAL): correlates with rollout stability
loss_m7 = -correlation(slot[7], rollout_error)
```

**Success Criteria:**
- Enforced roles improve metrics beyond Stage 0
- Interpretability maintained

---

## Baseline Reference (clean-200M)

**Model:** 6L/8H/256E, K=8 MDN, 4.8M params
**Training:** 20k steps on data/continuous_clean

| Metric | Baseline Value |
|--------|----------------|
| Best NLL | -0.0690 |
| PIT mean | 0.481 (target: 0.5) |
| PIT std | 0.279 (target: 0.289) |
| CRPS | 0.129 |
| Err@10 (mean) | 0.18 |
| Err@25 (mean) | 0.20 |
| Err@50 (mean) | 0.23 |
| Err@100 (mean) | 0.22 |

**Key Observation:** Proper context (128) is critical for autoregressive stability. Baseline error stays below 0.25 even at h=100.

---

## Stage 0 Training Command

```bash
source .venv/bin/activate

python train_memory_q3.py \
  --data-dir data/continuous_clean \
  --out-dir out/mdn_memory_q3 \
  --n-layer 6 --n-head 8 --n-embd 256 \
  --n-components 8 \
  --n-memory 8 \
  --memory-dropout 0.1 \
  --memory-cap 2.0 \
  --diversity-weight 0.01 \
  --memory-lr-mult 0.1 \
  --max-steps 20000 \
  --eval-interval 1000
```

---

## Evaluation Protocol (Stage 0 Final Results)

**Date:** 2026-01-02

| Metric | Baseline (step 19k) | Memory Q3 (step 20k) | Δ |
|--------|---------------------|----------------------|---|
| NLL | -0.0690 | **-0.6889** | -0.62 (better) |
| CRPS | 0.1290 | **0.0940** | -27% (better) |
| PIT mean | 0.481 | 0.475 | -0.006 (worse) |
| Err@10 | **0.18** | 0.29 | +0.11 (worse) |
| Err@100 | **0.22** | 2.41 | +2.19 (11x worse) |

**Conclusion:** Stage 0 shows that while Memory Bank Q3 significantly improves local probabilistic accuracy, it creates massive instability in autoregressive rollout. This "over-specialization" suggests that the learnable memory slots are not yet capturing global invariants in a stable way.

### Next Steps After Stage 0

1. **Diversity weight scan:** Does higher diversity regularization help stability?
2. **Memory Dropout tuning:** Increase dropout to force more robust representations.
3. **Stage 1 (Probing):** Identify which of the 8 slots is causing the drift.

---

## Files Created

```
model/memory_mdn.py     # MemoryMDN + MemoryBank classes
train_memory_q3.py      # Training script with Q3 args
docs/model_200M_memory_test.md  # This document
```

---

## Technical Notes

### LR Multiplier Fix
Memory params identified by `is_memory` flag (not index) to avoid bugs:

```python
for param_group in optimizer.param_groups:
    if param_group.get('is_memory', False):
        param_group['lr'] = lr * memory_lr_mult
```

### Permutation Symmetry
Current slots are **not** role-enforced. Model can internally "rename" slots.
This is intentional for Stage 0 — proving memory helps before adding complexity.

### Why 8 Slots?
Maps to Q3's 8 key invariants. But could test 4 slots (M1/M4/M5/M7) for faster iteration.

---

## Next Steps After Stage 0

1. **If success:** Move to Stage 1 (probe slot specialization)
2. **If partial success:** Tune regularization weights
3. **If failure:**
   - Try fewer slots (4)
   - Try different initialization
   - Reconsider if global memory is the right approach

---

## Q3 Glossary

| Term | Meaning |
|------|---------|
| **Transformer** | Learnable "lens" that aggregates context |
| **MDN** | Probabilistic "focus function" F: outputs distribution |
| **Memory Bank** | Global "registers" for invariants |
| **PIT** | Probability Integral Transform: calibration test |
| **CRPS** | Continuous Ranked Probability Score |
| **Rollout** | Multi-step prediction on own outputs |
| **Bias** | Systematic prediction offset (focus shift) |
| **Drift** | Accumulated bias over rollout (defocus) |
| **Spectral Gap** | Stability margin (rollout health) |
