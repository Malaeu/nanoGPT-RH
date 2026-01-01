# PROJECT MAP — nanoGPT_RH

> Автогенерируемая карта проекта. Обновляется после каждого значимого изменения.

**Last Updated:** 2026-01-01 (E4 implementation)
**Current Focus:** E4 Register Stabilization & ID-Detox

---

## Experiments Timeline

| ID | Architecture | Status | Best NLL | Key Insight |
|----|--------------|--------|----------|-------------|
| E0 | SpacingMDN (no memory) | ✅ completed | ~-0.25 | Baseline MDN |
| E1 | PREFIX Memory (8 slots) | ✅ completed | -0.3793 | Memory "decorative" |
| E2 | PREFIX Memory (8 slots) | ✅ completed | -0.3840 | Same as E1, seed variance |
| E3 | POSTFIX Memory | ✅ completed | 0.304 | +14% vs E2, but ID-crutch! |
| **E4** | **POSTFIX + ID-Detox** | ✅ **DONE** | **0.1942** | **+36% vs E3! No ID-crutch!** |

### E3 Final Results
| Seed | Best NLL | Ablation Δ | Grad Corr | Perm Inc | Issue |
|------|----------|------------|-----------|----------|-------|
| 7 | 0.304 | +0.035 | 0.417 | 93% | ID-reliance |
| 42 | 0.340 | +0.005 | 0.253 | 117% | ID-reliance |
| 1337 | 0.394 | -0.013 | 0.234 | 13% | Weak causality |

**E3 Conclusion:** NLL improved, but Permutation test shows model relies on slot-ID embeddings.

### E4 Final Results ✅ COMPLETED!
| Seed | Best NLL | vs E3 | Steps | Status |
|------|----------|-------|-------|--------|
| 7 | **0.1942** | **+36%** 🏆 | 10000 | BEST |
| 42 | 0.3294 | -8% | 7000 | stuck |
| 1337 | 0.3431 | -13% | 9000 | stuck |

**E4 Configuration:**
- `slot_id_mode=permute_per_batch` (ID-detox)
- `use_aux_loss=True` (Q3-proxy supervision)
- `early_stop=True, patience=800`

**E4 Conclusion:**
- s7 achieved **0.1942** — 36% better than E3 best!
- Model learns WITHOUT ID-crutch
- permute_per_batch works!
- High seed variance (only 1/3 found good minimum)

### E4 s7 Diagnostics ✅
| Metric | Value | Status | Comment |
|--------|-------|--------|---------|
| **NLL** | 0.1721 | ✅ | Best ever |
| **Slot-ID Reliance** | 1.0% | ✅ **VICTORY!** | ID-Detox works! |
| **Grad Corr** | 0.296 | ✅ | Was 0.9 in E1/E2 |
| **Mean Slot Sim** | 0.167 | ✅ | Slots differentiated |
| **Effect Entropy** | 1.97 | ✅ | Good distribution |
| **Error Growth** | 0.076 | ✅ | Low drift |
| **Max Ablation Δ** | 0.0058 | ⚠️ | Target >= 0.02 |
| **Cross-Block CV** | 0.644 | ⚠️ | Distribution shift |
| **CoM std** | 4.3 | ⚠️ | Slots don't specialize |

**Key Finding:** ID-Detox **works**! Perm Inc dropped from 93-117% (E3) to **1.0%** (E4).
Model uses slot **content**, not slot **ID**. This was the main E4 goal.

**Remaining Issue:** Low ablation Δ means slots are redundant (robust but not specialized)

---

## Architecture Evolution

```
E1/E2 PREFIX (broken):
┌─────────────────────────────────────────────────────┐
│ [M0..M7] [s₁, s₂, ..., s_T]                         │
│    ↑          ↑                                      │
│  Memory     Data                                     │
│    │          │                                      │
│    ╳──────────┘  Memory CAN'T see data (causal!)   │
│                   Data CAN see memory (useless)     │
└─────────────────────────────────────────────────────┘
Problem: Memory is "blind" → becomes decoration

E3 POSTFIX (working):
┌─────────────────────────────────────────────────────┐
│ [s₁, s₂, ..., s_T] [M0..M7]                         │
│        ↑              ↑                              │
│      Data          Memory                            │
│        │              │                              │
│        └──────────────┤  Memory CAN see data ✓      │
│                       │  Data CAN'T see memory ✓    │
│                       ↓                              │
│                   [READOUT]  ← bottleneck!          │
│                       ↓                              │
│                   MDN Head → predict s_{T+1}        │
└─────────────────────────────────────────────────────┘
Result: Memory becomes ESSENTIAL (true register)
```

---

## Repository Structure (Jan 2026)

```
nanoGpt_RH/
├── 📁 src/                      # MAIN CODE
│   ├── train_mdn.py             # Base SpacingMDN (MDNConfig)
│   ├── train_mdn_postfix.py     # ⭐ E4 training (ID-Detox, aux-loss)
│   ├── train_mdn_memory.py      # PREFIX memory (deprecated)
│   ├── eval_mdn.py              # Evaluation (NLL, CRPS, PIT)
│   ├── diagnose_memory.py       # PREFIX diagnostics
│   └── diagnose_memory_postfix.py # ⭐ POSTFIX diagnostics (A-K)
│
├── 📁 scripts/                  # UTILITIES
│   ├── prepare_continuous_2M.py # Unfolding zeros → spacings
│   ├── prepare_zeros.py         # Raw zeros processing
│   ├── prepare_primes.py        # Prime gaps dataset
│   └── runpod_setup.sh          # RunPod setup script
│
├── 📁 checkpoints/              # TRAINED MODELS (Git LFS)
│   ├── E0_baseline_best.pt      # SpacingMDN no memory
│   ├── E1_prefix_best.pt        # PREFIX (decorative)
│   ├── E2_prefix_best.pt        # PREFIX (seed variance)
│   ├── E3_postfix_s1337_best.pt # POSTFIX (ID-crutch)
│   ├── E4_s7_best.pt            # ⭐ BEST! NLL=0.1942
│   └── E4_s1337_best.pt         # E4 (stuck seed)
│
├── 📁 data/                     # DATASET
│   └── continuous_2M/           # (7035, 256) train, (781, 256) val
│
├── 📁 docs/                     # DOCUMENTATION
│   ├── PROJECT_MAP.md           # ⭐ THIS FILE
│   ├── E4_SPEC.md               # E4 specification
│   └── runpod_specs.md          # GPU comparison
│
├── 📁 results/                  # DIAGNOSTICS OUTPUT
│   └── E4_s7/                   # jsonl + png
│
└── 📁 archive/                  # OLD CODE (gitignored)
```

### E4 Training Flags
```bash
python src/train_mdn_postfix.py \
  --slot-id-mode permute_per_batch  # ID-detox
  --content-mode normal             # or zeroed
  --use-aux-loss                    # Q3-proxy supervision
  --early-stop --patience 800       # early stopping
```

---

## Implementation Checklist

### E3 Core Architecture ✅
- [x] POSTFIX layout: `[data..., memory...]`
- [x] Memory sees data (causal mask OK)
- [x] Data doesn't see memory (blocked)
- [x] Memory-only readout (bottleneck)
- [x] Learnable weighted pooling
- [x] Slot-ID embeddings
- [x] Single-step prediction (seq_len=257)
- [x] Ablation support (slot_off param)

### E4 ID-Detox ✅ (implemented!)
- [x] `--slot-id-mode fixed|off|permute_per_batch`
- [x] `--content-mode normal|zeroed`
- [x] Permute slot-IDs per batch during training
- [x] Zeroed content mode for ID-only test

### E4 Q3 Aux Supervision ✅ (implemented!)
- [x] `compute_q3_targets()` function
- [x] M0: mean(x)-1 (T0 normalization)
- [x] M1: hist_entropy (A1' coverage)
- [x] M2: max|dx| (A2 Lipschitz)
- [x] M3: quantile 0.01 (A3 floor)
- [x] M4: mean|d²x| (smoothness)
- [x] M5: half_window_divergence (Toeplitz)
- [x] M6: high_freq_energy (RKHS cap)
- [x] M7: local_rigidity (spectral gap)
- [x] z-score normalization
- [x] aux_loss MSE
- [x] Ramp schedule (0→1e-3→1e-2)

### E4 Early Stopping ✅
- [x] `--early-stop --patience 800`
- [x] Monitor val_nll

### Regularization ❌ (not planned for E4)
- [ ] Orthogonality loss (slots don't collapse)
- [ ] Norm cap (slots don't dominate)

### Extended Diagnostics ✅ (diagnose_memory_postfix.py)

**Core Metrics (A-F):**
- [x] A) Ablation study (slot_off → NLL delta)
- [x] B) Slot similarity matrix
- [x] C) Gradient correlation between slots
- [x] D) Readout weights visualization
- [x] E) Slot effect norm (entropy)
- [x] F) Slot norms visualization

**Extended Metrics (G-K):**
- [x] G) Rollout Drift (Err@h horizons + error_growth_slope)
- [x] H) Cross-Block Test (distribution shift detection)
- [x] I) Slot Attention Profile (CoM, receptive field)
- [x] J) Permutation Sanity (slot-ID reliance test)
- [x] K) Gradient Rank/PCA (effective dimensionality)

**In eval_mdn.py:**
- [ ] CRPS metric
- [ ] PIT calibration

---

## Key Insights Log

### 2025-12-28: E1/E2 Analysis
- **Problem identified:** PREFIX memory can't see data due to causal mask
- **Evidence:** Ablation Δ ≈ 0, grad_corr ≈ 0.9
- **Conclusion:** Memory is decorative, not causal

### 2025-12-31: E3 POSTFIX Design
- **Solution:** Put memory AFTER data
- **Key change:** Memory-only readout creates bottleneck
- **Prediction:** Ablation Δ should increase, grad_corr should decrease

### 2026-01-01: E3 Early Results
- **Observation:** NLL improved by 5-10% vs E2
- **Status:** Training in progress, 3 seeds running
- **Next:** Wait for completion, run diagnostics

### 2026-01-01: E3 Progress Update
- **Results:** s7 leading with -0.3286 (+14.4% vs E2!)
- **Created:** `diagnose_memory_postfix.py` for POSTFIX diagnostics
- **Next:** Run diagnostics when training completes

### 2026-01-01: Extended Diagnostics G-K
- **Goal:** Distinguish "seed variance luck" from "architecture found real law"
- **Added:** G) Rollout Drift, H) Cross-Block, I) Attention Profile, J) Permutation, K) Gradient Rank
- **Key tests:**
  - G) Error growth slope < 0.5 → stable predictions
  - H) Cross-block CV < 0.1 → no distribution shift
  - I) CoM std > 10 → slots specialize to different positions
  - J) Permutation NLL increase < 5% → not relying on slot-ID
  - K) rank_ratio > 0.5 → slots learn independently

### 2026-01-01: Bug Fixes in diagnose_memory_postfix.py
- **Fixed:** Unified sampler `sample_xy()` — no more (x,y) misalignment
- **Fixed:** MDNConfig import from `train_mdn.py`
- **Fixed:** JSONL keys (`blocks`, `slot_profiles`)
- **Fixed:** 2D data handling in rollout/cross_block (`s = data.flatten()`)
- **Fixed:** Permutation test wrapped in `torch.no_grad()`
- **Added:** `--attn-layers last|mean3|meanAll` for stable attention profile
- **Added:** `--ckpt-glob` for automatic seed aggregation with mean±std table

### 2026-01-01: E4 Implementation
- **Problem:** E3 Permutation test shows 93-117% NLL increase → ID-crutch detected
- **Solution:** ID-Detox via `--slot-id-mode permute_per_batch`
- **Added features:**
  - `slot_id_mode`: fixed | off | permute_per_batch
  - `content_mode`: normal | zeroed (sanity tests)
  - Q3-proxy aux loss with ramp schedule
  - Early stopping with patience
- **Created:** `docs/E4_SPEC.md` with full specification
- **Success criteria:**
  - Ablation Δ >= 0.02 on 2/3 seeds
  - ID-only test: ΔNLL >= 0.02
  - Content-only degradation <= 30%

### 2026-01-01: E4 Diagnostics Complete
- **Result:** E4 s7 achieved NLL=0.1721 (36% better than E3!)
- **ID-Detox:** ✅ **WORKS!** Perm Inc dropped from 93-117% → **1.0%**
- **Partial success:**
  - ✅ Grad Corr = 0.296 (was 0.9 in E1/E2)
  - ✅ Effect Entropy = 1.97 (good distribution)
  - ✅ Error Growth = 0.076 (low drift)
  - ⚠️ Ablation Δ = 0.0058 (target was 0.02)
  - ⚠️ Cross-Block CV = 0.644 (distribution shift)
- **Conclusion:** Model learns from slot content, not ID. Slots are redundant (robust).

---

## Terminology

| Term | Meaning |
|------|---------|
| PREFIX | Memory slots BEFORE data (E1/E2, broken) |
| POSTFIX | Memory slots AFTER data (E3, working) |
| Bottleneck | Prediction only from memory (no shortcut) |
| Ablation Δ | NLL change when slot zeroed out |
| Grad corr | Gradient correlation between slots (~1 = same learning) |
| MDN | Mixture Density Network (outputs distribution) |
| NLL | Negative Log-Likelihood (lower = better) |

---

## RunPod Commands

```bash
# Package for RunPod (E4)
tar czf runpod_e4.tar.gz train_mdn_postfix.py train_mdn.py data/continuous_2M
runpodctl send runpod_e4.tar.gz

# On Pod
runpodctl receive <CODE> && tar xzf runpod_e4.tar.gz
pip install torch numpy scipy matplotlib rich

# E4 Training (ID-Detox + Aux Loss)
python train_mdn_postfix.py \
    --data-dir data/continuous_2M \
    --out-dir out/mdn_postfix_E4_s1337 \
    --seed 1337 \
    --slot-id-mode permute_per_batch \
    --use-aux-loss \
    --early-stop \
    --patience 800 \
    --batch-size 512 \
    --use-amp

# Download results
tar czf e4_results.tar.gz out/ && runpodctl send e4_results.tar.gz
```

---

## Next Steps

1. ✅ **E3 completed** — NLL improved but ID-crutch detected
2. ✅ **E4 implemented** — ID-Detox + Q3-aux loss ready
3. ✅ **E4 trained on RunPod** — s7 achieved 0.1942 (BEST!)
4. ✅ **E4 diagnostics** — ID-Detox works! Perm Inc = 1.0%
5. **E4 Partial Success:**
   - ✅ ID-only: model uses content, not ID (Perm Inc 1%)
   - ⚠️ Ablation Δ = 0.0058 < 0.02 target (slots redundant)
   - ⚠️ Cross-block CV = 0.644 (distribution shift)
6. **Options for E5:**
   - A) **Slot specialization** — add orthogonality loss to force different roles
   - B) **Slot dropout** — randomly drop slots during training
   - C) **Block normalization** — fix distribution shift issue
   - D) **Move to extraction** — model is good enough, start symbolic regression
7. **Default GPU:** L40S @ $0.86/hr (ML-optimized, high availability)
