# PROJECT MAP — nanoGPT_RH

> Автогенерируемая карта проекта. Обновляется после каждого значимого изменения.

**Last Updated:** 2026-01-01
**Current Focus:** E3 POSTFIX Memory Experiments

---

## Experiments Timeline

| ID | Architecture | Status | Best NLL | Key Insight |
|----|--------------|--------|----------|-------------|
| E0 | SpacingMDN (no memory) | ✅ completed | ~-0.25 | Baseline MDN |
| E1 | PREFIX Memory (8 slots) | ✅ completed | -0.3793 | Memory "decorative" |
| E2 | PREFIX Memory (8 slots) | ✅ completed | -0.3840 | Same as E1, seed variance |
| **E3** | **POSTFIX Memory** | 🔄 running | **-0.3453** | **+10% vs E2!** |

### E3 Seeds Status (RunPod RTX 6000 Ada 48GB)
| Seed | Step | Best NLL | vs E2 (-0.384) |
|------|------|----------|----------------|
| 1337 | 4500 | -0.3453 | +10.1% |
| 42 | 3000 | -0.3484 | +9.3% |
| 7 | 1000 | -0.3646 | +5.1% |

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

## File Map

### Core Training Scripts
| File | Purpose | Architecture | Status |
|------|---------|--------------|--------|
| `train_mdn.py` | Base SpacingMDN | No memory | ✅ stable |
| `train_mdn_memory.py` | PREFIX memory | E1/E2 | ✅ deprecated |
| `train_mdn_postfix.py` | **POSTFIX memory** | **E3** | ✅ active |

### Evaluation & Diagnostics
| File | Purpose | Status |
|------|---------|--------|
| `eval_mdn.py` | NLL, CRPS, PIT metrics | ✅ works |
| `diagnose_memory.py` | Ablation, grad corr, slot analysis | ⚠️ needs postfix update |
| `test_slot_effect.py` | Per-slot ablation | ⚠️ needs postfix update |

### Data
| Path | Contents | Shape |
|------|----------|-------|
| `data/continuous_2M/train.pt` | Training spacings | (7035, 256) |
| `data/continuous_2M/val.pt` | Validation spacings | (781, 256) |

### Checkpoints (RunPod)
```
/workspace/out/
├── mdn_postfix_E3_s1337/
│   ├── best.pt
│   ├── train.log
│   └── memory_diagnostics.pt
├── mdn_postfix_E3_s42/
└── mdn_postfix_E3_s7/
```

---

## Implementation Checklist (E3 ТЗ)

### Core Architecture ✅
- [x] POSTFIX layout: `[data..., memory...]`
- [x] Memory sees data (causal mask OK)
- [x] Data doesn't see memory (blocked)
- [x] Memory-only readout (bottleneck)
- [x] Learnable weighted pooling
- [x] Slot-ID embeddings
- [x] Single-step prediction (seq_len=257)
- [x] Ablation support (slot_off param)

### Q3 Aux Supervision ❌ (planned for E4)
- [ ] `compute_q3_invariants()` function
- [ ] M0: mean(x)-1 (T0 normalization)
- [ ] M1: hist_entropy (A1' coverage)
- [ ] M2: max|dx| (A2 Lipschitz)
- [ ] M3: quantile 0.01 (A3 floor)
- [ ] M4: mean|d²x| (smoothness)
- [ ] M5: half_window_divergence (Toeplitz)
- [ ] M6: high_freq_energy (RKHS cap)
- [ ] M7: local_rigidity (spectral gap)
- [ ] z-score normalization
- [ ] aux_loss MSE/Huber

### Regularization ❌ (planned)
- [ ] Orthogonality loss (slots don't collapse)
- [ ] Norm cap (slots don't dominate)

### Extended Diagnostics ⚠️ (partial)
- [x] Val NLL
- [x] Readout weights visualization
- [x] Slot norms visualization
- [x] Timing summary
- [ ] CRPS metric
- [ ] PIT calibration
- [ ] Rollout Err@h
- [ ] Grad correlation logging
- [ ] Attention mass memory→data

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
# Package for RunPod
tar czf runpod_postfix.tar.gz train_mdn_postfix.py data/continuous_2M
runpodctl send runpod_postfix.tar.gz

# On Pod
runpodctl receive <CODE> && tar xzf runpod_postfix.tar.gz
pip install torch numpy scipy matplotlib rich
python train_mdn_postfix.py --data-dir data/continuous_2M --out-dir out/mdn_postfix_E3_s1337 --seed 1337 --max-steps 20000 --batch-size 512 --use-amp

# Download results
tar czf e3_results.tar.gz out/ && runpodctl send e3_results.tar.gz
```

---

## Next Steps

1. **Wait for E3 to complete** (~20k steps each seed)
2. **Run diagnostics** on E3 checkpoints
3. **Compare metrics** with E1/E2 (ablation Δ, grad corr)
4. **If E3 successful:** Add Q3-proxy aux loss for E4
5. **If E3 marginal:** Investigate attention patterns
