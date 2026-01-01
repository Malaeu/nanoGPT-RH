# Project Status: Honest Assessment (2025-12-28)

## Executive Summary

**SpacingGPT** is a neural spectroscopy experiment training small transformers on Riemann zeta zeros to discover spectral properties. The project has produced both validated results and exposed false positives through rigorous null hypothesis testing.

---

## Validated Results (Proven)

### 1. Kernel Extraction
- **What:** Attention logits fit to symbolic formula via PySR
- **Result:** μ(d) = (0.127·d + 0.062) × exp(-1.16·√d) + 0.0017
- **Evidence:** R² = 0.9927
- **Interpretation:** Level repulsion (linear term) + spectral rigidity (stretched exp)

### 2. RMT Memory Effect
- **What:** Memory mechanism improves long-range correlations
- **Results:**
  - Elephant (memory): +63% SFF improvement
  - Placebo (noise): -18% degradation
  - Elephant/Placebo ratio: 1.98x
- **Evidence:** Ablation study with 50 trajectories
- **Conclusion:** Model reads information from memory, not just noise

### 3. Training with Memory
- **What:** RMTSpacingGPT with learnable memory token
- **Results:**
  - RMT plateau: 1.72 (69% of real zeros ≈2.5)
  - Ablation (no memory, same context): 0.85
  - Pure memory effect: +103.5%
- **Conclusion:** Memory effect is real, not confound from longer context

### 4. SFF Level Repulsion
- **What:** Null hypothesis test against Poisson/shuffled
- **Results:**
  - Real zeros: plateau = 0.49
  - Poisson: plateau = 1.04
  - Suppression: 53%
- **Conclusion:** Level repulsion confirmed, not artifact

### 5. 2π Spike is Real
- **What:** Jitter robustness test
- **Method:** Add noise within bin width to de-quantize
- **Result:** Spike INCREASES with jitter (458% retention)
- **Conclusion:** Real physics, not binning artifact

---

## Disproven Claims (False Positives)

### 1. Memory Bank Discovers Prime Logarithms
- **Claim:** Memory slots tune to ln(23), 4·ln(29)
- **Problem:** Arbitrary `/10` scaling in FFT analysis
- **Null Test:** p = 0.60, indistinguishable from random
- **Status:** DEBUNKED

### 2. Hermitian Emergence (Hilbert-Pólya)
- **Claim:** Training on Riemann data creates Hermitian weights
- **Test:** ||H - H^T|| / ||H + H^T|| for W_q^T @ W_k
- **Result:** score = 0.994 (non-Hermitian), z = 0.19 vs random
- **Status:** NEGATIVE RESULT

### 3. Toeplitz Operator Produces GUE
- **Claim:** Build H[i,j] = μ(|i-j|) and get RMT statistics
- **Test:** Compare sine kernel (THE GUE kernel) in Toeplitz form
- **Result:** Even sine kernel gives Poisson, not GUE
- **Reason:** Toeplitz = translational invariance = integrable = Poisson
- **Status:** FUNDAMENTAL LIMITATION

---

## Partially Validated (Needs More Work)

### SFF Spikes at m·ln(p)
- **Claim:** Peaks at τ = 2·ln(19), 3·ln(11) match Selberg trace formula
- **Status:** Observed but NOT statistically validated
- **Problem:** 100+ candidates (m, p combinations) → any peak matches something
- **Needed:** Proper p-values, multiple testing correction, null dictionary

---

## Current Codebase

### Models
- `out/best.pt` — SpacingGPT (0.85M params)
- `out/memory_bank_best.pt` — MemoryBankGPT (4 slots)
- `out/final.pt` — Final training checkpoint

### Key Scripts
| File | Purpose | Status |
|------|---------|--------|
| `train.py` | Basic SpacingGPT training | ✅ Works |
| `train_memory_bank.py` | MemoryBank training | ✅ Works |
| `extract_kernel.py` | PySR symbolic regression | ✅ Works |
| `check_jitter_robustness.py` | Validates 2π spike | ✅ Passed |
| `check_hermiticity.py` | Hilbert-Pólya test | ❌ Negative |
| `probe_brain.py` | Memory slot analysis | ⚠️ Null test added |
| `construct_operator.py` | Toeplitz from kernel | ❌ Doesn't work |
| `compare_kernels.py` | Kernel vs GUE vs GOE | ✅ Revealed limitation |

### Analysis Reports
- `reports/artifact_check.png` — Real vs Poisson vs Shuffled
- `reports/jitter_test.png` — 2π spike robustness
- `reports/kernel_comparison.png` — Why Toeplitz fails
- `reports/phase_transition.png` — Coupling scan (no transition found)

---

## Key Lessons Learned

1. **Always do null hypothesis testing** — Many candidates + any signal = false match
2. **Arbitrary scaling = result fitting** — `/10` magic number created fake primes
3. **Toeplitz ≠ RMT** — Translational invariance kills level repulsion
4. **Statistics first, interpretation second** — Celebrate after p-values, not before

---

## Honest Assessment

### What We Have:
- A working neural telescope for spectral analysis
- Proven memory mechanism (+103.5% effect)
- Validated SFF level repulsion (53% suppression)
- Extracted kernel formula (R²=0.99)

### What We Don't Have:
- Automatic prime discovery
- Hermitian emergence
- Operator construction that works
- Statistical validation of m·ln(p) spikes

### Bottom Line:
The foundation is solid for spectral analysis. The "AI discovers primes" narrative collapsed under scrutiny. Honest science requires admitting when we're wrong.

---

## Files for External Review

1. `PAPER_DRAFT.md` — Current paper with all results
2. `SESSION_SUMMARY_2025_12_28.md` — Today's session log
3. `PROJECT_STATUS_HONEST.md` — This file
4. All `reports/*.png` — Visualization evidence
5. All `*.py` scripts — Reproducible code
