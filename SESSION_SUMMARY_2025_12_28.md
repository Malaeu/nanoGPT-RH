# Session Summary: 2025-12-28

## What We Did

### 1. Memory Bank Experiment
- Trained MemoryBankGPT with 4 learnable memory slots
- Initially "discovered" that slots resonate with ln(23), 4·ln(29)
- **TURNED OUT TO BE FALSE POSITIVE** — arbitrary `/10` scaling matched frequencies artificially

### 2. Code Review and Fixes
- Found bug: `scaled_freqs = freq_pos * 2π * dim / 10` — magic number with no justification
- Added null hypothesis test (1000 random vectors)
- **Result:** p=0.60 — memory structure indistinguishable from random noise

### 3. Hermiticity Test (Hilbert-Pólya)
- Tested symmetry of W_q^T @ W_k (effective Hamiltonian)
- **Result:** score=0.994, z=0.19 — NOT Hermitian, not more symmetric than random

### 4. Jitter Robustness Test
- Tested if 2π spike is binning artifact
- Added noise within bin width to de-quantize data
- **Result:** spike INCREASES with jitter (458% retention)
- **CONFIRMED: Real physics, not artifact!**

---

## Validated Results (Still Valid)

| Result | Evidence |
|--------|----------|
| SFF level repulsion | 53% suppression vs Poisson |
| 2π spike is real | Jitter test: 458% retention |
| GUE correlations learned | ACF MSE = 0.005 |
| RMT Memory effect | Elephant +63% vs Placebo -18% |
| Kernel extraction | μ(d) ~ d·exp(-γ√d), R²=0.99 |

## Disproven Claims (False Positives)

| Claim | Problem |
|-------|---------|
| Memory Bank → prime logs | Arbitrary /10 scaling created fake matches |
| "AI discovers arithmetic" | No null hypothesis test, 100+ candidates |
| Hermitian emergence | z=0.19, not statistically significant |

---

## Where to Go Next

### Valid Assets:
1. **SpacingGPT** — works, learns GUE correlations
2. **SFF analysis pipeline** — formulas are correct
3. **Jitter test** — proved 2π spike is real physics
4. **RMT Memory** — Elephant vs Placebo mechanism works

### Possible Directions:

**Option A: Deepen SFF Analysis**
- Spike detection on larger datasets
- More rigorous m·ln(p) matching with proper p-values
- Formal connection to Selberg trace formula

**Option B: Different Architecture for Hermitian**
- Add symmetry regularization
- Try symmetric attention mechanisms
- Or abandon this approach

**Option C: Kernel → Operator**
- Use μ(d) ~ d·exp(-γ√d) directly
- Build Toeplitz matrix from kernel
- Check its spectrum

---

## Key Lesson Learned

**Always do null hypothesis testing!**
- Many candidates (m·ln(p) for various m,p) + any frequency = false match
- Arbitrary scaling = result fitting
- Statistics first, interpretation second

---

## Files Changed This Session

- `probe_brain.py` — fixed, removed arbitrary scaling, added null test
- `math_mining.py` — fixed, removed hardcoded conclusions
- `check_jitter_robustness.py` — new, validates 2π spike
- `check_hermiticity.py` — new, tests Hilbert-Pólya hypothesis
- `PAPER_DRAFT.md` — updated with honest results

## Commits

1. `aee6f83` — Add jitter robustness test
2. `485dfc8` — Fix FFT analysis + add Hermiticity test
3. `ddd1137` — Update paper: remove false positives

---

## Bottom Line

The foundation remains solid:
- SpacingGPT works
- SFF analysis is valid
- 2π spike is real physics
- Memory Bank "discovery" was a mirage due to flawed methodology

Honest science requires admitting when we're wrong.
