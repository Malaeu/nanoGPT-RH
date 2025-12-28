# Causal Zeta MVP Report - Round 001

Generated: 2025-12-27T01:33:10.553582

---

## 1. Configuration

- Checkpoint: `out/best.pt`
- Data: `data`
- Seed: 42
- N windows: 2000
- PCA fit samples: 10000
- Rigidity window: 10
- CI permutations: 100
- do(S) delta: 0.05
- do(S) steps: 10

## 2. Variable Definitions

- **Z_t** = PCA(n=2) of hidden_states[-1][:, -1, :] (last layer, last position)
- **R_t** = Var(s[t-10:t]) / 0.178 (on actual spacing values)
- **S_t, Y_t** = spacing values via bin_centers lookup

## 3. Sanity Checks

**Overall: PASS**

| Check | Status | Value |
|-------|--------|-------|
| Mean spacing ≈ 1.0 | ✓ PASS | 0.9991 |
| Min spacing > 0 | ✓ PASS | 0.1016 |
| No NaN/Inf | ✓ PASS | clean |

- S mean: 0.9991 (target: 1.0)
- S std: 0.4054 (target: ~0.42)
- R mean: 1.0196 (target: ~1.0)

## 4. CI Test Results

| Test | X | Y | Z | HSIC | p-value | Independent? |
|------|---|---|---|------|---------|--------------|
| R_t ⊥ S_{t-1} | R_t | S_{t-1} | Z_t | 0.001524 | 0.0000 | ✗ No |
| Y_t ⊥ Z_t | Y_t | Z_t | ∅ | 0.017051 | 0.0000 | ✗ No |

**Interpretation:**

- **CI-A FAIL**: Rigidity depends on local spacing even after conditioning on Z_t. Consider adding edge S_{t-1} → R_t.
- **CI-B PASS**: Z_t is informative about Y_t. Latent mode captures meaningful structure.

## 5. Intervention do(S)

- Delta: 0.05 (spacing scale)
- Steps generated: 10
- **Mean healing time: 0.0 ± 0.0 steps**

**Interpretation**: Fast healing suggests strong spectral rigidity.

## 6. Verdict

- Sanity: ✓ PASS
- CI-A (R⊥S|Z): ✗ FAIL
- CI-B (Y~Z): ✓ PASS

**Graph update decision: NO CHANGE** (stable)

---

*End of report*