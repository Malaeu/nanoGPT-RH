# Causal Zeta MVP Report

Generated: 2025-12-27T02:08:48.654355

---

## 1. Configuration

- Checkpoint: `out/best.pt`
- Data: `data`
- Seed: 46
- N windows: 2000
- PCA fit samples: 10000
- Rigidity window: 10
- CI permutations: 200
- do(S) delta: 0.2
- do(S) steps: 25

## 2. Variable Definitions

- **Z_t** = PCA(n=2) of hidden_states[-1][:, -1, :] (last layer, last position)
- **R_t** = Var(s[t-10:t]) / 0.178 (on actual spacing values)
- **S_t, Y_t** = spacing values via bin_centers lookup

## 3. Sanity Checks

**Overall: PASS**

| Check | Status | Value |
|-------|--------|-------|
| Mean spacing ≈ 1.0 | ✓ PASS | 1.0001 |
| Min spacing > 0 | ✓ PASS | 0.1016 |
| No NaN/Inf | ✓ PASS | clean |

- S mean: 1.0001 (target: 1.0)
- S std: 0.4048 (target: ~0.42)
- R mean: 1.0122 (target: ~1.0)

## 4. CI Test Results

| Test | X | Y | Z | HSIC | p-value | Independent? |
|------|---|---|---|------|---------|--------------|
| R_t ⊥ S_deep | R_t | S_deep | S_{t-1}, Z_t | 0.000155 | 0.1050 | ✓ Yes |
| Y_t ⊥ Z_t | Y_t | Z_t | ∅ | 0.017284 | 0.0000 | ✗ No |

**Interpretation:**

- **CI-1 PASS**: R_t ⊥ S_deep | (S_{t-1}, Z_t). Window L_R=10 is sufficient.
- **CI-2 PASS**: Z_t is informative about Y_t. Latent mode captures meaningful structure.

## 4.1 Effect Size (CI-1)

Ridge regression: R_t ~ f(features)

| Model | Features | R² | MAE |
|-------|----------|-----|-----|
| Baseline | S_{t-1}, Z_t | 0.1129 | 0.3585 |
| With S_deep | S_{t-1}, Z_t, S_deep | 0.1166 | 0.3574 |

- **ΔR² = 0.003734**
- **ΔMAE = 0.001086**
- **Verdict: PASS-ok**

S_deep adds minimal predictive power → memory is mostly short-range.

## 5. Intervention do(S)

- Delta: 0.2 (spacing scale)
- Steps generated: 25
- **Mean healing time: 8.2 ± 8.6 steps**

**Interpretation**: Slow healing may indicate weak rigidity or model issues.

## 6. Verdict

- Sanity: ✓ PASS
- CI-1 (R⊥S₂|S₁,Z): ✓ PASS
- CI-2 (Y~Z): ✓ PASS

**Graph update decision: NO CHANGE** (all tests passed)

---

*End of report*