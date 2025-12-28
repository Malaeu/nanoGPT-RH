# Causal Zeta MVP Report - Round 001

Generated: 2025-12-27T02:01:59.583299

---

## 1. Configuration

- Checkpoint: `out/best.pt`
- Data: `data`
- Seed: 46
- N windows: 2000
- PCA fit samples: 10000
- Rigidity window: 20
- CI permutations: 200
- do(S) delta: 0.2
- do(S) steps: 25

## 2. Variable Definitions

- **Z_t** = PCA(n=2) of hidden_states[-1][:, -1, :] (last layer, last position)
- **R_t** = Var(s[t-20:t]) / 0.178 (on actual spacing values)
- **S_t, Y_t** = spacing values via bin_centers lookup

## 3. Sanity Checks

**Overall: PASS**

| Check | Status | Value |
|-------|--------|-------|
| Mean spacing ≈ 1.0 | ✓ PASS | 0.9997 |
| Min spacing > 0 | ✓ PASS | 0.1016 |
| No NaN/Inf | ✓ PASS | clean |

- S mean: 0.9997 (target: 1.0)
- S std: 0.4053 (target: ~0.42)
- R mean: 0.9608 (target: ~1.0)

## 4. CI Test Results

| Test | X | Y | Z | HSIC | p-value | Independent? |
|------|---|---|---|------|---------|--------------|
| R_t ⊥ S_deep | R_t | S_deep | S_{t-1}, Z_t | 0.000072 | 0.6700 | ✓ Yes |
| Y_t ⊥ Z_t | Y_t | Z_t | ∅ | 0.018262 | 0.0000 | ✗ No |

**Interpretation:**

- **CI-2 PASS**: Z_t is informative about Y_t. Latent mode captures meaningful structure.

## 5. Intervention do(S)

- Delta: 0.2 (spacing scale)
- Steps generated: 25
- **Mean healing time: 8.2 ± 8.6 steps**

**Interpretation**: Slow healing may indicate weak rigidity or model issues.

## 6. Verdict

- Sanity: ✓ PASS
- CI-2 (Y~Z): ✓ PASS

**Graph update decision: NO CHANGE** (all tests passed)

---

*End of report*