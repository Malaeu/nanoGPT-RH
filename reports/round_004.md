# Causal Zeta MVP Report - Round 001

Generated: 2025-12-27T01:52:27.603737

---

## 1. Configuration

- Checkpoint: `out/best.pt`
- Data: `data`
- Seed: 45
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
| Mean spacing ≈ 1.0 | ✓ PASS | 1.0002 |
| Min spacing > 0 | ✓ PASS | 0.1016 |
| No NaN/Inf | ✓ PASS | clean |

- S mean: 1.0002 (target: 1.0)
- S std: 0.4050 (target: ~0.42)
- R mean: 1.0161 (target: ~1.0)

## 4. CI Test Results

| Test | X | Y | Z | HSIC | p-value | Independent? |
|------|---|---|---|------|---------|--------------|
| R_t ⊥ S_{t-2} | R_t | S_{t-2} | S_{t-1}, Z_t | 0.001953 | 0.0000 | ✗ No |
| Y_t ⊥ Z_t | Y_t | Z_t | ∅ | 0.017122 | 0.0000 | ✗ No |

**Interpretation:**

- **CI-1 FAIL**: R_t depends on longer history (S_{t-2}). Consider expanding R_t definition or adding memory.
- **CI-2 PASS**: Z_t is informative about Y_t. Latent mode captures meaningful structure.

## 5. Intervention do(S)

- Delta: 0.2 (spacing scale)
- Steps generated: 25
- **Mean healing time: 12.4 ± 8.6 steps**

**Interpretation**: Slow healing may indicate weak rigidity or model issues.

## 6. Verdict

- Sanity: ✓ PASS
- CI-1 (R⊥S₂|S₁,Z): ✗ FAIL
- CI-2 (Y~Z): ✓ PASS

**Graph update decision: UPDATE REQUIRED**
- ✏️ EXPAND: R_t has longer memory, consider wider window

---

*End of report*