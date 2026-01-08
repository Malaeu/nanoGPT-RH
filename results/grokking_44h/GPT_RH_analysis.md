# GPT 5.2 Pro Analysis: RH Operator Extraction Questions

**Date:** 2026-01-08
**Thinking time:** ~30 minutes
**Context:** Frozen-Backbone Grokking experiment (44h) results

---

## Summary

### Key Findings

1. **GOE/GUE Bug Found!**
   - In pipeline plotting code: "GUE Wigner surmise" uses formula `(pi/2)s exp(-pi*s^2/4)`
   - This is **GOE** surmise, not GUE!
   - GUE std should be ~0.422, GOE std is ~0.523

2. **std mismatch (4x) is NOT a GUE failure - it's a scale mismatch:**
   - Unfolded spacings should have std ~ 0.41-0.43
   - Your 3.22 and 0.80 are likely different quantities (raw gaps vs unfolded vs delta*)

3. **Memory is decorative (Ablation Delta = 0.0058):**
   - Target: Ablation Delta >= 0.02
   - Current: attention -> operator **doesn't work** causally

4. **Even perfect GUE capture != RH proof:**
   - "Even if you perfectly captured GUE statistics, this itself doesn't equal RH - it's at most empirics about the distribution of zeros you fed to the model"

---

## 6 Action Items

| # | Action | Acceptance Criteria |
|---|--------|---------------------|
| 1 | Fix normalization and recalculate std on correct quantity | std_unfolded ~ 0.41-0.43 |
| 2 | Switch evaluation from corr to NLL/CRPS/PIT + KS | PIT mean~0.5, std~0.289 |
| 3 | block-CV diagnostics for height-shift cheating | nll_cv < 0.1 |
| 4 | Force causal memory (FiLM/gating) | Ablation Delta >= 0.02 |
| 5 | **Explicit kernel-operator layer** | rank_ratio > 0.5 |
| 6 | Close grokking as dead end | -- |

---

## Key Metrics Protocol

### Pass/Fail Thresholds
- Perm Inc <= 5% (currently: 1% PASS)
- block-CV < 0.1 (currently: 0.644 FAIL)
- rollout-drift slope < 0.5
- Grad Corr < 0.7 (currently: 0.296 PASS)
- Ablation Delta >= 0.02/slot (currently: 0.0058 FAIL)
- rank_ratio > 0.5

---

## Recommended Architecture Change

**Instead of "extracting attention post-hoc":**

Parameterize K(d) directly (symmetric/Hermitian) as the operator layer, with losses:
1. one-step NLL/CRPS/PIT
2. multi-scale statistics (rigidity/Delta3-proxy)
3. block-CV transferability

Then attention becomes THE object to extract, not an internal artifact.

---

## Full Response

Frozen-Backbone CorrectionNet (delta*=True-Flash; std_flash=0.8059 vs std_true=3.2200; corr=0.8838; Perm Inc=1%; Ablation Delta=0.0058): EMPIRICAL

### Context

You made the "Flash-generator -> CorrectionNet learns delta*=True-Flash" pipeline and saw two signals:

1. Flash rollouts are severely underdispersed: std=0.8059 vs std_true=3.2200 (~x4). (EMPIRICAL; std)

2. CorrectionNet hit a ceiling: corr~0.8838 by epoch~20 and then 44 hours without growth, while loss falls 0.17->0.06, and corr even drops 0.88->0.85. (EMPIRICAL; corr+loss)

**Key anchor:** in your own pipeline "2M continuous spacings" after unfolding, expected mean~1.0 and std~0.41-0.43 (and autocorr lag-1 < 0). (PROVEN; sanity target)

**Hard conclusion:** the number std_true=3.2200 is almost certainly NOT about unfolded spacings (and/or you're comparing different quantities/scales: raw gaps, cumulative gamma, or delta* in different normalization). (HYPOTHESIS; std mismatch <-> normalization)

**On "RH-perspective":** even if you perfectly captured GUE statistics, this itself doesn't equal RH - it's at most empirics about the distribution of those zeros you fed to the model (and it doesn't fix the absence of "bad" zeros off the line). (PROVEN; task logic)

**And also:** you yourself gave critical metrics on memory "operatorness": Perm Inc=1% (ID-detox ok), but Ablation Delta=0.0058 (memory is decorative). (EMPIRICAL; Perm Inc/Ablation Delta) This means: "attention_weights -> operator" currently doesn't work causally - operator is not necessary for loss. (EMPIRICAL->HYPOTHESIS; Ablation Delta low => non-operator)

### Definitions

- rho_n = 1/2 + i*gamma_n - non-trivial zeros of zeta, parameterized by heights gamma_n
- s_n = gamma_{n+1} - gamma_n - raw gap in gamma coordinate
- Unfolding - transition to coordinate u(gamma) where local density is flattened and mean spacing~1
- Variant B unfolding: u(gamma) = (gamma/2pi) * log(gamma/(2pi*e)), s_tilde_n = u(gamma_{n+1}) - u(gamma_n)
- m(T) = 2pi/log(T/2pi) - asymptotic mean distance in raw gamma-scale at T~gamma_n
- Flash (SpacingMDN) - base probabilistic predictor of next spacing via mixture (MDN)
- Rollout - auto-generation of multi-step trajectory from one-step model
- delta* - target correction: True - Flash
- corr - correlation of prediction with target (usually Pearson); doesn't measure distribution calibration
- NLL - negative log-likelihood of probabilistic predictor; key metric for MDN
- CRPS - calibration/quality score of predicted distribution
- PIT - probability integral transform; shows under/over-confidence (ideal: mean~0.5, std~0.289)
- KS p-value - test for PIT uniformity/calibration
- Perm Inc - NLL growth (%) with random permutation of memory slots; test for "ID cheating"
- Ablation Delta - delta_NLL when turning off slot: NLL(slot_off) - NLL(base); slot causality
- Grad Corr - gradient correlation between slots; high => co-adaptation
- rollout-drift slope - error growth over horizon
- block-CV - quality variation across height-blocks; if >0.1 - shift/cheating by position
- rank_ratio - fraction of "signal" in top ranks
- POSTFIX memory - order [data..., memory...] with causal mask; memory sees data, data doesn't see memory; readout only from memory
- ID-DETOX - slot_id_mode=permute_per_batch to keep Perm Inc <= 5%
- GUE (unitary) - class of spectra statistics of random Hermitian matrices; for unfolded spacings mean=1, std~0.422 (Wigner surmise)
- Sine-kernel: K(s) = sin(pi*s)/(pi*s), defines 2-point correlations in GUE limit
- Operator extraction - attempt to recover "kernel/operator" from attention/logits

### Lemmas/Checks

**A) No-leakage:**
- Perm Inc = 1% => no slot-ID leakage, permutation invariance exists. PASS

**B) Mask-invariant (POSTFIX):**
- Architecture PASS, but formally check gradients "head->data tokens" = 0

**C) No-shortcut:**
- Need ID-only/content-only tests (delta_NLL_id_only and delta_NLL_content_only)

**D) Causality:**
- Ablation Delta = 0.0058 => causality is weak; "attention=operator" currently doesn't extract, this is decorative memory. FAIL

### Phenomenon Analysis

**Phenomenon 1: "4x std mismatch"**
- Symptom: std_flash=0.8059, std_true=3.2200 (x3.996)
- Diagnosis hypotheses:
  1. Scales are mixed (unfolded vs raw vs delta*)
  2. Rollout collapses variance (one-step may be ok, but auto-generation compresses)
  3. MDN sampling not random (taking mean/mode, low temperature, or entropy_reg suppresses sigma)
- Cheapest test: Print mean/std for (a) train target, (b) val target, (c) Flash one-step samples, (d) Flash rollouts - all in same scale

**Phenomenon 2: "0.88 corr ceiling and no grokking"**
- Symptom: corr doesn't grow after epoch~20, loss falls, corr even falls
- Diagnosis hypotheses:
  1. 12% is "process entropy" at your conditioning - you predict conditional mean, remainder is randomness
  2. Remaining 12% is tails/extreme gaps - MLP learns "bulk", cuts tails -> corr stalls
  3. Missing global context/height T: corrections depend on T and slow modes (rigidity), but your memory is "decorative"
- Cheapest test: block-CV by height, performance by target quantiles

**Phenomenon 3: "What does 4x mismatch say about GUE hypothesis?"**
- GUE comparison is only correct in unfolded scale
- In your pipeline unfolding is fixed (Variant B) and sanity-std targets 0.41-0.43
- You probably missed not GUE, but the scale

**Phenomenon 4: "Operator extraction: how to make model learn exactly the operator?"**
- Current "operatorness" failed by Ablation Delta (0.0058) => attention may not be causal
- Diagnosis hypotheses:
  1. Identifiability killed by universality: even perfect GUE statistics don't fix unique Hamiltonian
  2. You're learning transfer operator of spacing process, not Hamiltonian
  3. Architecture doesn't force "operator" to be unique solution
- Evaluation of your options a/b/c:
  - a) Hermitian constraint on attention - useful as structural constraint, but doesn't solve identifiability alone
  - b) Spectral loss to GUE kernel - good as sanity regularizer, but as endgame almost guarantees you just recover universal K(d) (and this is not RH)
  - c) Training on raw zeros - closer to Hilbert-Polya, but huge risk of cheating by index/height

**My choice for "extracting operator":**
- Not "attention-based correction" on top of everything, but explicit operator layer: parameterize K(d) (or Toeplitz symbol) and force model to explain data through it
- Second layer: add arithmetic supervision (at least proxy M0-M7 like rigidity/Toeplitz-stability) as "anchors" against pure GUE universality

**Phenomenon 5: "Is there sense to continue grokking?"**
- Symptom: "grokking didn't happen", correlation plateaued early
- Diagnosis: grokking is useful when there's algorithmic structure + regime overfit->late generalization. Here you see "loss down without corr up" - this is more often "fitting variance/noise component" regime, not discovering new structure
- Fix: grokking as "run infinitely long" - low ROI; better short runs + structural constraints (operator/memory/calibration)

### Audit/Edge-check

1. **Unfolding sanity:** check mean~1 and std~0.41-0.43 on what you call True
2. **Unfolding formula:** Variant B u(gamma)=(gamma/2pi)log(gamma/(2pi*e)) - if you use m(T) backwards somewhere, you get "magic" x4 scale
3. **GOE/GUE confusion:** in pipeline on "GUE Wigner surmise" plot stands formula (pi/2)s exp(-pi*s^2/4) - this is **GOE** surmise, not GUE!
4. **One-step vs rollout mismatch:** without PIT/CRPS/KS on one-step and without rollout ensemble you don't know where variance breaks
5. **Height-shift:** block-CV; threshold nll_cv>0.1 is red flag
6. **Mask/readout shortcut:** POSTFIX + readout only from memory must be literal
7. **Slot co-adaptation:** Grad Corr; if stable >0.8 - slots not independent
8. **ID-cheating:** delta_NLL_id_only and delta_NLL_content_only must be logged

### Next Actions

**Action 1: Fix normalization and recalculate std on correct quantity**
- Why: your "4x mismatch" is not interpretable without scale; in unfolded world std should be ~0.42
- Acceptance: logs explicitly say "raw vs unfolded", and for unfolded True: mean~1, std~0.41-0.43
- Failure mode: after fix still std_true~3.2 => you're actually measuring not spacings, but something else

**Action 2: Switch evaluation from corr to NLL/CRPS/PIT + KS (one-step) and add rollout ensemble**
- Why: corr doesn't see calibration and easily "freezes"; NLL/CRPS/PIT will show underdispersion and tails
- Acceptance: on eval always have PIT mean/std, KS p-value, CRPS and gamma-rollout up to h=1000
- Failure mode: one-step calibrated, but rollout falls apart => exposure bias, need multi-step training

**Action 3: Height transferability diagnostics (block-CV) and cheating cutoff**
- Why: if model "learned index/height", operator extraction is meaningless
- Acceptance: cross_block_test gives nll_cv<0.1 and has_distribution_shift=False
- Failure mode: nll_cv>0.1 => add explicit T conditioning or harder dataset cut/mask positional

**Action 4: Force causal memory (otherwise "operator" doesn't learn)**
- Why: Ablation Delta=0.0058 is "memory decorative"
- What to do: FiLM/gating so slots modulate backbone; aux-proxy M0-M7 with ramp schedule; ID-only/content-only tests
- Acceptance: Ablation Delta_i >= 0.02 at least for some seeds; Grad Corr mean <= 0.6; Perm Inc <= 5% doesn't grow
- Failure mode: Ablation Delta doesn't grow => backbone "eats" everything, operator layer needs to be explicit

**Action 5: Next experiment "for RH": explicit kernel-operator instead of "extracting attention post-hoc"**
- Why: otherwise you'll again get universal "almost-sine" without identifiability and without causality
- Plan: parameterize K(d) directly (symmetric/Hermitian), train on (i) one-step NLL/CRPS/PIT, (ii) multi-scale statistics (rigidity/Delta3-proxy), (iii) block-CV transferability
- Acceptance: rank_ratio>0.5, block-CV<0.1, rollout-drift slope<0.5, Ablation Delta>=0.02
- Failure mode: K(d) collapses to trivial => your losses don't force it to be operator; strengthen spectral/rigidity-loss

**Action 6: Grokking - keep as tool, but not as "path"**
- Why: 44 hours without corr growth is not "a bit more and it'll see the light", it's "objective/loss looks wrong way"
- Acceptance: grokking makes sense only if you see late break in block-CV/rollout-drift slope under fixed conditions
- Failure mode: no break => close topic and move to operator-layer + explicit constraints

### "Lens Language"

- Background is currently floating: scale/calibration not fixed - so lens "compresses" variance (rollout-drift)
- Focus doesn't hold: Ablation Delta small => sieve registers don't carry causal mass, operator doesn't manifest
- Sieve should work like: window -> memory-registers (M0-M7) -> next step, and each register is irreplaceable (Ablation Delta up) with low sticking (Grad Corr down)
- When this happens, attention/logits can be interpreted as kernel K(d) and then try to extract formula "sin(pi*d)/(pi*d)+..." without self-deception

---

## Action Items Status (2026-01-08)

| # | Action | Status | Notes |
|---|--------|--------|-------|
| 1 | Fix GOE/GUE formula bug | **DONE** | Fixed in 3 files, committed 2f95af6 |
| 2 | Switch eval to NLL/CRPS/PIT | **EXISTS** | Already in `src/eval_mdn.py` |
| 3 | block-CV diagnostics | **EXISTS** | Already in `src/diagnose_memory_postfix.py` |
| 4 | Force causal memory | PENDING | Needs FiLM/gating architecture |
| 5 | Explicit kernel-operator layer | PENDING | Major architecture change |
| 6 | Close grokking as dead end | **DONE** | See below |

---

## GROKKING EXPERIMENT: CLOSED

**Decision:** Grokking approach is a **dead end** for this problem.

**Evidence:**
- 44 hours of training with no correlation improvement after epoch ~20
- Loss decreased (0.17 → 0.06) but correlation dropped (0.88 → 0.85)
- This pattern indicates "fitting variance/noise" not "discovering structure"
- Ablation Δ = 0.0058 shows memory is decorative, not causal

**Conclusion:**
Grokking works when there's hidden algorithmic structure that emerges after overfitting phase.
For RH/GUE statistics, the "structure" is already continuous (sine-kernel), not discrete algorithm.
Running longer won't help — the architecture needs fundamental change.

**Next Steps:**
1. Explicit kernel-operator layer (parameterize K(d) directly)
2. Multi-scale losses (rigidity, Delta3-proxy, block-CV)
3. FiLM/gating for causal memory if sticking with attention approach
