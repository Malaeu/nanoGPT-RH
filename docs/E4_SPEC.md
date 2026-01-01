# E4 — REGISTER STABILIZATION & ID-DETox

**Goal:** Make **Memory Slots** true **operator state registers**, not ID-crutches, and stabilize causality across seeds.

---

## 0) Input Data

* Architecture: **POSTFIX** (data → memory → readout)
* Base: `E3 best.pt` (3 seeds)
* Task: **one-step spacing prediction** (MDN)

---

## 1) New Flags (Required)

### Slot-ID Modes

```bash
--slot-id-mode fixed | off | permute_per_batch
```

* `fixed` — as before (E3 default)
* `off` — slot-ID embeddings = 0
* `permute_per_batch` — random slot permutation per batch

### Content Modes (for sanity)

```bash
--content-mode normal | zeroed
```

* `normal` — as is
* `zeroed` — memory embeddings = 0 (ID-only test)

---

## 2) Mandatory Sanity Tests (in diagnostics)

### A) **ID-only**

* `slot_id_mode=fixed`
* `content_mode=zeroed`
* Log: `ΔNLL_id_only = NLL(id_only) - NLL(base)`

**Expected:**
`ΔNLL_id_only >= +0.02` → **ID is not a crutch**
(if ≈0 → FAIL, model cheats via ID)

---

### B) **Content-only**

* `slot_id_mode=off`
* `content_mode=normal`
* Log: `ΔNLL_content_only`

**Expected:**
degradation <= **30%** of base → **content rules**

---

## 3) Aux-loss: Q3-proxy Supervision

```bash
--use-aux-loss
```

### Proxies (Fixed)

| Slot | Proxy | Description |
|------|-------|-------------|
| M0 | mean(x) − 1.0 | T0 normalization |
| M1 | hist_entropy(x) | A1' coverage |
| M2 | max\|dx\| | A2 Lipschitz |
| M3 | q1% | A3 floor |
| M4 | mean\|d²x\| | Smoothness |
| M5 | half_window_divergence | Toeplitz |
| M6 | high_freq_energy(dx) | RKHS cap |
| M7 | local_rigidity | Δ3-proxy |

**Important:** aux-head reads **memory hidden states**, not data.

### Ramp Schedule (Required)

```
aux_weight(t):
  0 → 1e-3   by step 500
  1e-3 → 1e-2 by step 3000
  hold 1e-2 until early-stop
```

---

## 4) Early Stopping (Required)

```bash
--early-stop --patience 800
```

Monitor: `val_nll`

---

## 5) Diagnostics (What to Log)

Minimum set (all numeric):

* `val_nll`
* `Ablation Δ` (max across slots)
* `Grad Corr (mean)`
* `Err@h` and `Err slope`
* `Perm Inc`
* `ΔNLL_id_only`
* `ΔNLL_content_only`

---

## 6) Success Criteria (Strict)

E4 is **PASS** if on **3 seeds**:

1. **Ablation Δ (max)** >= **0.02** for >= **2/3** seeds
2. **Grad Corr mean** <= **0.6**
3. **ID-only**: `ΔNLL_id_only >= 0.02`
4. **Content-only**: degradation <= **30%** of base
5. **Err slope** stays low (no explosion)

> **Note on Perm Inc:**
> High `Perm Inc` is **acceptable** if **Content-only works**.
> If `Perm Inc` high **AND** Content-only dead → **FAIL** (ID-crutch).

---

## 7) Run Plan

* Seeds: **3**
* Steps: until **early-stop** (usually 3k–6k)
* Modes:

  1. `slot_id_mode=permute_per_batch` (primary)
  2. (optional) `slot_id_mode=off` (confirmation)

---

## 8) Output Artifacts

* `best.pt` for each seed
* `diagnostics.jsonl`
* Table **mean±std** by metrics (seed-variance)

---

## 9) What NOT to Do

* ❌ Slot dropout
* ❌ Change POSTFIX-readout
* ❌ Symbolic regression **before PASS E4**

---

## 10) After PASS E4

* Transition to **extraction**:

  * `(z1..zk) → residual to RMT`
  * **symbolic regression / SINDy / Koopman**
* Transfer to **clean-200M**

---

## CLI Examples

### E3 Mode (backward compatible)
```bash
python train_mdn_postfix.py \
    --data-dir data/continuous_2M \
    --out-dir out/mdn_postfix_E3_s1337 \
    --seed 1337
```

### E4 Mode (ID-detox + aux-loss)
```bash
python train_mdn_postfix.py \
    --data-dir data/continuous_2M \
    --out-dir out/mdn_postfix_E4_s1337 \
    --seed 1337 \
    --slot-id-mode permute_per_batch \
    --use-aux-loss \
    --early-stop \
    --patience 800
```

### Sanity: ID-only Test
```bash
python train_mdn_postfix.py \
    --data-dir data/continuous_2M \
    --out-dir out/mdn_postfix_E4_idonly \
    --slot-id-mode fixed \
    --content-mode zeroed \
    --max-steps 5000
```

### Sanity: Content-only Test
```bash
python train_mdn_postfix.py \
    --data-dir data/continuous_2M \
    --out-dir out/mdn_postfix_E4_contentonly \
    --slot-id-mode off \
    --content-mode normal \
    --max-steps 5000
```
