# E5.5 Stabilization Plan

> **Status:** MVP + 2nd commit done, code on RunPod
> **Date:** 2026-01-01

---

## Current E5 Status (old code, still running)

| Seed | Step  | Best NLL | Stale         |
|------|-------|----------|---------------|
| 7    | 17500 | 0.2369   | 500/10000     |
| 42   | 15500 | 0.3284   | 500/10000     |
| 1337 | 13000 | 0.2992   | 7500/10000    |

**Reference:** E4 best = 0.1942 (seed 7)

---

## Problems Fixed

### Issue A: Noisy Validation
- `permute_per_batch` ran DURING eval via `torch.randperm()`
- Every validation batch got DIFFERENT random slot IDs
- val_nll fluctuated ~0.01-0.05 from pure randomness

**Fix:** `eval_slot_id_mode='fixed'` - separate from training mode

### Issue B: Misleading Speed Metric
- `steps_per_sec = step / elapsed` was AVERAGE from start
- Eval time (~2.5s per eval) mixed into denominator

**Fix:** Separate `train_time_total` and `eval_time_total`, plus interval speed

---

## Racing vs A/B Test

| | Racing (PHASE 3) | A/B Test (PHASE 0) |
|---|---|---|
| **Goal** | Find lucky seeds | Understand regression |
| **Method** | 4 seeds parallel to 6k | 2 runs: E4 config vs E5 config |
| **Question** | "Which seed is best?" | "Why 0.19 → 0.33?" |
| **When** | After understanding regression | FIRST (critical!) |

---

## Implementation Status

### MVP Commit (done)
- [x] PHASE 1.1: Deterministic eval (`eval_slot_id_mode='fixed'`)
- [x] PHASE 2: Separate train/eval timing (`train:` vs `int:` in logs)

### 2nd Commit (done)
- [x] PHASE 1.2: `--eval-perm-average N` (average NLL over N permutations)
- [x] PHASE 4.1: `--perm-warmup-steps 2000` (warmup without permutation)
- [x] PHASE 3: `scripts/race_seeds.sh` (racing 4 seeds)

### Pending
- [ ] PHASE 0: Regression A/B test (E4 vs E5)

---

## New Flags

```bash
--eval-perm-average 5    # Average NLL over 5 seeded permutations
--perm-warmup-steps 2000 # First 2000 steps use fixed IDs
```

---

## New Log Format

```
Step 500: val_nll=0.40 (best=0.40) | train:5.6 s/s | int:4.2 s/s | elapsed=1.5m | stale=0/10000
```

Where:
- `train:` = pure training speed (without eval)
- `int:` = interval speed (500 steps / time since last eval)

---

## Action Plan

### Step 1: Wait for E5 (old code)
- Seed 7 at 0.2369 - approaching E4!
- Let it run to completion or early stop

### Step 2: PHASE 0 - A/B Test (critical!)
Find regression root cause:
```bash
# Run A (E4 config)
python src/train_mdn_postfix.py \
  --data-dir data/continuous_2M \
  --out-dir out/AB_E4_config \
  --seed 7 \
  --slot-id-mode permute_per_batch \
  --use-aux-loss \
  --max-steps 10000 \
  --batch-size 512 --use-amp

# Run B (E5 config) - compare differences
# If B worse, bisect: diagonal aux? patience? other?
```

### Step 3: Racing (new code)
```bash
cd /workspace/nanoGpt_RH
./scripts/race_seeds.sh   # 4 seeds to 6k steps with warmup
```

Pick top 2 seeds, continue:
```bash
python src/train_mdn_postfix.py \
  --resume out/racing/s<SEED>/best.pt \
  --out-dir out/E5_s<SEED>_full \
  --early-stop --patience 10000
```

---

## Files Modified

| File | Changes |
|------|---------|
| `src/train_mdn_postfix.py` | eval_slot_id_mode, timing, warmup, multi-perm |
| `scripts/race_seeds.sh` | NEW: 4-seed racing script |

---

## Success Criteria

- [ ] Same checkpoint gives same val_nll on repeated evals
- [ ] Train speed vs eval time clearly separated in logs
- [ ] Racing finds NLL < 0.25 in at least 2 seeds within 6k steps
- [ ] Understand WHY E4→E5 regressed (not just find lucky seeds)

---

*Last updated: 2026-01-01*
