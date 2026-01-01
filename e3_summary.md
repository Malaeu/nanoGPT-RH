# E3 POSTFIX Experiment Summary

**Date:** 2026-01-01
**Architecture:** POSTFIX memory (memory AFTER data)
**GPU:** NVIDIA RTX 6000 Ada (48GB)

## Key Innovation

PREFIX memory (E1/E2) was "blind" - couldn't see data due to causal mask.
POSTFIX fixes this:
- Memory tokens AFTER data: `[s1..sT, M0..M7]`
- Memory CAN attend to all data (they're to the left)
- Data CANNOT attend to memory (blocked by causal mask)
- Readout ONLY from memory outputs (bottleneck!)

## Results (in progress)

| Experiment | Step | Best NLL | vs E2 (-0.384) |
|------------|------|----------|----------------|
| **s1337** | 4500 | **-0.3453** | **+10.1%** |
| **s42** | 3000 | **-0.3484** | **+9.3%** |
| **s7** | 1000 | **-0.3646** | **+5.1%** |

## Comparison with Previous

| Experiment | Architecture | Best NLL | Status |
|------------|--------------|----------|--------|
| E1 | PREFIX | -0.3793 | completed |
| E2 | PREFIX | -0.3840 | completed |
| **E3 s1337** | **POSTFIX** | **-0.3453** | running |
| **E3 s42** | **POSTFIX** | **-0.3484** | running |
| **E3 s7** | **POSTFIX** | **-0.3646** | running |

## Why POSTFIX Works Better

1. **Memory sees data** - can aggregate window statistics
2. **Bottleneck readout** - prediction MUST go through memory
3. **No shortcut** - model can't bypass memory using data hidden states

## Expected Diagnostics Improvements

After E3 completes, we expect:
- **Ablation Δ** should INCREASE (memory is essential now)
- **Grad correlation** should DECREASE (slots learn different things)
- **Slot effect norm** should be more distributed

## Files

- `train_mdn_postfix.py` - training script (committed)
- `out/mdn_postfix_E3_s{1337,42,7}/` - checkpoints on RunPod

## Next Steps

1. Wait for E3 to complete (~20k steps each)
2. Run `diagnose_memory.py` on E3 checkpoints
3. Compare ablation/grad_corr metrics with E1/E2
4. If successful, add Q3-proxy aux loss
