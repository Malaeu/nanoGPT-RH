# MDN Eval Comparison: Baseline vs Memory v0

**Date:** 2024-12-31
**RunPod GPU:** H100/A100
**Dataset:** 2M unfolded zeta zeros

## Models

| Model | Parameters | Training Steps | Config |
|-------|------------|----------------|--------|
| Baseline | 4.79M | 20k | Standard SpacingMDN |
| Memory v0 | 4.80M | 10k | 8 slots, slot-ID=False, aux_loss=False |

## Results

### One-Step Metrics

| Metric | Baseline | Memory v0 | Winner |
|--------|----------|-----------|--------|
| NLL | **-0.5607** | -0.1272 | Baseline |
| CRPS | **0.0835** | 0.1240 | Baseline |
| PIT mean | 0.4669 | **0.4780** | Memory (closer to 0.5) |
| PIT std | 0.2730 | 0.2831 | ~same |
| Bias | **0.0172** | 0.0214 | Baseline |

### Rollout Errors (Err@median)

| Horizon | Baseline | Memory v0 |
|---------|----------|-----------|
| h=10 | 0.3178 | 0.3225 |
| h=25 | 0.3067 | 0.3281 |
| h=50 | 0.2953 | 0.3256 |
| h=100 | **0.2789** | 0.3105 |

## Conclusion

**Baseline wins** on almost all metrics. Memory Bank v0 did NOT improve performance.

### Possible reasons:
1. Memory v0 trained only 10k steps vs 20k for baseline
2. Key features disabled: `slot-ID=False`, `aux_loss=False`
3. Memory bank adds complexity without proper supervision

### Next steps:
- [ ] Train Memory with `--use-slot-id --use-aux-loss` for 20k steps
- [ ] Compare with same training budget
- [ ] Consider if Memory Bank architecture is needed at all

## Raw Output

### Baseline
```
NLL: -0.5607
CRPS: 0.0835
PIT: mean=0.4669, std=0.2730
Bias: 0.017199
Err@h=100: 0.2789
```

### Memory v0
```
NLL: -0.1272
CRPS: 0.1240
PIT: mean=0.4780, std=0.2831
Bias: 0.021392
Err@h=100: 0.3105
```
