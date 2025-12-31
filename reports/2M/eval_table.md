# Eval Report: SpacingMDN

## Model Info
- Checkpoint: `out/mdn_baseline/best.pt`
- Architecture: 6L/8H/256E
- MDN Components: 8

## One-Step Metrics
| Metric | Value | Target |
|--------|-------|--------|
| NLL | -0.3984 | lower is better |
| CRPS | 0.0961 | lower is better |
| PIT mean | 0.4819 | 0.5 |
| PIT std | 0.2797 | ~0.29 |
| KS p-value | 0.0000 | >0.05 |
| Bias | 0.011513 | 0 |

## Rollout Metrics
| Horizon | Err@mean | Err@median | Spread |
|---------|----------|------------|--------|
| 10 | 0.3397 | 0.3429 | 0.3946 |
| 25 | 0.3226 | 0.3164 | 0.3724 |
| 50 | 0.3182 | 0.3110 | 0.3434 |
| 100 | 0.3074 | 0.2978 | 0.3321 |

## Sanity Check
- GT rollout error: 0.000000 (should be ~0) ✓

## Notes
- PIT KS test: FAIL (miscalibrated)
- Bias: significant
