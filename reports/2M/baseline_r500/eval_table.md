# Eval Report: SpacingMDN

## Model Info
- Checkpoint: `out/mdn_baseline/best.pt`
- Architecture: 6L/8H/256E
- MDN Components: 8

## One-Step Metrics
| Metric | Value | Target |
|--------|-------|--------|
| NLL | -0.5607 | lower is better |
| CRPS | 0.0836 | lower is better |
| PIT mean | 0.4669 | 0.5 |
| PIT std | 0.2725 | ~0.29 |
| KS p-value | 0.0000 | >0.05 |
| Bias | 0.017339 | 0 |

## Rollout Metrics
| Horizon | Err@mean | Err@median | Spread |
|---------|----------|------------|--------|
| 10 | 0.3288 | 0.3228 | 0.3948 |
| 25 | 0.3208 | 0.3145 | 0.3669 |
| 50 | 0.2944 | 0.2887 | 0.3333 |
| 100 | 0.2814 | 0.2743 | 0.3234 |
| 200 | 0.2590 | 0.2583 | 0.3128 |

## Sanity Check
- GT rollout error: 0.000000 (should be ~0) ✓

## Notes
- PIT KS test: FAIL (miscalibrated)
- Bias: significant
