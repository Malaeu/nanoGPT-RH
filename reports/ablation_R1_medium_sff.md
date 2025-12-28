# Ablation Study: R1

Generated: 2025-12-27T02:49:23.895645

---

## Configuration

- Mode: **R1**
- Checkpoint: `out/best.pt`
- N trajectories: 100
- Trajectory length: 512
- Context length: 64
- Rigidity window: 10
- Seed: 100

## Rejector Calibration (from real-val)

- R_t P5: 0.4824
- R_t P95: 1.6660
- Healing baseline: 8.2
- Healing max (5x): 41.0

## Results

| Metric | Value |
|--------|-------|
| Trajectories | 97/100 |
| Acceptance Rate | 97.0% |
| R_t mean ± std | 0.9764 ± 0.4464 |
| R_t [P5, P50, P95] | [0.373, 0.913, 1.803] |
| S mean ± std | 1.0013 ± 0.3995 |
| Healing mean ± std | 3.6 ± 4.7 |
| Healing [P50, P95] | [2.0, 13.0] |

## Rejection Reasons

| Reason | Count |
|--------|-------|
| R_t out of range | 3 |

## SFF (Spectral Form Factor)

Long-range spectral rigidity test.

| Metric | Real | Generated | Ratio |
|--------|------|-----------|-------|
| Ramp Slope | 0.061811 | 0.102746 | 1.662 |
| Ramp RMSE | 0.091901 | 0.126117 | - |
| Plateau Level | 3.558079 | 0.663958 | 0.187 |
| N spacings | 51200 | 49664 | - |

**Ramp**: ✗ FAIL (ratio = 1.66)

---
*End of report*