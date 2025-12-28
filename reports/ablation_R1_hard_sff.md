# Ablation Study: R1

Generated: 2025-12-27T02:44:46.274732

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

- R_t P5: 0.6224
- R_t P95: 1.3948
- Healing baseline: 8.2
- Healing max (5x): 41.0

## Results

| Metric | Value |
|--------|-------|
| Trajectories | 80/100 |
| Acceptance Rate | 80.0% |
| R_t mean ± std | 0.9687 ± 0.4342 |
| R_t [P5, P50, P95] | [0.374, 0.912, 1.763] |
| S mean ± std | 1.0011 ± 0.3979 |
| Healing mean ± std | 3.4 ± 4.3 |
| Healing [P50, P95] | [2.0, 11.1] |

## Rejection Reasons

| Reason | Count |
|--------|-------|
| R_t out of range | 20 |

## SFF (Spectral Form Factor)

Long-range spectral rigidity test.

| Metric | Real | Generated | Ratio |
|--------|------|-----------|-------|
| Ramp Slope | 0.061811 | 0.342033 | 5.533 |
| Ramp RMSE | 0.091901 | 0.362551 | - |
| Plateau Level | 3.558079 | 0.793007 | 0.223 |
| N spacings | 51200 | 40960 | - |

**Ramp**: ✗ FAIL (ratio = 5.53)

---
*End of report*