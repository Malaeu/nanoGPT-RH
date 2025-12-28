# Ablation Study: R1

Generated: 2025-12-27T02:35:05.135293

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

- R_t P5: 0.3850
- R_t P95: 1.9159
- Healing baseline: 8.2
- Healing max (5x): 41.0

## Results

| Metric | Value |
|--------|-------|
| Trajectories | 100/100 |
| Acceptance Rate | 100.0% |
| R_t mean ± std | 0.9786 ± 0.4508 |
| R_t [P5, P50, P95] | [0.374, 0.913, 1.817] |
| S mean ± std | 1.0013 ± 0.4000 |
| Healing mean ± std | 3.8 ± 4.7 |
| Healing [P50, P95] | [2.0, 13.0] |

## SFF (Spectral Form Factor)

Long-range spectral rigidity test.

| Metric | Real | Generated | Ratio |
|--------|------|-----------|-------|
| Ramp Slope | 0.000001 | 0.000002 | 1.640 |
| Ramp RMSE | 0.000002 | 0.000004 | - |
| Plateau Level | 0.000069 | 0.000016 | 0.236 |
| N spacings | 51200 | 51200 | - |

**Ramp**: ✗ FAIL (ratio = 1.64)

---
*End of report*