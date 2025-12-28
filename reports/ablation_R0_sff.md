# Ablation Study: R0

Generated: 2025-12-27T02:39:40.633733

---

## Configuration

- Mode: **R0**
- Checkpoint: `out/best.pt`
- N trajectories: 100
- Trajectory length: 512
- Context length: 64
- Rigidity window: 10
- Seed: 100

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
| Ramp Slope | 0.061811 | 0.101342 | 1.640 |
| Ramp RMSE | 0.091901 | 0.206741 | - |
| Plateau Level | 3.558079 | 0.838794 | 0.236 |
| N spacings | 51200 | 51200 | - |

**Ramp**: ✗ FAIL (ratio = 1.64)

---
*End of report*