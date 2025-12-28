# Ablation Study: R1

Generated: 2025-12-27T02:20:57.593343

---

## Configuration

- Mode: **R1**
- Checkpoint: `out/best.pt`
- N trajectories: 100
- Trajectory length: 128
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
| Trajectories | 91/100 |
| Acceptance Rate | 91.0% |
| R_t mean ± std | 1.0095 ± 0.4691 |
| R_t [P5, P50, P95] | [0.369, 0.943, 1.928] |
| S mean ± std | 1.0026 ± 0.4073 |
| Healing mean ± std | 3.7 ± 4.8 |
| Healing [P50, P95] | [2.0, 13.0] |

## Rejection Reasons

| Reason | Count |
|--------|-------|
| R_t out of range | 9 |

---
*End of report*