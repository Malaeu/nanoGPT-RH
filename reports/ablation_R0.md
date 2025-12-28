# Ablation Study: R0

Generated: 2025-12-27T02:19:30.040323

---

## Configuration

- Mode: **R0**
- Checkpoint: `out/best.pt`
- N trajectories: 100
- Trajectory length: 128
- Context length: 64
- Rigidity window: 10
- Seed: 100

## Results

| Metric | Value |
|--------|-------|
| Trajectories | 100/100 |
| Acceptance Rate | 100.0% |
| R_t mean ± std | 1.0176 ± 0.4912 |
| R_t [P5, P50, P95] | [0.368, 0.940, 1.979] |
| S mean ± std | 1.0027 ± 0.4087 |
| Healing mean ± std | 3.8 ± 4.7 |
| Healing [P50, P95] | [2.0, 13.0] |

---
*End of report*