# Hardware Specs: RunPod A40 Container (Flash Reality Check)

**Date:** 2026-01-02
**Container ID:** `root@69.30.85.23`

## Comparison: Dashboard vs. Reality

| Resource | RunPod Dashboard (Marketing) | Internal Cgroups (Reality) | Discrepancy |
| :--- | :--- | :--- | :--- |
| **GPU** | 1x NVIDIA A40 (48GB) | 1x NVIDIA A40 (48GB) | 0% (Exact) |
| **vCPU** | 9 Cores | **7.65 Cores** | -15% (Overhead?) |
| **RAM** | 50.0 GB | **46.57 GiB** | -7% (System reserved) |
| **Disk** | 20.0 GB | 20.0 GB | 0% |

## Detailed Technical Limits

### CPU
- **CFS Quota:** 765,000
- **CFS Period:** 100,000
- **Calculation:** `765000 / 100000 = 7.65` concurrent cores.

### Memory (RAM)
- **Limit in bytes:** 49,999,998,976 bytes.
- **Conversion:** `49,999,998,976 / 1024^3 ≈ 46.566 GiB`.

### GPU
- **UUID:** `GPU-5b71d2f3-60e1-72d6-5a03-5bd0b3ff34d9`
- **Memory:** 48 GB GDDR6
- **Power Cap:** 300W

---
*Note: The host machine itself is a massive Xeon Gold 6342 with 503GB RAM and 48 physical cores, but this container is strictly limited to the values above.*
