# RunPod GPU Specifications & Benchmarks

> Полный справочник по GPU на RunPod (Jan 2026)

**Last Updated:** 2026-01-01

---

## GPU ARCHITECTURES

### Timeline:
```
2020: Ampere (GA100, GA102)     → A100, A40, RTX 3090
2022: Ada Lovelace (AD102)      → RTX 4090, RTX 6000 Ada, L4, L40, L40S
2022: Hopper (GH100)            → H100, H200 (datacenter only!)
2024: Blackwell (GB100, GB202)  → B200, RTX 5090
```

### Architecture Details:

| GPU | Arch | Year | CUDA Cores | Tensor Cores | Memory Type | Bandwidth | TDP | Type |
|-----|------|------|------------|--------------|-------------|-----------|-----|------|
| A40 | Ampere | 2020 | 10,752 | 336 (3rd) | GDDR6 ECC | 696 GB/s | 300W | Datacenter |
| RTX 4090 | Ada | 2022 | 16,384 | 512 (4th) | GDDR6X | 1,008 GB/s | 450W | Consumer |
| RTX 6000 Ada | Ada | 2022 | 18,176 | 568 (4th) | GDDR6 ECC | 960 GB/s | 300W | Workstation |
| L4 | Ada | 2023 | 7,424 | 240 (4th) | GDDR6 | 300 GB/s | 72W | Inference |
| L40 | Ada | 2023 | 18,176 | 568 (4th) | GDDR6 ECC | 864 GB/s | 300W | Datacenter |
| L40S | Ada | 2023 | 18,176 | 568 (4th) | GDDR6 ECC | 864 GB/s | 350W | **ML Optimized** |
| H100 PCIe | Hopper | 2022 | 16,896 | 528 (4th+FP8) | HBM3 | 2,000 GB/s | 350W | Datacenter |
| H100 SXM | Hopper | 2022 | 16,896 | 528 (4th+FP8) | HBM3 | 3,350 GB/s | 700W | Datacenter |
| H200 | Hopper+ | 2024 | 16,896 | 528 | HBM3e | 4,800 GB/s | 700W | Datacenter |
| B200 | Blackwell | 2024 | ~20,000 | 5th (FP4) | HBM3e | ~8,000 GB/s | 1000W | Datacenter |
| RTX 5090 | Blackwell | 2024 | ~21,000 | 5th | GDDR7 | ~1,800 GB/s | 575W | Consumer |

### Key Insights:
```
Ampere (A40):     Old but gold. Best $/perf for budget.
Ada (L40S):       ML-optimized datacenter. BEST MID-TIER!
Ada (RTX 6000):   Workstation drivers, not ideal for ML.
Hopper (H100):    Monster bandwidth (HBM3), expensive.
Blackwell (B200): Bleeding edge, overkill pricing.
```

---

## MASTER TABLE — ALL GPUs (sorted by price)

| GPU | $/hr On-Demand | $/hr Spot | VRAM | RAM | vCPU | Max Pods | Avail | $/GB VRAM |
|-----|----------------|-----------|------|-----|------|----------|-------|-----------|
| RTX 2000 Ada | $0.24 | $0.18 | 16 GB | 15 GB | 6 | 6 | Medium | $0.015 |
| RTX 4000 Ada | $0.26 | $0.20 | 20 GB | 50 GB | 8 | 7 | Low | $0.013 |
| L4 | $0.39 | $0.32 | 24 GB | 50 GB | 12 | 6 | Medium | $0.016 |
| **A40** | **$0.40** | **$0.20** | **48 GB** | 48 GB | 9 | 10 | **High** | **$0.008** 🏆 |
| RTX 4090 | $0.59 | $0.50 | 24 GB | 41 GB | 6 | 6 | High | $0.025 |
| **RTX 6000 Ada** | **$0.77** | **$0.63** | **48 GB** | 62 GB | 16 | 8 | Low | $0.016 |
| L40S | $0.86 | $0.71 | 48 GB | 62 GB | 16 | 8 | **High** | $0.018 |
| RTX 5090 | $0.89 | $0.76 | 32 GB | 92 GB | 12 | 10 | **High** | $0.028 |
| L40 | $0.99 | $0.81 | 48 GB | 250 GB | 16 | 6 | Low | $0.021 |
| **RTX PRO 6000** | **$1.84** | **$1.56** | **96 GB** | 188 GB | 16 | 8 | **High** | **$0.019** |
| RTX PRO 6000 WK | $2.09 | $1.78 | 96 GB | 188 GB | 16 | 8 | Low | $0.022 |
| H100 PCIe | $2.39 | $2.03 | 80 GB | 176 GB | 16 | 7 | Low | $0.030 |
| H100 SXM | $2.69 | — | 80 GB | 125 GB | 20 | 8 | **High** | $0.034 |
| H100 NVL | $3.07 | $2.61 | 94 GB | 94 GB | 16 | 7 | Low | $0.033 |
| H200 NVL | $3.39 | $2.88 | 143 GB | 283 GB | 16 | 2 | Low | $0.024 |
| H200 SXM | $3.59 | $3.05 | 141 GB | 188 GB | 12 | 8 | **High** | $0.025 |
| B200 | $5.19 | $4.41 | 180 GB | 180 GB | 28 | 8 | Low | $0.029 |

---

## QUICK COMPARISON CHARTS

### By VRAM (sorted)
```
180 GB ████████████████████████████████████  B200         $5.19/hr
143 GB ████████████████████████████         H200 NVL     $3.39/hr
141 GB ████████████████████████████         H200 SXM     $3.59/hr
 96 GB ███████████████████                  RTX PRO 6000 $1.84/hr  ← BEST $/GB!
 94 GB ██████████████████                   H100 NVL     $3.07/hr
 80 GB ████████████████                     H100 PCIe    $2.39/hr
 80 GB ████████████████                     H100 SXM     $2.69/hr
 48 GB ██████████                           A40          $0.40/hr  ← CHEAPEST!
 48 GB ██████████                           RTX 6000 Ada $0.77/hr
 48 GB ██████████                           L40S         $0.86/hr
 48 GB ██████████                           L40          $0.99/hr
 32 GB ███████                              RTX 5090     $0.89/hr
 24 GB █████                                RTX 4090     $0.59/hr
 24 GB █████                                L4           $0.39/hr
 20 GB ████                                 RTX 4000 Ada $0.26/hr
 16 GB ███                                  RTX 2000 Ada $0.24/hr
```

### By Price (sorted)
```
$0.24/hr █                    RTX 2000 Ada  16 GB
$0.26/hr █                    RTX 4000 Ada  20 GB
$0.39/hr ██                   L4            24 GB
$0.40/hr ██                   A40           48 GB  ← BEST VALUE!
$0.59/hr ███                  RTX 4090      24 GB
$0.77/hr ████                 RTX 6000 Ada  48 GB  ← CURRENT
$0.86/hr ████                 L40S          48 GB
$0.89/hr ████                 RTX 5090      32 GB
$0.99/hr █████                L40           48 GB
$1.84/hr █████████            RTX PRO 6000  96 GB  ← HIDDEN GEM!
$2.09/hr ██████████           RTX PRO 6000 WK 96 GB
$2.39/hr ████████████         H100 PCIe     80 GB
$2.69/hr █████████████        H100 SXM      80 GB
$3.07/hr ███████████████      H100 NVL      94 GB
$3.39/hr ████████████████     H200 NVL      143 GB
$3.59/hr █████████████████    H200 SXM      141 GB
$5.19/hr █████████████████████████ B200     180 GB
```

### By vCPU Count
```
28 vCPU ████████████████████████████  B200
20 vCPU ████████████████████          H100 SXM
16 vCPU ████████████████              RTX 6000 Ada, L40S, L40, PRO 6000, H100 PCIe/NVL, H200
12 vCPU ████████████                  RTX 5090, H200 SXM
 9 vCPU █████████                     A40
 8 vCPU ████████                      RTX 4000 Ada
 6 vCPU ██████                        RTX 2000 Ada, RTX 4090, L4
```

### By System RAM
```
283 GB ██████████████████████████████████████████████████████  H200 NVL
250 GB ██████████████████████████████████████████████         L40
188 GB █████████████████████████████████████                  H200 SXM, PRO 6000
180 GB ████████████████████████████████████                   B200
176 GB ███████████████████████████████████                    H100 PCIe
125 GB █████████████████████████                              H100 SXM
 94 GB ███████████████████                                    H100 NVL
 92 GB ██████████████████                                     RTX 5090
 62 GB ████████████                                           RTX 6000 Ada, L40S
 50 GB ██████████                                             RTX 4000 Ada, L4
 48 GB ██████████                                             A40
 41 GB ████████                                               RTX 4090
 15 GB ███                                                    RTX 2000 Ada
```

---

## TIER RATINGS

### 🏆 Tier S — BEST VALUE
| GPU | Price | VRAM | Why |
|-----|-------|------|-----|
| **A40** | $0.40/hr | 48 GB | Cheapest $/VRAM, High avail, TESTED! |
| **RTX PRO 6000** | $1.84/hr | 96 GB | 96GB for less than H100! |

### ⭐ Tier A — SOLID CHOICES
| GPU | Price | VRAM | Why |
|-----|-------|------|-----|
| L40S | $0.86/hr | 48 GB | High availability |
| RTX 6000 Ada | $0.77/hr | 48 GB | Current pick, 16 vCPU |
| RTX 4000 Ada | $0.26/hr | 20 GB | Ultra cheap for small models |
| L4 | $0.39/hr | 24 GB | Good balance |

### 👍 Tier B — SITUATIONAL
| GPU | Price | VRAM | Why |
|-----|-------|------|-----|
| H100 PCIe | $2.39/hr | 80 GB | When speed matters |
| RTX 5090 | $0.89/hr | 32 GB | New gen, good avail |
| RTX 4090 | $0.59/hr | 24 GB | Consumer card |

### ❌ Tier C — AVOID
| GPU | Price | VRAM | Why NOT |
|-----|-------|------|---------|
| B200 | $5.19/hr | 180 GB | Insanely expensive |
| H200 NVL | $3.39/hr | 143 GB | Only 2 pods max |
| L40 | $0.99/hr | 48 GB | Pay more for same VRAM as A40 |
| H100 SXM | $2.69/hr | 80 GB | H100 PCIe cheaper |

---

## RECOMMENDATIONS

### For nanoGPT_RH (5M params)
```
1st: A40 @ $0.40/hr        — BEST, 5.8 steps/s tested
2nd: L40S @ $0.86/hr       — If A40 busy
3rd: RTX 6000 Ada @ $0.77  — Good for parallel seeds
```

### For Medium Models (50-100M params)
```
1st: RTX PRO 6000 @ $1.84  — 96GB!!! Hidden gem
2nd: H100 PCIe @ $2.39     — Fast
3rd: L40S @ $0.86          — Budget
```

### For Large Models (100M+ params)
```
1st: RTX PRO 6000 @ $1.84  — Best $/GB
2nd: H100 PCIe @ $2.39     — 80GB + speed
3rd: H200 SXM @ $3.59      — If 80GB not enough
```

---

## Our Benchmarks (SpacingMDN+Memory, ~5M params)

### Speed Comparison

| GPU | $/hr | steps/s | batch | time/500 steps | samples/s | Tested |
|-----|------|---------|-------|----------------|-----------|--------|
| Mac M4 Max | FREE | 1.3 | 128 | ~6.4 min | 166 | ✅ |
| A40 48GB | $0.40 | 5.8 | 512 | ~1.4 min | 2,969 | ✅ |
| RTX 6000 Ada | $0.77 | ~2.5* | 512 | ~3.3 min | ~1,280 | ✅ E4 |
| H100 80GB | $2.69 | ~15** | 512 | ~33 sec | ~7,680 | estimated |

*RTX 6000 Ada: 2.5 steps/s with 3 parallel jobs. Single job ~6-7 steps/s.
**H100 estimated based on typical 2.5x speedup over A40

### Cost per 10k Steps

| GPU | Time | Cost | $/10k steps | Efficiency |
|-----|------|------|-------------|------------|
| Mac M4 Max | ~2.1 hr | $0 | $0 | - |
| A40 | ~29 min | $0.19 | $0.19 | **baseline** |
| RTX 6000 Ada (3 jobs) | ~66 min | $0.85 | $0.28/job | 1.5x A40 |
| RTX 6000 Ada (1 job) | ~25 min | $0.32 | $0.32 | 1.7x A40 |
| H100 SXM | ~11 min | $0.49 | $0.49 | 2.6x A40 |

**Conclusion:** A40 still best for single job. RTX 6000 Ada good for parallel jobs (3x throughput).

---

## Recommendations

### For Small Models (<10M params)
**A40 = Best price/performance!**
- $0.40/hr (or $0.20 spot!)
- 48GB VRAM enough for batch_size=512
- High availability

### For Medium Models (10-100M params)
**L40S or RTX 6000 Ada**
- 48GB VRAM
- Better compute than A40
- L40S has better availability

### For Large Models (100M+ params)
**H100 PCIe** (cheaper than SXM)
- $2.39/hr vs $2.69/hr
- Same 80GB VRAM
- Good for batch_size=2048+

### For Maximum VRAM
**H200 NVL** (143GB) or **B200** (180GB)
- Only if you REALLY need it
- Very expensive ($3-5/hr)
- Low availability

---

## Quick Reference: VRAM Requirements

| batch_size | Estimated VRAM (5M model) |
|------------|---------------------------|
| 128 | ~4 GB |
| 256 | ~8 GB |
| 512 | ~16 GB |
| 1024 | ~30 GB |
| 2048 | ~55 GB |

---

## Notes

- **Spot pricing** can be 50% cheaper but pods can be preempted
- **Availability** matters: "Low" means you might wait hours for a pod
- Always check **actual speed** on first run, not just specs
- 3 parallel seeds @ batch=512 need ~30GB VRAM total
