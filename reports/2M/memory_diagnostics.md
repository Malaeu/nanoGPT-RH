# Memory Bank Diagnostics Report

## Model Info
- Checkpoint: `out/mdn_memory_v0/best.pt`
- Memory slots: 8

## A) Shortcut / Concentration
| Metric | Value | Status |
|--------|-------|--------|
| Attention entropy | 2.08 / 2.08 | ✓ |
| Concentration ratio | 0.00 | OK |
| Top slot | 7 (mass=0.13) | |

### Ablation Study
| Slot | Δ NLL | Interpretation |
|------|-------|---------------|
| 0 | +0.0015 | minor |
| 1 | +0.0015 | minor |
| 2 | +0.0001 | minor |
| 3 | -0.0005 | minor |
| 4 | +0.0007 | minor |
| 5 | +0.0020 | minor |
| 6 | -0.0004 | minor |
| 7 | +0.0003 | minor |

## B) Symmetry / Redundancy
| Metric | Value | Status |
|--------|-------|--------|
| Mean slot similarity | 0.088 | ✓ |
| Max slot similarity | 0.224 | |
| Permutation invariant | YES ⚠ | |

## C) Co-adaptation / Entanglement
| Metric | Value | Status |
|--------|-------|--------|
| Mean slot correlation | 0.098 | |
| Mean redistribution | 0.000 | ✓ |
| Co-adaptation detected | NO ✓ | |

## Summary
- **Shortcut**: ✓ OK
- **Symmetry**: ✓ OK
- **Co-adaptation**: ✓ OK
