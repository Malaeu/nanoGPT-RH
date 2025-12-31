# Sanity Report: 2M Continuous Spacings

## Dataset Info
- **Source**: `zeros/zeros2M.txt`
- **Zeros**: 2,001,052
- **γ range**: [14.13, 1132490.66]

## Unfolding Stats
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Mean | 1.000000 | ≈ 1.0 | ✓ |
| Std | 0.407523 | 0.41-0.43 | ✓ |
| Autocorr(1) | -0.348893 | < 0 | ✓ |
| Min | 0.005447 | > 0 | ✓ |
| Max | 3.303473 | < 10 | ✓ |

## Data Shapes
- **Train**: torch.Size([7035, 256]) (`float32`)
- **Val**: torch.Size([781, 256]) (`float32`, val_tail)

## Split Details
- Train: первые 90% окон (низкая высота)
- Val: последние 10% окон (высокая высота) — **val_tail**

## Quality Summary
- Монотонность: ✓
- Аномальные gaps: нет
- Stationarity: mean≈1.0 ✓
- Level repulsion: autocorr(1) < 0 ✓
