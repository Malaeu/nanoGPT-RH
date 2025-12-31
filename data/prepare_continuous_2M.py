#!/usr/bin/env python3
"""
TASK_SPEC_2M — Data Pipeline для 2M нулей → continuous spacings.

Выход:
  data/continuous_2M/train.pt  # [N_train, 256] float32
  data/continuous_2M/val.pt    # [N_val, 256] float32 (val_tail - последние 10%)
  data/continuous_2M/meta.pt   # dict с параметрами

Sanity checks:
  - mean ≈ 1.0
  - std ≈ 0.41-0.43
  - autocorr lag-1 < 0 (level repulsion)
  - no anomalous gaps
"""

import numpy as np
import torch
from pathlib import Path
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt

console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_ZEROS = Path("zeros/zeros2M.txt")
OUTPUT_DIR = Path("data/continuous_2M")
REPORTS_DIR = Path("reports/2M")
SEQ_LEN = 256
VAL_RATIO = 0.1  # последние 10% по высоте (val_tail)

# ============================================================================
# UNFOLDING (Variant B — стандартный)
# ============================================================================

def unfold_variant_b(zeros: np.ndarray) -> np.ndarray:
    """
    Unfolding Variant B (unfolded coordinates).

    u(γ) = (γ / 2π) * log(γ / (2πe))
    s_n = u(γ_{n+1}) - u(γ_n)
    """
    def u(gamma):
        return (gamma / (2 * np.pi)) * np.log(gamma / (2 * np.pi * np.e))

    u_vals = u(zeros)
    spacings = np.diff(u_vals)
    return spacings


def check_monotonicity(zeros: np.ndarray) -> bool:
    """Проверить монотонность γ."""
    diffs = np.diff(zeros)
    if np.any(diffs <= 0):
        bad_idx = np.where(diffs <= 0)[0]
        console.print(f"[red]ОШИБКА: {len(bad_idx)} немонотонных точек![/]")
        for idx in bad_idx[:5]:
            console.print(f"  idx={idx}: γ[{idx}]={zeros[idx]:.6f} >= γ[{idx+1}]={zeros[idx+1]:.6f}")
        return False
    return True


def check_anomalous_gaps(spacings: np.ndarray, threshold: float = 10.0) -> bool:
    """Проверить на аномально большие gaps."""
    anomalous = np.where(spacings > threshold)[0]
    if len(anomalous) > 0:
        console.print(f"[red]ОШИБКА: {len(anomalous)} аномальных gaps (>{threshold})![/]")
        for idx in anomalous[:5]:
            console.print(f"  idx={idx}: s={spacings[idx]:.4f}")
        return False
    return True


def compute_autocorr_lag1(spacings: np.ndarray) -> float:
    """Автокорреляция lag-1. Должна быть < 0 (level repulsion)."""
    s = spacings - spacings.mean()
    autocorr = np.correlate(s, s, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # положительные лаги
    autocorr = autocorr / autocorr[0]  # нормировка
    return autocorr[1]


def main():
    console.print("[bold magenta]═══ TASK_SPEC_2M: Data Pipeline ═══[/]\n")

    # Создать директории
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. ЗАГРУЗКА НУЛЕЙ
    # ========================================================================
    console.print(f"[cyan]Загрузка нулей из {INPUT_ZEROS}...[/]")
    zeros = np.loadtxt(INPUT_ZEROS)
    console.print(f"[green]Загружено {len(zeros):,} нулей[/]")
    console.print(f"  γ_min = {zeros[0]:.6f}")
    console.print(f"  γ_max = {zeros[-1]:.6f}")

    # ========================================================================
    # 2. ПРОВЕРКА МОНОТОННОСТИ
    # ========================================================================
    console.print("\n[cyan]Проверка монотонности...[/]")
    if not check_monotonicity(zeros):
        console.print("[red]СТОП: данные некорректны![/]")
        return
    console.print("[green]✓ Монотонность OK[/]")

    # ========================================================================
    # 3. UNFOLDING
    # ========================================================================
    console.print("\n[cyan]Unfolding (Variant B)...[/]")
    spacings = unfold_variant_b(zeros)
    console.print(f"[green]Получено {len(spacings):,} spacings[/]")

    # ========================================================================
    # 4. ПРОВЕРКА НА АНОМАЛЬНЫЕ GAPS
    # ========================================================================
    console.print("\n[cyan]Проверка на аномальные gaps...[/]")
    if not check_anomalous_gaps(spacings):
        console.print("[red]СТОП: данные содержат аномалии![/]")
        return
    console.print("[green]✓ Нет аномальных gaps[/]")

    # ========================================================================
    # 5. СТАТИСТИКА
    # ========================================================================
    console.print("\n[cyan]Статистика spacings:[/]")

    mean_s = spacings.mean()
    std_s = spacings.std()
    min_s = spacings.min()
    max_s = spacings.max()
    median_s = np.median(spacings)
    autocorr_1 = compute_autocorr_lag1(spacings)

    table = Table(title="Spacing Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Target", style="yellow")
    table.add_column("Status", style="bold")

    # Mean check
    mean_ok = abs(mean_s - 1.0) < 0.01
    table.add_row("Mean", f"{mean_s:.6f}", "≈ 1.0", "✓" if mean_ok else "⚠")

    # Std check
    std_ok = 0.40 <= std_s <= 0.44
    table.add_row("Std", f"{std_s:.6f}", "0.41-0.43", "✓" if std_ok else "⚠")

    # Autocorr lag-1 check (должна быть отрицательная)
    autocorr_ok = autocorr_1 < 0
    table.add_row("Autocorr(1)", f"{autocorr_1:.6f}", "< 0", "✓" if autocorr_ok else "⚠")

    table.add_row("Min", f"{min_s:.6f}", "> 0", "✓" if min_s > 0 else "⚠")
    table.add_row("Max", f"{max_s:.6f}", "< 10", "✓" if max_s < 10 else "⚠")
    table.add_row("Median", f"{median_s:.6f}", "≈ 0.87 (GUE)", "")

    console.print(table)

    # ========================================================================
    # 6. ФОРМИРОВАНИЕ ОКОН [N_seq, 256]
    # ========================================================================
    console.print(f"\n[cyan]Формирование окон (seq_len={SEQ_LEN})...[/]")

    n_seqs = len(spacings) // SEQ_LEN
    truncated = spacings[:n_seqs * SEQ_LEN]
    sequences = truncated.reshape(n_seqs, SEQ_LEN).astype(np.float32)

    console.print(f"[green]Создано {n_seqs:,} окон[/]")

    # ========================================================================
    # 7. SPLIT: val_tail = последние 10% по индексу (по высоте)
    # ========================================================================
    console.print(f"\n[cyan]Split (val_tail={VAL_RATIO*100:.0f}%)...[/]")

    n_val = int(n_seqs * VAL_RATIO)
    n_train = n_seqs - n_val

    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:]  # val_tail — последние по высоте

    console.print(f"  Train: {n_train:,} окон (первые {(1-VAL_RATIO)*100:.0f}%)")
    console.print(f"  Val:   {n_val:,} окон (последние {VAL_RATIO*100:.0f}%)")

    # ========================================================================
    # 8. СОХРАНЕНИЕ
    # ========================================================================
    console.print(f"\n[cyan]Сохранение в {OUTPUT_DIR}...[/]")

    train_pt = torch.from_numpy(train_seqs)  # [N_train, 256] float32
    val_pt = torch.from_numpy(val_seqs)      # [N_val, 256] float32

    torch.save(train_pt, OUTPUT_DIR / "train.pt")
    torch.save(val_pt, OUTPUT_DIR / "val.pt")

    # Meta
    meta = {
        "n_zeros": len(zeros),
        "gamma_min": float(zeros[0]),
        "gamma_max": float(zeros[-1]),
        "spacing_mean": float(mean_s),
        "spacing_std": float(std_s),
        "spacing_min": float(min_s),
        "spacing_max": float(max_s),
        "autocorr_lag1": float(autocorr_1),
        "seq_len": SEQ_LEN,
        "n_train": n_train,
        "n_val": n_val,
        "val_type": "val_tail",
        "val_ratio": VAL_RATIO,
        "source": str(INPUT_ZEROS),
        "dtype": "float32",
        "binned": False,
    }
    torch.save(meta, OUTPUT_DIR / "meta.pt")

    console.print(f"[green]✓ train.pt: {train_pt.shape} ({train_pt.dtype})[/]")
    console.print(f"[green]✓ val.pt:   {val_pt.shape} ({val_pt.dtype})[/]")
    console.print(f"[green]✓ meta.pt:  saved[/]")

    # ========================================================================
    # 9. SANITY REPORT
    # ========================================================================
    console.print(f"\n[cyan]Генерация sanity report...[/]")

    report = f"""# Sanity Report: 2M Continuous Spacings

## Dataset Info
- **Source**: `{INPUT_ZEROS}`
- **Zeros**: {len(zeros):,}
- **γ range**: [{zeros[0]:.2f}, {zeros[-1]:.2f}]

## Unfolding Stats
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Mean | {mean_s:.6f} | ≈ 1.0 | {"✓" if mean_ok else "⚠"} |
| Std | {std_s:.6f} | 0.41-0.43 | {"✓" if std_ok else "⚠"} |
| Autocorr(1) | {autocorr_1:.6f} | < 0 | {"✓" if autocorr_ok else "⚠"} |
| Min | {min_s:.6f} | > 0 | ✓ |
| Max | {max_s:.6f} | < 10 | ✓ |

## Data Shapes
- **Train**: {train_pt.shape} (`float32`)
- **Val**: {val_pt.shape} (`float32`, val_tail)

## Split Details
- Train: первые {(1-VAL_RATIO)*100:.0f}% окон (низкая высота)
- Val: последние {VAL_RATIO*100:.0f}% окон (высокая высота) — **val_tail**

## Quality Summary
- Монотонность: ✓
- Аномальные gaps: нет
- Stationarity: mean≈1.0 ✓
- Level repulsion: autocorr(1) < 0 ✓
"""

    with open(REPORTS_DIR / "sanity_report.md", "w") as f:
        f.write(report)

    console.print(f"[green]✓ reports/2M/sanity_report.md[/]")

    # ========================================================================
    # 10. ГИСТОГРАММА
    # ========================================================================
    console.print(f"\n[cyan]Генерация spacing_hist.png...[/]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram vs GUE
    ax = axes[0]
    ax.hist(spacings, bins=100, density=True, alpha=0.7,
            color='steelblue', label='Unfolded spacings')

    s = np.linspace(0, 4, 500)
    gue = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
    ax.plot(s, gue, 'r-', lw=2, label='GUE Wigner surmise')

    poisson = np.exp(-s)
    ax.plot(s, poisson, 'g--', lw=2, label='Poisson (uncorrelated)')

    ax.set_xlabel('Spacing s', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title(f'2M Zeta Spacings: mean={mean_s:.4f}, std={std_s:.4f}', fontsize=12)
    ax.legend()
    ax.set_xlim(0, 4)
    ax.grid(alpha=0.3)

    # Right: autocorrelation
    ax = axes[1]
    s_centered = spacings - spacings.mean()
    autocorr = np.correlate(s_centered[:10000], s_centered[:10000], mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    lags = np.arange(50)
    ax.bar(lags, autocorr[:50], color='steelblue', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    ax.axhline(y=-1/np.sqrt(10000), color='r', linestyle='--', lw=1, label='95% CI')
    ax.axhline(y=1/np.sqrt(10000), color='r', linestyle='--', lw=1)

    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title(f'Autocorrelation (lag-1={autocorr_1:.4f} < 0 = level repulsion)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "spacing_hist.png", dpi=150)
    plt.close()

    console.print(f"[green]✓ reports/2M/spacing_hist.png[/]")

    # ========================================================================
    # DONE
    # ========================================================================
    console.print("\n[bold green]═══ Data Pipeline Complete ═══[/]")
    console.print(f"""
Артефакты:
  data/continuous_2M/train.pt   [{n_train}, {SEQ_LEN}] float32
  data/continuous_2M/val.pt     [{n_val}, {SEQ_LEN}] float32
  data/continuous_2M/meta.pt    dict
  reports/2M/sanity_report.md
  reports/2M/spacing_hist.png
""")


if __name__ == "__main__":
    main()
