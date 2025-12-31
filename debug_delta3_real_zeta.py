#!/usr/bin/env python3
"""
DEBUG: Проверяем Δ3 на РЕАЛЬНЫХ zeta zeros.

Вопросы:
1. Правильно ли unfolded данные?
2. Правильно ли считается Δ3?
3. Соответствует ли real zeta теории GUE?
"""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
from pathlib import Path

console = Console()


def compute_delta3_verbose(spacings, L, n_samples=100):
    """
    Verbose Δ3 computation with debug output.
    """
    positions = np.cumsum(spacings)
    N = len(positions)

    if L >= positions[-1] - positions[0]:
        return np.nan, {}

    d3_samples = []
    debug_info = {"n_levels": [], "residuals": []}

    max_start = positions[-1] - L
    start_positions = np.linspace(positions[0], max_start, n_samples)

    for x0 in start_positions:
        mask = (positions >= x0) & (positions < x0 + L)
        levels_in_window = positions[mask]
        n_levels = len(levels_in_window)

        if n_levels < 3:
            continue

        debug_info["n_levels"].append(n_levels)

        n_i = np.arange(1, n_levels + 1)
        E_i = levels_in_window - x0

        # Least squares
        sum_E = np.sum(E_i)
        sum_E2 = np.sum(E_i ** 2)
        sum_n = np.sum(n_i)
        sum_nE = np.sum(n_i * E_i)

        denom = n_levels * sum_E2 - sum_E ** 2
        if abs(denom) < 1e-10:
            continue

        a = (n_levels * sum_nE - sum_E * sum_n) / denom
        b = (sum_n * sum_E2 - sum_E * sum_nE) / denom

        residuals = n_i - a * E_i - b
        d3 = np.sum(residuals ** 2) / L

        d3_samples.append(d3)
        debug_info["residuals"].append(np.mean(np.abs(residuals)))

    if d3_samples:
        return np.mean(d3_samples), debug_info
    return np.nan, debug_info


def main():
    console.print(Panel.fit(
        "[bold cyan]🔍 DEBUG: Δ3 on REAL ZETA ZEROS[/]\n"
        "Is the data correct? Is the metric correct?",
        title="SANITY CHECK"
    ))

    # Load data
    console.print("\n[cyan]Loading data...[/]")
    val_data = torch.load("data/val.pt", weights_only=False)
    bin_centers = np.load("data/bin_centers.npy")

    console.print(f"  Val sequences: {val_data.shape}")
    console.print(f"  Bin centers: {len(bin_centers)}, range [{bin_centers[0]:.4f}, {bin_centers[-1]:.4f}]")

    # Extract long sequence of real spacings
    n_seqs = min(50, len(val_data))  # Use many sequences
    tokens = val_data[:n_seqs].flatten().numpy()
    spacings = bin_centers[tokens]

    console.print(f"\n[cyan]Raw spacing statistics:[/]")
    console.print(f"  Total spacings: {len(spacings)}")
    console.print(f"  Mean: {spacings.mean():.4f}")
    console.print(f"  Std: {spacings.std():.4f}")
    console.print(f"  Min: {spacings.min():.4f}")
    console.print(f"  Max: {spacings.max():.4f}")

    # Normalize to mean 1
    spacings_norm = spacings / spacings.mean()

    console.print(f"\n[cyan]Normalized spacing statistics:[/]")
    console.print(f"  Mean: {spacings_norm.mean():.4f} (should be 1.0)")
    console.print(f"  Std: {spacings_norm.std():.4f} (GUE ~ 0.42)")

    # GUE theoretical std
    gue_std = np.sqrt(4/np.pi - 1)  # ≈ 0.523 for Wigner surmise
    console.print(f"  GUE Wigner std: {gue_std:.4f}")

    # Check spacing distribution
    console.print(f"\n[cyan]Spacing distribution check:[/]")
    # Wigner surmise: P(s) = (π/2) s exp(-π s²/4)
    # Mode at s = sqrt(2/π) ≈ 0.798
    # Mean = 1 (by normalization)

    hist, edges = np.histogram(spacings_norm, bins=50, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    # Wigner surmise
    s = np.linspace(0.01, 4, 100)
    wigner = (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4)

    # KL divergence approximation
    wigner_at_centers = (np.pi * centers / 2) * np.exp(-np.pi * centers**2 / 4)
    kl = np.sum(hist * np.log((hist + 1e-10) / (wigner_at_centers + 1e-10)) * (edges[1] - edges[0]))
    console.print(f"  KL divergence from Wigner: {kl:.4f} (lower is better)")

    # Now compute Δ3 carefully
    console.print(f"\n[cyan]Computing Δ3 with verbose output...[/]")

    L_values = [5, 10, 20, 50, 100, 200, 500, 1000]
    L_values = [L for L in L_values if L < len(spacings_norm) // 3]

    results = []

    for L in L_values:
        d3, debug = compute_delta3_verbose(spacings_norm, L, n_samples=100)

        if debug.get("n_levels"):
            avg_levels = np.mean(debug["n_levels"])
            avg_res = np.mean(debug["residuals"]) if debug["residuals"] else 0
        else:
            avg_levels = 0
            avg_res = 0

        # GUE theory
        gue_theory = (1/np.pi**2) * (np.log(2*np.pi*L) + np.euler_gamma - 5/4)

        results.append({
            "L": L,
            "delta3": d3,
            "gue_theory": gue_theory,
            "avg_levels": avg_levels,
            "avg_residual": avg_res,
        })

    # Display
    console.print("\n")
    table = Table(title="📊 Δ3(L) DETAILED ANALYSIS")
    table.add_column("L", style="bold", justify="right")
    table.add_column("Δ3 (data)", justify="right")
    table.add_column("Δ3 (GUE)", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Avg levels", justify="right")
    table.add_column("Avg |res|", justify="right")

    for r in results:
        ratio = r["delta3"] / r["gue_theory"] if r["gue_theory"] > 0 else np.nan
        table.add_row(
            f"{r['L']}",
            f"{r['delta3']:.4f}" if not np.isnan(r['delta3']) else "N/A",
            f"{r['gue_theory']:.4f}",
            f"{ratio:.2f}" if not np.isnan(ratio) else "N/A",
            f"{r['avg_levels']:.1f}",
            f"{r['avg_residual']:.4f}",
        )

    console.print(table)

    # Check if Δ3 grows as log(L)
    console.print("\n[cyan]Checking log(L) growth:[/]")

    valid_results = [r for r in results if not np.isnan(r["delta3"])]
    if len(valid_results) >= 3:
        L_arr = np.array([r["L"] for r in valid_results])
        d3_arr = np.array([r["delta3"] for r in valid_results])
        log_L = np.log(L_arr)

        # Linear fit: Δ3 = a * log(L) + b
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_L, d3_arr)

        console.print(f"  Fit: Δ3 = {slope:.4f} * log(L) + {intercept:.4f}")
        console.print(f"  R²: {r_value**2:.4f}")
        console.print(f"  Expected slope (GUE): {1/np.pi**2:.4f} = 0.1013")
        console.print(f"  Actual slope: {slope:.4f}")

        if abs(slope - 1/np.pi**2) < 0.05:
            console.print(f"  [green]✓ Slope matches GUE theory![/]")
        else:
            console.print(f"  [red]✗ Slope differs from GUE theory[/]")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Spacing histogram vs Wigner
    ax = axes[0, 0]
    ax.hist(spacings_norm, bins=50, density=True, alpha=0.7, label='Real Zeta')
    ax.plot(s, wigner, 'r-', linewidth=2, label='Wigner Surmise')
    ax.set_xlabel('Spacing s')
    ax.set_ylabel('P(s)')
    ax.set_title('Spacing Distribution')
    ax.legend()
    ax.set_xlim(0, 4)

    # 2. Δ3 vs L
    ax = axes[0, 1]
    L_plot = [r["L"] for r in valid_results]
    d3_plot = [r["delta3"] for r in valid_results]
    gue_plot = [r["gue_theory"] for r in valid_results]

    ax.loglog(L_plot, d3_plot, 'bo-', label='Real Zeta Data', linewidth=2, markersize=8)
    ax.loglog(L_plot, gue_plot, 'r--', label='GUE Theory', linewidth=2)
    ax.set_xlabel('L')
    ax.set_ylabel('Δ3(L)')
    ax.set_title('Spectral Rigidity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Δ3 vs log(L) linear scale
    ax = axes[1, 0]
    ax.plot(log_L, d3_arr, 'bo-', label='Real Zeta Data', linewidth=2, markersize=8)
    ax.plot(log_L, slope * log_L + intercept, 'g--', label=f'Fit: {slope:.3f}*log(L)+{intercept:.3f}')
    ax.plot(log_L, [r["gue_theory"] for r in valid_results], 'r--', label='GUE Theory')
    ax.set_xlabel('log(L)')
    ax.set_ylabel('Δ3(L)')
    ax.set_title('Δ3 vs log(L) (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Ratio to GUE
    ax = axes[1, 1]
    ratios = [r["delta3"] / r["gue_theory"] for r in valid_results]
    ax.semilogx(L_plot, ratios, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Perfect GUE match')
    ax.set_xlabel('L')
    ax.set_ylabel('Δ3(data) / Δ3(GUE theory)')
    ax.set_title('Ratio to GUE Theory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/debug_delta3_real_zeta.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/debug_delta3_real_zeta.png[/]")

    # Final diagnosis
    console.print("\n")
    if len(valid_results) >= 3 and abs(slope - 1/np.pi**2) < 0.05:
        diagnosis = (
            "[bold green]✓ REAL ZETA DATA MATCHES GUE THEORY![/]\n"
            f"Δ3 grows as ~{slope:.4f}*log(L), close to 1/π² = 0.1013.\n"
            "The data is correctly unfolded and has proper rigidity."
        )
    elif len(valid_results) >= 3 and slope > 0.5:
        diagnosis = (
            "[bold red]✗ REAL ZETA DATA IS NOT GUE-LIKE![/]\n"
            f"Δ3 grows as ~{slope:.4f}*log(L), much faster than 1/π².\n"
            "Either data is not properly unfolded, or something is wrong."
        )
    else:
        diagnosis = (
            "[bold yellow]⚠️ INCONCLUSIVE[/]\n"
            "Not enough data points for reliable analysis."
        )

    console.print(Panel.fit(diagnosis, title="💡 DIAGNOSIS", border_style="cyan"))


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    main()
