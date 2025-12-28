#!/usr/bin/env python3
"""
RESONANCE PHASE ANALYSIS

Hypothesis: Riemann zeros are a QUASICRYSTAL, not random chaos.
The 147x spike at tau=2π is the strongest signal in the system.

Key insight: If mean spacing = 1, then u_n ≈ n (integers).
The deviation from integers = Resonance Phase = u mod 1

Question: Can we predict spacing from resonance phase?
s_t = 1 + f(φ_t) where φ_t = u_t mod 1
"""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
from scipy import stats

console = Console()


def run_resonance_analysis():
    console.print(Panel.fit(
        "[bold magenta]🌀 RESONANCE PHASE ANALYSIS[/]\n"
        "Testing the Quasicrystal Hypothesis",
        title="PARADIGM SHIFT"
    ))

    # 1. Load data
    console.print("\n[yellow]Loading data...[/]")
    val_data = torch.load('data/val.pt', weights_only=False)
    bin_centers = np.load('data/bin_centers.npy')

    # Extract ALL spacings
    spacings = []
    for i in range(len(val_data)):
        s = bin_centers[val_data[i].numpy()]
        spacings.extend(s)
    spacings = np.array(spacings)

    # Normalize (mean = 1)
    spacings = spacings / np.mean(spacings)

    # Compute unfolded positions
    u = np.concatenate([[0], np.cumsum(spacings)])
    N = len(spacings)

    console.print(f"[green]Loaded {N:,} spacings[/]")
    console.print(f"[cyan]Mean spacing: {np.mean(spacings):.6f}[/]")

    # 2. Compute Resonance Phase
    # φ = u mod 1 (deviation from ideal lattice)
    phi = u[:-1] % 1  # resonance phase at each position

    console.print(f"\n[bold]Resonance Phase Statistics:[/]")
    console.print(f"  Mean φ: {np.mean(phi):.4f} (should be ~0.5 if uniform)")
    console.print(f"  Std φ:  {np.std(phi):.4f}")

    # 3. Is φ uniformly distributed? (Null hypothesis for random)
    # Kolmogorov-Smirnov test against uniform [0,1]
    ks_stat, ks_pval = stats.kstest(phi, 'uniform')
    console.print(f"\n[bold]KS Test (φ vs Uniform):[/]")
    console.print(f"  KS statistic: {ks_stat:.6f}")
    console.print(f"  p-value: {ks_pval:.2e}")

    if ks_pval < 0.01:
        console.print("[green]  → φ is NOT uniform! There's structure![/]")
    else:
        console.print("[yellow]  → φ appears uniform (random-like)[/]")

    # 4. Correlation: φ_t vs s_t (spacing)
    # If quasicrystal: spacing should depend on phase
    corr_phi_s = np.corrcoef(phi, spacings)[0, 1]
    console.print(f"\n[bold]Correlation φ_t ↔ s_t:[/]")
    console.print(f"  Pearson r: {corr_phi_s:.6f}")

    if abs(corr_phi_s) > 0.1:
        console.print("[green]  → Spacing correlates with phase![/]")
    else:
        console.print("[yellow]  → Weak correlation[/]")

    # 5. More sophisticated: φ_t vs s_{t+1} (predictive)
    corr_phi_s_next = np.corrcoef(phi[:-1], spacings[1:])[0, 1]
    console.print(f"\n[bold]Predictive Correlation φ_t → s_next:[/]")
    console.print(f"  Pearson r: {corr_phi_s_next:.6f}")

    # 6. Autocorrelation of phase
    phi_centered = phi - np.mean(phi)
    acf_phi = np.correlate(phi_centered[:10000], phi_centered[:10000], mode='full')
    acf_phi = acf_phi[len(acf_phi)//2:]  # positive lags only
    acf_phi = acf_phi / acf_phi[0]  # normalize

    console.print(f"\n[bold]Phase Autocorrelation:[/]")
    console.print(f"  ACF(1): {acf_phi[1]:.4f}")
    console.print(f"  ACF(2): {acf_phi[2]:.4f}")
    console.print(f"  ACF(5): {acf_phi[5]:.4f}")

    # 7. Compare with Poisson
    console.print("\n[yellow]Comparing with Poisson baseline...[/]")
    np.random.seed(42)
    poisson_spacings = np.random.exponential(1.0, N)
    poisson_u = np.concatenate([[0], np.cumsum(poisson_spacings)])
    poisson_phi = poisson_u[:-1] % 1

    ks_poisson, pval_poisson = stats.kstest(poisson_phi, 'uniform')
    corr_poisson = np.corrcoef(poisson_phi, poisson_spacings)[0, 1]

    # 8. Summary Table
    console.print("\n")
    table = Table(title="🔬 RESONANCE PHASE: REAL vs POISSON")
    table.add_column("Metric", style="bold")
    table.add_column("Real Zeros", justify="right", style="cyan")
    table.add_column("Poisson", justify="right", style="red")
    table.add_column("Interpretation")

    table.add_row(
        "KS stat (φ vs uniform)",
        f"{ks_stat:.4f}",
        f"{ks_poisson:.4f}",
        "Lower = more uniform"
    )
    table.add_row(
        "Corr(φ, spacing)",
        f"{corr_phi_s:.4f}",
        f"{corr_poisson:.4f}",
        "Higher = predictable"
    )
    table.add_row(
        "Phase ACF(1)",
        f"{acf_phi[1]:.4f}",
        "-",
        "Higher = memory"
    )

    console.print(table)

    # 9. VERDICT
    console.print("\n")
    if ks_stat > ks_poisson * 2 or abs(corr_phi_s) > 0.05:
        console.print(Panel.fit(
            "[bold green]✅ QUASICRYSTAL STRUCTURE DETECTED![/]\n\n"
            f"Phase deviation from uniform: {ks_stat:.4f} (vs Poisson {ks_poisson:.4f})\n"
            f"Phase-spacing correlation: {corr_phi_s:.4f}\n\n"
            "[green]The resonance phase carries information![/]\n"
            "[dim]Next step: Train model to predict φ → s[/]",
            title="🎉 PARADIGM SHIFT CONFIRMED",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[yellow]Results inconclusive[/]\n\n"
            "Phase appears mostly uniform.\n"
            "May need different analysis approach.",
            title="⚠️ NEEDS MORE INVESTIGATION",
            border_style="yellow"
        ))

    # 10. Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Phase distribution
    ax1 = axes[0, 0]
    ax1.hist(phi, bins=50, density=True, alpha=0.7, label='Real zeros')
    ax1.hist(poisson_phi, bins=50, density=True, alpha=0.5, label='Poisson')
    ax1.axhline(y=1.0, color='red', linestyle='--', label='Uniform')
    ax1.set_xlabel('Resonance Phase φ = u mod 1')
    ax1.set_ylabel('Density')
    ax1.set_title('Phase Distribution')
    ax1.legend()

    # Plot 2: Phase vs Spacing scatter
    ax2 = axes[0, 1]
    # Sample for visibility
    idx = np.random.choice(len(phi), min(5000, len(phi)), replace=False)
    ax2.scatter(phi[idx], spacings[idx], alpha=0.3, s=1)
    ax2.set_xlabel('Resonance Phase φ')
    ax2.set_ylabel('Spacing s')
    ax2.set_title(f'Phase vs Spacing (r={corr_phi_s:.4f})')

    # Plot 3: Phase autocorrelation
    ax3 = axes[1, 0]
    lags = np.arange(100)
    ax3.plot(lags, acf_phi[:100], 'b-', linewidth=1)
    ax3.axhline(y=0, color='gray', linestyle='--')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('ACF')
    ax3.set_title('Phase Autocorrelation')
    ax3.set_xlim(0, 100)

    # Plot 4: Binned mean spacing by phase
    ax4 = axes[1, 1]
    n_bins = 20
    phi_bins = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_stds = []
    bin_centers_plot = []
    for i in range(n_bins):
        mask = (phi >= phi_bins[i]) & (phi < phi_bins[i+1])
        if mask.sum() > 0:
            bin_means.append(np.mean(spacings[mask]))
            bin_stds.append(np.std(spacings[mask]) / np.sqrt(mask.sum()))
            bin_centers_plot.append((phi_bins[i] + phi_bins[i+1]) / 2)

    ax4.errorbar(bin_centers_plot, bin_means, yerr=bin_stds, fmt='o-', capsize=3)
    ax4.axhline(y=1.0, color='red', linestyle='--', label='Mean = 1')
    ax4.set_xlabel('Resonance Phase φ')
    ax4.set_ylabel('Mean Spacing')
    ax4.set_title('Mean Spacing by Phase Bin')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('reports/resonance_phase_analysis.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]📊 Plot saved: reports/resonance_phase_analysis.png[/]")


if __name__ == "__main__":
    run_resonance_analysis()
