#!/usr/bin/env python3
"""
SFF Scaling Analysis: Is 4.57 a Universal Constant or Finite-Size Effect?

Critical question: Does the SFF plateau depend on sample size N?
- If plateau → 1.0 as N → ∞: Standard GUE, 4.57 is artifact
- If plateau stays high or grows: Anomalous spectral rigidity

Test sizes: [512, 1024, 4096, 10000, 50000, 100000]
"""

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import track
import matplotlib.pyplot as plt

console = Console()

def compute_sff_large(spacings: np.ndarray, tau_values: np.ndarray) -> dict:
    """Compute SFF for potentially large arrays.

    Using SAME formula as benchmark_memory_sff.py for consistency!
    K(τ) = |Σ exp(iτu)|² / N
    """
    # Unfolded coordinates (same as benchmark: start from 0, no centering)
    u = np.concatenate([[0], np.cumsum(spacings)])
    N = len(u)

    # SFF at each tau (SAME as benchmark_memory_sff.py)
    sff = np.zeros(len(tau_values))
    for i, tau in enumerate(tau_values):
        phases = np.exp(1j * tau * u)  # NO 2π - same as benchmark!
        sff[i] = np.abs(phases.sum())**2 / N

    # Find plateau (τ > 2)
    plateau_mask = tau_values > 2.0
    if plateau_mask.sum() > 0:
        plateau = sff[plateau_mask].mean()
    else:
        plateau = sff[-10:].mean()

    return {
        'tau': tau_values,
        'sff': sff,
        'plateau': plateau,
        'N': N
    }

def main():
    console.print("[bold cyan]═══════════════════════════════════════════════════[/]")
    console.print("[bold cyan]  SFF SCALING ANALYSIS: Is 4.57 Real or Artifact?  [/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/]\n")

    # Load all data
    console.print("[yellow]Loading data...[/]")
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data = torch.load('data/val.pt', weights_only=False)
    bin_centers = np.load('data/bin_centers.npy')

    # Concatenate all data
    all_bins = torch.cat([train_data.flatten(), val_data.flatten()]).numpy()
    all_spacings = bin_centers[all_bins]

    console.print(f"[green]Total spacings available: {len(all_spacings):,}[/]\n")

    # Test different sample sizes
    sample_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 100000, 200000]
    sample_sizes = [s for s in sample_sizes if s <= len(all_spacings)]

    tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    results = []

    console.print("[bold]Computing SFF for different sample sizes...[/]\n")

    for N in track(sample_sizes, description="Processing..."):
        # Take contiguous block (not random sample - preserves correlations!)
        spacings = all_spacings[:N]

        result = compute_sff_large(spacings, tau_values)
        results.append(result)

    # Display results
    table = Table(title="🔬 SFF PLATEAU vs SAMPLE SIZE N")
    table.add_column("N (samples)", justify="right", style="cyan")
    table.add_column("SFF Plateau", justify="right", style="green")
    table.add_column("Δ from N=1024", justify="right")
    table.add_column("Trend", justify="center")

    # Reference: N=1024
    ref_plateau = None
    for r in results:
        if r['N'] == 1024:
            ref_plateau = r['plateau']
            break
    if ref_plateau is None:
        ref_plateau = results[1]['plateau']  # Use second smallest

    prev_plateau = None
    for r in results:
        N = r['N']
        plateau = r['plateau']

        delta = (plateau / ref_plateau - 1) * 100
        delta_str = f"{delta:+.1f}%" if delta != 0 else "REF"

        if prev_plateau is not None:
            if plateau > prev_plateau * 1.05:
                trend = "📈 UP"
            elif plateau < prev_plateau * 0.95:
                trend = "📉 DOWN"
            else:
                trend = "➡️ STABLE"
        else:
            trend = "-"

        table.add_row(
            f"{N:,}",
            f"{plateau:.4f}",
            delta_str,
            trend
        )
        prev_plateau = plateau

    console.print(table)

    # Key analysis
    console.print("\n[bold]═══ INTERPRETATION ═══[/]\n")

    first_plateau = results[0]['plateau']
    last_plateau = results[-1]['plateau']

    # Linear regression on log(N) vs plateau
    log_N = np.log10([r['N'] for r in results])
    plateaus = np.array([r['plateau'] for r in results])

    slope, intercept = np.polyfit(log_N, plateaus, 1)

    console.print(f"[cyan]Smallest N ({results[0]['N']:,}):[/] plateau = {first_plateau:.4f}")
    console.print(f"[cyan]Largest N ({results[-1]['N']:,}):[/] plateau = {last_plateau:.4f}")
    console.print(f"[cyan]Ratio (largest/smallest):[/] {last_plateau/first_plateau:.2f}x")
    console.print(f"[cyan]Linear trend (slope on log₁₀N):[/] {slope:.4f}")

    console.print("\n[bold]═══ VERDICT ═══[/]\n")

    if abs(slope) < 0.1:
        console.print("[bold green]✅ PLATEAU IS STABLE![/]")
        console.print("   SFF plateau does NOT depend on sample size.")
        console.print(f"   Value ≈ {plateaus.mean():.2f} is a REAL spectral property.")
        console.print("   This is NOT a finite-size artifact!")
    elif slope < -0.2:
        console.print("[bold red]⚠️ PLATEAU DECREASES WITH N![/]")
        console.print("   4.57 may be finite-size effect.")
        console.print("   Need even larger samples to find asymptote.")
    elif slope > 0.2:
        console.print("[bold yellow]📈 PLATEAU INCREASES WITH N![/]")
        console.print("   Unusual: spectral rigidity grows with scale.")
        console.print("   Might indicate non-ergodic structure.")
    else:
        console.print("[bold cyan]➡️ WEAK TREND[/]")
        console.print(f"   Slope = {slope:.3f} - borderline.")
        console.print("   Need larger samples for definitive answer.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: SFF curves for different N
    ax1 = axes[0]
    cmap = plt.cm.viridis
    for i, r in enumerate(results[::2]):  # Every other for clarity
        color = cmap(i / (len(results)//2))
        ax1.semilogx(r['tau'], r['sff'], label=f"N={r['N']:,}", color=color, alpha=0.8)
    ax1.axhline(y=1.0, color='red', linestyle='--', label='GUE limit')
    ax1.set_xlabel('τ (Heisenberg time units)')
    ax1.set_ylabel('SFF K(τ)')
    ax1.set_title('SFF Curves for Different Sample Sizes')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Plateau vs log(N)
    ax2 = axes[1]
    Ns = [r['N'] for r in results]
    plats = [r['plateau'] for r in results]
    ax2.semilogx(Ns, plats, 'bo-', markersize=8)
    ax2.axhline(y=1.0, color='red', linestyle='--', label='GUE expected')
    ax2.axhline(y=4.57, color='green', linestyle='--', label='Our measurement (4.57)')

    # Fit line
    fit_x = np.logspace(np.log10(min(Ns)), np.log10(max(Ns)), 100)
    fit_y = slope * np.log10(fit_x) + intercept
    ax2.semilogx(fit_x, fit_y, 'r-', alpha=0.5, label=f'Trend: slope={slope:.3f}')

    ax2.set_xlabel('Sample Size N')
    ax2.set_ylabel('SFF Plateau')
    ax2.set_title('SFF Plateau vs Sample Size: Scaling Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/sff_scaling_analysis.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]📊 Plot saved: reports/sff_scaling_analysis.png[/]")

    # Save data
    np.savez('reports/sff_scaling_data.npz',
             sample_sizes=Ns,
             plateaus=plats,
             slope=slope,
             intercept=intercept)
    console.print(f"[green]💾 Data saved: reports/sff_scaling_data.npz[/]")

if __name__ == "__main__":
    main()
