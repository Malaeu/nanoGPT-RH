#!/usr/bin/env python3
"""
OPERATION ANTI-MIRAGE: Testing Normalization Artifacts

Critical Question: Is SFF plateau ≈2.5 a real property of Riemann zeros,
or just an artifact of the unfolding formula?

Null Hypothesis Test:
1. Real zeros → SFF plateau should be ~2.5 (our claim)
2. Poisson process → SFF plateau should be ~1.0 (random baseline)
3. Shuffled zeros → SFF plateau should be ~1.0 (destroy correlations)

If Poisson/Shuffled also give 2.5 → WE ARE WRONG (artifact)
If Poisson/Shuffled give 1.0 → WE ARE RIGHT (real physics)
"""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt

console = Console()


def compute_sff(spacings: np.ndarray, tau_values: np.ndarray = None) -> dict:
    """Compute Spectral Form Factor (same as benchmark_memory_sff.py)."""
    u = np.concatenate([[0], np.cumsum(spacings)])
    N = len(u)

    if tau_values is None:
        tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    K_values = np.zeros(len(tau_values))
    for i, tau in enumerate(tau_values):
        phases = np.exp(1j * tau * u)
        K_values[i] = np.abs(np.sum(phases))**2 / N

    # Plateau: τ > 4
    plateau_mask = tau_values > 4.0
    plateau = np.mean(K_values[plateau_mask]) if np.sum(plateau_mask) > 0 else K_values[-1]

    return {
        "tau": tau_values,
        "K": K_values,
        "plateau": plateau,
        "N": N,
    }


def run_sanity_check():
    console.print(Panel.fit(
        "[bold red]🕵️ OPERATION ANTI-MIRAGE[/]\n"
        "Testing if SFF plateau is real or normalization artifact",
        title="NULL HYPOTHESIS TEST"
    ))

    # 1. Load Real Data (Reference)
    console.print("\n[yellow]Loading real data...[/]")
    val_data = torch.load('data/val.pt', weights_only=False)
    bin_centers = np.load('data/bin_centers.npy')

    # Flatten all validation data into spacings
    real_spacings = []
    for i in range(len(val_data)):
        s = bin_centers[val_data[i].numpy()]
        real_spacings.extend(s)

    real_spacings = np.array(real_spacings)
    N = len(real_spacings)
    mean_spacing = np.mean(real_spacings)

    console.print(f"[green]Loaded {N:,} real spacings[/]")
    console.print(f"[cyan]Mean spacing: {mean_spacing:.4f} (should be ≈1.0)[/]")

    # 2. Create "Fake Physics" (Poisson Process)
    # Poisson process = Exponentially distributed spacings with same Mean
    console.print("\n[yellow]Generating Poisson (random) spacings...[/]")
    np.random.seed(42)  # Reproducibility
    fake_poisson = np.random.exponential(scale=mean_spacing, size=N)

    # 3. Create "Shuffled Physics" (Destroy correlations, keep distribution)
    console.print("[yellow]Shuffling real spacings...[/]")
    fake_shuffled = real_spacings.copy()
    np.random.shuffle(fake_shuffled)

    # 4. Compute SFF for all
    console.print("\n[yellow]Computing SFF for all datasets...[/]")
    tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    sff_real = compute_sff(real_spacings, tau_values)
    sff_poisson = compute_sff(fake_poisson, tau_values)
    sff_shuffled = compute_sff(fake_shuffled, tau_values)

    # 5. Display Results
    console.print("\n")
    table = Table(title="🧪 ARTIFACT CHECK RESULTS", show_header=True)
    table.add_column("Dataset", style="bold cyan")
    table.add_column("Description", style="dim")
    table.add_column("SFF Plateau", justify="right", style="magenta")
    table.add_column("Verdict", justify="center")

    real_p = sff_real['plateau']
    poisson_p = sff_poisson['plateau']
    shuffle_p = sff_shuffled['plateau']

    # Real zeros - baseline
    table.add_row(
        "Real Zeros",
        "True Riemann physics",
        f"{real_p:.4f}",
        "[bold white]BASELINE[/]"
    )

    # Poisson verdict
    if poisson_p > 2.0:
        verdict_p = "[bold red]❌ ARTIFACT![/]"
    elif poisson_p > 1.5:
        verdict_p = "[yellow]⚠️ SUSPICIOUS[/]"
    else:
        verdict_p = "[bold green]✅ CLEAN[/]"
    table.add_row(
        "Poisson (Random)",
        "Same mean, no correlations",
        f"{poisson_p:.4f}",
        verdict_p
    )

    # Shuffled verdict
    if shuffle_p > 2.0:
        verdict_s = "[bold red]❌ ARTIFACT![/]"
    elif shuffle_p > 1.5:
        verdict_s = "[yellow]⚠️ SUSPICIOUS[/]"
    else:
        verdict_s = "[bold green]✅ CLEAN[/]"
    table.add_row(
        "Shuffled",
        "Same values, wrong order",
        f"{shuffle_p:.4f}",
        verdict_s
    )

    console.print(table)

    # 6. Final Verdict
    console.print("\n")
    if poisson_p < 1.5 and shuffle_p < 1.5:
        console.print(Panel.fit(
            f"[bold green]✅ FORMULA IS CLEAN![/]\n\n"
            f"Real zeros plateau:    [cyan]{real_p:.2f}[/]\n"
            f"Poisson plateau:       [cyan]{poisson_p:.2f}[/]\n"
            f"Shuffled plateau:      [cyan]{shuffle_p:.2f}[/]\n\n"
            f"[green]The 2.5x enhancement is REAL PHYSICS, not artifact![/]\n"
            f"[dim]Poisson/Shuffled → ~1.0 (as expected for random)[/]\n"
            f"[dim]Real → ~{real_p:.1f} (genuine spectral rigidity)[/]",
            title="🎉 VERDICT: REAL PHYSICS",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            f"[bold red]❌ ARTIFACT DETECTED![/]\n\n"
            f"Real zeros plateau:    [cyan]{real_p:.2f}[/]\n"
            f"Poisson plateau:       [red]{poisson_p:.2f}[/]\n"
            f"Shuffled plateau:      [red]{shuffle_p:.2f}[/]\n\n"
            f"[red]The formula creates fake structure![/]\n"
            f"[dim]We need to investigate the unfolding process.[/]",
            title="⚠️ VERDICT: POSSIBLE ARTIFACT",
            border_style="red"
        ))

    # 7. Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: SFF curves
    ax1 = axes[0]
    ax1.semilogx(tau_values, sff_real['K'], 'b-', label=f'Real (plateau={real_p:.2f})', linewidth=2)
    ax1.semilogx(tau_values, sff_poisson['K'], 'r--', label=f'Poisson (plateau={poisson_p:.2f})', linewidth=2)
    ax1.semilogx(tau_values, sff_shuffled['K'], 'g:', label=f'Shuffled (plateau={shuffle_p:.2f})', linewidth=2)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='GUE baseline')
    ax1.set_xlabel('τ (Heisenberg time)')
    ax1.set_ylabel('SFF K(τ)')
    ax1.set_title('SFF Comparison: Real vs Null Models')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(real_p, poisson_p, shuffle_p) * 1.5)

    # Right: Bar chart of plateaus
    ax2 = axes[1]
    labels = ['Real\nZeros', 'Poisson\n(Random)', 'Shuffled']
    values = [real_p, poisson_p, shuffle_p]
    colors = ['blue', 'red', 'green']
    bars = ax2.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1.0, color='gray', linestyle='--', label='GUE baseline = 1.0')
    ax2.axhline(y=2.5, color='blue', linestyle=':', alpha=0.5, label='Expected real ≈ 2.5')
    ax2.set_ylabel('SFF Plateau')
    ax2.set_title('Plateau Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('reports/artifact_check.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]📊 Plot saved: reports/artifact_check.png[/]")


if __name__ == "__main__":
    run_sanity_check()
