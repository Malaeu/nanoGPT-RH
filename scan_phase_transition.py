#!/usr/bin/env python3
"""
PHASE TRANSITION SCAN: Anderson Localization → GUE Chaos

Problem: Our operator shows Poisson statistics (plateau=0.82, peak at s=0).
This is Anderson Localization: Disorder >> Interaction.

Solution: Scan coupling constant λ to find phase transition.
- λ small → Poisson (localized, no level repulsion)
- λ large → GUE (chaotic, level repulsion)

We're looking for the critical λ_c where our kernel produces Riemann-like physics.
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def scan_phase_transition():
    console.print(Panel.fit(
        "[bold cyan]🌊 PHASE TRANSITION SCAN[/]\n"
        "Anderson Localization → GUE Chaos",
        title="SEARCHING FOR RIEMANN"
    ))

    N = 1000
    d = np.arange(N)

    # 1. Base Kernel (Modulated for repulsion)
    # μ(d) = (0.127·d + 0.062) * exp(-1.16·√d) * cos(πd)
    kernel_base = (0.127 * d + 0.062) * np.exp(-1.16 * np.sqrt(d)) * np.cos(np.pi * d)
    kernel_base[0] = 0  # No self-interaction

    # 2. Diagonal Disorder (Fixed scale = 1.0)
    np.random.seed(42)  # Reproducibility
    disorder = np.random.normal(0, 1.0, N)

    # 3. Scan Coupling Strength λ
    lambdas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    def get_spacings(coupling):
        # Build Matrix: H = disorder + λ * Toeplitz(kernel)
        H = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                H[i, j] = coupling * kernel_base[abs(i - j)]

        # Add diagonal disorder
        np.fill_diagonal(H, disorder)

        # Diagonalize
        eigs = np.linalg.eigvalsh(H)

        # Unfold: take middle 60% to avoid edge effects
        center = eigs[int(N * 0.2):int(N * 0.8)]
        center = np.sort(center)
        spacings = np.diff(center)
        spacings /= np.mean(spacings)  # Normalize to mean=1
        return spacings

    # Results table
    table = Table(title="🔬 COUPLING SCAN RESULTS")
    table.add_column("λ", style="bold cyan", justify="right")
    table.add_column("Mean s", justify="right")
    table.add_column("Var s", justify="right")
    table.add_column("P(s<0.5)", style="bold yellow", justify="right")
    table.add_column("Regime", justify="center")

    # Reference values:
    # Poisson: P(s<0.5) = 1 - exp(-0.5) ≈ 0.393
    # GOE: P(s<0.5) ≈ 0.11
    # GUE: P(s<0.5) ≈ 0.14

    console.print("\n[bold]Reference: P(s<0.5)[/]")
    console.print("  Poisson: 0.393 (no repulsion)")
    console.print("  GOE:     0.11  (strong repulsion)")
    console.print("  GUE:     0.14  (level repulsion)\n")

    results = []
    plot_data = {}

    for lam in lambdas:
        s = get_spacings(lam)
        mean_s = np.mean(s)
        var_s = np.var(s)
        prob_small = np.sum(s < 0.5) / len(s)

        # Determine regime
        if prob_small > 0.35:
            regime = "[red]Poisson[/]"
        elif prob_small > 0.25:
            regime = "[yellow]Transition[/]"
        elif prob_small > 0.18:
            regime = "[green]Near-GUE[/]"
        else:
            regime = "[bold green]GUE![/]"

        table.add_row(
            f"{lam:.1f}",
            f"{mean_s:.3f}",
            f"{var_s:.3f}",
            f"{prob_small:.3f}",
            regime
        )

        results.append((lam, prob_small, var_s))

        # Store for plotting
        if lam in [0.5, 5.0, 20.0, 100.0]:
            plot_data[lam] = s

    console.print(table)

    # Find critical λ (interpolate where P(s<0.5) crosses 0.25)
    lambdas_arr = np.array([r[0] for r in results])
    probs_arr = np.array([r[1] for r in results])

    # Linear interpolation to find λ_c
    for i in range(len(probs_arr) - 1):
        if probs_arr[i] > 0.25 and probs_arr[i + 1] <= 0.25:
            # Interpolate
            lambda_c = lambdas_arr[i] + (0.25 - probs_arr[i]) * (lambdas_arr[i + 1] - lambdas_arr[i]) / (probs_arr[i + 1] - probs_arr[i])
            console.print(f"\n[bold magenta]⚡ CRITICAL COUPLING: λ_c ≈ {lambda_c:.2f}[/]")
            break
    else:
        console.print("\n[yellow]Phase transition not found in scan range[/]")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Spacing distributions
    ax = axes[0]
    x = np.linspace(0, 4, 100)

    # Reference curves
    p_poisson = np.exp(-x)
    p_goe = (np.pi * x / 2) * np.exp(-np.pi * x ** 2 / 4)

    ax.plot(x, p_poisson, 'k--', alpha=0.5, lw=2, label='Poisson')
    ax.plot(x, p_goe, 'k-', alpha=0.3, lw=3, label='Wigner (GOE)')

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_data)))
    for (lam, s), color in zip(plot_data.items(), colors):
        hist, bins = np.histogram(s, bins=50, density=True, range=(0, 4))
        centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(centers, hist, color=color, lw=2, label=f'λ={lam}')

    ax.set_xlabel('Normalized Spacing s')
    ax.set_ylabel('P(s)')
    ax.set_title('Phase Transition: Spacing Distribution')
    ax.legend()
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # Right: P(s<0.5) vs λ
    ax = axes[1]
    ax.semilogx(lambdas_arr, probs_arr, 'bo-', lw=2, markersize=8)
    ax.axhline(y=0.393, color='r', linestyle='--', label='Poisson (0.393)')
    ax.axhline(y=0.14, color='g', linestyle='--', label='GUE (0.14)')
    ax.axhline(y=0.25, color='orange', linestyle=':', alpha=0.7, label='Transition (0.25)')
    ax.set_xlabel('Coupling λ')
    ax.set_ylabel('P(s < 0.5)')
    ax.set_title('Level Repulsion vs Coupling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/phase_transition.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/phase_transition.png[/]")

    # Summary
    console.print("\n")
    console.print(Panel.fit(
        "[bold]PHYSICS INTERPRETATION:[/]\n\n"
        "H = Disorder + λ × Kernel\n\n"
        "• λ << 1: Disorder dominates → Anderson Localization → Poisson\n"
        "• λ ~ λ_c: Critical point → Metal-Insulator Transition\n"
        "• λ >> 1: Kernel dominates → Level Repulsion → GUE/GOE\n\n"
        "If λ_c exists, our learned kernel CAN produce RMT statistics!",
        title="📊 RESULT",
        border_style="cyan"
    ))


if __name__ == "__main__":
    scan_phase_transition()
