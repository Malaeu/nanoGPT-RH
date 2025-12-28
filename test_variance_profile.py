#!/usr/bin/env python3
"""
VARIANCE PROFILE TEST: From Crystal to Chaos

Key Insight: The neural network learned the ENVELOPE of interactions,
not the exact values. In quantum chaos, the phase/sign is random!

H_ij = Gaussian(0, 1) × μ(|i-j|)

This is a Random Banded Matrix with our learned kernel as variance profile.
If this produces GOE/GUE statistics → we found the Hamiltonian skeleton!
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def test_variance_profile():
    console.print(Panel.fit(
        "[bold cyan]🎲 VARIANCE PROFILE TEST[/]\n"
        "From Crystal to Chaos",
        title="RANDOM BANDED MATRIX"
    ))

    N = 1000
    d_vals = np.arange(N)

    # 1. Our Learned Kernel (The Envelope)
    # μ(d) = (0.127·d + 0.062) * exp(-1.16·√d)
    # This is the STANDARD DEVIATION of interaction at distance d
    kernel_envelope = (0.127 * d_vals + 0.062) * np.exp(-1.16 * np.sqrt(d_vals))
    kernel_envelope[0] = 0  # Diagonal handled separately

    # 2. Construct Random Hamiltonian
    # H_ij = Gaussian(0, 1) * Kernel(|i-j|)
    console.print("Generating Random Matrix with Learned Profile...")

    np.random.seed(42)

    # Create random matrix
    H_rand = np.random.randn(N, N)

    # Symmetrize for GOE (Real Symmetric)
    H_sym = (H_rand + H_rand.T) / np.sqrt(2)

    # Apply variance profile
    console.print("Applying variance profile...")
    Profile = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Profile[i, j] = kernel_envelope[abs(i - j)]

    H_final = H_sym * Profile

    # Add diagonal disorder (standard GOE diagonal)
    np.fill_diagonal(H_final, np.random.randn(N))

    # 3. Analyze Spectrum
    console.print("Diagonalizing...")
    eigs = np.linalg.eigvalsh(H_final)

    # Unfold - take center 40% to avoid edge effects
    center = eigs[int(N * 0.3):int(N * 0.7)]
    center = np.sort(center)
    spacings = np.diff(center)
    spacings = spacings / np.mean(spacings)  # Normalize to mean=1

    # 4. Check Statistics
    prob_small = np.sum(spacings < 0.5) / len(spacings)
    var_s = np.var(spacings)

    # SFF check
    u = np.concatenate([[0], np.cumsum(spacings)])
    tau_vals = np.linspace(2.0, 10.0, 50)
    sff_vals = []
    for tau in tau_vals:
        val = np.abs(np.sum(np.exp(1j * tau * u))) ** 2 / len(u)
        sff_vals.append(val)
    sff_plateau = np.mean(sff_vals)

    # 5. Compare with pure GOE
    console.print("Generating pure GOE for comparison...")
    A = np.random.randn(N, N)
    H_goe = (A + A.T) / 2 / np.sqrt(N)
    eigs_goe = np.linalg.eigvalsh(H_goe)
    center_goe = eigs_goe[int(N * 0.3):int(N * 0.7)]
    spacings_goe = np.diff(np.sort(center_goe))
    spacings_goe = spacings_goe / np.mean(spacings_goe)
    prob_small_goe = np.sum(spacings_goe < 0.5) / len(spacings_goe)

    # 6. Report
    console.print("\n")
    table = Table(title="🎲 CHAOS CHECK: Variance Profile vs Pure GOE")
    table.add_column("Model", style="bold")
    table.add_column("P(s<0.5)", style="yellow", justify="right")
    table.add_column("Var(s)", justify="right")
    table.add_column("Verdict", justify="center")

    # Reference: Poisson ~0.39, GOE ~0.11, GUE ~0.14
    def get_verdict(p):
        if p < 0.15:
            return "[bold green]GOE/GUE![/]"
        elif p < 0.20:
            return "[green]Near-RMT[/]"
        elif p < 0.30:
            return "[yellow]Transition[/]"
        else:
            return "[red]Poisson[/]"

    table.add_row(
        "Variance Profile",
        f"{prob_small:.3f}",
        f"{var_s:.3f}",
        get_verdict(prob_small)
    )
    table.add_row(
        "Pure GOE",
        f"{prob_small_goe:.3f}",
        f"{np.var(spacings_goe):.3f}",
        get_verdict(prob_small_goe)
    )
    table.add_row(
        "[dim]Poisson ref[/]",
        "[dim]0.393[/]",
        "[dim]1.0[/]",
        "[dim]—[/]"
    )
    table.add_row(
        "[dim]GOE ref[/]",
        "[dim]0.11[/]",
        "[dim]0.27[/]",
        "[dim]—[/]"
    )

    console.print(table)

    # Verdict
    console.print("\n")
    if prob_small < 0.20:
        verdict = (
            "[bold green]SUCCESS![/] Variance Profile produces RMT statistics!\n\n"
            "The neural network learned the [bold]envelope of quantum chaos[/]:\n"
            "• μ(d) = variance profile, not deterministic interaction\n"
            "• Random phases create level repulsion\n"
            "• We found the [bold]skeleton of the Hamiltonian[/]!"
        )
    elif prob_small < 0.30:
        verdict = (
            "[yellow]PARTIAL SUCCESS[/] - Some level repulsion detected.\n"
            "May need tuning of diagonal vs off-diagonal strength."
        )
    else:
        verdict = (
            "[red]FAILED[/] - Still Poisson-like.\n"
            "Variance profile alone may not be enough."
        )

    console.print(Panel.fit(verdict, title="📊 RESULT", border_style="cyan"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Spacing distribution
    ax = axes[0]
    x = np.linspace(0, 4, 100)
    p_poisson = np.exp(-x)
    p_goe = (np.pi * x / 2) * np.exp(-np.pi * x ** 2 / 4)

    ax.hist(spacings, bins=40, density=True, alpha=0.6, color='blue',
            label=f'Variance Profile (P={prob_small:.2f})')
    ax.hist(spacings_goe, bins=40, density=True, alpha=0.4, color='green',
            label=f'Pure GOE (P={prob_small_goe:.2f})')
    ax.plot(x, p_poisson, 'r--', lw=2, label='Poisson')
    ax.plot(x, p_goe, 'k-', lw=2, label='Wigner-Dyson (GOE)')

    ax.set_xlabel('Spacing s')
    ax.set_ylabel('P(s)')
    ax.set_title('Level Spacing Distribution')
    ax.legend()
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # Right: Variance profile visualization
    ax = axes[1]
    d_plot = np.arange(50)
    k_plot = (0.127 * d_plot + 0.062) * np.exp(-1.16 * np.sqrt(d_plot))

    ax.bar(d_plot, k_plot, alpha=0.7, color='purple')
    ax.set_xlabel('Distance d = |i-j|')
    ax.set_ylabel('σ(d) = Variance Profile')
    ax.set_title('Learned Variance Profile μ(d)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/variance_profile_test.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/variance_profile_test.png[/]")


if __name__ == "__main__":
    test_variance_profile()
