#!/usr/bin/env python3
"""
CONSTRUCT OPERATOR V2: Adding The Phase

Problem: Our kernel μ(d) = d·exp(-γ√d) is always positive.
This creates ATTRACTION (ferromagnet) → eigenvalues cluster.

Solution: Add oscillation cos(πd) to create REPULSION (antiferromagnet).
μ_mod(d) = μ(d) · cos(πd) = amplitude × alternating signs

If plateau drops from 19 to ~2.5, we found the Generator!
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def construct_operator_v2():
    console.print(Panel.fit(
        "[bold cyan]🏗️ CONSTRUCT OPERATOR V2[/]\n"
        "Adding Phase: cos(πd) modulation",
        title="ANTIFERROMAGNET FIX"
    ))

    N = 1000  # Matrix size

    # 1. Define Kernels
    d = np.arange(N)

    # A. Our Learned Decay (Amplitude Only)
    # μ(d) = (0.127·d + 0.062) * exp(-1.16·√d)
    kernel_amp = (0.127 * d + 0.062) * np.exp(-1.16 * np.sqrt(d))
    kernel_amp[0] = 0  # Diagonal filled with noise separately

    # B. Modulated Kernel (Amplitude + Phase)
    # Alternating signs: + - + -
    kernel_mod = kernel_amp * np.cos(np.pi * d)

    # C. Sine Kernel (Theoretical Ideal for GUE)
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel_sine = np.sin(np.pi * d) / (np.pi * d)
    kernel_sine[0] = 1.0

    # 2. Build Matrices (Toeplitz)
    def build_toeplitz(k_vals):
        H = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dist = abs(i - j)
                H[i, j] = k_vals[dist]
        return H

    console.print("Building Matrices...")
    H_positive = build_toeplitz(kernel_amp)
    H_modulated = build_toeplitz(kernel_mod)

    # Add random diagonal noise (Disorder) - Wigner-Dyson requires disorder
    disorder = np.random.normal(0, 1.0, N)
    np.fill_diagonal(H_positive, disorder.copy())
    np.fill_diagonal(H_modulated, disorder.copy())

    # 3. Diagonalize
    console.print("Diagonalizing...")
    eigs_pos = np.linalg.eigvalsh(H_positive)
    eigs_mod = np.linalg.eigvalsh(H_modulated)

    # Normalize Eigenvalues (Unfold locally)
    def unfold_spectrum(eigs):
        eigs = np.sort(eigs)
        # Remove edges (20% from each side)
        center = eigs[int(N * 0.2):int(N * 0.8)]
        spacing = (center[-1] - center[0]) / len(center)
        return center / spacing

    levels_pos = unfold_spectrum(eigs_pos)
    levels_mod = unfold_spectrum(eigs_mod)

    # 4. Compute SFF
    def compute_sff_plateau(levels):
        spacings = np.diff(levels)
        u = np.concatenate([[0], np.cumsum(spacings)])
        tau_vals = np.linspace(2.0, 10.0, 100)
        sff_vals = []
        for tau in tau_vals:
            val = np.abs(np.sum(np.exp(1j * tau * u))) ** 2 / len(u)
            sff_vals.append(val)
        return np.mean(sff_vals)

    p_pos = compute_sff_plateau(levels_pos)
    p_mod = compute_sff_plateau(levels_mod)

    # 5. Report
    console.print("\n")
    table = Table(title="🧲 MAGNETISM CHECK: Attraction vs Repulsion")
    table.add_column("Model", style="bold")
    table.add_column("Kernel Type", style="cyan")
    table.add_column("SFF Plateau", style="bold yellow", justify="right")
    table.add_column("Verdict", justify="center")

    table.add_row("Positive", "d·exp(-γ√d)", f"{p_pos:.2f}", "[red]Clustering[/]")
    table.add_row("Modulated", "d·exp(-γ√d)·cos(πd)", f"{p_mod:.2f}",
                  "[green]Repulsion![/]" if p_mod < 5 else "[yellow]Partial[/]")
    table.add_row("[dim]Target[/]", "[dim]Riemann Zeros[/]", "[dim]≈2.50[/]", "[dim]Goal[/]")

    console.print(table)

    # Spacing statistics
    spacings_pos = np.diff(levels_pos)
    spacings_mod = np.diff(levels_mod)

    console.print(f"\n[bold]Spacing Statistics:[/]")
    console.print(f"  Positive:  mean={np.mean(spacings_pos):.3f}, var={np.var(spacings_pos):.3f}")
    console.print(f"  Modulated: mean={np.mean(spacings_mod):.3f}, var={np.var(spacings_mod):.3f}")
    console.print(f"  [dim]GUE target: mean=1.0, var≈0.27[/]")

    # Verdict
    console.print("\n")
    if p_mod < 3.0:
        verdict = "[bold green]SUCCESS![/] Phase modulation creates proper repulsion!"
    elif p_mod < p_pos / 2:
        verdict = "[yellow]Partial success[/] - repulsion improved but not perfect"
    else:
        verdict = "[red]Phase modulation didn't help much[/]"

    console.print(Panel.fit(
        f"{verdict}\n\n"
        f"Positive kernel: {p_pos:.2f} (ferromagnet, clustering)\n"
        f"Modulated kernel: {p_mod:.2f} (antiferromagnet, repulsion)\n"
        f"Target: ≈2.5",
        title="📊 RESULT",
        border_style="cyan"
    ))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Spacing histograms
    ax = axes[0]
    ax.hist(spacings_pos, bins=50, density=True, alpha=0.5, label=f'Positive (P={p_pos:.1f})')
    ax.hist(spacings_mod, bins=50, density=True, alpha=0.5, label=f'Modulated (P={p_mod:.1f})')

    # Wigner Surmise (GOE)
    x = np.linspace(0, 4, 100)
    p_wigner = (np.pi * x / 2) * np.exp(-np.pi * x ** 2 / 4)
    ax.plot(x, p_wigner, 'k--', linewidth=2, label='Wigner-Dyson (GOE)')

    ax.set_xlabel('Spacing s')
    ax.set_ylabel('P(s)')
    ax.set_title('Level Spacing Distribution')
    ax.legend()
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # Right: Kernel comparison
    ax = axes[1]
    d_plot = np.arange(30)
    k_pos = (0.127 * d_plot + 0.062) * np.exp(-1.16 * np.sqrt(d_plot))
    k_mod = k_pos * np.cos(np.pi * d_plot)

    ax.stem(d_plot, k_pos, 'b', markerfmt='bo', label='Positive (attraction)', basefmt=' ')
    ax.stem(d_plot + 0.2, k_mod, 'r', markerfmt='ro', label='Modulated (repulsion)', basefmt=' ')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Distance d')
    ax.set_ylabel('Kernel μ(d)')
    ax.set_title('Kernel: Positive vs Modulated')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/operator_v2.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/operator_v2.png[/]")


if __name__ == "__main__":
    construct_operator_v2()
