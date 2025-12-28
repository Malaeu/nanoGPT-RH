#!/usr/bin/env python3
"""
BANDWIDTH SCAN: How wide does the kernel need to be?

Our kernel exp(-1.16√d) decays too fast → localization
Let's scan different decay rates to find the chaos threshold.

γ small → wide band → GOE
γ large → narrow band → Poisson (localization)
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def test_bandwidth_scan():
    console.print(Panel.fit(
        "[bold cyan]📊 BANDWIDTH SCAN[/]\n"
        "Finding the Chaos Threshold",
        title="LOCALIZATION → DELOCALIZATION"
    ))

    N = 1000
    d_vals = np.arange(N)

    # Scan decay parameter γ
    # Our learned: γ = 1.16
    # Let's try smaller (wider band)
    gammas = [0.1, 0.2, 0.5, 0.8, 1.16, 2.0, 5.0]

    np.random.seed(42)

    # Pre-generate random matrix
    H_rand = np.random.randn(N, N)
    H_sym = (H_rand + H_rand.T) / np.sqrt(2)

    table = Table(title="🔬 DECAY RATE SCAN")
    table.add_column("γ (decay)", style="cyan", justify="right")
    table.add_column("Bandwidth", justify="right")
    table.add_column("P(s<0.5)", style="yellow", justify="right")
    table.add_column("Verdict", justify="center")

    results = []

    for gamma in gammas:
        # Kernel with variable decay
        kernel = (0.127 * d_vals + 0.062) * np.exp(-gamma * np.sqrt(d_vals))
        kernel[0] = 0

        # Estimate effective bandwidth (where kernel drops to 1% of max)
        max_k = np.max(kernel[1:])
        if max_k > 0:
            bandwidth = np.sum(kernel > 0.01 * max_k)
        else:
            bandwidth = 1

        # Build profile matrix
        Profile = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Profile[i, j] = kernel[abs(i - j)]

        H_final = H_sym * Profile
        np.fill_diagonal(H_final, np.random.randn(N))

        # Diagonalize
        eigs = np.linalg.eigvalsh(H_final)
        center = eigs[int(N * 0.3):int(N * 0.7)]
        spacings = np.diff(np.sort(center))
        spacings = spacings / np.mean(spacings)

        prob_small = np.sum(spacings < 0.5) / len(spacings)

        # Verdict
        if prob_small < 0.15:
            verdict = "[bold green]GOE![/]"
        elif prob_small < 0.20:
            verdict = "[green]Near-RMT[/]"
        elif prob_small < 0.30:
            verdict = "[yellow]Transition[/]"
        else:
            verdict = "[red]Poisson[/]"

        is_ours = " (OURS)" if abs(gamma - 1.16) < 0.01 else ""
        table.add_row(f"{gamma:.2f}{is_ours}", f"{bandwidth}", f"{prob_small:.3f}", verdict)

        results.append((gamma, bandwidth, prob_small, spacings))

    console.print(table)

    # Find critical γ
    for i in range(len(results) - 1):
        if results[i][2] < 0.25 and results[i + 1][2] >= 0.25:
            gamma_c = (results[i][0] + results[i + 1][0]) / 2
            console.print(f"\n[bold magenta]⚡ CRITICAL DECAY: γ_c ≈ {gamma_c:.2f}[/]")
            break

    # Interpretation
    console.print("\n")
    our_gamma = 1.16
    our_result = [r for r in results if abs(r[0] - our_gamma) < 0.01][0]

    if our_result[2] > 0.30:
        msg = (
            f"[yellow]Our learned γ = {our_gamma} is in the LOCALIZED regime![/]\n\n"
            "The neural network learned a kernel that's too narrow for chaos.\n"
            "This might mean:\n"
            "• Riemann zeros have FINITE-RANGE correlations\n"
            "• Or the network couldn't capture long-range structure\n"
            "• Or we need a different interpretation of the kernel"
        )
    else:
        msg = f"[green]Our γ = {our_gamma} produces chaos![/]"

    console.print(Panel.fit(msg, title="📊 INTERPRETATION", border_style="cyan"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: P(s<0.5) vs γ
    ax = axes[0]
    gammas_arr = [r[0] for r in results]
    probs_arr = [r[2] for r in results]

    ax.semilogx(gammas_arr, probs_arr, 'bo-', lw=2, markersize=8)
    ax.axhline(y=0.393, color='r', linestyle='--', label='Poisson')
    ax.axhline(y=0.11, color='g', linestyle='--', label='GOE')
    ax.axvline(x=1.16, color='purple', linestyle=':', alpha=0.7, label='Our γ=1.16')
    ax.set_xlabel('Decay rate γ')
    ax.set_ylabel('P(s < 0.5)')
    ax.set_title('Localization Transition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Spacing distributions for select γ
    ax = axes[1]
    x = np.linspace(0, 4, 100)
    p_goe = (np.pi * x / 2) * np.exp(-np.pi * x ** 2 / 4)
    ax.plot(x, p_goe, 'k-', lw=2, label='GOE (target)')

    colors = ['green', 'blue', 'red']
    for (gamma, bw, prob, spacings), color in zip(
            [results[0], results[4], results[-1]], colors):
        ax.hist(spacings, bins=40, density=True, alpha=0.4, color=color,
                label=f'γ={gamma:.2f} (P={prob:.2f})')

    ax.set_xlabel('Spacing s')
    ax.set_ylabel('P(s)')
    ax.set_title('Spacing Distribution vs Decay Rate')
    ax.legend()
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/bandwidth_scan.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/bandwidth_scan.png[/]")


if __name__ == "__main__":
    test_bandwidth_scan()
