#!/usr/bin/env python3
"""
KERNEL COMPARISON: Why doesn't our kernel produce level repulsion?

Test 3 kernels:
1. Our learned kernel (modulated): d·exp(-γ√d)·cos(πd)
2. GUE Sine kernel: sin(πd)/(πd)
3. Pure GOE (random symmetric matrix)

If sine kernel works but ours doesn't → problem is kernel shape
If sine kernel also fails → problem is Toeplitz approach
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def compare_kernels():
    console.print(Panel.fit(
        "[bold cyan]🔬 KERNEL SHOWDOWN[/]\n"
        "Our Kernel vs Sine Kernel vs GOE",
        title="DIAGNOSIS"
    ))

    N = 1000
    d = np.arange(N)

    # 1. Our Learned Kernel (modulated)
    kernel_ours = (0.127 * d + 0.062) * np.exp(-1.16 * np.sqrt(d)) * np.cos(np.pi * d)
    kernel_ours[0] = 0

    # 2. GUE Sine Kernel: sin(πd)/(πd)
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel_sine = np.sin(np.pi * d) / (np.pi * d)
    kernel_sine[0] = 1.0  # lim_{d→0} sinc(d) = 1

    # 3. Our kernel WITHOUT modulation (raw)
    kernel_raw = (0.127 * d + 0.062) * np.exp(-1.16 * np.sqrt(d))
    kernel_raw[0] = 0

    np.random.seed(42)
    disorder = np.random.normal(0, 1.0, N)

    def build_and_analyze(kernel, name, coupling=10.0):
        """Build Toeplitz matrix and compute spacing statistics."""
        H = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                H[i, j] = coupling * kernel[abs(i - j)]
        np.fill_diagonal(H, disorder)

        eigs = np.linalg.eigvalsh(H)
        center = eigs[int(N * 0.2):int(N * 0.8)]
        center = np.sort(center)
        spacings = np.diff(center)
        spacings /= np.mean(spacings)

        prob_small = np.sum(spacings < 0.5) / len(spacings)
        var_s = np.var(spacings)

        return spacings, prob_small, var_s

    # Also test GOE (no Toeplitz structure)
    def goe_spacings():
        A = np.random.randn(N, N)
        H = (A + A.T) / 2 / np.sqrt(N)
        eigs = np.linalg.eigvalsh(H)
        center = eigs[int(N * 0.2):int(N * 0.8)]
        center = np.sort(center)
        spacings = np.diff(center)
        spacings /= np.mean(spacings)
        return spacings, np.sum(spacings < 0.5) / len(spacings), np.var(spacings)

    # Run tests
    console.print("\n[bold]Testing kernels with λ=10.0...[/]\n")

    results = []

    # Test our kernel
    s_ours, p_ours, v_ours = build_and_analyze(kernel_ours, "Ours (mod)")
    results.append(("Our kernel (cos mod)", s_ours, p_ours, v_ours))

    # Test raw kernel (no modulation)
    s_raw, p_raw, v_raw = build_and_analyze(kernel_raw, "Raw")
    results.append(("Raw kernel (no cos)", s_raw, p_raw, v_raw))

    # Test sine kernel
    s_sine, p_sine, v_sine = build_and_analyze(kernel_sine, "Sine")
    results.append(("Sine kernel", s_sine, p_sine, v_sine))

    # Test GOE
    s_goe, p_goe, v_goe = goe_spacings()
    results.append(("GOE (true RMT)", s_goe, p_goe, v_goe))

    # Results table
    table = Table(title="🎯 KERNEL COMPARISON (λ=10)")
    table.add_column("Kernel", style="bold")
    table.add_column("P(s<0.5)", style="yellow", justify="right")
    table.add_column("Var(s)", justify="right")
    table.add_column("Verdict", justify="center")

    for name, s, p, v in results:
        if p < 0.18:
            verdict = "[bold green]GUE![/]"
        elif p < 0.25:
            verdict = "[green]Near-GUE[/]"
        elif p < 0.35:
            verdict = "[yellow]Transition[/]"
        else:
            verdict = "[red]Poisson[/]"

        table.add_row(name, f"{p:.3f}", f"{v:.3f}", verdict)

    # Reference
    table.add_row("[dim]Poisson ref[/]", "[dim]0.393[/]", "[dim]1.0[/]", "[dim]—[/]")
    table.add_row("[dim]GUE ref[/]", "[dim]0.14[/]", "[dim]0.27[/]", "[dim]—[/]")

    console.print(table)

    # Test with NO diagonal disorder (pure kernel)
    console.print("\n[bold magenta]Test WITHOUT diagonal disorder:[/]")

    def pure_kernel_test(kernel, name):
        H = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                H[i, j] = kernel[abs(i - j)]
        # NO diagonal noise!

        eigs = np.linalg.eigvalsh(H)
        # For pure Toeplitz, eigenvalues may be degenerate or weird
        center = eigs[int(N * 0.2):int(N * 0.8)]
        center = np.sort(center)
        spacings = np.diff(center)
        if np.mean(spacings) > 0:
            spacings /= np.mean(spacings)
        else:
            spacings = np.ones_like(spacings)  # fallback

        prob_small = np.sum(spacings < 0.5) / len(spacings)
        return spacings, prob_small

    s_pure_sine, p_pure_sine = pure_kernel_test(kernel_sine, "Pure Sine")
    s_pure_ours, p_pure_ours = pure_kernel_test(kernel_ours, "Pure Ours")

    console.print(f"  Pure Sine kernel: P(s<0.5) = {p_pure_sine:.3f}")
    console.print(f"  Pure Our kernel:  P(s<0.5) = {p_pure_ours:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: Kernel shapes
    ax = axes[0]
    d_plot = np.arange(30)
    k_ours_plot = (0.127 * d_plot + 0.062) * np.exp(-1.16 * np.sqrt(d_plot)) * np.cos(np.pi * d_plot)
    k_ours_plot[0] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        k_sine_plot = np.sin(np.pi * d_plot) / (np.pi * d_plot)
    k_sine_plot[0] = 1.0

    ax.stem(d_plot, k_ours_plot, 'b', markerfmt='bo', label='Our kernel', basefmt=' ')
    ax.stem(d_plot + 0.2, k_sine_plot, 'g', markerfmt='go', label='Sine kernel', basefmt=' ')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_xlabel('Distance d')
    ax.set_ylabel('K(d)')
    ax.set_title('Kernel Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: Spacing histograms
    ax = axes[1]
    x = np.linspace(0, 4, 100)
    p_poisson = np.exp(-x)
    p_wigner = (np.pi * x / 2) * np.exp(-np.pi * x ** 2 / 4)

    ax.plot(x, p_poisson, 'k--', alpha=0.5, lw=2, label='Poisson')
    ax.plot(x, p_wigner, 'k-', alpha=0.3, lw=3, label='Wigner')

    for name, s, p, v in results:
        hist, bins = np.histogram(s, bins=50, density=True, range=(0, 4))
        centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(centers, hist, lw=2, label=f'{name} (P={p:.2f})')

    ax.set_xlabel('Spacing s')
    ax.set_ylabel('P(s)')
    ax.set_title('Spacing Distributions')
    ax.legend()
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # Right: P(s<0.5) bar chart
    ax = axes[2]
    names = [r[0] for r in results]
    probs = [r[2] for r in results]
    colors = ['red' if p > 0.35 else 'orange' if p > 0.25 else 'lightgreen' if p > 0.18 else 'green' for p in probs]

    bars = ax.barh(names, probs, color=colors)
    ax.axvline(x=0.393, color='r', linestyle='--', label='Poisson')
    ax.axvline(x=0.14, color='g', linestyle='--', label='GUE')
    ax.set_xlabel('P(s < 0.5)')
    ax.set_title('Level Repulsion Metric')
    ax.legend()
    ax.set_xlim(0, 0.5)

    plt.tight_layout()
    plt.savefig('reports/kernel_comparison.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/kernel_comparison.png[/]")

    # Diagnosis
    console.print("\n")

    if p_goe < 0.2 and p_sine > 0.3:
        diagnosis = (
            "[bold red]PROBLEM: Toeplitz structure doesn't produce GUE![/]\n\n"
            "Even the sine kernel (which is THE GUE kernel) fails in Toeplitz form.\n"
            "This means: Toeplitz matrices ≠ Random Matrix Theory.\n\n"
            "Toeplitz = translational invariance → band structure → Poisson\n"
            "RMT needs GLOBAL correlations, not just local interactions."
        )
    elif p_sine < 0.2 and p_ours > 0.3:
        diagnosis = (
            "[bold yellow]PROBLEM: Our kernel shape is wrong![/]\n\n"
            "Sine kernel works but ours doesn't.\n"
            "Our kernel: d·exp(-γ√d)·cos(πd)\n"
            "GUE kernel: sin(πd)/(πd)\n\n"
            "We need to find the RIGHT oscillation, not just cos(πd)."
        )
    else:
        diagnosis = (
            f"GOE: {p_goe:.3f}, Sine: {p_sine:.3f}, Ours: {p_ours:.3f}\n\n"
            "Need further analysis..."
        )

    console.print(Panel.fit(diagnosis, title="🔍 DIAGNOSIS", border_style="cyan"))


if __name__ == "__main__":
    compare_kernels()
