#!/usr/bin/env python3
"""
CONSTRUCT OPERATOR: Build Hamiltonian from Learned Kernel

We have the kernel: μ(d) = (0.127·d + 0.062) × exp(-1.16·√d) + 0.0017
Now we build a Toeplitz matrix H where H[i,j] = μ(|i-j|)

If the eigenvalues of H have the same SFF statistics as Riemann zeros,
we have reverse-engineered the generator!

No magic numbers. Pure math.
"""

import numpy as np
import torch
from scipy.linalg import eigvalsh  # For symmetric matrices
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt

console = Console()


def kernel_pysr(d):
    """
    Our learned kernel from PySR (R² = 0.9927):
    μ(d) = (0.127·d + 0.062) × exp(-1.16·√d) + 0.0017
    """
    return (0.127 * d + 0.062) * np.exp(-1.16 * np.sqrt(d)) + 0.0017


def kernel_sine(d, beta=np.pi):
    """
    GUE sine kernel for comparison:
    K(d) = sin(πd) / (πd)  for d > 0
    K(0) = 1
    """
    if np.isscalar(d):
        if d == 0:
            return 1.0
        return np.sin(beta * d) / (beta * d)
    else:
        result = np.zeros_like(d, dtype=float)
        mask = d != 0
        result[mask] = np.sin(beta * d[mask]) / (beta * d[mask])
        result[~mask] = 1.0
        return result


def build_toeplitz_operator(N, kernel_func, diagonal_noise=0.0):
    """
    Build symmetric Toeplitz matrix from kernel.
    H[i,j] = kernel(|i-j|)
    """
    H = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            H[i, j] = kernel_func(d)

    # Add diagonal noise if specified
    if diagonal_noise > 0:
        H += np.diag(np.random.randn(N) * diagonal_noise)

    # Ensure symmetry (should already be symmetric)
    H = (H + H.T) / 2

    return H


def compute_sff_from_eigenvalues(eigenvalues, tau_values):
    """
    Compute SFF from eigenvalues.
    K(τ) = |Σ exp(iτλ)|² / N
    """
    N = len(eigenvalues)

    # Normalize eigenvalues to mean spacing = 1
    sorted_eigs = np.sort(eigenvalues)
    spacings = np.diff(sorted_eigs)
    mean_spacing = np.mean(spacings)
    normalized_eigs = sorted_eigs / mean_spacing

    sff = []
    for tau in tau_values:
        phases = tau * normalized_eigs
        complex_sum = np.sum(np.exp(1j * phases))
        k = (np.abs(complex_sum) ** 2) / N
        sff.append(k)

    return np.array(sff), spacings / mean_spacing


def analyze_spacing_distribution(spacings):
    """Compare spacing distribution to GUE Wigner surmise."""
    # Wigner surmise: P(s) = (π/2) s exp(-πs²/4)
    s = np.linspace(0, 4, 100)
    wigner = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

    # Histogram of actual spacings
    hist, bin_edges = np.histogram(spacings, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, hist, s, wigner


def construct_and_analyze():
    console.print(Panel.fit(
        "[bold cyan]🏗️ CONSTRUCT OPERATOR FROM KERNEL[/]\n"
        "Building Toeplitz Hamiltonian from μ(d)",
        title="REVERSE ENGINEERING"
    ))

    # Parameters
    N = 500  # Matrix size
    tau_values = np.linspace(0.1, 10, 500)

    # Build operators with different kernels
    console.print("\n[bold]Building Operators...[/]")

    # 1. Our learned kernel
    console.print("  Building H_learned (PySR kernel)...")
    H_learned = build_toeplitz_operator(N, kernel_pysr)
    eigs_learned = eigvalsh(H_learned)

    # 2. Sine kernel (GUE reference)
    console.print("  Building H_sine (GUE sine kernel)...")
    H_sine = build_toeplitz_operator(N, kernel_sine)
    eigs_sine = eigvalsh(H_sine)

    # 3. Random matrix (GOE baseline)
    console.print("  Building H_random (GOE baseline)...")
    A = np.random.randn(N, N)
    H_random = (A + A.T) / 2 / np.sqrt(N)
    eigs_random = eigvalsh(H_random)

    # Compute SFF
    console.print("\n[bold]Computing SFF...[/]")
    sff_learned, spacings_learned = compute_sff_from_eigenvalues(eigs_learned, tau_values)
    sff_sine, spacings_sine = compute_sff_from_eigenvalues(eigs_sine, tau_values)
    sff_random, spacings_random = compute_sff_from_eigenvalues(eigs_random, tau_values)

    # Compute plateaus (τ > 5)
    plateau_mask = tau_values > 5
    plateau_learned = np.mean(sff_learned[plateau_mask])
    plateau_sine = np.mean(sff_sine[plateau_mask])
    plateau_random = np.mean(sff_random[plateau_mask])

    # Results table
    console.print("\n")
    table = Table(title="⚛️ OPERATOR SFF COMPARISON")
    table.add_column("Operator", style="bold")
    table.add_column("Kernel", style="cyan")
    table.add_column("SFF Plateau", justify="right")
    table.add_column("vs GOE", justify="right")

    table.add_row("H_learned", "d·exp(-γ√d)", f"{plateau_learned:.3f}", f"{plateau_learned/plateau_random:.2f}x")
    table.add_row("H_sine", "sin(πd)/(πd)", f"{plateau_sine:.3f}", f"{plateau_sine/plateau_random:.2f}x")
    table.add_row("H_random", "GOE", f"{plateau_random:.3f}", "1.00x")
    table.add_row("[dim]Real Zeros[/]", "[dim]Target[/]", "[dim]≈2.5[/]", "[dim]—[/]")

    console.print(table)

    # Spacing statistics
    console.print("\n[bold]Spacing Statistics:[/]")
    console.print(f"  H_learned: mean={np.mean(spacings_learned):.4f}, var={np.var(spacings_learned):.4f}")
    console.print(f"  H_sine:    mean={np.mean(spacings_sine):.4f}, var={np.var(spacings_sine):.4f}")
    console.print(f"  H_random:  mean={np.mean(spacings_random):.4f}, var={np.var(spacings_random):.4f}")

    # Verdict
    console.print("\n")
    if plateau_learned > 1.5:
        verdict = "[bold green]SIGNIFICANT STRUCTURE![/]"
        msg = f"Learned kernel produces SFF plateau {plateau_learned:.2f} (vs GOE {plateau_random:.2f})"
    elif plateau_learned > plateau_random * 1.2:
        verdict = "[yellow]Some structure detected[/]"
        msg = f"Learned kernel shows {plateau_learned/plateau_random:.1f}x enhancement over GOE"
    else:
        verdict = "[red]No significant structure[/]"
        msg = "Learned kernel produces GOE-like spectrum"

    console.print(Panel.fit(
        f"[bold]Verdict:[/] {verdict}\n\n"
        f"{msg}\n\n"
        f"[dim]Target for Riemann zeros: plateau ≈ 2.5[/]",
        title="📊 RESULT",
        border_style="cyan"
    ))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. SFF comparison
    ax = axes[0, 0]
    ax.plot(tau_values, sff_learned, 'b-', label=f'Learned (plateau={plateau_learned:.2f})', linewidth=2)
    ax.plot(tau_values, sff_sine, 'g--', label=f'Sine (plateau={plateau_sine:.2f})', linewidth=1.5)
    ax.plot(tau_values, sff_random, 'r:', label=f'GOE (plateau={plateau_random:.2f})', linewidth=1.5)
    ax.axhline(y=2.5, color='purple', linestyle='--', alpha=0.5, label='Riemann target')
    ax.set_xlabel('τ')
    ax.set_ylabel('SFF K(τ)')
    ax.set_title('Spectral Form Factor Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Kernel visualization
    ax = axes[0, 1]
    d_vals = np.linspace(0, 50, 200)
    ax.plot(d_vals, kernel_pysr(d_vals), 'b-', label='Learned μ(d)', linewidth=2)
    ax.plot(d_vals, kernel_sine(d_vals), 'g--', label='Sine kernel', linewidth=1.5)
    ax.set_xlabel('Distance d')
    ax.set_ylabel('Kernel μ(d)')
    ax.set_title('Kernel Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Spacing distribution (learned)
    ax = axes[1, 0]
    bc, hist, s, wigner = analyze_spacing_distribution(spacings_learned)
    ax.bar(bc, hist, width=bc[1]-bc[0], alpha=0.7, label='H_learned')
    ax.plot(s, wigner, 'r-', linewidth=2, label='Wigner surmise')
    ax.set_xlabel('Spacing s')
    ax.set_ylabel('P(s)')
    ax.set_title('Spacing Distribution (Learned)')
    ax.legend()
    ax.set_xlim(0, 4)

    # 4. Eigenvalue spectrum
    ax = axes[1, 1]
    ax.hist(eigs_learned, bins=50, alpha=0.7, density=True, label='H_learned')
    ax.hist(eigs_random, bins=50, alpha=0.5, density=True, label='GOE')
    ax.set_xlabel('Eigenvalue λ')
    ax.set_ylabel('Density')
    ax.set_title('Eigenvalue Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig('reports/operator_construction.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/operator_construction.png[/]")


if __name__ == "__main__":
    construct_and_analyze()
