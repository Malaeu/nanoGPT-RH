#!/usr/bin/env python3
"""
SFF Validation Script.

Tests:
1. GUE baseline - synthetic GUE matrix eigenvalues
2. Shuffle baseline - i.i.d. shuffled spacings (should break ramp)
3. Real stability - check plateau consistency across blocks

Usage:
    python -m causal_zeta.validate_sff --data-dir data
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def compute_sff(spacings: np.ndarray, tau_values: np.ndarray = None, connected: bool = False) -> dict:
    """
    Compute Spectral Form Factor from spacings.
    K(τ) = (1/N) |Σ exp(i τ u_n)|²
    where u_n = cumsum(spacings) are unfolded coordinates.

    If connected=True, compute connected SFF:
    K_c(τ) = K(τ) - |<exp(iτu)>|² = K(τ) - (sin(τN/2)/(N sin(τ/2)))²
    This removes the "trivial" contribution and highlights correlations.
    """
    u = np.concatenate([[0], np.cumsum(spacings)])
    N = len(u)

    if tau_values is None:
        tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    K_values = np.zeros(len(tau_values))
    for i, tau in enumerate(tau_values):
        phases = np.exp(1j * tau * u)
        K_values[i] = np.abs(np.sum(phases))**2 / N

    if connected:
        # Connected SFF: subtract Poisson baseline
        # For uniform spacing (mean=1), the "trivial" contribution is:
        # K_0(τ) ≈ 1 for large N (when τ << N)
        # Simple approximation: subtract 1
        K_values = K_values - 1.0
        K_values = np.maximum(K_values, 0)  # Clip negative values

    # Ramp region: τ in [0.5, 3]
    ramp_mask = (tau_values >= 0.5) & (tau_values <= 3.0)
    if np.sum(ramp_mask) > 2:
        tau_ramp = tau_values[ramp_mask]
        K_ramp = K_values[ramp_mask]
        slope, intercept = np.polyfit(tau_ramp, K_ramp, 1)
        K_fit = slope * tau_ramp + intercept
        rmse = np.sqrt(np.mean((K_ramp - K_fit)**2))
    else:
        slope, rmse = 0.0, 0.0

    # Plateau region: τ > 4
    plateau_mask = tau_values > 4.0
    plateau_level = np.mean(K_values[plateau_mask]) if np.sum(plateau_mask) > 0 else K_values[-1]

    return {
        "tau_values": tau_values,
        "K_values": K_values,
        "ramp_slope": slope,
        "ramp_rmse": rmse,
        "plateau_level": plateau_level,
        "N": N,
        "connected": connected,
    }


def generate_gue_spacings(n_eigenvalues: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Generate spacings from GUE (Gaussian Unitary Ensemble).

    GUE = (H + H†) / 2 where H_ij ~ CN(0, 1/n)
    """
    np.random.seed(seed)

    # Generate complex Gaussian matrix
    H_real = np.random.randn(n_eigenvalues, n_eigenvalues)
    H_imag = np.random.randn(n_eigenvalues, n_eigenvalues)
    H = (H_real + 1j * H_imag) / np.sqrt(2 * n_eigenvalues)

    # Make Hermitian
    H = (H + H.conj().T) / 2

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues = np.sort(eigenvalues)

    # Unfold: for GUE, mean spacing ~ 1/(n * semicircle density)
    # Simple unfolding: rescale to mean spacing = 1
    spacings = np.diff(eigenvalues)
    spacings = spacings / np.mean(spacings)  # Normalize to mean = 1

    return spacings


def generate_poisson_spacings(n_spacings: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Generate i.i.d. exponential spacings (Poisson process).
    This is the "no correlation" baseline.
    """
    np.random.seed(seed)
    spacings = np.random.exponential(1.0, n_spacings)
    return spacings


def shuffle_spacings(spacings: np.ndarray, seed: int = 42) -> np.ndarray:
    """Shuffle spacings to destroy correlations."""
    np.random.seed(seed)
    shuffled = spacings.copy()
    np.random.shuffle(shuffled)
    return shuffled


def load_real_spacings(data_dir: str, n_spacings: int = 10000) -> np.ndarray:
    """Load real spacings from validation data."""
    import torch

    data_path = Path(data_dir)
    val_data = torch.load(data_path / "val.pt", weights_only=False)
    bin_centers = np.load(data_path / "bin_centers.npy")

    if isinstance(val_data, torch.Tensor):
        val_data = val_data.numpy()

    all_spacings = []
    for i in range(len(val_data)):
        tokens = val_data[i]
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
        spacings = bin_centers[tokens]
        all_spacings.extend(spacings)
        if len(all_spacings) >= n_spacings:
            break

    return np.array(all_spacings[:n_spacings])


def test_plateau_stability(real_spacings: np.ndarray, block_sizes: list = [512, 1024, 2048, 4096]) -> dict:
    """Test plateau stability across different block sizes."""
    results = {}
    tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    for block_size in block_sizes:
        if block_size > len(real_spacings):
            continue

        # Compute SFF on multiple non-overlapping blocks
        n_blocks = len(real_spacings) // block_size
        if n_blocks < 2:
            continue

        plateaus = []
        ramps = []
        for i in range(min(n_blocks, 10)):  # Max 10 blocks
            start = i * block_size
            end = start + block_size
            block_spacings = real_spacings[start:end]
            sff = compute_sff(block_spacings, tau_values)
            plateaus.append(sff["plateau_level"])
            ramps.append(sff["ramp_slope"])

        results[block_size] = {
            "n_blocks": len(plateaus),
            "plateau_mean": np.mean(plateaus),
            "plateau_std": np.std(plateaus),
            "plateau_cv": np.std(plateaus) / np.mean(plateaus) if np.mean(plateaus) > 0 else 0,
            "ramp_mean": np.mean(ramps),
            "ramp_std": np.std(ramps),
            "ramp_cv": np.std(ramps) / np.mean(ramps) if np.mean(ramps) > 0 else 0,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="SFF Validation")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-spacings", type=int, default=10000)
    parser.add_argument("--gue-size", type=int, default=1000)
    args = parser.parse_args()

    console.print(Panel.fit("[bold blue]SFF VALIDATION[/]", title="Causal Zeta"))

    tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    # =========================================================================
    # Test 1: GUE Baseline
    # =========================================================================
    console.print("\n[bold cyan]Test 1: GUE Baseline[/]")
    console.print(f"  Generating {args.gue_size}x{args.gue_size} GUE matrix...")

    gue_spacings = generate_gue_spacings(args.gue_size)
    sff_gue = compute_sff(gue_spacings, tau_values)

    console.print(f"  N spacings: {len(gue_spacings)}")
    console.print(f"  Mean spacing: {np.mean(gue_spacings):.4f}")
    console.print(f"  Ramp slope: {sff_gue['ramp_slope']:.6f}")
    console.print(f"  Ramp RMSE: {sff_gue['ramp_rmse']:.6f}")
    console.print(f"  Plateau level: {sff_gue['plateau_level']:.4f}")

    # =========================================================================
    # Test 2: Poisson Baseline (no correlations)
    # =========================================================================
    console.print("\n[bold cyan]Test 2: Poisson Baseline (i.i.d. exponential)[/]")

    poisson_spacings = generate_poisson_spacings(args.n_spacings)
    sff_poisson = compute_sff(poisson_spacings, tau_values)

    console.print(f"  N spacings: {len(poisson_spacings)}")
    console.print(f"  Mean spacing: {np.mean(poisson_spacings):.4f}")
    console.print(f"  Ramp slope: {sff_poisson['ramp_slope']:.6f}")
    console.print(f"  Ramp RMSE: {sff_poisson['ramp_rmse']:.6f}")
    console.print(f"  Plateau level: {sff_poisson['plateau_level']:.4f}")

    # =========================================================================
    # Test 3: Real Data
    # =========================================================================
    console.print("\n[bold cyan]Test 3: Real Data[/]")

    try:
        import torch
        real_spacings = load_real_spacings(args.data_dir, args.n_spacings)
        sff_real = compute_sff(real_spacings, tau_values)

        console.print(f"  N spacings: {len(real_spacings)}")
        console.print(f"  Mean spacing: {np.mean(real_spacings):.4f}")
        console.print(f"  Ramp slope: {sff_real['ramp_slope']:.6f}")
        console.print(f"  Ramp RMSE: {sff_real['ramp_rmse']:.6f}")
        console.print(f"  Plateau level: {sff_real['plateau_level']:.4f}")

        # Shuffle test
        console.print("\n[bold cyan]Test 4: Shuffled Real Data[/]")
        shuffled_spacings = shuffle_spacings(real_spacings)
        sff_shuffled = compute_sff(shuffled_spacings, tau_values)

        console.print(f"  Ramp slope: {sff_shuffled['ramp_slope']:.6f}")
        console.print(f"  Plateau level: {sff_shuffled['plateau_level']:.4f}")

        # Plateau stability
        console.print("\n[bold cyan]Test 5: Plateau Stability Across Block Sizes[/]")
        stability = test_plateau_stability(real_spacings, [512, 1024, 2048, 4096])

        table = Table(title="Plateau Stability (CV = std/mean)")
        table.add_column("Block Size", style="cyan")
        table.add_column("N Blocks", style="green")
        table.add_column("Plateau Mean", style="yellow")
        table.add_column("Plateau CV", style="magenta")
        table.add_column("Ramp Mean", style="yellow")
        table.add_column("Ramp CV", style="magenta")

        for block_size, stats in sorted(stability.items()):
            table.add_row(
                str(block_size),
                str(stats["n_blocks"]),
                f"{stats['plateau_mean']:.4f}",
                f"{stats['plateau_cv']:.2%}",
                f"{stats['ramp_mean']:.6f}",
                f"{stats['ramp_cv']:.2%}",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Could not load real data: {e}[/]")
        sff_real = None

    # =========================================================================
    # Summary Table
    # =========================================================================
    console.print("\n[bold magenta]SUMMARY[/]")

    table = Table(title="SFF Validation Results")
    table.add_column("Source", style="cyan")
    table.add_column("Ramp Slope", style="green")
    table.add_column("Ramp RMSE", style="yellow")
    table.add_column("Plateau", style="blue")
    table.add_column("Verdict", style="magenta")

    table.add_row(
        "GUE (expected ramp)",
        f"{sff_gue['ramp_slope']:.6f}",
        f"{sff_gue['ramp_rmse']:.6f}",
        f"{sff_gue['plateau_level']:.4f}",
        "✓ Baseline" if sff_gue['ramp_slope'] > 0 else "?"
    )

    table.add_row(
        "Poisson (no ramp)",
        f"{sff_poisson['ramp_slope']:.6f}",
        f"{sff_poisson['ramp_rmse']:.6f}",
        f"{sff_poisson['plateau_level']:.4f}",
        "✓ No ramp" if sff_poisson['ramp_slope'] < sff_gue['ramp_slope'] * 0.3 else "? Check"
    )

    if sff_real:
        table.add_row(
            "Real zeta",
            f"{sff_real['ramp_slope']:.6f}",
            f"{sff_real['ramp_rmse']:.6f}",
            f"{sff_real['plateau_level']:.4f}",
            "Reference"
        )
        table.add_row(
            "Shuffled real",
            f"{sff_shuffled['ramp_slope']:.6f}",
            f"{sff_shuffled['ramp_rmse']:.6f}",
            f"{sff_shuffled['plateau_level']:.4f}",
            "✓ Ramp broken" if sff_shuffled['ramp_slope'] < sff_real['ramp_slope'] * 0.5 else "? Check"
        )

    console.print(table)

    # Interpretation
    console.print("\n[bold]Interpretation:[/]")
    console.print("  - GUE should show clear ramp (positive slope)")
    console.print("  - Poisson should have weak/no ramp (destroys correlations)")
    console.print("  - Shuffled real should break ramp (same marginals, no sequence)")
    console.print("  - If plateau CV > 50%, plateau is unstable (finite-size effects)")


if __name__ == "__main__":
    main()
