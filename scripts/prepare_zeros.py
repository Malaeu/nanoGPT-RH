#!/usr/bin/env python3
"""
Zeta zeros unfolding pipeline.
Transforms raw imaginary parts of Riemann zeta zeros into unfolded spacings.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


def load_zeros(path: Path) -> np.ndarray:
    """Load raw zeros from text file."""
    console.print(f"[cyan]Loading zeros from {path}...[/]")
    zeros = np.loadtxt(path)
    console.print(f"[green]Loaded {len(zeros):,} zeros[/]")
    console.print(f"  γ_min = {zeros[0]:.6f}, γ_max = {zeros[-1]:.6f}")
    return zeros


def unfold_variant_a(zeros: np.ndarray) -> np.ndarray:
    """
    Unfolding Variant A (local density normalization).

    s_n = Δ_n * log(γ_n) / (2π)

    where Δ_n = γ_{n+1} - γ_n
    """
    gaps = np.diff(zeros)  # Δ_n = γ_{n+1} - γ_n
    # Use midpoint for density estimation
    midpoints = (zeros[:-1] + zeros[1:]) / 2
    spacings = gaps * np.log(midpoints) / (2 * np.pi)
    return spacings


def unfold_variant_b(zeros: np.ndarray) -> np.ndarray:
    """
    Unfolding Variant B (unfolded coordinates).

    u(γ) = (γ / 2π) * log(γ / (2πe))
    s_n = u(γ_{n+1}) - u(γ_n)
    """
    def u(gamma):
        return (gamma / (2 * np.pi)) * np.log(gamma / (2 * np.pi * np.e))

    u_vals = u(zeros)
    spacings = np.diff(u_vals)
    return spacings


def compute_statistics(spacings: np.ndarray, name: str) -> dict:
    """Compute and display statistics for unfolded spacings."""
    stats = {
        "mean": np.mean(spacings),
        "std": np.std(spacings),
        "min": np.min(spacings),
        "max": np.max(spacings),
        "median": np.median(spacings),
    }

    table = Table(title=f"[bold]{name} Statistics[/]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Target", style="yellow")

    table.add_row("Mean", f"{stats['mean']:.6f}", "≈ 1.0")
    table.add_row("Std", f"{stats['std']:.6f}", "≈ 0.42 (GUE)")
    table.add_row("Median", f"{stats['median']:.6f}", "≈ 0.87 (GUE)")
    table.add_row("Min", f"{stats['min']:.6f}", "> 0 (repulsion)")
    table.add_row("Max", f"{stats['max']:.6f}", "")

    console.print(table)
    return stats


def quantize_spacings(spacings: np.ndarray, n_bins: int = 256, max_val: float = 4.0) -> tuple:
    """
    Quantize continuous spacings into discrete bins for classification.

    Args:
        spacings: continuous spacing values
        n_bins: number of bins (vocabulary size)
        max_val: maximum spacing value (clipped above this)

    Returns:
        tokens: integer tokens 0..n_bins-1
        bin_edges: edges for decoding back to continuous
    """
    # Clip to [0, max_val]
    clipped = np.clip(spacings, 0, max_val - 1e-6)

    # Uniform bins from 0 to max_val
    bin_edges = np.linspace(0, max_val, n_bins + 1)
    tokens = np.digitize(clipped, bin_edges[1:-1])  # 0 to n_bins-1

    # Stats
    unique, counts = np.unique(tokens, return_counts=True)
    n_used = len(unique)
    entropy = -np.sum((counts / len(tokens)) * np.log2(counts / len(tokens) + 1e-10))

    console.print(f"[cyan]Quantized to {n_bins} bins (max={max_val})[/]")
    console.print(f"  Bins used: {n_used}/{n_bins}")
    console.print(f"  Empirical entropy: {entropy:.2f} bits")
    console.print(f"  Theoretical perplexity lower bound: {2**entropy:.1f}")

    return tokens.astype(np.int64), bin_edges


def create_sequences(spacings: np.ndarray, seq_len: int = 256) -> np.ndarray:
    """Split spacings into sequences of fixed length."""
    n_seqs = len(spacings) // seq_len
    truncated = spacings[:n_seqs * seq_len]
    sequences = truncated.reshape(n_seqs, seq_len)
    console.print(f"[cyan]Created {n_seqs:,} sequences of length {seq_len}[/]")
    return sequences


def train_val_split(sequences: np.ndarray, val_ratio: float = 0.1) -> tuple:
    """
    Split sequences into train/val BY BLOCKS (no shuffling).
    This preserves spectral structure.
    """
    n_val = int(len(sequences) * val_ratio)
    n_train = len(sequences) - n_val

    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:]

    console.print(f"[cyan]Train: {len(train_seqs):,} sequences[/]")
    console.print(f"[cyan]Val: {len(val_seqs):,} sequences[/]")

    return train_seqs, val_seqs


def save_tensors(train: np.ndarray, val: np.ndarray, output_dir: Path,
                  binned: bool = False, bin_edges: np.ndarray = None, n_bins: int = None):
    """Save as PyTorch tensors."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if binned:
        train_pt = torch.from_numpy(train).long()
        val_pt = torch.from_numpy(val).long()
    else:
        train_pt = torch.from_numpy(train).float()
        val_pt = torch.from_numpy(val).float()

    torch.save(train_pt, output_dir / "train.pt")
    torch.save(val_pt, output_dir / "val.pt")

    # Save metadata
    meta = {
        "train_shape": list(train_pt.shape),
        "val_shape": list(val_pt.shape),
        "seq_len": train_pt.shape[1],
        "dtype": str(train_pt.dtype),
        "binned": binned,
        "n_bins": n_bins,
        "vocab_size": n_bins if binned else None,
    }
    if bin_edges is not None:
        meta["bin_edges"] = bin_edges.tolist()

    torch.save(meta, output_dir / "meta.pt")

    console.print(f"[green]Saved tensors to {output_dir}[/]")
    console.print(f"  train.pt: {train_pt.shape} ({train_pt.dtype})")
    console.print(f"  val.pt: {val_pt.shape} ({val_pt.dtype})")
    if binned:
        console.print(f"  vocab_size: {n_bins}")


def plot_distribution(spacings: np.ndarray, output_path: Path):
    """Plot spacing distribution vs GUE Wigner surmise."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed, skipping plot[/]")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of actual spacings
    ax.hist(spacings, bins=100, density=True, alpha=0.7,
            label="Unfolded spacings", color="steelblue")

    # GUE Wigner surmise: P(s) = (32/π²) s² exp(-4s²/π), std ≈ 0.422
    s = np.linspace(0, 4, 500)
    gue = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    ax.plot(s, gue, 'r-', lw=2, label="GUE Wigner surmise")

    # Poisson: P(s) = exp(-s)
    poisson = np.exp(-s)
    ax.plot(s, poisson, 'g--', lw=2, label="Poisson (uncorrelated)")

    ax.set_xlabel("Spacing s", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title("Zeta Zero Spacing Distribution", fontsize=14)
    ax.legend()
    ax.set_xlim(0, 4)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    console.print(f"[green]Saved distribution plot to {output_path}[/]")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Prepare zeta zeros for training")
    parser.add_argument("--input", type=Path, required=True, help="Path to zeros file")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Output directory")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--variant", choices=["a", "b"], default="b", help="Unfolding variant")
    parser.add_argument("--plot", action="store_true", help="Generate distribution plot")
    parser.add_argument("--binned", action="store_true", help="Quantize to discrete bins")
    parser.add_argument("--n-bins", type=int, default=256, help="Number of bins for quantization")
    parser.add_argument("--max-val", type=float, default=4.0, help="Max spacing value for binning")
    args = parser.parse_args()

    console.print("[bold magenta]═══ Zeta Zeros Unfolding Pipeline ═══[/]\n")

    # Load
    zeros = load_zeros(args.input)

    # Unfold
    console.print(f"\n[bold]Unfolding (Variant {args.variant.upper()})...[/]")
    if args.variant == "a":
        spacings = unfold_variant_a(zeros)
    else:
        spacings = unfold_variant_b(zeros)

    # Statistics
    console.print()
    stats = compute_statistics(spacings, f"Variant {args.variant.upper()}")

    # Quality check
    if abs(stats["mean"] - 1.0) > 0.01:
        console.print("[yellow]⚠ Warning: mean deviates from 1.0[/]")
    else:
        console.print("[green]✓ Unfolding quality OK (mean ≈ 1)[/]")

    # Binning (optional)
    bin_edges = None
    if args.binned:
        console.print()
        spacings, bin_edges = quantize_spacings(spacings, args.n_bins, args.max_val)

    # Create sequences
    console.print()
    sequences = create_sequences(spacings, args.seq_len)

    # Split
    train, val = train_val_split(sequences, args.val_ratio)

    # Save
    console.print()
    save_tensors(train, val, args.output,
                 binned=args.binned, bin_edges=bin_edges, n_bins=args.n_bins)

    # Plot (only for continuous spacings)
    if args.plot and not args.binned:
        plot_distribution(spacings, args.output / "spacing_distribution.png")

    console.print("\n[bold green]✓ Done![/]")


if __name__ == "__main__":
    main()
