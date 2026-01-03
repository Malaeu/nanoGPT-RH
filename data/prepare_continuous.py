#!/usr/bin/env python3
"""
Prepare CONTINUOUS spacing data for MDN training.

Unlike binned data, this preserves exact float values for regression/MDN.
Output: train.pt, val.pt with float32 tensors of shape (N, seq_len).
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


def unfold(zeros: np.ndarray) -> np.ndarray:
    """
    Unfold zeros using Variant B (unfolded coordinates).
    u(gamma) = (gamma / 2pi) * log(gamma / (2*pi*e))
    s_n = u(gamma_{n+1}) - u(gamma_n)
    """
    def u(gamma):
        return (gamma / (2 * np.pi)) * np.log(gamma / (2 * np.pi * np.e))

    u_vals = u(zeros)
    spacings = np.diff(u_vals)
    return spacings.astype(np.float32)


def compute_stats(spacings: np.ndarray, name: str = "Spacings"):
    """Display statistics."""
    stats = {
        "mean": np.mean(spacings),
        "std": np.std(spacings),
        "min": np.min(spacings),
        "max": np.max(spacings),
        "median": np.median(spacings),
    }

    table = Table(title=f"[bold]{name}[/]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Target", style="dim")

    table.add_row("Mean", f"{stats['mean']:.6f}", "~1.0")
    table.add_row("Std", f"{stats['std']:.6f}", "~0.42 (GUE)")
    table.add_row("Median", f"{stats['median']:.6f}", "~0.87 (GUE)")
    table.add_row("Min", f"{stats['min']:.6f}", ">0 (repulsion)")
    table.add_row("Max", f"{stats['max']:.6f}", "")

    console.print(table)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare continuous spacing data")
    parser.add_argument("--input", type=Path, default=Path("data/raw/zeros_100M.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/continuous_100M"))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-zeros", type=int, default=None,
                        help="Limit number of zeros (for testing)")
    args = parser.parse_args()

    console.print(f"[bold cyan]Preparing Continuous Data for MDN[/]")
    console.print(f"Input: {args.input}")
    console.print(f"Output: {args.output}")
    console.print(f"Seq len: {args.seq_len}")

    # Load zeros
    console.print(f"\n[cyan]Loading zeros...[/]")

    if args.max_zeros:
        # Stream limited number
        zeros = []
        with open(args.input, 'r') as f:
            for i, line in enumerate(f):
                if i >= args.max_zeros:
                    break
                zeros.append(float(line.strip()))
        zeros = np.array(zeros)
    else:
        zeros = np.loadtxt(args.input)

    console.print(f"[green]Loaded {len(zeros):,} zeros[/]")
    console.print(f"Range: [{zeros[0]:.2f}, {zeros[-1]:.2f}]")

    # Unfold
    console.print(f"\n[cyan]Unfolding...[/]")
    spacings = unfold(zeros)
    console.print(f"[green]Created {len(spacings):,} spacings[/]")

    # Stats
    compute_stats(spacings)

    # Create sequences
    n_seqs = len(spacings) // args.seq_len
    spacings_truncated = spacings[:n_seqs * args.seq_len]
    sequences = spacings_truncated.reshape(n_seqs, args.seq_len)
    console.print(f"\n[cyan]Created {n_seqs:,} sequences of length {args.seq_len}[/]")

    # Split
    n_val = int(n_seqs * args.val_ratio)
    n_train = n_seqs - n_val

    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:]

    console.print(f"[green]Train: {len(train_seqs):,} sequences[/]")
    console.print(f"[green]Val: {len(val_seqs):,} sequences[/]")

    # Convert to tensors
    train_pt = torch.from_numpy(train_seqs).float()
    val_pt = torch.from_numpy(val_seqs).float()

    # Save
    args.output.mkdir(parents=True, exist_ok=True)

    torch.save(train_pt, args.output / "train.pt")
    torch.save(val_pt, args.output / "val.pt")

    # Meta
    meta = {
        "train_shape": list(train_pt.shape),
        "val_shape": list(val_pt.shape),
        "seq_len": args.seq_len,
        "dtype": "float32",
        "continuous": True,
        "mean_spacing": float(spacings.mean()),
        "std_spacing": float(spacings.std()),
    }
    torch.save(meta, args.output / "meta.pt")

    console.print(f"\n[bold green]Saved to {args.output}/[/]")
    console.print(f"  train.pt: {train_pt.shape}")
    console.print(f"  val.pt: {val_pt.shape}")
    console.print(f"  meta.pt: statistics")

    # Quick sanity check
    console.print(f"\n[cyan]Sanity check:[/]")
    console.print(f"  Train[0,:5] = {train_pt[0,:5].tolist()}")
    console.print(f"  Val[0,:5] = {val_pt[0,:5].tolist()}")


if __name__ == "__main__":
    main()
