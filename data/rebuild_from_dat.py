#!/usr/bin/env python3
"""
Rebuild training data from clean Platt .dat files.

Reads all zeros_*.dat files, computes unfolded spacings,
and creates train/val splits for MDN training.
"""

import struct
import argparse
from pathlib import Path
import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def read_zeros_from_dat(filepath: Path) -> np.ndarray:
    """Read all zeros from a single .dat file (Platt binary format)."""
    zeros = []

    with open(filepath, 'rb') as f:
        num_blocks = struct.unpack('Q', f.read(8))[0]

        for block_idx in range(num_blocks):
            header = f.read(32)
            t0, t1, Nt0, Nt1 = struct.unpack('ddQQ', header)
            num_zeros = Nt1 - Nt0

            Z = 0
            for i in range(num_zeros):
                data = f.read(13)
                z1, z2, z3 = struct.unpack('QIB', data)
                Z = Z + (z3 << 96) + (z2 << 64) + z1
                zero = t0 + Z * (2 ** -101)
                zeros.append(zero)

    return np.array(zeros, dtype=np.float64)


def unfolding(gamma: np.ndarray) -> np.ndarray:
    """
    Unfolding transformation: γ → u(γ)

    u(γ) = (γ / 2π) * log(γ / 2πe)

    This makes the mean spacing = 1.
    """
    return (gamma / (2 * np.pi)) * np.log(gamma / (2 * np.pi * np.e))


def main():
    parser = argparse.ArgumentParser(description="Rebuild training data from .dat files")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                        help="Directory containing zeros_*.dat files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/continuous_clean"),
                        help="Output directory for processed data")
    parser.add_argument("--max-zeros", type=int, default=200_000_000,
                        help="Maximum zeros to use (default: 200M)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length for training")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--max-spacing", type=float, default=10.0,
                        help="Clip spacings to this maximum")
    args = parser.parse_args()

    console.print("[bold cyan]═══ REBUILD TRAINING DATA FROM CLEAN .DAT FILES ═══[/]")

    # Find all .dat files
    dat_files = sorted(
        args.raw_dir.glob("zeros_*.dat"),
        key=lambda x: int(x.stem.split('_')[1])
    )
    console.print(f"[green]Found {len(dat_files)} .dat files[/]")

    # Read zeros from files
    console.print(f"\n[cyan]Reading zeros (max {args.max_zeros:,})...[/]")

    all_zeros = []
    total_read = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:,}/{task.total:,}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Reading files...", total=len(dat_files))

        for dat_file in dat_files:
            if total_read >= args.max_zeros:
                break

            zeros = read_zeros_from_dat(dat_file)

            # Take only what we need
            remaining = args.max_zeros - total_read
            if len(zeros) > remaining:
                zeros = zeros[:remaining]

            all_zeros.append(zeros)
            total_read += len(zeros)

            progress.update(task, advance=1, description=f"[cyan]{dat_file.name} ({total_read:,} zeros)")

    # Concatenate all zeros
    console.print(f"\n[cyan]Concatenating {total_read:,} zeros...[/]")
    zeros = np.concatenate(all_zeros)
    del all_zeros  # Free memory

    console.print(f"[green]Loaded {len(zeros):,} zeros[/]")
    console.print(f"[dim]γ range: [{zeros[0]:.2f}, {zeros[-1]:.2f}][/]")

    # Verify monotonicity
    console.print("\n[cyan]Verifying monotonicity...[/]")
    diffs = np.diff(zeros)
    non_mono = np.sum(diffs <= 0)
    if non_mono > 0:
        console.print(f"[red]WARNING: {non_mono} non-monotonic pairs![/]")
    else:
        console.print("[green]✓ All zeros monotonically increasing[/]")

    # Check for anomalous gaps
    expected_gaps = 2 * np.pi / np.log(zeros[:-1] / (2 * np.pi))
    ratio = diffs / expected_gaps
    anomalous = np.sum(ratio > 10)
    if anomalous > 0:
        console.print(f"[red]WARNING: {anomalous} anomalous gaps (ratio > 10)![/]")
    else:
        console.print("[green]✓ No anomalous gaps detected[/]")

    # Unfold
    console.print("\n[cyan]Computing unfolded coordinates...[/]")
    u = unfolding(zeros)

    # Compute spacings
    console.print("[cyan]Computing spacings...[/]")
    spacings = np.diff(u).astype(np.float32)

    console.print(f"[green]Spacings: {len(spacings):,}[/]")
    console.print(f"[dim]Mean: {spacings.mean():.6f} (should be ~1.0)[/]")
    console.print(f"[dim]Std: {spacings.std():.6f}[/]")
    console.print(f"[dim]Min: {spacings.min():.6f}, Max: {spacings.max():.4f}[/]")

    # Clip outliers
    console.print(f"\n[cyan]Clipping spacings to [0, {args.max_spacing}]...[/]")
    n_clipped = np.sum(spacings > args.max_spacing)
    spacings = np.clip(spacings, 0, args.max_spacing)
    console.print(f"[dim]Clipped {n_clipped:,} values ({100*n_clipped/len(spacings):.4f}%)[/]")

    # Create sequences
    console.print(f"\n[cyan]Creating sequences (length={args.seq_len})...[/]")
    n_seqs = len(spacings) // args.seq_len
    spacings_truncated = spacings[:n_seqs * args.seq_len]
    sequences = spacings_truncated.reshape(n_seqs, args.seq_len)

    console.print(f"[green]Created {n_seqs:,} sequences[/]")

    # Train/val split (by blocks, no shuffle)
    n_val = int(n_seqs * args.val_ratio)
    n_train = n_seqs - n_val

    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:]

    console.print(f"[green]Train: {n_train:,} sequences[/]")
    console.print(f"[green]Val: {n_val:,} sequences[/]")

    # Convert to tensors
    train_tensor = torch.from_numpy(train_seqs)
    val_tensor = torch.from_numpy(val_seqs)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[cyan]Saving to {args.output_dir}/...[/]")
    torch.save(train_tensor, args.output_dir / "train.pt")
    torch.save(val_tensor, args.output_dir / "val.pt")

    # Save metadata
    meta = {
        'n_zeros': len(zeros),
        'gamma_min': float(zeros[0]),
        'gamma_max': float(zeros[-1]),
        'n_spacings': len(spacings),
        'spacing_mean': float(spacings.mean()),
        'spacing_std': float(spacings.std()),
        'seq_len': args.seq_len,
        'n_train': n_train,
        'n_val': n_val,
        'max_spacing': args.max_spacing,
        'source_files': len(dat_files),
    }
    torch.save(meta, args.output_dir / "meta.pt")

    # Summary table
    console.print("\n")
    table = Table(title="[bold green]Data Rebuild Complete[/]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Source files", f"{len(dat_files)}")
    table.add_row("Total zeros", f"{len(zeros):,}")
    table.add_row("γ range", f"[{zeros[0]:.1f}, {zeros[-1]:.1f}]")
    table.add_row("Spacings", f"{len(spacings):,}")
    table.add_row("Mean spacing", f"{spacings.mean():.6f}")
    table.add_row("Sequences", f"{n_seqs:,}")
    table.add_row("Train", f"{n_train:,}")
    table.add_row("Val", f"{n_val:,}")
    table.add_row("Output", str(args.output_dir))

    console.print(table)

    # File sizes
    train_size = (args.output_dir / "train.pt").stat().st_size / 1e6
    val_size = (args.output_dir / "val.pt").stat().st_size / 1e6
    console.print(f"\n[dim]train.pt: {train_size:.1f} MB[/]")
    console.print(f"[dim]val.pt: {val_size:.1f} MB[/]")


if __name__ == "__main__":
    main()
