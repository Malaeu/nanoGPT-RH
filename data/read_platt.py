#!/usr/bin/env python3
"""
Reader for Dave Platt's binary zeta zeros files.

Binary format:
- 8 bytes: number of blocks (uint64)
- For each block:
  - 32 byte header: t0 (double), t1 (double), Nt0 (uint64), Nt1 (uint64)
  - 13 bytes per zero: encoded as (uint64, uint32, uint8)

Zero reconstruction:
  z1, z2, z3 = struct.unpack('QIB', data)
  Z = (z3 << 96) + (z2 << 64) + z1
  zero = t0 + Z * 2^(-101)
"""

import argparse
import struct
import sqlite3
from pathlib import Path
from decimal import Decimal, getcontext

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

# High precision for zero reconstruction
getcontext().prec = 50
EPS = Decimal(2) ** Decimal(-101)


def read_zeros_from_file(filepath: Path, max_zeros: int = None) -> list[float]:
    """Read zeros from a single .dat file."""
    zeros = []

    with open(filepath, 'rb') as f:
        # Read number of blocks
        num_blocks = struct.unpack('Q', f.read(8))[0]
        console.print(f"  [dim]File has {num_blocks} blocks[/]")

        for block_idx in range(num_blocks):
            # Read block header
            header = f.read(32)
            t0, t1, Nt0, Nt1 = struct.unpack('ddQQ', header)
            num_zeros_in_block = Nt1 - Nt0

            # console.print(f"  [dim]Block {block_idx}: t0={t0:.2f}, N={Nt0}-{Nt1} ({num_zeros_in_block} zeros)[/]")

            # Read zeros in this block
            Z = 0  # Cumulative offset
            for i in range(num_zeros_in_block):
                data = f.read(13)
                z1, z2, z3 = struct.unpack('QIB', data)
                Z = Z + (z3 << 96) + (z2 << 64) + z1

                # Reconstruct zero value (use float64 for speed, precision is ~10^-15)
                # For full precision would use: t0 + float(Decimal(Z) * EPS)
                zero = t0 + Z * (2 ** -101)
                zeros.append(zero)

                if max_zeros and len(zeros) >= max_zeros:
                    return zeros

    return zeros


def read_zeros_with_index(data_dir: Path, db_path: Path, start_n: int = 1, num_zeros: int = 1000000) -> np.ndarray:
    """Read zeros using the SQLite index to find the right files."""

    console.print(f"[cyan]Reading {num_zeros:,} zeros starting from N={start_n}[/]")

    # Connect to index
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Find starting position
    c.execute('SELECT * FROM zero_index WHERE N <= ? ORDER BY N DESC LIMIT 1', (start_n,))
    result = c.fetchone()

    if result is None:
        console.print("[red]Could not find starting position in index[/]")
        return np.array([])

    t0_start, N0, filename, offset, block_number = result
    console.print(f"[dim]Starting from file={filename}, offset={offset}, block={block_number}[/]")

    zeros = []
    current_N = N0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:,}/{task.total:,}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Reading zeros...", total=num_zeros)

        while len(zeros) < num_zeros:
            filepath = data_dir / filename

            if not filepath.exists():
                console.print(f"[yellow]File not found: {filepath}[/]")
                # Try to find next file in index
                c.execute('SELECT * FROM zero_index WHERE N > ? ORDER BY N ASC LIMIT 1', (current_N,))
                result = c.fetchone()
                if result is None:
                    break
                t0_start, N0, filename, offset, block_number = result
                continue

            with open(filepath, 'rb') as f:
                num_blocks = struct.unpack('Q', f.read(8))[0]
                f.seek(offset)

                while block_number < num_blocks and len(zeros) < num_zeros:
                    # Read block header
                    header = f.read(32)
                    if len(header) < 32:
                        break
                    t0, t1, Nt0, Nt1 = struct.unpack('ddQQ', header)

                    # Read zeros
                    Z = 0
                    for i in range(Nt1 - Nt0):
                        data = f.read(13)
                        if len(data) < 13:
                            break
                        z1, z2, z3 = struct.unpack('QIB', data)
                        Z = Z + (z3 << 96) + (z2 << 64) + z1

                        current_N = Nt0 + i + 1

                        if current_N >= start_n:
                            zero = t0 + Z * (2 ** -101)
                            zeros.append(zero)
                            progress.update(task, completed=len(zeros))

                            if len(zeros) >= num_zeros:
                                break

                    block_number += 1

            # Move to next file
            c.execute('SELECT * FROM zero_index WHERE N > ? ORDER BY N ASC LIMIT 1', (current_N,))
            result = c.fetchone()
            if result is None:
                break
            t0_start, N0, filename, offset, block_number = result

    conn.close()
    return np.array(zeros)


def read_zeros_simple(data_dir: Path, num_zeros: int = 1000000) -> np.ndarray:
    """Read zeros directly from .dat files without index (assumes files are in order)."""

    # Get all dat files sorted by name
    dat_files = sorted(data_dir.glob('zeros_*.dat'),
                       key=lambda p: int(p.stem.split('_')[1]))

    if not dat_files:
        console.print("[red]No .dat files found[/]")
        return np.array([])

    console.print(f"[cyan]Found {len(dat_files)} .dat files[/]")
    console.print(f"[cyan]Reading up to {num_zeros:,} zeros...[/]")

    zeros = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:,}/{task.total:,}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Reading zeros...", total=num_zeros)

        for filepath in dat_files:
            console.print(f"[dim]Reading {filepath.name}...[/]")

            with open(filepath, 'rb') as f:
                num_blocks = struct.unpack('Q', f.read(8))[0]

                for block_idx in range(num_blocks):
                    header = f.read(32)
                    if len(header) < 32:
                        break
                    t0, t1, Nt0, Nt1 = struct.unpack('ddQQ', header)

                    Z = 0
                    for i in range(Nt1 - Nt0):
                        data = f.read(13)
                        if len(data) < 13:
                            break
                        z1, z2, z3 = struct.unpack('QIB', data)
                        Z = Z + (z3 << 96) + (z2 << 64) + z1

                        zero = t0 + Z * (2 ** -101)
                        zeros.append(zero)
                        progress.update(task, completed=len(zeros))

                        if len(zeros) >= num_zeros:
                            return np.array(zeros)

    return np.array(zeros)


def main():
    parser = argparse.ArgumentParser(description="Read Platt binary zeta zeros")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory with .dat files")
    parser.add_argument("--index", type=Path, default=None, help="Path to index.db (optional)")
    parser.add_argument("--start", type=int, default=1, help="Starting zero index")
    parser.add_argument("--num", type=int, default=100_000_000, help="Number of zeros to read")
    parser.add_argument("--output", type=Path, default=Path("data/raw/zeros_100M.txt"), help="Output file")
    args = parser.parse_args()

    console.print("[bold magenta]Platt Binary Zeros Reader[/]")
    console.print()

    if args.index and args.index.exists():
        zeros = read_zeros_with_index(args.data_dir, args.index, args.start, args.num)
    else:
        zeros = read_zeros_simple(args.data_dir, args.num)

    console.print(f"\n[green]Read {len(zeros):,} zeros[/]")

    if len(zeros) > 0:
        console.print(f"  First: {zeros[0]:.10f}")
        console.print(f"  Last:  {zeros[-1]:.10f}")

        # Save to file
        console.print(f"\n[cyan]Saving to {args.output}...[/]")
        np.savetxt(args.output, zeros, fmt='%.15f')
        console.print(f"[green]Done![/]")


if __name__ == "__main__":
    main()
