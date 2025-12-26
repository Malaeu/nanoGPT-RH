#!/usr/bin/env python3
"""
Prepare prime number dataset for training.

Generates prime gaps and bins them similar to zeta zeros.
"""

import numpy as np
import torch
from pathlib import Path
from rich.console import Console
from rich.progress import track

console = Console()

# Config
N_PRIMES = 2_000_000  # Generate 2M primes (comparable to zeta zeros)
N_BINS = 256          # Same binning as zeta
SEQ_LEN = 256         # Sequence length
OUTPUT_DIR = Path(".")


def sieve_of_eratosthenes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    return np.where(is_prime)[0]


def generate_primes(n_primes):
    """Generate first n_primes prime numbers."""
    console.print(f"[cyan]Generating {n_primes:,} primes...[/]")

    # Estimate upper bound using prime number theorem: p_n ~ n * ln(n)
    if n_primes < 6:
        limit = 15
    else:
        limit = int(n_primes * (np.log(n_primes) + np.log(np.log(n_primes)))) + 100

    primes = sieve_of_eratosthenes(limit)

    while len(primes) < n_primes:
        limit = int(limit * 1.5)
        primes = sieve_of_eratosthenes(limit)

    return primes[:n_primes]


def compute_gaps(primes):
    """Compute gaps between consecutive primes."""
    return np.diff(primes)


def bin_gaps(gaps, n_bins=N_BINS):
    """
    Bin gaps into discrete tokens.

    Prime gaps have different distribution than zeta:
    - Most gaps are small (2, 4, 6)
    - Occasional large gaps
    - All gaps are even (except first gap = 1)
    """
    # Use logarithmic binning for better resolution
    # Gap 2 -> bin 0-10
    # Gap 4 -> bin 10-20
    # etc.

    max_gap = np.max(gaps)
    console.print(f"[cyan]Gap statistics: min={np.min(gaps)}, max={max_gap}, mean={np.mean(gaps):.2f}[/]")

    # Linear binning with clipping
    # Scale so that median gap maps to middle bin
    median_gap = np.median(gaps)
    scale = (n_bins // 2) / median_gap

    binned = np.clip((gaps * scale).astype(int), 0, n_bins - 1)

    return binned


def create_sequences(binned_gaps, seq_len=SEQ_LEN):
    """Create training sequences from binned gaps."""
    n_seqs = len(binned_gaps) // seq_len

    # Reshape into sequences
    data = binned_gaps[:n_seqs * seq_len].reshape(n_seqs, seq_len)

    return data


def main():
    console.print("[bold magenta]═══ PRIME DATA PREPARATION ═══[/]\n")

    # 1. Generate primes
    primes = generate_primes(N_PRIMES)
    console.print(f"[green]✓ Generated {len(primes):,} primes[/]")
    console.print(f"  First 10: {primes[:10]}")
    console.print(f"  Last prime: {primes[-1]:,}")

    # 2. Compute gaps
    gaps = compute_gaps(primes)
    console.print(f"\n[green]✓ Computed {len(gaps):,} gaps[/]")

    # 3. Bin gaps
    binned = bin_gaps(gaps, N_BINS)
    console.print(f"[green]✓ Binned gaps into {N_BINS} bins[/]")

    # Check distribution
    unique, counts = np.unique(binned, return_counts=True)
    console.print(f"  Unique bins used: {len(unique)}")
    console.print(f"  Most common bins: {unique[np.argsort(-counts)[:5]]}")

    # 4. Create sequences
    data = create_sequences(binned)
    console.print(f"\n[green]✓ Created {len(data):,} sequences of length {SEQ_LEN}[/]")

    # 5. Train/val split (90/10)
    n_train = int(len(data) * 0.9)
    train_data = data[:n_train]
    val_data = data[n_train:]

    console.print(f"  Train: {len(train_data):,} sequences")
    console.print(f"  Val:   {len(val_data):,} sequences")

    # 6. Save
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    val_tensor = torch.tensor(val_data, dtype=torch.long)

    torch.save(train_tensor, OUTPUT_DIR / "train_primes.pt")
    torch.save(val_tensor, OUTPUT_DIR / "val_primes.pt")

    # Also save raw data for analysis
    np.save(OUTPUT_DIR / "primes.npy", primes)
    np.save(OUTPUT_DIR / "prime_gaps.npy", gaps)

    console.print(f"\n[green]✓ Saved:[/]")
    console.print(f"  train_primes.pt ({train_tensor.shape})")
    console.print(f"  val_primes.pt ({val_tensor.shape})")
    console.print(f"  primes.npy, prime_gaps.npy")

    console.print("\n[bold green]═══ DONE ═══[/]")


if __name__ == "__main__":
    main()
