#!/usr/bin/env python3
"""
Create shuffled baseline datasets.
Used to verify that model cannot beat entropy lower bound without temporal structure.
"""

import argparse
from pathlib import Path
import torch
import numpy as np
from rich.console import Console

console = Console()


def shuffle_per_sequence(data: torch.Tensor) -> torch.Tensor:
    """
    Variant A: Shuffle tokens within each sequence independently.
    Preserves per-sequence histogram but destroys temporal order.
    """
    shuffled = data.clone()
    for i in range(len(shuffled)):
        perm = torch.randperm(shuffled.shape[1])
        shuffled[i] = shuffled[i, perm]
    return shuffled


def shuffle_global_pool(data: torch.Tensor) -> torch.Tensor:
    """
    Variant B: Flatten all tokens, shuffle globally, reshape.
    Completely destroys all structure, preserves only global histogram.
    """
    flat = data.flatten()
    perm = torch.randperm(len(flat))
    shuffled = flat[perm].reshape(data.shape)
    return shuffled


def compute_entropy(data: torch.Tensor, n_bins: int = 256) -> float:
    """Compute empirical entropy of token distribution."""
    flat = data.flatten().numpy()
    counts = np.bincount(flat, minlength=n_bins)
    probs = counts / len(flat)
    probs = probs[probs > 0]  # avoid log(0)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def main():
    parser = argparse.ArgumentParser(description="Create shuffled baseline data")
    parser.add_argument("--input-dir", type=Path, default=Path("data"), help="Input directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data_shuffled"), help="Output directory")
    parser.add_argument("--variant", choices=["a", "b", "both"], default="both", help="Shuffle variant")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    console.print("[bold magenta]═══ Creating Shuffled Baseline ═══[/]\n")

    # Load original data
    train = torch.load(args.input_dir / "train.pt")
    val = torch.load(args.input_dir / "val.pt")
    meta = torch.load(args.input_dir / "meta.pt")

    console.print(f"[cyan]Original data:[/]")
    console.print(f"  Train: {train.shape}")
    console.print(f"  Val: {val.shape}")

    # Compute original entropy
    orig_entropy = compute_entropy(train, meta["n_bins"])
    console.print(f"  Entropy: {orig_entropy:.2f} bits")
    console.print(f"  Perplexity lower bound: {2**orig_entropy:.1f}")

    variants = []
    if args.variant in ["a", "both"]:
        variants.append(("a", "per-sequence", shuffle_per_sequence))
    if args.variant in ["b", "both"]:
        variants.append(("b", "global-pool", shuffle_global_pool))

    for var_id, var_name, shuffle_fn in variants:
        console.print(f"\n[bold]Variant {var_id.upper()}: {var_name}[/]")

        out_dir = args.output_dir / f"variant_{var_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle
        train_shuffled = shuffle_fn(train)
        val_shuffled = shuffle_fn(val)

        # Verify entropy preserved
        new_entropy = compute_entropy(train_shuffled, meta["n_bins"])
        console.print(f"  Entropy after shuffle: {new_entropy:.2f} bits")
        console.print(f"  Perplexity lower bound: {2**new_entropy:.1f}")

        if abs(new_entropy - orig_entropy) > 0.01:
            console.print("[yellow]  ⚠ Warning: entropy changed![/]")
        else:
            console.print("[green]  ✓ Entropy preserved[/]")

        # Save
        torch.save(train_shuffled, out_dir / "train.pt")
        torch.save(val_shuffled, out_dir / "val.pt")
        torch.save(meta, out_dir / "meta.pt")

        console.print(f"  Saved to {out_dir}")

    console.print("\n[bold green]✓ Done![/]")
    console.print("\n[cyan]To train on shuffled data:[/]")
    console.print(f"  python train.py --data-dir {args.output_dir}/variant_a --out-dir out_shuffled_a")
    console.print(f"  python train.py --data-dir {args.output_dir}/variant_b --out-dir out_shuffled_b")


if __name__ == "__main__":
    main()
