#!/usr/bin/env python3
"""
🔬 Q3 Beam Filter: Neural + Theory Hybrid

Uses GUE level repulsion as a physics-based filter on neural predictions.

Key insight: GUE says P(s→0) → 0 (level repulsion).
Very small spacings are forbidden. We use this to filter candidates.

This is NOT magic - it's applying known physics to constrain neural output.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from scipy.stats import spearmanr

from model.gpt import SpacingGPT

console = Console()

# ============================================================
# CONFIG
# ============================================================
CHECKPOINT = Path("out/best.pt")
VAL_DATA = Path("data/val.pt")
TEST_SIZE = 2000
N_BINS = 256

# Physics-based thresholds (from GUE)
# Wigner surmise: P(s) = (π/2)s exp(-πs²/4)
# P(s < 0.3) ≈ 3.5% (rare but possible)
# P(s < 0.1) ≈ 0.4% (very rare)
MIN_SPACING_BIN = 10   # ~0.1 in normalized units (bin 10/256 ≈ 0.04)
SOFT_MIN_BIN = 30      # ~0.3 in normalized units


def load_model_and_data():
    """Load trained model and validation data."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SpacingGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_data = torch.load(VAL_DATA)

    return model, val_data, device


def wigner_surmise_weight(bin_idx, n_bins=256):
    """
    Weight based on Wigner surmise P(s) = (π/2)s exp(-πs²/4).

    Low bins (small spacings) get lower weight due to level repulsion.
    """
    # Convert bin to approximate spacing (assuming bins span 0-4 range)
    s = (bin_idx / n_bins) * 4.0

    if s < 0.01:
        return 0.0  # Essentially forbidden

    # Wigner surmise
    p = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
    return p


def apply_q3_filter(probs, top_k=10, method="soft"):
    """
    Apply physics-based filter to probability distribution.

    Methods:
    - "hard": Zero out bins below MIN_SPACING_BIN
    - "soft": Multiply by Wigner surmise weights
    - "hybrid": Hard cutoff + soft weighting
    """
    filtered = probs.clone()

    if method == "hard":
        # Hard cutoff: zero out very small spacings
        filtered[:MIN_SPACING_BIN] = 0

    elif method == "soft":
        # Soft weighting: multiply by Wigner surmise
        for i in range(len(filtered)):
            filtered[i] *= wigner_surmise_weight(i)

    elif method == "hybrid":
        # Hard cutoff + soft weighting
        filtered[:MIN_SPACING_BIN] = 0
        for i in range(MIN_SPACING_BIN, SOFT_MIN_BIN):
            # Gradual penalty in transition zone
            penalty = (i - MIN_SPACING_BIN) / (SOFT_MIN_BIN - MIN_SPACING_BIN)
            filtered[i] *= penalty

    # Renormalize
    if filtered.sum() > 0:
        filtered = filtered / filtered.sum()
    else:
        filtered = probs  # Fallback to original if all filtered out

    return filtered


@torch.no_grad()
def run_comparison(model, data, device, n_samples=TEST_SIZE):
    """
    Compare predictions with and without Q3 filter.
    """
    results = {
        "no_filter": {"correct": 0, "top5": 0, "top10": 0, "preds": [], "acts": []},
        "hard": {"correct": 0, "top5": 0, "top10": 0, "preds": [], "acts": []},
        "soft": {"correct": 0, "top5": 0, "top10": 0, "preds": [], "acts": []},
        "hybrid": {"correct": 0, "top5": 0, "top10": 0, "preds": [], "acts": []},
    }

    n_seqs, seq_len = data.shape
    total = 0

    for seq_idx in track(range(min(n_seqs, n_samples // (seq_len - 1) + 1)),
                         description="[green]Comparing methods..."):
        seq = data[seq_idx].unsqueeze(0).to(device)
        logits, _ = model(seq)

        for pos in range(seq_len - 1):
            if total >= n_samples:
                break

            pred_logits = logits[0, pos, :]
            target = seq[0, pos + 1].item()

            # Original probabilities
            probs = torch.softmax(pred_logits, dim=-1).cpu()

            # Test each method
            for method in results.keys():
                if method == "no_filter":
                    filtered_probs = probs
                else:
                    filtered_probs = apply_q3_filter(probs, method=method)

                # Get top predictions
                topk = torch.topk(filtered_probs, k=10)
                top1 = topk.indices[0].item()
                top5 = topk.indices[:5].tolist()
                top10 = topk.indices[:10].tolist()

                # Record results
                results[method]["preds"].append(top1)
                results[method]["acts"].append(target)

                if top1 == target:
                    results[method]["correct"] += 1
                if target in top5:
                    results[method]["top5"] += 1
                if target in top10:
                    results[method]["top10"] += 1

            total += 1

        if total >= n_samples:
            break

    # Calculate metrics
    for method in results:
        n = total
        results[method]["top1_acc"] = results[method]["correct"] / n * 100
        results[method]["top5_acc"] = results[method]["top5"] / n * 100
        results[method]["top10_acc"] = results[method]["top10"] / n * 100

        preds = np.array(results[method]["preds"])
        acts = np.array(results[method]["acts"])
        results[method]["spearman"], _ = spearmanr(preds, acts)
        results[method]["n_samples"] = n

    return results


def print_comparison_table(results):
    """Print comparison table."""
    console.print("\n")
    table = Table(title="🔬 Q3 BEAM FILTER COMPARISON",
                  title_style="bold magenta")

    table.add_column("Method", style="cyan")
    table.add_column("Top-1 Acc", justify="right")
    table.add_column("Top-5 Acc", justify="right")
    table.add_column("Top-10 Acc", justify="right")
    table.add_column("Spearman ρ", justify="right")
    table.add_column("Δ vs Base", justify="right", style="yellow")

    base_top1 = results["no_filter"]["top1_acc"]

    for method, data in results.items():
        delta = data["top1_acc"] - base_top1
        delta_str = f"{delta:+.2f}%" if method != "no_filter" else "—"

        table.add_row(
            method.replace("_", " ").title(),
            f"{data['top1_acc']:.2f}%",
            f"{data['top5_acc']:.2f}%",
            f"{data['top10_acc']:.2f}%",
            f"{data['spearman']:.4f}",
            delta_str
        )

    console.print(table)


def create_visualization(results, output_path="q3_filter_comparison.png"):
    """Create comparison visualization."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = list(results.keys())
    colors = ['blue', 'red', 'green', 'orange']

    # Panel A: Top-K Accuracy comparison
    ax = axes[0]
    x = np.arange(3)
    width = 0.2

    for i, method in enumerate(methods):
        accs = [results[method]["top1_acc"],
                results[method]["top5_acc"],
                results[method]["top10_acc"]]
        ax.bar(x + i * width, accs, width, label=method.replace("_", " ").title(),
               color=colors[i], alpha=0.8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(A) Accuracy by Method')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Top-1', 'Top-5', 'Top-10'])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # Panel B: Spearman correlation
    ax = axes[1]
    spearman_vals = [results[m]["spearman"] for m in methods]
    bars = ax.bar(methods, spearman_vals, color=colors, alpha=0.8)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('(B) Correlation by Method')
    ax.set_xticklabels([m.replace("_", " ").title() for m in methods], rotation=15)
    ax.grid(alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, spearman_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel C: Improvement delta
    ax = axes[2]
    base = results["no_filter"]["top1_acc"]
    deltas = [results[m]["top1_acc"] - base for m in methods]
    colors_delta = ['gray' if d <= 0 else 'green' for d in deltas]
    bars = ax.bar(methods, deltas, color=colors_delta, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Δ Top-1 Accuracy (%)')
    ax.set_title('(C) Improvement vs Baseline')
    ax.set_xticklabels([m.replace("_", " ").title() for m in methods], rotation=15)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Q3 Physics Filter: Neural + Theory Hybrid',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved {output_path}[/]")


def main():
    console.print("[bold magenta]═══ 🔬 Q3 BEAM FILTER EXPERIMENT ═══[/]\n")

    console.print("[cyan]Loading model and data...[/]")
    model, val_data, device = load_model_and_data()
    console.print(f"[green]Data shape: {val_data.shape}[/]\n")

    console.print("[bold]Running comparison of filtering methods...[/]\n")
    results = run_comparison(model, val_data, device, n_samples=TEST_SIZE)

    print_comparison_table(results)
    create_visualization(results)

    # Verdict
    console.print("\n" + "=" * 60)
    best_method = max(results.keys(), key=lambda m: results[m]["top1_acc"])
    best_acc = results[best_method]["top1_acc"]
    base_acc = results["no_filter"]["top1_acc"]
    improvement = best_acc - base_acc

    if improvement > 0.5:
        console.print(f"[bold green]✅ Q3 FILTER HELPS![/]")
        console.print(f"   Best method: {best_method}")
        console.print(f"   Improvement: +{improvement:.2f}% accuracy")
    elif improvement > 0:
        console.print(f"[bold yellow]🟡 MARGINAL IMPROVEMENT[/]")
        console.print(f"   Best method: {best_method}")
        console.print(f"   Improvement: +{improvement:.2f}%")
    else:
        console.print(f"[bold red]❌ NO IMPROVEMENT FROM Q3 FILTER[/]")
        console.print(f"   Neural predictions already respect physics!")

    console.print("=" * 60)
    console.print("\n[bold green]═══ EXPERIMENT COMPLETE ═══[/]")


if __name__ == "__main__":
    main()
