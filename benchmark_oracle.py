#!/usr/bin/env python3
"""
🏆 ZETA BENCHMARK: Neural Oracle Validation

This script validates the trained model against known Odlyzko zeros.
It provides transparent, reproducible metrics that any skeptic can verify.

Key insight: We DON'T claim 99.9% accuracy on exact values.
We measure how well the model captures CORRELATIONS and STRUCTURE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from scipy.stats import spearmanr, pearsonr
from collections import Counter

from model.gpt import SpacingGPT

console = Console()

# ============================================================
# CONFIG
# ============================================================
CHECKPOINT = Path("out/best.pt")
VAL_DATA = Path("data/val.pt")
TEST_SIZE = 5000  # Number of predictions to make
CONTEXT_LEN = 256  # Must match training


def load_model_and_data():
    """Load trained model and validation data."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    console.print(f"[cyan]Loading model from {CHECKPOINT}...[/]")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SpacingGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    console.print(f"[cyan]Loading validation data from {VAL_DATA}...[/]")
    val_data = torch.load(VAL_DATA)

    return model, val_data, device, ckpt["config"]


@torch.no_grad()
def run_predictions(model, data, device, n_samples=TEST_SIZE):
    """
    Run model predictions and collect statistics.

    Data format: [n_sequences, seq_len] where each row is a sequence of token IDs.

    For each position in each sequence, we:
    1. Take context (all tokens before position)
    2. Get model's probability distribution over 256 bins
    3. Compare with actual next token
    """
    results = {
        "predictions": [],      # Top-1 predicted bin
        "actuals": [],          # Actual bin
        "top5_hits": 0,         # Count of top-5 hits
        "top10_hits": 0,        # Count of top-10 hits
        "exact_hits": 0,        # Count of exact matches
        "log_probs": [],        # Log probability of actual token
        "pred_probs": [],       # Probability assigned to actual token
    }

    n_seqs, seq_len = data.shape
    total_predictions = 0

    # Iterate over sequences
    for seq_idx in track(range(min(n_seqs, n_samples // (seq_len - 1) + 1)),
                         description="[green]Predicting..."):
        seq = data[seq_idx].unsqueeze(0).to(device)  # [1, seq_len]

        # Get all logits at once
        logits, _ = model(seq)  # [1, seq_len, vocab_size]

        # For positions 0..seq_len-2, predict position 1..seq_len-1
        for pos in range(seq_len - 1):
            if total_predictions >= n_samples:
                break

            pred_logits = logits[0, pos, :]  # Predict next token
            target = seq[0, pos + 1].item()  # Actual next token

            # Probabilities
            probs = torch.softmax(pred_logits, dim=-1)
            log_probs_t = torch.log_softmax(pred_logits, dim=-1)

            # Top-k predictions
            topk_probs, topk_indices = torch.topk(probs, k=10)
            top1_pred = topk_indices[0].item()

            # Collect results
            results["predictions"].append(top1_pred)
            results["actuals"].append(target)
            results["log_probs"].append(log_probs_t[target].item())
            results["pred_probs"].append(probs[target].item())

            # Check hits
            if top1_pred == target:
                results["exact_hits"] += 1
            if target in topk_indices[:5].tolist():
                results["top5_hits"] += 1
            if target in topk_indices[:10].tolist():
                results["top10_hits"] += 1

            total_predictions += 1

        if total_predictions >= n_samples:
            break

    results["n_samples"] = total_predictions
    return results


def compute_metrics(results):
    """Compute comprehensive metrics from prediction results."""
    n = results["n_samples"]

    metrics = {
        "n_samples": n,
        "top1_accuracy": results["exact_hits"] / n * 100,
        "top5_accuracy": results["top5_hits"] / n * 100,
        "top10_accuracy": results["top10_hits"] / n * 100,
        "mean_log_prob": np.mean(results["log_probs"]),
        "perplexity": np.exp(-np.mean(results["log_probs"])),
        "mean_prob": np.mean(results["pred_probs"]),
    }

    # Correlation between predicted and actual
    preds = np.array(results["predictions"])
    acts = np.array(results["actuals"])

    metrics["spearman_corr"], metrics["spearman_p"] = spearmanr(preds, acts)
    metrics["pearson_corr"], metrics["pearson_p"] = pearsonr(preds, acts)

    # Baseline comparison
    # Random baseline: 1/256 = 0.39% for top-1
    metrics["random_top1"] = 100 / 256
    metrics["random_top5"] = 500 / 256
    metrics["random_ppl"] = 256.0

    # Improvement factors
    metrics["top1_vs_random"] = metrics["top1_accuracy"] / metrics["random_top1"]
    metrics["ppl_improvement"] = (1 - metrics["perplexity"] / metrics["random_ppl"]) * 100

    return metrics


def create_visualization(results, metrics, output_path="benchmark_proof.png"):
    """Create publication-quality visualization of results."""

    preds = np.array(results["predictions"])
    acts = np.array(results["actuals"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: Predicted vs Actual (first 200 points) ---
    ax = axes[0, 0]
    n_show = min(200, len(preds))
    ax.plot(acts[:n_show], 'b-', alpha=0.7, linewidth=1.5, label='Actual')
    ax.plot(preds[:n_show], 'r--', alpha=0.7, linewidth=1.5, label='Predicted')
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Bin Index (0-255)')
    ax.set_title(f'(A) Predictions vs Reality (first {n_show} tokens)')
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Panel B: Accuracy Bars ---
    ax = axes[0, 1]
    categories = ['Top-1', 'Top-5', 'Top-10']
    model_acc = [metrics["top1_accuracy"], metrics["top5_accuracy"], metrics["top10_accuracy"]]
    random_acc = [metrics["random_top1"], metrics["random_top5"], 10 * 100 / 256]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, model_acc, width, label='Model', color='green', alpha=0.8)
    bars2 = ax.bar(x + width/2, random_acc, width, label='Random', color='dimgray', alpha=0.8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(B) Model vs Random Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars1, model_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # --- Panel C: Bin Distribution ---
    ax = axes[1, 0]
    ax.hist(acts, bins=50, alpha=0.6, label='Actual', color='blue', density=True)
    ax.hist(preds, bins=50, alpha=0.6, label='Predicted', color='red', density=True)
    ax.set_xlabel('Bin Index')
    ax.set_ylabel('Density')
    ax.set_title('(C) Distribution: Predicted vs Actual Bins')
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Panel D: Summary Stats ---
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    ═══════════════════════════════════════
           🏆 ZETA BENCHMARK RESULTS 🏆
    ═══════════════════════════════════════

    Samples Tested:     {metrics['n_samples']:,}

    ACCURACY:
      Top-1:            {metrics['top1_accuracy']:.2f}%
      Top-5:            {metrics['top5_accuracy']:.2f}%
      Top-10:           {metrics['top10_accuracy']:.2f}%

    PERPLEXITY:
      Model:            {metrics['perplexity']:.2f}
      Random Baseline:  {metrics['random_ppl']:.1f}
      Improvement:      {metrics['ppl_improvement']:.1f}%

    CORRELATION:
      Spearman ρ:       {metrics['spearman_corr']:.4f}
      Pearson r:        {metrics['pearson_corr']:.4f}

    VERDICT:
      {metrics['top1_vs_random']:.1f}× better than random!
    ═══════════════════════════════════════
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Neural Oracle Benchmark: Zeta Zero Prediction',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved {output_path}[/]")


def print_results_table(metrics):
    """Print rich formatted results table."""

    console.print("\n")
    table = Table(title="🏆 ZETA BENCHMARK RESULTS",
                  title_style="bold magenta")

    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Model", style="green", justify="right")
    table.add_column("Random", style="dim", justify="right")
    table.add_column("Factor", style="yellow", justify="right")

    table.add_row(
        "Top-1 Accuracy",
        f"{metrics['top1_accuracy']:.2f}%",
        f"{metrics['random_top1']:.2f}%",
        f"{metrics['top1_vs_random']:.1f}×"
    )
    table.add_row(
        "Top-5 Accuracy",
        f"{metrics['top5_accuracy']:.2f}%",
        f"{metrics['random_top5']:.2f}%",
        f"{metrics['top5_accuracy'] / metrics['random_top5']:.1f}×"
    )
    table.add_row(
        "Top-10 Accuracy",
        f"{metrics['top10_accuracy']:.2f}%",
        f"{10 * 100 / 256:.2f}%",
        f"{metrics['top10_accuracy'] / (10 * 100 / 256):.1f}×"
    )
    table.add_row("─" * 20, "─" * 10, "─" * 10, "─" * 10)
    table.add_row(
        "Perplexity",
        f"{metrics['perplexity']:.2f}",
        f"{metrics['random_ppl']:.1f}",
        f"{metrics['ppl_improvement']:.1f}% better"
    )
    table.add_row(
        "Spearman Correlation",
        f"{metrics['spearman_corr']:.4f}",
        "0.0000",
        f"p={metrics['spearman_p']:.2e}"
    )

    console.print(table)

    # Verdict
    console.print("\n" + "=" * 60)
    if metrics['ppl_improvement'] > 20:
        console.print("[bold green]✅ VERDICT: Model captures significant structure![/]")
        console.print(f"   The model is {metrics['top1_vs_random']:.1f}× better than random guessing.")
        console.print(f"   Perplexity {metrics['ppl_improvement']:.1f}% below entropy floor.")
    elif metrics['ppl_improvement'] > 10:
        console.print("[bold yellow]🟡 VERDICT: Model shows some structure learning.[/]")
    else:
        console.print("[bold red]❌ VERDICT: Model near random baseline.[/]")
    console.print("=" * 60)


def main():
    console.print("[bold magenta]═══ 🏆 ZETA BENCHMARK: Neural Oracle Validation ═══[/]\n")

    # Load
    model, val_data, device, config = load_model_and_data()
    console.print(f"[green]Model: {config.n_layer}L × {config.n_head}H × {config.n_embd}D[/]")
    console.print(f"[green]Data: {len(val_data):,} tokens[/]\n")

    # Run predictions
    results = run_predictions(model, val_data, device, n_samples=TEST_SIZE)

    # Compute metrics
    metrics = compute_metrics(results)

    # Display results
    print_results_table(metrics)

    # Create visualization
    create_visualization(results, metrics)

    console.print("\n[bold green]═══ BENCHMARK COMPLETE ═══[/]")
    console.print("Run this on your own machine to verify!")


if __name__ == "__main__":
    main()
