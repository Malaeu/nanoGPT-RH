#!/usr/bin/env python3
"""
🔮 INVERSE UNFOLDING: Predict Actual Zeta Zeros

This script implements the key missing piece:
bin_prediction → spacing → inverse_unfold → actual γ value

Newton-Raphson for: u(γ_{n+1}) - u(γ_n) = s_pred
Where: u(γ) = (γ/2π) × log(γ/(2πe))
"""

import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
import time

from model.gpt import SpacingGPT, GPTConfig

console = Console()

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("data")
CKPT_PATH = Path("out/best.pt")
ZEROS_PATH = Path("zeros/zeros2M.txt")
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# INVERSE UNFOLDING LOGIC
# ============================================================
def unfold_val(gamma):
    """
    Direct unfolding: u(γ) = (γ/2π) × log(γ/(2πe))

    This is the smooth counting function for zeta zeros.
    """
    return (gamma / (2 * np.pi)) * np.log(gamma / (2 * np.pi * np.e))


def inverse_unfold(gamma_prev, spacing_pred, n_iter=10):
    """
    Find γ_next such that: u(γ_next) - u(γ_prev) = spacing_pred

    Uses Newton-Raphson method.

    Derivative: u'(γ) = (1/2π) × log(γ/2π)
    """
    u_prev = unfold_val(gamma_prev)
    u_target = u_prev + spacing_pred

    def u_prime(t):
        """Derivative of unfolding function."""
        return (1.0 / (2 * np.pi)) * np.log(t / (2 * np.pi))

    # Initial guess: linear approximation
    # spacing ≈ u'(γ_prev) × Δγ  =>  Δγ ≈ spacing / u'(γ_prev)
    deriv_prev = u_prime(gamma_prev)
    gamma_curr = gamma_prev + spacing_pred / deriv_prev

    # Newton-Raphson refinement
    for _ in range(n_iter):
        u_curr = unfold_val(gamma_curr)
        error = u_curr - u_target
        if abs(error) < 1e-12:
            break
        deriv = u_prime(gamma_curr)
        gamma_curr = gamma_curr - error / deriv

    return gamma_curr


# ============================================================
# DATA UTILITIES
# ============================================================
def load_meta():
    """Load bin edges used during training."""
    meta_path = DATA_DIR / "meta.pt"
    if meta_path.exists():
        return torch.load(meta_path, weights_only=False)
    else:
        console.print("[red]meta.pt not found! Using linear bins fallback (DANGEROUS).[/]")
        return {"bin_edges": np.linspace(0, 4, 257)}


def bin_to_spacing(bin_idx, bin_edges):
    """Convert bin index to continuous spacing value (using bin center)."""
    idx = max(0, min(bin_idx, len(bin_edges) - 2))
    left = bin_edges[idx]
    right = bin_edges[idx + 1]
    return (left + right) / 2.0


def load_model():
    """Load trained SpacingGPT model."""
    console.print(f"[cyan]Loading model from {CKPT_PATH}...[/]")

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]

    model = SpacingGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    console.print(f"[green]Model loaded: {sum(p.numel() for p in model.parameters()):,} params[/]")
    return model, config


def load_zeros():
    """Load ground truth zeros."""
    console.print(f"[cyan]Loading zeros from {ZEROS_PATH}...[/]")

    zeros = np.loadtxt(ZEROS_PATH)
    console.print(f"[green]Loaded {len(zeros):,} zeros[/]")
    console.print(f"[dim]Range: {zeros[0]:.4f} to {zeros[-1]:.4f}[/]")

    return zeros


# ============================================================
# MAIN PREDICTION BENCHMARK
# ============================================================
def run_prediction_benchmark(start_idx=1500000, num_tests=1000):
    """
    Test inverse unfolding on validation region.

    For each position:
    1. Take context of 256 spacings
    2. Predict next bin
    3. Convert bin → spacing → γ via inverse unfold
    4. Compare with ground truth γ
    """
    console.print("\n[bold magenta]═══ 🔮 INVERSE UNFOLDING BENCHMARK ═══[/]\n")
    console.print(f"[cyan]Device: {DEVICE}[/]")

    # Load everything
    model, config = load_model()
    zeros = load_zeros()
    meta = load_meta()
    bin_edges = np.array(meta["bin_edges"])

    # Compute unfolded spacings
    console.print("[cyan]Computing unfolded spacings...[/]")
    unfolded = unfold_val(zeros)
    gaps = np.diff(unfolded)

    console.print(f"[green]Spacing stats: mean={gaps.mean():.4f}, std={gaps.std():.4f}[/]")

    # Validation region
    context_len = config.seq_len
    console.print(f"\n[bold]Testing {num_tests} predictions starting at index {start_idx}[/]")
    console.print(f"[dim]Context length: {context_len}[/]\n")

    # Metrics
    errors = []
    rel_errors = []
    bin_correct = 0
    top5_correct = 0

    start_time = time.time()

    with torch.no_grad():
        for i in range(num_tests):
            # Context window of spacings
            ctx_gaps = gaps[start_idx + i : start_idx + i + context_len]

            # Digitize context to bins
            ctx_bins = np.digitize(ctx_gaps, bin_edges) - 1
            ctx_bins = np.clip(ctx_bins, 0, config.vocab_size - 1)

            x = torch.tensor(ctx_bins, dtype=torch.long).unsqueeze(0).to(DEVICE)

            # Predict
            logits, _ = model(x)
            next_token_logits = logits[0, -1, :]

            # Argmax prediction
            pred_bin = torch.argmax(next_token_logits).item()

            # Top-5 for accuracy
            top5_bins = torch.topk(next_token_logits, 5).indices.tolist()

            # Ground truth bin
            true_gap = gaps[start_idx + i + context_len]
            true_bin = np.digitize([true_gap], bin_edges)[0] - 1
            true_bin = np.clip(true_bin, 0, config.vocab_size - 1)

            # Bin accuracy
            if pred_bin == true_bin:
                bin_correct += 1
            if true_bin in top5_bins:
                top5_correct += 1

            # Convert to spacing
            s_pred = bin_to_spacing(pred_bin, bin_edges)

            # Inverse unfold to get γ
            gamma_prev = zeros[start_idx + i + context_len]
            gamma_true = zeros[start_idx + i + context_len + 1]

            gamma_pred = inverse_unfold(gamma_prev, s_pred)

            # Metrics
            abs_err = abs(gamma_pred - gamma_true)
            local_gap = gamma_true - gamma_prev
            rel_err = abs_err / local_gap if local_gap > 0 else 0

            errors.append(abs_err)
            rel_errors.append(rel_err)

            if i % 200 == 0:
                console.print(
                    f"  [{i:4d}/{num_tests}] "
                    f"γ_true={gamma_true:.4f} γ_pred={gamma_pred:.4f} "
                    f"err={abs_err:.4f} ({rel_err*100:.1f}%)"
                )

    elapsed = time.time() - start_time

    # Final statistics
    mae = np.mean(errors)
    mre = np.mean(rel_errors)
    std_err = np.std(errors)
    bin_acc = bin_correct / num_tests * 100
    top5_acc = top5_correct / num_tests * 100

    # Results table
    console.print("\n")
    table = Table(title="🏆 PREDICTION RESULTS", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Notes", style="dim")

    table.add_row("MAE (γ error)", f"{mae:.6f}", "Absolute gamma error")
    table.add_row("Std (γ error)", f"{std_err:.6f}", "Standard deviation")
    table.add_row("MRE", f"{mre:.4f} ({mre*100:.2f}%)", "Relative to gap size")
    table.add_row("Bin Accuracy", f"{bin_acc:.2f}%", f"vs random {100/256:.2f}%")
    table.add_row("Top-5 Accuracy", f"{top5_acc:.2f}%", f"vs random {5*100/256:.2f}%")
    table.add_row("Time", f"{elapsed:.1f}s", f"{num_tests/elapsed:.0f} pred/s")

    console.print(table)

    # Verdict
    random_bin_acc = 100 / 256
    if bin_acc > random_bin_acc * 5:
        console.print("\n[bold green]✅ SUCCESS: Model significantly beats random![/]")
    elif bin_acc > random_bin_acc * 2:
        console.print("\n[yellow]⚠️ MODERATE: Model beats random but could be better[/]")
    else:
        console.print("\n[red]❌ WARNING: Model barely beats random[/]")

    if mre < 0.3:
        console.print("[bold green]✅ Relative error < 30% - Good precision![/]")
    elif mre < 0.5:
        console.print("[yellow]⚠️ Relative error 30-50% - Moderate precision[/]")
    else:
        console.print("[red]❌ Relative error > 50% - Poor precision[/]")

    return {
        "mae": mae,
        "mre": mre,
        "bin_acc": bin_acc,
        "top5_acc": top5_acc,
        "errors": errors,
        "rel_errors": rel_errors,
    }


if __name__ == "__main__":
    results = run_prediction_benchmark()
    console.print("\n[bold green]═══ COMPLETE ═══[/]")
