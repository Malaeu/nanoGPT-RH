#!/usr/bin/env python3
"""
🔮 TRAJECTORY GENERATION: Autoregressive Zero Prediction

The "Money Shot" - generates future zeros using only model predictions.
Shows if the model can maintain trajectory or drifts into chaos.

Key test: Does the cumulative error grow linearly (stable) or exponentially (unstable)?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
import argparse

from model.gpt import SpacingGPT, GPTConfig
from predict_zeros import inverse_unfold, bin_to_spacing, unfold_val

console = Console()

# ============================================================
# CONFIG
# ============================================================
CKPT_PATH = Path("out/best.pt")
ZEROS_PATH = Path("zeros/zeros2M.txt")
DATA_DIR = Path("data")
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load trained SpacingGPT model."""
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]

    model = SpacingGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    return model, config


def load_meta():
    """Load bin edges."""
    meta_path = DATA_DIR / "meta.pt"
    return torch.load(meta_path, weights_only=False)


def generate_trajectory(start_idx=1800000, n_steps=50, sampling="greedy"):
    """
    Generate trajectory of zeros starting from start_idx.

    Args:
        start_idx: Starting position in zeros array
        n_steps: Number of zeros to generate
        sampling: "greedy" (argmax) or "sample" (from distribution)

    Returns:
        dict with generated and true zeros
    """
    console.print(f"\n[bold magenta]═══ 🔮 TRAJECTORY GENERATION ═══[/]\n")
    console.print(f"[cyan]Device: {DEVICE}[/]")
    console.print(f"[cyan]Sampling: {sampling}[/]")

    # Load everything
    model, config = load_model()
    console.print(f"[green]Model loaded: seq_len={config.seq_len}[/]")

    zeros = np.loadtxt(ZEROS_PATH)
    console.print(f"[green]Zeros loaded: {len(zeros):,}[/]")

    meta = load_meta()
    bin_edges = np.array(meta["bin_edges"])

    # Compute all spacings
    unfolded = unfold_val(zeros)
    all_gaps = np.diff(unfolded)

    # Initialize context
    context_len = config.seq_len
    curr_gaps = list(all_gaps[start_idx : start_idx + context_len])
    curr_gamma = zeros[start_idx + context_len]

    # Ground truth
    true_gammas = zeros[start_idx + context_len + 1 : start_idx + context_len + 1 + n_steps]

    console.print(f"\n[bold]Starting from zero #{start_idx + context_len}[/]")
    console.print(f"[dim]γ_start = {curr_gamma:.6f}[/]")
    console.print(f"[dim]Generating {n_steps} future zeros...[/]\n")

    generated_gammas = []
    predicted_spacings = []

    with torch.no_grad():
        for i in range(n_steps):
            # Prepare input
            ctx = curr_gaps[-context_len:]
            ctx_bins = np.digitize(ctx, bin_edges) - 1
            ctx_bins = np.clip(ctx_bins, 0, config.vocab_size - 1)

            x = torch.tensor(ctx_bins, dtype=torch.long).unsqueeze(0).to(DEVICE)

            # Predict
            logits, _ = model(x)
            probs = torch.softmax(logits[0, -1, :], dim=0)

            if sampling == "greedy":
                pred_bin = torch.argmax(probs).item()
            else:  # sample
                pred_bin = torch.multinomial(probs, 1).item()

            # Convert to spacing and gamma
            s_pred = bin_to_spacing(pred_bin, bin_edges)
            next_gamma = inverse_unfold(curr_gamma, s_pred)

            generated_gammas.append(next_gamma)
            predicted_spacings.append(s_pred)

            # Update state for autoregression
            curr_gaps.append(s_pred)
            curr_gamma = next_gamma

            if i % 10 == 0:
                true_g = true_gammas[i] if i < len(true_gammas) else None
                err = abs(next_gamma - true_g) if true_g else None
                err_str = f"err={err:.4f}" if err else ""
                console.print(f"  Step {i:3d}: γ_pred={next_gamma:.4f} {err_str}")

    # Compute errors
    generated_gammas = np.array(generated_gammas)
    diffs = generated_gammas - true_gammas
    cum_drift = np.cumsum(diffs)

    console.print(f"\n[bold]Final Drift after {n_steps} steps:[/]")
    console.print(f"  Last error: {diffs[-1]:.6f}")
    console.print(f"  Cumulative drift: {cum_drift[-1]:.6f}")
    console.print(f"  Mean abs error: {np.abs(diffs).mean():.6f}")

    return {
        "generated": generated_gammas,
        "true": true_gammas,
        "diffs": diffs,
        "cum_drift": cum_drift,
        "spacings": predicted_spacings,
        "start_idx": start_idx,
        "n_steps": n_steps,
    }


def visualize_trajectory(results, save_path="trajectory_demo.png"):
    """Create visualization of trajectory generation."""
    console.print(f"\n[cyan]Creating visualization...[/]")

    generated = results["generated"]
    true = results["true"]
    diffs = results["diffs"]
    cum_drift = results["cum_drift"]
    n_steps = len(generated)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========================================
    # Top Left: Trajectory (relative to start)
    # ========================================
    ax = axes[0, 0]
    steps = np.arange(n_steps)
    ax.plot(steps, true - true[0], "k-o", label="Ground Truth", alpha=0.6, markersize=3)
    ax.plot(steps, generated - true[0], "r--x", label="Neural Prediction", linewidth=2, markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Δγ from start")
    ax.set_title("Trajectory: True vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========================================
    # Top Right: Step-by-step error
    # ========================================
    ax = axes[0, 1]
    colors = ["green" if d < 0 else "red" for d in diffs]
    ax.bar(steps, diffs, color=colors, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Error (Pred - True)")
    ax.set_title("Prediction Error per Step")
    ax.grid(True, alpha=0.3)

    # ========================================
    # Bottom Left: Cumulative drift
    # ========================================
    ax = axes[1, 0]
    ax.plot(steps, cum_drift, "b-", linewidth=2)
    ax.fill_between(steps, cum_drift, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Drift")
    ax.set_title("Cumulative Error (Stability Test)")
    ax.grid(True, alpha=0.3)

    # Linear fit for drift rate
    drift_rate = np.polyfit(steps, cum_drift, 1)[0]
    ax.annotate(
        f"Drift rate: {drift_rate:.4f}/step",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        color="blue",
    )

    # ========================================
    # Bottom Right: Error distribution
    # ========================================
    ax = axes[1, 1]
    ax.hist(diffs, bins=20, color="purple", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.axvline(np.mean(diffs), color="blue", linestyle="-", linewidth=2, label=f"Mean: {np.mean(diffs):.4f}")
    ax.set_xlabel("Error")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Neural Trajectory Generation ({n_steps} steps)\n"
        f"MAE: {np.abs(diffs).mean():.4f}, Final Drift: {cum_drift[-1]:.4f}",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    console.print(f"[green]✅ Saved to {save_path}[/]")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate zeta zero trajectory")
    parser.add_argument("--start", type=int, default=1800000, help="Starting index in zeros array")
    parser.add_argument("--steps", type=int, default=50, help="Number of zeros to generate")
    parser.add_argument("--sampling", type=str, default="greedy", choices=["greedy", "sample"])
    parser.add_argument("--output", type=str, default="trajectory_demo.png")

    args = parser.parse_args()

    results = generate_trajectory(
        start_idx=args.start,
        n_steps=args.steps,
        sampling=args.sampling,
    )

    visualize_trajectory(results, save_path=args.output)

    console.print("\n[bold green]═══ TRAJECTORY COMPLETE ═══[/]")


if __name__ == "__main__":
    main()
