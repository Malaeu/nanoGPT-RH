#!/usr/bin/env python3
"""
🚀 TRAJECTORY BENCHMARK: Autoregressive Generation Test

The ultimate test: can the model predict a sequence of zeros autoregressively?

1. Start with real context (256 tokens)
2. Predict next token, add to context
3. Repeat N times
4. Compare trajectory with reality

If model captures local geometry, trajectories should track each other.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.progress import track

from model.gpt import SpacingGPT

console = Console()

# ============================================================
# CONFIG
# ============================================================
CHECKPOINT = Path("out/best.pt")
VAL_DATA = Path("data/val.pt")
STEPS_TO_PREDICT = 50  # How many steps to predict autoregressively
N_TRAJECTORIES = 5     # How many different starting points to test


def load_model_and_data():
    """Load trained model and validation data."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SpacingGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_data = torch.load(VAL_DATA)

    return model, val_data, device


@torch.no_grad()
def generate_trajectory(model, context, steps, device, temperature=1.0, greedy=True):
    """
    Generate trajectory autoregressively.

    Args:
        context: Starting context [1, seq_len] tensor of token IDs
        steps: Number of tokens to generate
        temperature: Sampling temperature (1.0 = no change)
        greedy: If True, use argmax; if False, sample

    Returns:
        List of predicted token IDs
    """
    predictions = []
    current_ctx = context.clone()

    for _ in range(steps):
        # Get logits for last position
        logits, _ = model(current_ctx)
        next_logits = logits[0, -1, :] / temperature

        if greedy:
            next_token = torch.argmax(next_logits).item()
        else:
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        predictions.append(next_token)

        # Shift context and append new token
        next_tensor = torch.tensor([[next_token]], device=device)
        current_ctx = torch.cat([current_ctx[:, 1:], next_tensor], dim=1)

    return predictions


def run_trajectory_test():
    console.print("[bold magenta]═══ 🚀 TRAJECTORY BENCHMARK ═══[/]\n")

    # Load model and data
    console.print("[cyan]Loading model and data...[/]")
    model, val_data, device = load_model_and_data()

    n_seqs, seq_len = val_data.shape
    console.print(f"[green]Data: {n_seqs} sequences × {seq_len} tokens[/]")
    console.print(f"[green]Will test {N_TRAJECTORIES} trajectories, {STEPS_TO_PREDICT} steps each[/]\n")

    # Results storage
    all_results = []

    # Test multiple starting points
    for traj_idx in track(range(N_TRAJECTORIES), description="[green]Generating trajectories..."):
        # Pick a random sequence that has enough tokens
        seq_idx = np.random.randint(0, n_seqs)
        start_pos = 0  # Start from beginning of sequence

        # Get context (first seq_len - STEPS_TO_PREDICT tokens)
        max_context = seq_len - STEPS_TO_PREDICT
        context = val_data[seq_idx, :max_context].unsqueeze(0).to(device)

        # Get real future tokens
        real_future = val_data[seq_idx, max_context:max_context + STEPS_TO_PREDICT].numpy()

        # Generate predictions (both greedy and sampled)
        pred_greedy = generate_trajectory(model, context, STEPS_TO_PREDICT, device, greedy=True)
        pred_sampled = generate_trajectory(model, context, STEPS_TO_PREDICT, device, greedy=False, temperature=0.8)

        all_results.append({
            "seq_idx": seq_idx,
            "real": real_future,
            "pred_greedy": np.array(pred_greedy),
            "pred_sampled": np.array(pred_sampled),
        })

    # ============================================================
    # ANALYSIS
    # ============================================================
    console.print("\n[bold]Analyzing trajectories...[/]\n")

    # Compute cumulative sums (trajectories)
    for r in all_results:
        r["real_traj"] = np.cumsum(r["real"])
        r["pred_greedy_traj"] = np.cumsum(r["pred_greedy"])
        r["pred_sampled_traj"] = np.cumsum(r["pred_sampled"])

        # Normalize to start at 0
        r["real_traj"] = r["real_traj"] - r["real_traj"][0]
        r["pred_greedy_traj"] = r["pred_greedy_traj"] - r["pred_greedy_traj"][0]
        r["pred_sampled_traj"] = r["pred_sampled_traj"] - r["pred_sampled_traj"][0]

        # Compute drift (MSE from real)
        r["drift_greedy"] = np.mean((r["real_traj"] - r["pred_greedy_traj"])**2)
        r["drift_sampled"] = np.mean((r["real_traj"] - r["pred_sampled_traj"])**2)

        # Compute correlation
        r["corr_greedy"] = np.corrcoef(r["real_traj"], r["pred_greedy_traj"])[0, 1]
        r["corr_sampled"] = np.corrcoef(r["real_traj"], r["pred_sampled_traj"])[0, 1]

    # Print summary
    avg_drift_greedy = np.mean([r["drift_greedy"] for r in all_results])
    avg_drift_sampled = np.mean([r["drift_sampled"] for r in all_results])
    avg_corr_greedy = np.mean([r["corr_greedy"] for r in all_results])
    avg_corr_sampled = np.mean([r["corr_sampled"] for r in all_results])

    console.print(f"[bold]Average Results ({N_TRAJECTORIES} trajectories):[/]")
    console.print(f"  Greedy:  Drift MSE = {avg_drift_greedy:.2f}, Correlation = {avg_corr_greedy:.4f}")
    console.print(f"  Sampled: Drift MSE = {avg_drift_sampled:.2f}, Correlation = {avg_corr_sampled:.4f}")

    # ============================================================
    # VISUALIZATION
    # ============================================================
    console.print("\n[cyan]Creating visualization...[/]")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot first 3 trajectories
    for i in range(min(3, N_TRAJECTORIES)):
        ax = axes[0, i]
        r = all_results[i]

        ax.plot(r["real_traj"], 'b-', linewidth=2, label='Reality', alpha=0.8)
        ax.plot(r["pred_greedy_traj"], 'r--', linewidth=2, label='Greedy', alpha=0.8)
        ax.plot(r["pred_sampled_traj"], 'g:', linewidth=2, label='Sampled', alpha=0.8)

        ax.set_title(f'Trajectory #{i+1} (Seq {r["seq_idx"]})\n'
                     f'Corr: {r["corr_greedy"]:.3f} (G), {r["corr_sampled"]:.3f} (S)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Cumulative Position')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Plot step-by-step comparison for first trajectory
    ax = axes[1, 0]
    r = all_results[0]
    x = np.arange(STEPS_TO_PREDICT)
    ax.bar(x - 0.2, r["real"], 0.4, label='Real', alpha=0.7)
    ax.bar(x + 0.2, r["pred_greedy"], 0.4, label='Predicted', alpha=0.7)
    ax.set_title('Step-by-Step: Real vs Predicted (Traj #1)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Token (Bin)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Summary panel
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    ════════════════════════════════════════
           🚀 TRAJECTORY BENCHMARK RESULTS
    ════════════════════════════════════════

    Trajectories tested: {N_TRAJECTORIES}
    Steps per trajectory: {STEPS_TO_PREDICT}

    GREEDY GENERATION:
      Average Drift MSE:   {avg_drift_greedy:.2f}
      Average Correlation: {avg_corr_greedy:.4f}

    SAMPLED GENERATION (T=0.8):
      Average Drift MSE:   {avg_drift_sampled:.2f}
      Average Correlation: {avg_corr_sampled:.4f}

    VERDICT:
    {"✅ Model tracks trajectory!" if avg_corr_greedy > 0.5 else "⚠️ Trajectory diverges (expected for chaos)"}
    ════════════════════════════════════════
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Correlation distribution
    ax = axes[1, 2]
    corrs_greedy = [r["corr_greedy"] for r in all_results]
    corrs_sampled = [r["corr_sampled"] for r in all_results]
    ax.hist(corrs_greedy, bins=10, alpha=0.6, label='Greedy', color='red')
    ax.hist(corrs_sampled, bins=10, alpha=0.6, label='Sampled', color='green')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trajectory Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Correlation Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Autoregressive Trajectory Generation: Neural vs Reality',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('trajectory_benchmark.png', dpi=300, bbox_inches='tight')
    console.print("[green]✓ Saved trajectory_benchmark.png[/]")

    # Final verdict
    console.print("\n" + "=" * 60)
    if avg_corr_greedy > 0.7:
        console.print("[bold green]✅ EXCELLENT: Model captures trajectory structure![/]")
    elif avg_corr_greedy > 0.4:
        console.print("[bold yellow]🟡 MODERATE: Partial trajectory tracking[/]")
    else:
        console.print("[bold red]⚠️ EXPECTED: Chaotic divergence (this is normal for GUE!)[/]")
        console.print("   Autoregressive generation in chaotic systems accumulates errors.")
        console.print("   The model still captures LOCAL correlations (see benchmark_oracle.py).")
    console.print("=" * 60)

    console.print("\n[bold green]═══ TRAJECTORY BENCHMARK COMPLETE ═══[/]")


if __name__ == "__main__":
    run_trajectory_test()
