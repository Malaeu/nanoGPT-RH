#!/usr/bin/env python3
"""
📊 SPECTRAL FORM FACTOR TEST

Честный тест: сравниваем SFF генераций vs true zeros vs GUE теория.

SFF(τ) = |∑_n exp(2πi·u_n·τ)|² / N

GUE prediction:
- τ < 1: K(τ) ≈ 2τ (ramp)
- τ > 1: K(τ) ≈ 1 (plateau)

Если модель выучила правильные корреляции → SFF близок к GUE.
Если модель генерит случайный шум → SFF = 1 везде (Poisson).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table

from model.gpt import SpacingGPT
from predict_zeros import inverse_unfold, bin_to_spacing, unfold_val

console = Console()

# ============================================================
# CONFIG
# ============================================================
CKPT_PATH = Path("out/best.pt")
ZEROS_PATH = Path("zeros/zeros2M.txt")
DATA_DIR = Path("data")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================
# SFF COMPUTATION
# ============================================================
def compute_sff(unfolded_positions, tau_values):
    """
    Compute Spectral Form Factor.

    K(τ) = |∑_n exp(2πi·u_n·τ)|² / N

    Args:
        unfolded_positions: array of unfolded coordinates u_n
        tau_values: array of τ values to compute SFF at

    Returns:
        SFF values at each τ
    """
    N = len(unfolded_positions)
    u = np.array(unfolded_positions)

    # Center positions (remove mean drift)
    u_centered = u - np.mean(u)

    sff = []
    for tau in tau_values:
        # Sum of complex exponentials
        phases = 2 * np.pi * u_centered * tau
        sum_exp = np.sum(np.exp(1j * phases))
        K_tau = np.abs(sum_exp)**2 / N
        sff.append(K_tau)

    return np.array(sff)


def gue_sff_theory(tau):
    """
    GUE/CUE theoretical SFF (connected part).

    K_conn(τ) = 2τ for τ < 1
    K_conn(τ) = 2 - τ for τ > 1 (simplified)

    Full formula more complex, this is approximation.
    """
    tau = np.array(tau)
    K = np.where(tau < 1, 2 * tau, np.ones_like(tau) * 2 - tau)
    K = np.maximum(K, 1)  # plateau at 1
    return K


def poisson_sff():
    """Poisson (uncorrelated) gives K(τ) = 1 everywhere."""
    return 1.0


# ============================================================
# GENERATION
# ============================================================
def load_model():
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = SpacingGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model, config


def load_meta():
    return torch.load(DATA_DIR / "meta.pt", weights_only=False)


def generate_zeros(start_idx, n_zeros, sampling="greedy"):
    """Generate zeros autoregressively."""
    model, config = load_model()
    zeros = np.loadtxt(ZEROS_PATH)
    meta = load_meta()
    bin_edges = np.array(meta["bin_edges"])

    # Compute spacings
    unfolded = unfold_val(zeros)
    all_gaps = np.diff(unfolded)

    context_len = config.seq_len
    curr_gaps = list(all_gaps[start_idx : start_idx + context_len])
    curr_gamma = zeros[start_idx + context_len]

    generated_gammas = [curr_gamma]

    with torch.no_grad():
        for i in range(n_zeros - 1):
            ctx = curr_gaps[-context_len:]
            ctx_bins = np.digitize(ctx, bin_edges) - 1
            ctx_bins = np.clip(ctx_bins, 0, config.vocab_size - 1)

            x = torch.tensor(ctx_bins, dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits, _ = model(x)
            probs = torch.softmax(logits[0, -1, :], dim=0)

            if sampling == "greedy":
                pred_bin = torch.argmax(probs).item()
            else:
                pred_bin = torch.multinomial(probs, 1).item()

            s_pred = bin_to_spacing(pred_bin, bin_edges)
            next_gamma = inverse_unfold(curr_gamma, s_pred)

            generated_gammas.append(next_gamma)
            curr_gaps.append(s_pred)
            curr_gamma = next_gamma

    return np.array(generated_gammas)


# ============================================================
# MAIN TEST
# ============================================================
def run_sff_test(start_idx=1800000, n_zeros=500, n_tau=100):
    """
    Run SFF comparison test.
    """
    console.print("\n[bold magenta]═══ 📊 SPECTRAL FORM FACTOR TEST ═══[/]\n")
    console.print(f"[cyan]Generating {n_zeros} zeros from idx {start_idx}...[/]")

    # Load true zeros
    zeros = np.loadtxt(ZEROS_PATH)
    true_gammas = zeros[start_idx : start_idx + n_zeros]
    true_unfolded = unfold_val(true_gammas)

    # Generate zeros
    console.print("[cyan]Running neural generation (greedy)...[/]")
    gen_gammas = generate_zeros(start_idx - 256, n_zeros, sampling="greedy")
    gen_unfolded = unfold_val(gen_gammas)

    # Also generate with sampling
    console.print("[cyan]Running neural generation (sampling T=0.8)...[/]")
    gen_gammas_sample = generate_zeros(start_idx - 256, n_zeros, sampling="sample")
    gen_unfolded_sample = unfold_val(gen_gammas_sample)

    # Compute SFF
    tau_values = np.linspace(0.01, 2.0, n_tau)

    console.print("[cyan]Computing SFF...[/]")
    sff_true = compute_sff(true_unfolded, tau_values)
    sff_gen = compute_sff(gen_unfolded, tau_values)
    sff_gen_sample = compute_sff(gen_unfolded_sample, tau_values)
    sff_gue = gue_sff_theory(tau_values)

    # Metrics
    # Compare in ramp region (τ < 1)
    ramp_mask = tau_values < 1.0

    mse_true_gue = np.mean((sff_true[ramp_mask] - sff_gue[ramp_mask])**2)
    mse_gen_gue = np.mean((sff_gen[ramp_mask] - sff_gue[ramp_mask])**2)
    mse_gen_sample_gue = np.mean((sff_gen_sample[ramp_mask] - sff_gue[ramp_mask])**2)
    mse_gen_true = np.mean((sff_gen[ramp_mask] - sff_true[ramp_mask])**2)

    # Results table
    console.print("\n")
    table = Table(title="📊 SFF COMPARISON (Ramp Region τ < 1)", show_header=True)
    table.add_column("Comparison", style="cyan")
    table.add_column("MSE", style="green", justify="right")
    table.add_column("Interpretation", style="dim")

    table.add_row("True vs GUE Theory", f"{mse_true_gue:.6f}", "How GUE-like are real zeros")
    table.add_row("Generated (greedy) vs GUE", f"{mse_gen_gue:.6f}", "Model learns GUE?")
    table.add_row("Generated (sample) vs GUE", f"{mse_gen_sample_gue:.6f}", "Sampling helps?")
    table.add_row("Generated vs True", f"{mse_gen_true:.6f}", "Direct match")

    console.print(table)

    # Interpretation
    console.print("\n[bold]═══ INTERPRETATION ═══[/]")

    if mse_gen_gue < 2 * mse_true_gue:
        console.print("[green]✅ Generated zeros show GUE-like correlations![/]")
        console.print("[dim]→ Model learned long-range structure[/]")
    else:
        ratio = mse_gen_gue / mse_true_gue
        console.print(f"[yellow]⚠️ Generated SFF deviates from GUE ({ratio:.1f}x worse than true)[/]")
        console.print("[dim]→ Model may not capture full correlation structure[/]")

    # Check if better than Poisson (K=1 everywhere)
    # For ramp region, GUE goes from 0 to 2, mean ~1
    # Poisson = 1 everywhere, so MSE vs GUE ramp is high
    poisson_mse = np.mean((np.ones_like(sff_gue[ramp_mask]) - sff_gue[ramp_mask])**2)
    console.print(f"\n[dim]Poisson (K=1) vs GUE MSE: {poisson_mse:.4f}[/]")

    if mse_gen_gue < poisson_mse:
        console.print(f"[green]✅ Generated better than Poisson: {mse_gen_gue:.4f} < {poisson_mse:.4f}[/]")
    else:
        console.print(f"[yellow]⚠️ Generated similar to Poisson[/]")

    # Compare sampling vs greedy
    if mse_gen_sample_gue < mse_gen_gue:
        console.print(f"[green]✅ Sampling beats greedy: {mse_gen_sample_gue:.4f} < {mse_gen_gue:.4f}[/]")

    # Ultimate test: does generated match true zeros' SFF?
    console.print(f"\n[bold]Key metric: Generated vs True SFF MSE = {mse_gen_true:.4f}[/]")

    return {
        "tau": tau_values,
        "sff_true": sff_true,
        "sff_gen": sff_gen,
        "sff_gen_sample": sff_gen_sample,
        "sff_gue": sff_gue,
        "mse_true_gue": mse_true_gue,
        "mse_gen_gue": mse_gen_gue,
        "mse_gen_true": mse_gen_true,
    }


def visualize_sff(results, save_path="sff_comparison.png"):
    """Visualize SFF comparison."""
    console.print("\n[cyan]Creating visualization...[/]")

    tau = results["tau"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Full SFF comparison
    ax = axes[0]
    ax.plot(tau, results["sff_true"], 'k-', linewidth=2, label='True Zeros', alpha=0.8)
    ax.plot(tau, results["sff_gen"], 'r--', linewidth=2, label='Generated (greedy)')
    ax.plot(tau, results["sff_gen_sample"], 'b:', linewidth=2, label='Generated (sample)')
    ax.plot(tau, results["sff_gue"], 'g-', linewidth=3, label='GUE Theory', alpha=0.5)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Poisson')
    ax.axvline(1.0, color='orange', linestyle=':', alpha=0.5, label='τ = 1 (Heisenberg)')

    ax.set_xlabel('τ (scaled time)')
    ax.set_ylabel('K(τ) (SFF)')
    ax.set_title('Spectral Form Factor Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)

    # Right: Ramp region zoom
    ax = axes[1]
    ramp_mask = tau < 1.0
    ax.plot(tau[ramp_mask], results["sff_true"][ramp_mask], 'k-', linewidth=2, label='True', alpha=0.8)
    ax.plot(tau[ramp_mask], results["sff_gen"][ramp_mask], 'r--', linewidth=2, label='Generated')
    ax.plot(tau[ramp_mask], results["sff_gue"][ramp_mask], 'g-', linewidth=3, label='GUE: K=2τ', alpha=0.5)

    ax.set_xlabel('τ')
    ax.set_ylabel('K(τ)')
    ax.set_title('Ramp Region (τ < 1) - Key Test')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add metrics annotation
    ax.annotate(
        f"MSE(Gen vs GUE): {results['mse_gen_gue']:.4f}\n"
        f"MSE(True vs GUE): {results['mse_true_gue']:.4f}",
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.suptitle(
        "SFF Test: Does Neural Model Learn GUE Correlations?",
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    console.print(f"[green]✅ Saved to {save_path}[/]")
    plt.close()


def main():
    results = run_sff_test(n_zeros=500)
    visualize_sff(results)

    console.print("\n[bold]═══ SFF TEST SUMMARY ═══[/]")
    console.print("SFF = |∑ exp(2πi·u_n·τ)|² / N")
    console.print("")
    console.print("[cyan]GUE prediction:[/]")
    console.print("  τ < 1: K(τ) = 2τ (linear ramp)")
    console.print("  τ > 1: K(τ) → 1 (plateau)")
    console.print("")
    console.print("[cyan]Poisson (random):[/]")
    console.print("  K(τ) = 1 everywhere")
    console.print("")
    console.print("[bold green]═══ COMPLETE ═══[/]")


if __name__ == "__main__":
    main()
