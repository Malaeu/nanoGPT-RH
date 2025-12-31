#!/usr/bin/env python3
"""
CAUSAL ORACLE v1.0

Проверяем: понимает ли модель ПРИЧИННОСТЬ или только корреляции?

Метод:
1. Берём последовательность spacings S_1, S_2, ..., S_T
2. Делаем intervention do(S_t := S_t + δ)
3. Смотрим как меняется prediction P(S_{t+1})
4. Сравниваем с RMT prediction (level repulsion → отрицательная корреляция)

Если модель понимает физику:
- do(S_t + δ) → prediction S_{t+1} должен сдвинуться в ПРОТИВОПОЛОЖНУЮ сторону
- Это level repulsion: большой gap → следующий меньше

Если модель только корреляции:
- Эффект будет слабым или случайным
"""

import torch
import torch.nn.functional as F
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import matplotlib.pyplot as plt
from scipy import stats

from train_snowball import SnowballGPT, SnowballConfig
from train_wigner import wigner_logprob

console = Console()


def load_model(ckpt_path="out/snowball_v8_best.pt"):
    """Load trained snowball model."""
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = SnowballConfig(**ckpt["config"])
    model = SnowballGPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    bin_centers = ckpt.get("bin_centers", np.load("data/bin_centers.npy"))

    return model, config, bin_centers, device


def get_prediction(model, X, bin_centers_t, device, memory_state=None):
    """Get model prediction for position T (next spacing after sequence)."""
    B = X.shape[0]

    if memory_state is None:
        memory_state = model.snowball.get_initial_state(B, device)

    scale_val = bin_centers_t[X]

    with torch.no_grad():
        # Model returns tuple: (pred, loss, mem_attn, new_memory)
        pred, loss, mem_attn, new_memory = model(
            X, targets=None, return_hidden=False, scale_val=scale_val,
            memory_state=memory_state, return_memory=True
        )
        mu = model.get_mu(pred)  # [B, T]

    # Return prediction for last position
    return mu[:, -1]  # [B]


def intervene(X, pos, delta, bin_centers, n_bins=256):
    """
    Do intervention: X[pos] := X[pos] + delta (in spacing space).

    Args:
        X: token indices [B, T]
        pos: position to intervene
        delta: change in spacing (continuous)
        bin_centers: mapping from token to spacing
        n_bins: number of bins

    Returns:
        X_new: modified token indices
    """
    X_new = X.clone()

    # Get current spacing value
    current_spacing = bin_centers[X[:, pos].cpu().numpy()]

    # Apply intervention
    new_spacing = current_spacing + delta
    new_spacing = np.clip(new_spacing, bin_centers[0], bin_centers[-1])

    # Find nearest bin
    new_tokens = np.abs(bin_centers[:, None] - new_spacing[None, :]).argmin(axis=0)
    X_new[:, pos] = torch.tensor(new_tokens, dtype=X.dtype, device=X.device)

    return X_new


def causal_test(
    model,
    val_data,
    bin_centers,
    device,
    n_samples=100,
    delta_range=(-0.3, 0.3),
    n_deltas=11,
    intervention_pos=-2,  # Second to last position
):
    """
    Run causal intervention test.

    For each sample:
    1. Get baseline prediction P(S_{t+1} | S_1, ..., S_t)
    2. Intervene on S_t: do(S_t := S_t + δ)
    3. Get new prediction P(S_{t+1} | S_1, ..., S_t + δ)
    4. Measure shift in prediction

    Returns:
        results: dict with intervention effects
    """
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)

    deltas = np.linspace(delta_range[0], delta_range[1], n_deltas)

    # Storage
    all_effects = []  # (delta, baseline_pred, intervened_pred, actual_next)

    console.print(f"\n[bold cyan]Running Causal Interventions[/]")
    console.print(f"  Samples: {n_samples}")
    console.print(f"  Deltas: {deltas}")
    console.print(f"  Intervention position: {intervention_pos}")

    for i in track(range(n_samples), description="Intervening..."):
        # Get a sequence
        seq = val_data[i:i+1]  # [1, T+1]
        X = seq[:, :-1].to(device)  # [1, T]
        Y = seq[:, 1:].to(device)   # [1, T]

        T = X.shape[1]
        pos = T + intervention_pos  # Convert negative index

        # Baseline prediction
        baseline_pred = get_prediction(model, X, bin_centers_t, device)
        baseline_pred = baseline_pred.item()

        # Actual next spacing
        actual_next = bin_centers[Y[0, -1].item()]

        # Current spacing at intervention point
        current_spacing = bin_centers[X[0, pos].item()]

        for delta in deltas:
            # Intervene
            X_int = intervene(X, pos, delta, bin_centers)

            # Get new prediction
            int_pred = get_prediction(model, X_int, bin_centers_t, device)
            int_pred = int_pred.item()

            # Record effect
            effect = int_pred - baseline_pred
            all_effects.append({
                "delta": delta,
                "baseline_pred": baseline_pred,
                "intervened_pred": int_pred,
                "effect": effect,
                "actual_next": actual_next,
                "current_spacing": current_spacing,
            })

    return all_effects


def analyze_causal_effects(effects):
    """Analyze intervention effects and compare to RMT predictions."""

    effects_df = {k: [e[k] for e in effects] for k in effects[0].keys()}

    deltas = np.array(effects_df["delta"])
    effects_arr = np.array(effects_df["effect"])

    # Group by delta
    unique_deltas = np.unique(deltas)
    mean_effects = []
    std_effects = []

    for d in unique_deltas:
        mask = deltas == d
        mean_effects.append(effects_arr[mask].mean())
        std_effects.append(effects_arr[mask].std())

    mean_effects = np.array(mean_effects)
    std_effects = np.array(std_effects)

    # Linear regression: effect = slope * delta + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(unique_deltas, mean_effects)

    # RMT expectation: negative slope (level repulsion)
    # If S_t increases, S_{t+1} should decrease (on average)

    return {
        "deltas": unique_deltas,
        "mean_effects": mean_effects,
        "std_effects": std_effects,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value": p_value,
    }


def run_causal_oracle(
    ckpt_path="out/snowball_v8_best.pt",
    n_samples=200,
    delta_range=(-0.5, 0.5),
    n_deltas=21,
):
    """Full causal oracle pipeline."""

    console.print(Panel.fit(
        "[bold cyan]🔬 CAUSAL ORACLE v1.0[/]\n"
        "Testing: Does the model understand CAUSALITY?",
        title="INTERVENTION MODE"
    ))

    # Load model
    console.print("\nLoading model...")
    model, config, bin_centers, device = load_model(ckpt_path)
    console.print(f"  Model: {ckpt_path}")

    # Load data
    console.print("Loading validation data...")
    val = torch.load("data/val.pt", weights_only=False)
    console.print(f"  Sequences: {len(val)}")

    # Run interventions
    effects = causal_test(
        model, val, bin_centers, device,
        n_samples=n_samples,
        delta_range=delta_range,
        n_deltas=n_deltas,
    )

    # Analyze
    console.print("\nAnalyzing causal effects...")
    results = analyze_causal_effects(effects)

    # Display results
    console.print("\n")
    table = Table(title="🔬 CAUSAL ORACLE RESULTS")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation")

    slope = results["slope"]
    r2 = results["r_squared"]
    p_val = results["p_value"]

    # Slope interpretation
    if slope < -0.1:
        slope_verdict = "[green]Negative (RMT-like!)[/]"
    elif slope > 0.1:
        slope_verdict = "[red]Positive (anti-RMT)[/]"
    else:
        slope_verdict = "[yellow]Near zero (no effect)[/]"

    # R² interpretation
    if r2 > 0.5:
        r2_verdict = "[green]Strong relationship[/]"
    elif r2 > 0.1:
        r2_verdict = "[yellow]Weak relationship[/]"
    else:
        r2_verdict = "[red]No relationship[/]"

    # P-value interpretation
    if p_val < 0.01:
        p_verdict = "[green]Significant[/]"
    elif p_val < 0.05:
        p_verdict = "[yellow]Marginal[/]"
    else:
        p_verdict = "[red]Not significant[/]"

    table.add_row("Slope (dPred/dS)", f"{slope:.4f}", slope_verdict)
    table.add_row("R²", f"{r2:.4f}", r2_verdict)
    table.add_row("p-value", f"{p_val:.4f}", p_verdict)

    console.print(table)

    # RMT comparison
    console.print("\n")
    rmt_table = Table(title="📊 RMT COMPARISON")
    rmt_table.add_column("Property", style="bold")
    rmt_table.add_column("RMT/GUE", justify="center")
    rmt_table.add_column("Model", justify="center")
    rmt_table.add_column("Match?", justify="center")

    # GUE neighbor correlation is approximately -0.27
    gue_neighbor_corr = -0.27

    if slope < 0:
        match_direction = "[green]✓[/]"
    else:
        match_direction = "[red]✗[/]"

    rmt_table.add_row(
        "Neighbor correlation",
        f"{gue_neighbor_corr:.2f}",
        f"{slope:.2f}",
        match_direction
    )

    console.print(rmt_table)

    # Diagnosis
    console.print("\n")
    if slope < -0.1 and p_val < 0.05:
        diagnosis = (
            "[bold green]🎉 MODEL UNDERSTANDS LEVEL REPULSION![/]\n"
            "Causal interventions show correct RMT-like response:\n"
            "  do(S_t + δ) → prediction shifts in OPPOSITE direction.\n"
            "The model learned PHYSICS, not just correlations!"
        )
    elif slope < 0 and p_val < 0.1:
        diagnosis = (
            "[bold yellow]⚠️ WEAK CAUSAL UNDERSTANDING[/]\n"
            "Direction is correct but effect is weak.\n"
            "Model partially captured level repulsion."
        )
    elif abs(slope) < 0.05:
        diagnosis = (
            "[bold red]🔴 NO CAUSAL EFFECT[/]\n"
            "Interventions have NO effect on predictions!\n"
            "Model treats each position independently.\n"
            "This is CORRELATION without CAUSATION."
        )
    else:
        diagnosis = (
            "[bold red]🔴 ANTI-PHYSICAL RESPONSE[/]\n"
            f"Slope is POSITIVE ({slope:.3f})!\n"
            "Model predicts ATTRACTION instead of REPULSION.\n"
            "Completely wrong physics."
        )

    console.print(Panel.fit(diagnosis, title="💡 DIAGNOSIS", border_style="cyan"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Mean effect vs delta
    ax = axes[0]
    ax.errorbar(
        results["deltas"],
        results["mean_effects"],
        yerr=results["std_effects"] / np.sqrt(n_samples),
        fmt='o-', capsize=3, color='blue'
    )
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Add regression line
    x_line = np.array([results["deltas"].min(), results["deltas"].max()])
    y_line = results["slope"] * x_line + results["intercept"]
    ax.plot(x_line, y_line, 'r--', label=f'slope={results["slope"]:.3f}')

    ax.set_xlabel('Intervention δ (spacing change)')
    ax.set_ylabel('Effect on prediction (Δμ)')
    ax.set_title('Causal Effect: do(S_t + δ) → Δ(S_{t+1})')
    ax.legend()

    # Plot 2: Scatter of all effects
    ax = axes[1]
    deltas_all = np.array([e["delta"] for e in effects])
    effects_all = np.array([e["effect"] for e in effects])

    ax.scatter(deltas_all, effects_all, alpha=0.1, s=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Intervention δ')
    ax.set_ylabel('Effect on prediction')
    ax.set_title('All Individual Effects')

    plt.tight_layout()
    plt.savefig('reports/causal_oracle_v1.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/causal_oracle_v1.png[/]")

    # Save results
    np.savez(
        "reports/causal_oracle_v1.npz",
        deltas=results["deltas"],
        mean_effects=results["mean_effects"],
        std_effects=results["std_effects"],
        slope=results["slope"],
        r_squared=results["r_squared"],
        p_value=results["p_value"],
    )
    console.print(f"[green]Results saved: reports/causal_oracle_v1.npz[/]")

    return results


if __name__ == "__main__":
    from pathlib import Path
    Path("reports").mkdir(exist_ok=True)
    run_causal_oracle()
