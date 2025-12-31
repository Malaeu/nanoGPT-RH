#!/usr/bin/env python3
"""
CAUSAL ORACLE v2.0 — MULTI-LAG + MODEL COMPARISON

Проверяем:
1. Затухает ли causal effect с лагом? (RMT: должен быстро)
2. Snowball усиливает overcompensation или нет?

Если slope остаётся ~-1 на дальних лагах — артефакт.
Если затухает до ~-0.3 к лагу 3-4 — настоящая физика.
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
from pathlib import Path

console = Console()


def load_snowball_model(ckpt_path="out/snowball_v8_best.pt"):
    """Load Snowball v8 model."""
    from train_snowball import SnowballGPT, SnowballConfig

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = SnowballConfig(**ckpt["config"])
    model = SnowballGPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    bin_centers = ckpt.get("bin_centers", np.load("data/bin_centers.npy"))
    return model, bin_centers, device, "snowball_v8"


def load_rope_model(ckpt_path="out/rope_v7_best.pt"):
    """Load RoPE v7 model (no snowball)."""
    from train_rope import WignerRoPEGPT, RoPEConfig

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = RoPEConfig(**ckpt["config"])
    model = WignerRoPEGPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    bin_centers = ckpt.get("bin_centers", np.load("data/bin_centers.npy"))
    return model, bin_centers, device, "rope_v7"


def load_wigner_model(ckpt_path="out/wigner_v6_best.pt"):
    """Load Wigner v6 model (baseline)."""
    from train_wigner import WignerGPT, WignerConfig

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    from train_wigner import WignerGPT

    # Recreate config
    class WignerConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    cfg = WignerConfig(**config)
    model = WignerGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    bin_centers = ckpt.get("bin_centers", np.load("data/bin_centers.npy"))
    return model, bin_centers, device, "wigner_v6"


def get_prediction_generic(model, X, bin_centers_t, device, model_type):
    """Get prediction from any model type."""
    scale_val = bin_centers_t[X]

    with torch.no_grad():
        if model_type == "snowball_v8":
            B = X.shape[0]
            memory_state = model.snowball.get_initial_state(B, device)
            pred, loss, mem_attn, new_memory = model(
                X, targets=None, return_hidden=False, scale_val=scale_val,
                memory_state=memory_state, return_memory=True
            )
        elif model_type == "rope_v7":
            out = model(X, targets=None, return_hidden=False, scale_val=scale_val)
            pred = out[0]
        else:  # wigner_v6
            out = model(X, targets=None, return_hidden=False, scale_val=scale_val)
            pred = out[0]

        mu = model.get_mu(pred)

    return mu[:, -1]


def intervene(X, pos, delta, bin_centers):
    """Apply intervention do(S_pos := S_pos + delta)."""
    X_new = X.clone()
    current_spacing = bin_centers[X[:, pos].cpu().numpy()]
    new_spacing = np.clip(current_spacing + delta, bin_centers[0], bin_centers[-1])
    new_tokens = np.abs(bin_centers[:, None] - new_spacing[None, :]).argmin(axis=0)
    X_new[:, pos] = torch.tensor(new_tokens, dtype=X.dtype, device=X.device)
    return X_new


def run_multi_lag_test(model, bin_centers, device, model_type, val_data,
                       lags=[1, 2, 3, 4, 5], deltas=None, n_samples=100):
    """Run interventions at multiple lags."""
    if deltas is None:
        deltas = np.linspace(-0.5, 0.5, 11)

    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)

    results = {}

    for lag in lags:
        effects = []

        for i in range(n_samples):
            seq = val_data[i:i+1]
            X = seq[:, :-1].to(device)
            T = X.shape[1]
            pos = T - lag - 1  # Position to intervene (lag before last)

            if pos < 0:
                continue

            # Baseline
            baseline_pred = get_prediction_generic(model, X, bin_centers_t, device, model_type)
            baseline_pred = baseline_pred.item()

            for delta in deltas:
                X_int = intervene(X, pos, delta, bin_centers)
                int_pred = get_prediction_generic(model, X_int, bin_centers_t, device, model_type)
                int_pred = int_pred.item()
                effect = int_pred - baseline_pred
                effects.append({"delta": delta, "effect": effect})

        # Compute slope for this lag
        if effects:
            deltas_arr = np.array([e["delta"] for e in effects])
            effects_arr = np.array([e["effect"] for e in effects])

            # Mean effect per delta
            unique_deltas = np.unique(deltas_arr)
            mean_effects = [effects_arr[deltas_arr == d].mean() for d in unique_deltas]

            slope, intercept, r_value, p_value, _ = stats.linregress(unique_deltas, mean_effects)
            results[lag] = {
                "slope": slope,
                "r_squared": r_value**2,
                "p_value": p_value,
                "n_samples": len(effects)
            }
        else:
            results[lag] = {"slope": 0, "r_squared": 0, "p_value": 1, "n_samples": 0}

    return results


def run_comparison():
    """Compare causal effects across models and lags."""

    console.print(Panel.fit(
        "[bold cyan]🔬 CAUSAL ORACLE v2.0[/]\n"
        "Multi-lag + Model Comparison",
        title="DEEP DIAGNOSTIC MODE"
    ))

    # Load validation data
    val = torch.load("data/val.pt", weights_only=False)
    console.print(f"Validation sequences: {len(val)}\n")

    # Models to compare
    models_to_test = []

    # Try loading each model
    if Path("out/snowball_v8_best.pt").exists():
        try:
            m, bc, dev, name = load_snowball_model()
            models_to_test.append((m, bc, dev, name))
            console.print(f"[green]✓ Loaded {name}[/]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load snowball_v8: {e}[/]")

    if Path("out/rope_v7_best.pt").exists():
        try:
            m, bc, dev, name = load_rope_model()
            models_to_test.append((m, bc, dev, name))
            console.print(f"[green]✓ Loaded {name}[/]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load rope_v7: {e}[/]")

    if Path("out/wigner_v6_best.pt").exists():
        try:
            m, bc, dev, name = load_wigner_model()
            models_to_test.append((m, bc, dev, name))
            console.print(f"[green]✓ Loaded {name}[/]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load wigner_v6: {e}[/]")

    if not models_to_test:
        console.print("[red]No models found![/]")
        return

    # Run tests
    lags = [1, 2, 3, 4, 5]
    all_results = {}

    for model, bin_centers, device, model_type in models_to_test:
        console.print(f"\n[cyan]Testing {model_type}...[/]")
        results = run_multi_lag_test(
            model, bin_centers, device, model_type, val,
            lags=lags, n_samples=100
        )
        all_results[model_type] = results

    # Display results
    console.print("\n")

    # Table: Slope by lag for each model
    table = Table(title="📊 CAUSAL EFFECT DECAY: Slope by Lag")
    table.add_column("Lag", style="bold", justify="center")

    for model_type in all_results.keys():
        table.add_column(model_type, justify="center")

    table.add_column("RMT Expected", justify="center", style="dim")

    # RMT expected correlation decay (approximate)
    rmt_expected = {1: -0.27, 2: -0.10, 3: -0.04, 4: -0.02, 5: -0.01}

    for lag in lags:
        row = [str(lag)]
        for model_type in all_results.keys():
            slope = all_results[model_type][lag]["slope"]
            r2 = all_results[model_type][lag]["r_squared"]

            # Color code based on deviation from RMT
            if abs(slope) > abs(rmt_expected[lag]) * 3:
                color = "red"
            elif abs(slope) > abs(rmt_expected[lag]) * 1.5:
                color = "yellow"
            else:
                color = "green"

            row.append(f"[{color}]{slope:.3f}[/] (R²={r2:.2f})")

        row.append(f"{rmt_expected[lag]:.2f}")
        table.add_row(*row)

    console.print(table)

    # Diagnosis
    console.print("\n")

    # Check decay pattern
    diagnoses = {}
    for model_type in all_results.keys():
        slopes = [abs(all_results[model_type][lag]["slope"]) for lag in lags]
        # Does it decay?
        if slopes[0] > 0 and slopes[-1] / slopes[0] < 0.3:
            decay_verdict = "DECAYS (good)"
        elif slopes[0] > 0 and slopes[-1] / slopes[0] < 0.6:
            decay_verdict = "PARTIAL DECAY"
        else:
            decay_verdict = "NO DECAY (artifact!)"

        # Is magnitude reasonable?
        if abs(all_results[model_type][1]["slope"]) < 0.5:
            mag_verdict = "REASONABLE"
        elif abs(all_results[model_type][1]["slope"]) < 1.0:
            mag_verdict = "HIGH"
        else:
            mag_verdict = "EXCESSIVE (artifact!)"

        diagnoses[model_type] = {"decay": decay_verdict, "magnitude": mag_verdict}

    diag_table = Table(title="💡 DIAGNOSIS")
    diag_table.add_column("Model", style="bold")
    diag_table.add_column("Decay Pattern")
    diag_table.add_column("Magnitude")
    diag_table.add_column("Verdict")

    for model_type, diag in diagnoses.items():
        if "artifact" in diag["decay"].lower() or "artifact" in diag["magnitude"].lower():
            verdict = "[red]ARTIFACT[/]"
        elif "good" in diag["decay"].lower() and diag["magnitude"] == "REASONABLE":
            verdict = "[green]REAL PHYSICS[/]"
        else:
            verdict = "[yellow]MIXED[/]"

        diag_table.add_row(model_type, diag["decay"], diag["magnitude"], verdict)

    console.print(diag_table)

    # Plot decay curves
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"snowball_v8": "blue", "rope_v7": "green", "wigner_v6": "orange"}
    markers = {"snowball_v8": "o", "rope_v7": "s", "wigner_v6": "^"}

    for model_type in all_results.keys():
        slopes = [abs(all_results[model_type][lag]["slope"]) for lag in lags]
        ax.plot(lags, slopes, f'{markers.get(model_type, "o")}-',
                label=model_type, color=colors.get(model_type, "gray"), linewidth=2, markersize=8)

    # RMT reference
    rmt_slopes = [abs(rmt_expected[lag]) for lag in lags]
    ax.plot(lags, rmt_slopes, 'k--', label='RMT Expected', linewidth=2, alpha=0.7)

    ax.set_xlabel('Lag (positions back from prediction)', fontsize=12)
    ax.set_ylabel('|Causal Effect Slope|', fontsize=12)
    ax.set_title('Causal Effect Decay with Lag', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('reports/causal_decay_comparison.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/causal_decay_comparison.png[/]")

    # Save results
    np.savez(
        "reports/causal_multi_lag.npz",
        lags=lags,
        **{f"{k}_slopes": [all_results[k][lag]["slope"] for lag in lags] for k in all_results}
    )
    console.print(f"[green]Results saved: reports/causal_multi_lag.npz[/]")

    return all_results


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    run_comparison()
