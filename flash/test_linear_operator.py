#!/usr/bin/env python3
"""
Test Linear Operator on Rollout.

Compare the discovered linear operator:
    rₙ = -0.45·r₋₁ - 0.28·r₋₂ - 0.16·r₋₃

Against the Flash model on rollout prediction.

Usage:
    python flash/test_linear_operator.py
"""

import sys
from pathlib import Path

import torch
import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from memory_mdn_flash import MemoryMDN, MemoryMDNConfig

console = Console()


# Discovered linear operator coefficients
LINEAR_COEFFS = np.array([-0.4467, -0.2819, -0.1586])


def load_model_and_data(ckpt_path: Path, data_dir: Path, device: str):
    """Load Flash MDN model and validation data."""
    console.print(f"[cyan]Loading checkpoint: {ckpt_path}[/]")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt['config']
    model = MemoryMDN(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    console.print(f"[green]MemoryMDN: {config.n_layer}L x {config.n_head}H[/]")

    # Load validation data (residuals)
    val_data = torch.load(data_dir / "val.pt", weights_only=True)
    console.print(f"[green]Val data: {val_data.shape}[/]")

    return model, config, val_data, device


def linear_operator_predict(history: np.ndarray) -> float:
    """
    Predict next residual using linear operator.

    rₙ = -0.45·r₋₁ - 0.28·r₋₂ - 0.16·r₋₃
    """
    if len(history) < 3:
        return 0.0  # Default for short history

    r1 = history[-1]  # r_{n-1}
    r2 = history[-2]  # r_{n-2}
    r3 = history[-3]  # r_{n-3}

    return LINEAR_COEFFS[0] * r1 + LINEAR_COEFFS[1] * r2 + LINEAR_COEFFS[2] * r3


def linear_operator_rollout(seed_seq: np.ndarray, n_steps: int) -> np.ndarray:
    """Generate sequence using linear operator."""
    history = list(seed_seq)
    predictions = []

    for _ in range(n_steps):
        pred = linear_operator_predict(np.array(history))
        predictions.append(pred)
        history.append(pred)

    return np.array(predictions)


@torch.no_grad()
def flash_model_rollout(model, seed_seq: torch.Tensor, n_steps: int, device: str) -> np.ndarray:
    """Generate sequence using Flash model."""
    # Ensure seed_seq is 1D
    if seed_seq.dim() > 1:
        seed_seq = seed_seq.flatten()

    history = seed_seq.clone().float()
    predictions = []

    for _ in range(n_steps):
        # Get prediction for next step - model expects (B, T)
        x = history.unsqueeze(0).to(device)
        result = model(x)

        # Weighted mean from MDN
        pi = result['pi'][0, -1]
        mu = result['mu'][0, -1]
        pred = (pi * mu).sum().item()

        predictions.append(pred)

        # Append prediction and slide window
        history = torch.cat([history[1:], torch.tensor([pred], dtype=history.dtype)])

    return np.array(predictions)


def compute_rollout_error(pred_seq: np.ndarray, true_seq: np.ndarray) -> dict:
    """Compute rollout error metrics."""
    # Cumulative sum for position tracking
    pred_cum = np.cumsum(pred_seq)
    true_cum = np.cumsum(true_seq)

    # Error at each step
    errors = np.abs(pred_cum - true_cum)

    return {
        'err_10': errors[9] if len(errors) >= 10 else np.nan,
        'err_50': errors[49] if len(errors) >= 50 else np.nan,
        'err_100': errors[99] if len(errors) >= 100 else np.nan,
        'mean_abs_error': np.mean(np.abs(pred_seq - true_seq)),
        'max_error': errors.max(),
    }


def run_rollout_comparison(model, val_data, device, n_trials: int = 100,
                           context_len: int = 128, rollout_len: int = 100):
    """Compare linear operator vs Flash model on rollout."""
    console.print(f"\n[bold cyan]=== ROLLOUT COMPARISON ===[/]")
    console.print(f"Trials: {n_trials}, Context: {context_len}, Rollout: {rollout_len}\n")

    # Storage for results
    linear_errors = []
    flash_errors = []
    baseline_errors = []  # Just predict 0 (mean of residuals)

    # Handle different data formats
    if val_data.dim() == 2:
        # Data is (N_sequences, seq_len) - use sequences directly
        n_seqs, seq_len = val_data.shape
        console.print(f"  Data format: {n_seqs} sequences x {seq_len} length")

        # Need to concatenate sequences or use them directly
        # Use first part of each sequence as seed, rest as target
        np.random.seed(42)
        seq_indices = np.random.choice(n_seqs, size=min(n_trials, n_seqs), replace=False)

        for i, seq_idx in enumerate(seq_indices):
            seq = val_data[seq_idx].numpy()

            # Use first context_len as seed
            seed = seq[:context_len]
            # Use next rollout_len as target
            target = seq[context_len:context_len + rollout_len]

            if len(target) < rollout_len:
                continue  # Skip if not enough data

            # Linear operator rollout
            linear_pred = linear_operator_rollout(seed, rollout_len)
            linear_err = compute_rollout_error(linear_pred, target)
            linear_errors.append(linear_err)

            # Flash model rollout
            seed_torch = val_data[seq_idx, :context_len]
            flash_pred = flash_model_rollout(model, seed_torch, rollout_len, device)
            flash_err = compute_rollout_error(flash_pred, target)
            flash_errors.append(flash_err)

            # Baseline: predict 0 (mean of residuals)
            baseline_pred = np.zeros(rollout_len)
            baseline_err = compute_rollout_error(baseline_pred, target)
            baseline_errors.append(baseline_err)

            if (i + 1) % 20 == 0:
                console.print(f"  Progress: {i+1}/{n_trials}")

    else:
        # Data is 1D - use sliding window
        np.random.seed(42)
        max_start = len(val_data) - context_len - rollout_len - 10
        starts = np.random.randint(0, max_start, size=n_trials)

        for i, start in enumerate(starts):
            seed = val_data[start:start + context_len].numpy()
            target = val_data[start + context_len:start + context_len + rollout_len].numpy()

            linear_pred = linear_operator_rollout(seed, rollout_len)
            linear_err = compute_rollout_error(linear_pred, target)
            linear_errors.append(linear_err)

            seed_torch = val_data[start:start + context_len]
            flash_pred = flash_model_rollout(model, seed_torch, rollout_len, device)
            flash_err = compute_rollout_error(flash_pred, target)
            flash_errors.append(flash_err)

            baseline_pred = np.zeros(rollout_len)
            baseline_err = compute_rollout_error(baseline_pred, target)
            baseline_errors.append(baseline_err)

            if (i + 1) % 20 == 0:
                console.print(f"  Progress: {i+1}/{n_trials}")

    return linear_errors, flash_errors, baseline_errors


def display_results(linear_errors, flash_errors, baseline_errors):
    """Display comparison results."""

    def avg_metric(errors, key):
        vals = [e[key] for e in errors if not np.isnan(e[key])]
        return np.mean(vals) if vals else np.nan

    table = Table(title="Rollout Error Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Linear Op", justify="right")
    table.add_column("Flash Model", justify="right")
    table.add_column("Baseline (0)", justify="right")
    table.add_column("Winner", justify="center")

    metrics = ['err_10', 'err_50', 'err_100', 'mean_abs_error']
    labels = ['Err@10', 'Err@50', 'Err@100', 'Mean Abs Error']

    for metric, label in zip(metrics, labels):
        lin = avg_metric(linear_errors, metric)
        flash = avg_metric(flash_errors, metric)
        base = avg_metric(baseline_errors, metric)

        # Determine winner
        values = [('Linear', lin), ('Flash', flash), ('Baseline', base)]
        winner = min(values, key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))

        table.add_row(
            label,
            f"{lin:.4f}",
            f"{flash:.4f}",
            f"{base:.4f}",
            f"[bold green]{winner[0]}[/]"
        )

    console.print(table)

    # Summary
    console.print("\n[bold]Summary:[/]")

    lin_100 = avg_metric(linear_errors, 'err_100')
    flash_100 = avg_metric(flash_errors, 'err_100')
    base_100 = avg_metric(baseline_errors, 'err_100')

    console.print(f"  Linear Operator Err@100: {lin_100:.4f}")
    console.print(f"  Flash Model Err@100:     {flash_100:.4f}")
    console.print(f"  Baseline Err@100:        {base_100:.4f}")

    if lin_100 < flash_100:
        improvement = (flash_100 - lin_100) / flash_100 * 100
        console.print(f"\n[bold green]Linear operator is {improvement:.1f}% better than Flash![/]")
    else:
        degradation = (lin_100 - flash_100) / flash_100 * 100
        console.print(f"\n[bold yellow]Flash model is {degradation:.1f}% better than linear operator[/]")

    return {
        'linear_err100': lin_100,
        'flash_err100': flash_100,
        'baseline_err100': base_100,
    }


def analyze_linear_operator():
    """Analyze the linear operator theoretically."""
    console.print("\n[bold cyan]=== LINEAR OPERATOR ANALYSIS ===[/]")

    # Coefficients
    a1, a2, a3 = LINEAR_COEFFS
    console.print(f"\nOperator: rₙ = {a1:.4f}·r₋₁ + {a2:.4f}·r₋₂ + {a3:.4f}·r₋₃")

    # Check stability (characteristic polynomial roots)
    # r^3 = a1*r^2 + a2*r + a3
    # r^3 - a1*r^2 - a2*r - a3 = 0
    coeffs = [1, -a1, -a2, -a3]
    roots = np.roots(coeffs)

    console.print(f"\nCharacteristic roots: {roots}")
    console.print(f"Max |root|: {np.max(np.abs(roots)):.4f}")

    if np.max(np.abs(roots)) < 1:
        console.print("[green]Operator is STABLE (|roots| < 1)[/]")
    else:
        console.print("[red]Operator is UNSTABLE (|roots| >= 1)[/]")

    # Sum of coefficients
    total = a1 + a2 + a3
    console.print(f"\nSum of coefficients: {total:.4f}")
    console.print("(Should be close to -1 for mean-reverting behavior)")

    # Compare with GUE theory
    console.print("\n[bold]GUE Comparison:[/]")
    console.print("  Expected: negative correlations that decay with lag")
    console.print(f"  Found: a1={a1:.3f}, a2={a2:.3f}, a3={a3:.3f}")
    console.print("  [green]All negative and decreasing - consistent with GUE![/]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Linear Operator on Rollout")
    parser.add_argument("--ckpt", type=Path, default=Path("out/mdn_memory_q3_flash/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/continuous_residuals"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--rollout-len", type=int, default=100)
    args = parser.parse_args()

    console.print("[bold]Linear Operator Rollout Test[/]")
    console.print(f"Device: {args.device}\n")

    # Analyze operator
    analyze_linear_operator()

    # Load model and data
    model, config, val_data, device = load_model_and_data(
        args.ckpt, args.data_dir, args.device
    )

    # Run comparison
    linear_errors, flash_errors, baseline_errors = run_rollout_comparison(
        model, val_data, device,
        n_trials=args.n_trials,
        context_len=args.context_len,
        rollout_len=args.rollout_len
    )

    # Display results
    results = display_results(linear_errors, flash_errors, baseline_errors)

    # Save results
    import json
    output_file = Path("results/linear_operator_rollout.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'operator_coeffs': LINEAR_COEFFS.tolist(),
            'n_trials': args.n_trials,
            'context_len': args.context_len,
            'rollout_len': args.rollout_len,
            'results': results,
        }, f, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/]")


if __name__ == "__main__":
    main()
