#!/usr/bin/env python3
"""
⚖️ KERNEL BASELINE: Neural vs Linear Filter Comparison

Scientific control: Is the neural network actually learning something
beyond what a simple linear filter with our extracted kernel can do?

Extracted kernel formula:
μ(d) = 1.20 × cos(0.357d - 2.05) × exp(-0.0024d) - 0.96

If Neural >> Linear → model learns nonlinearities
If Neural ≈ Linear → model is just a fancy filter
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

# ============================================================
# CONFIG
# ============================================================
ZEROS_PATH = Path("zeros/zeros2M.txt")
CONTEXT_LEN = 64  # Linear models work better with shorter context


# ============================================================
# KERNEL FORMULAS
# ============================================================
def neural_kernel_formula(d):
    """
    Extracted kernel from attention logits:
    μ(d) = 1.20 × cos(0.357d - 2.05) × exp(-0.0024d) - 0.96

    This is the "fingerprint" our neural network learned.
    """
    return 1.2 * np.cos(0.357 * d - 2.05) * np.exp(-0.0024 * d) - 0.96


def gue_kernel(d, alpha=0.3):
    """
    GUE-inspired kernel (simplified sine kernel).
    K(d) ~ sin(π×d) / (π×d) for large d
    """
    # Avoid division by zero
    d_safe = np.maximum(d, 0.01)
    return np.sinc(alpha * d_safe)  # sinc(x) = sin(πx)/(πx)


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    """Load and prepare spacing data."""
    console.print(f"[cyan]Loading zeros from {ZEROS_PATH}...[/]")

    zeros = np.loadtxt(ZEROS_PATH)

    # Unfold
    u = (zeros / (2 * np.pi)) * np.log(zeros / (2 * np.pi * np.e))
    gaps = np.diff(u)

    console.print(f"[green]Loaded {len(gaps):,} spacings[/]")
    console.print(f"[dim]Mean: {gaps.mean():.4f}, Std: {gaps.std():.4f}[/]")

    return gaps


def make_windows(seq, window_size):
    """Create sliding windows for regression."""
    X = []
    y = []
    for i in range(len(seq) - window_size):
        X.append(seq[i : i + window_size])
        y.append(seq[i + window_size])
    return np.array(X), np.array(y)


# ============================================================
# BASELINE MODELS
# ============================================================
def run_baselines():
    """Run all baseline comparisons."""
    console.print("\n[bold magenta]═══ ⚖️ KERNEL BASELINE COMPARISON ═══[/]\n")

    gaps = load_data()

    # Train/test split (90/10)
    split_idx = int(len(gaps) * 0.9)
    train_gaps = gaps[:split_idx]
    test_gaps = gaps[split_idx:]

    console.print(f"[cyan]Train: {len(train_gaps):,}, Test: {len(test_gaps):,}[/]")
    console.print(f"[cyan]Context length: {CONTEXT_LEN}[/]\n")

    # Create windows
    console.print("[dim]Creating sliding windows...[/]")
    X_train, y_train = make_windows(train_gaps, CONTEXT_LEN)
    X_test, y_test = make_windows(test_gaps, CONTEXT_LEN)

    results = {}

    # ========================================
    # 1. Naive Baseline (predict mean=1)
    # ========================================
    console.print("[cyan]1. Naive baseline (predict 1.0)...[/]")
    y_pred_naive = np.ones_like(y_test)
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    mse_naive = mean_squared_error(y_test, y_pred_naive)
    results["naive"] = {"mae": mae_naive, "mse": mse_naive}

    # ========================================
    # 2. Last-value baseline (predict previous)
    # ========================================
    console.print("[cyan]2. Last-value baseline...[/]")
    y_pred_last = X_test[:, -1]  # Last value in context
    mae_last = mean_absolute_error(y_test, y_pred_last)
    mse_last = mean_squared_error(y_test, y_pred_last)
    results["last_value"] = {"mae": mae_last, "mse": mse_last}

    # ========================================
    # 3. Moving average baseline
    # ========================================
    console.print("[cyan]3. Moving average baseline...[/]")
    y_pred_ma = X_test.mean(axis=1)
    mae_ma = mean_absolute_error(y_test, y_pred_ma)
    mse_ma = mean_squared_error(y_test, y_pred_ma)
    results["moving_avg"] = {"mae": mae_ma, "mse": mse_ma}

    # ========================================
    # 4. Optimal Linear Regression (full fit)
    # ========================================
    console.print("[cyan]4. Optimal Linear Regression (full fit)...[/]")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    results["optimal_linear"] = {"mae": mae_lr, "mse": mse_lr}

    # ========================================
    # 5. Neural Kernel Filter (fixed weights)
    # ========================================
    console.print("[cyan]5. Neural Kernel Filter (extracted formula)...[/]")

    # Weights from our extracted formula
    d = np.arange(1, CONTEXT_LEN + 1)
    weights = neural_kernel_formula(d)

    # Normalize weights
    weights = weights / np.abs(weights).sum()

    # Apply filter (reverse X to match d=1 being most recent)
    X_test_rev = X_test[:, ::-1]
    kernel_features = X_test_rev @ weights

    # Fit scale and bias
    lr_kernel = LinearRegression()
    lr_kernel.fit(kernel_features.reshape(-1, 1), y_test)
    y_pred_kernel = lr_kernel.predict(kernel_features.reshape(-1, 1))
    mae_kernel = mean_absolute_error(y_test, y_pred_kernel)
    mse_kernel = mean_squared_error(y_test, y_pred_kernel)
    results["neural_kernel"] = {"mae": mae_kernel, "mse": mse_kernel}

    # ========================================
    # 6. GUE Kernel Filter
    # ========================================
    console.print("[cyan]6. GUE Sine Kernel Filter...[/]")

    weights_gue = gue_kernel(d)
    weights_gue = weights_gue / np.abs(weights_gue).sum()

    kernel_features_gue = X_test_rev @ weights_gue
    lr_gue = LinearRegression()
    lr_gue.fit(kernel_features_gue.reshape(-1, 1), y_test)
    y_pred_gue = lr_gue.predict(kernel_features_gue.reshape(-1, 1))
    mae_gue = mean_absolute_error(y_test, y_pred_gue)
    mse_gue = mean_squared_error(y_test, y_pred_gue)
    results["gue_kernel"] = {"mae": mae_gue, "mse": mse_gue}

    # ========================================
    # Results Table
    # ========================================
    console.print("\n")
    table = Table(title="📊 BASELINE COMPARISON (Normalized Spacings)", show_header=True)
    table.add_column("Method", style="cyan")
    table.add_column("MAE", style="green", justify="right")
    table.add_column("MSE", style="yellow", justify="right")
    table.add_column("vs Naive", style="magenta", justify="right")

    for name, metrics in results.items():
        improvement = (mae_naive - metrics["mae"]) / mae_naive * 100
        table.add_row(
            name.replace("_", " ").title(),
            f"{metrics['mae']:.6f}",
            f"{metrics['mse']:.6f}",
            f"{improvement:+.1f}%",
        )

    console.print(table)

    # ========================================
    # Analysis
    # ========================================
    console.print("\n[bold]═══ ANALYSIS ═══[/]")

    # Compare neural kernel vs optimal linear
    if mae_kernel < mae_lr:
        console.print("[green]Neural Kernel < Optimal Linear[/]")
        console.print("[dim]→ Extracted kernel captures essential structure[/]")
    else:
        improvement = (mae_kernel - mae_lr) / mae_lr * 100
        console.print(f"[yellow]Optimal Linear beats Neural Kernel by {improvement:.1f}%[/]")
        console.print("[dim]→ Full linear fit finds more structure[/]")

    # What this means for the neural network
    console.print("\n[bold]Implications for SpacingGPT:[/]")
    console.print(f"  If Neural MAE < {mae_lr:.6f} → Model learns beyond linear")
    console.print(f"  If Neural MAE ≈ {mae_lr:.6f} → Model ≈ optimal linear filter")
    console.print(f"  If Neural MAE > {mae_lr:.6f} → Model underperforms linear!")

    # Kernel visualization data
    console.print("\n[bold]Kernel Weights (first 10):[/]")
    console.print(f"  Neural: {neural_kernel_formula(np.arange(1, 11))}")
    console.print(f"  GUE:    {gue_kernel(np.arange(1, 11))}")

    return results


if __name__ == "__main__":
    results = run_baselines()
    console.print("\n[bold green]═══ COMPLETE ═══[/]")
