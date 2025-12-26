#!/usr/bin/env python3
"""
🧬 PySR SYMBOLIC REGRESSION ON ATTENTION KERNEL

Extract symbolic formula for μ(d_unf) using genetic programming.

Target: Find analytic expression that matches attention weights
as function of unfolded distance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
import warnings
warnings.filterwarnings('ignore')

from model.gpt import SpacingGPT

console = Console()

CKPT_PATH = Path("out/best.pt")
ZEROS_PATH = Path("zeros/zeros2M.txt")
DATA_DIR = Path("data")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = SpacingGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model, config


def unfold_val(gamma):
    gamma = np.asarray(gamma)
    return (gamma / (2 * np.pi)) * np.log(gamma / (2 * np.pi * np.e))


def extract_kernel_data(n_samples=300):
    """Extract (d_unf, μ) pairs for PySR."""
    console.print("\n[bold magenta]═══ 🧬 EXTRACTING KERNEL DATA FOR PySR ═══[/]\n")

    model, config = load_model()
    console.print(f"[green]Model loaded: seq_len={config.seq_len}[/]")

    zeros = np.loadtxt(ZEROS_PATH)
    meta = torch.load(DATA_DIR / "meta.pt", weights_only=False)
    bin_edges = np.array(meta["bin_edges"])

    unfolded = unfold_val(zeros)
    all_gaps = np.diff(unfolded)

    np.random.seed(42)
    start_indices = np.random.randint(0, len(all_gaps) - config.seq_len - 1, n_samples)

    d_unf_all = []
    attn_all = []

    console.print(f"[cyan]Extracting attention from {n_samples} samples...[/]")

    with torch.no_grad():
        for idx, start in enumerate(start_indices):
            if idx % 100 == 0:
                console.print(f"  Processing {idx}/{n_samples}...")

            spacings = all_gaps[start:start + config.seq_len]
            bins = np.digitize(spacings, bin_edges) - 1
            bins = np.clip(bins, 0, config.vocab_size - 1)

            x = torch.tensor(bins, dtype=torch.long).unsqueeze(0).to(DEVICE)
            _, _, attentions = model(x, return_attention=True)

            # Average over all layers and heads
            avg_attn = np.mean([a[0].cpu().numpy() for a in attentions], axis=(0, 1))

            # Sample pairs
            n_pairs = 30
            for _ in range(n_pairs):
                i = np.random.randint(10, config.seq_len)
                j = np.random.randint(0, i)
                d_unf = np.sum(spacings[j:i])
                attn_weight = avg_attn[i, j]

                d_unf_all.append(d_unf)
                attn_all.append(attn_weight)

    d_unf_all = np.array(d_unf_all)
    attn_all = np.array(attn_all)

    console.print(f"[green]Collected {len(d_unf_all)} samples[/]")
    console.print(f"d_unf range: [{d_unf_all.min():.2f}, {d_unf_all.max():.2f}]")
    console.print(f"attn range: [{attn_all.min():.4f}, {attn_all.max():.4f}]")

    return d_unf_all, attn_all


def bin_and_smooth(d_unf, attn, n_bins=80):
    """Bin data for cleaner PySR input."""
    bins = np.linspace(d_unf.min(), min(d_unf.max(), 100), n_bins)
    d_centers = []
    mu_values = []

    for i in range(len(bins) - 1):
        mask = (d_unf >= bins[i]) & (d_unf < bins[i+1])
        if mask.sum() > 5:
            d_centers.append((bins[i] + bins[i+1]) / 2)
            mu_values.append(attn[mask].mean())

    return np.array(d_centers), np.array(mu_values)


def run_pysr(X, y):
    """Run PySR symbolic regression."""
    console.print("\n[bold cyan]═══ RUNNING PySR ═══[/]")

    try:
        from pysr import PySRRegressor
    except ImportError:
        console.print("[red]PySR not installed! Installing...[/]")
        import subprocess
        subprocess.run(["pip", "install", "pysr"], check=True)
        from pysr import PySRRegressor

    # Configure PySR
    model = PySRRegressor(
        niterations=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "sin", "cos", "exp", "log",
            "sqrt", "square"
        ],
        populations=15,
        population_size=50,
        maxsize=25,
        constraints={
            "sin": 5,
            "cos": 5,
            "exp": 5,
            "log": 5,
        },
        nested_constraints={
            "sin": {"sin": 0, "cos": 0},
            "cos": {"sin": 0, "cos": 0},
            "exp": {"exp": 0, "log": 0},
        },
        loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="best",
        progress=True,
        verbosity=1,
        random_state=42,
        deterministic=False,  # Allow parallel for speed
        procs=0,
        multithreading=False,
    )

    console.print("[cyan]Fitting PySR model...[/]")
    X_2d = X.reshape(-1, 1)
    model.fit(X_2d, y)

    return model


def analyze_pysr_results(model, X, y):
    """Analyze and display PySR results."""
    console.print("\n[bold magenta]═══ 🧬 PySR RESULTS ═══[/]\n")

    # Get equations
    equations = model.equations_

    console.print("[bold]Pareto Front (complexity vs accuracy):[/]")
    table = Table(show_header=True)
    table.add_column("Complexity", style="cyan")
    table.add_column("Loss", style="green")
    table.add_column("Equation", style="yellow")

    for _, row in equations.iterrows():
        table.add_row(
            str(int(row['complexity'])),
            f"{row['loss']:.6f}",
            str(row['equation'])
        )

    console.print(table)

    # Best equation
    best_eq = model.sympy()
    console.print(f"\n[bold green]Best equation:[/]")
    console.print(f"  μ(d) = {best_eq}")

    # Predictions
    X_2d = X.reshape(-1, 1)
    y_pred = model.predict(X_2d)

    # Metrics
    mse = np.mean((y - y_pred)**2)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

    console.print(f"\n[cyan]Fit quality:[/]")
    console.print(f"  MSE: {mse:.8f}")
    console.print(f"  R²:  {r2:.4f}")

    return best_eq, y_pred, r2


def visualize_pysr(X, y, y_pred, best_eq, save_path="pysr_kernel.png"):
    """Visualize PySR fit."""
    console.print("\n[cyan]Creating visualization...[/]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Data vs PySR fit
    ax = axes[0]
    ax.scatter(X, y, alpha=0.5, s=10, label='Data', color='blue')
    sort_idx = np.argsort(X)
    ax.plot(X[sort_idx], y_pred[sort_idx], 'r-', linewidth=2, label='PySR fit')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Unfolded distance d')
    ax.set_ylabel('Mean attention μ(d)')
    ax.set_title(f'PySR Symbolic Fit\nμ(d) = {best_eq}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Residuals
    ax = axes[1]
    residuals = y - y_pred
    ax.scatter(X, residuals, alpha=0.5, s=10, color='purple')
    ax.axhline(0, color='k', linestyle='-', linewidth=1)
    ax.axhline(residuals.std(), color='r', linestyle='--', alpha=0.5)
    ax.axhline(-residuals.std(), color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Unfolded distance d')
    ax.set_ylabel('Residual')
    ax.set_title(f'Residuals (std={residuals.std():.6f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    console.print(f"[green]✅ Saved to {save_path}[/]")
    plt.close()


def compare_with_theory(X, y, best_eq):
    """Compare PySR result with theoretical kernels."""
    console.print("\n[bold]═══ COMPARISON WITH THEORY ═══[/]")

    # Sine kernel
    def sine_kernel(d, alpha=0.04):
        return 0.044 * np.sinc(alpha * d) + 0.006

    # GUE sine kernel (normalized)
    def gue_sine(d):
        d_safe = np.maximum(d, 0.01)
        return np.sin(np.pi * d_safe) / (np.pi * d_safe)

    y_sine = sine_kernel(X)
    y_gue = gue_sine(X)

    # Normalize for comparison
    y_norm = (y - y.mean()) / y.std()
    y_sine_norm = (y_sine - y_sine.mean()) / y_sine.std()
    y_gue_norm = (y_gue - y_gue.mean()) / y_gue.std()

    corr_sine = np.corrcoef(y_norm, y_sine_norm)[0, 1]
    corr_gue = np.corrcoef(y_norm, y_gue_norm)[0, 1]

    console.print(f"Correlation with scaled sinc (α=0.04): {corr_sine:.4f}")
    console.print(f"Correlation with GUE sine kernel:      {corr_gue:.4f}")

    return corr_sine, corr_gue


def main():
    # Extract data
    d_unf, attn = extract_kernel_data(n_samples=300)

    # Bin and smooth
    X, y = bin_and_smooth(d_unf, attn, n_bins=60)

    console.print(f"\n[cyan]Binned data: {len(X)} points[/]")

    # Run PySR
    model = run_pysr(X, y)

    # Analyze
    best_eq, y_pred, r2 = analyze_pysr_results(model, X, y)

    # Visualize
    visualize_pysr(X, y, y_pred, best_eq)

    # Compare with theory
    corr_sine, corr_gue = compare_with_theory(X, y, best_eq)

    # Summary
    console.print("\n")
    table = Table(title="🧬 PySR KERNEL SUMMARY", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Best equation", str(best_eq))
    table.add_row("R²", f"{r2:.4f}")
    table.add_row("Corr with sinc(0.04d)", f"{corr_sine:.4f}")
    table.add_row("Corr with GUE sinc(d)", f"{corr_gue:.4f}")

    console.print(table)

    console.print("\n[bold green]═══ COMPLETE ═══[/]")

    return model, best_eq


if __name__ == "__main__":
    model, best_eq = main()
