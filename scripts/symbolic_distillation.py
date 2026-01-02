#!/usr/bin/env python3
"""
Symbolic Distillation via PySR.

Extract symbolic formula from trained transformer predictions.
Based on Masters paper approach.

Usage:
    python scripts/symbolic_distillation.py --ckpt checkpoints/E4_s7_best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from rich.console import Console
from rich.progress import track

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

console = Console()

# GPU optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


def load_model_and_data(ckpt_path: Path, data_dir: Path, device: str):
    """Load model and data."""
    from train_mdn_postfix import SpacingMDNPostfix, SpacingNextDataset
    from torch.utils.data import DataLoader

    console.print(f"[cyan]Loading checkpoint: {ckpt_path}[/]")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt['config']
    n_memory = ckpt.get('n_memory_slots', 8)
    slot_id_mode = ckpt.get('slot_id_mode', 'none')

    model = SpacingMDNPostfix(
        config,
        n_memory_slots=n_memory,
        slot_id_mode=slot_id_mode,
        content_mode='normal'
    ).to(device)

    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    if hasattr(model, 'memory_bank'):
        model.memory_bank.eval_slot_id_mode = 'fixed'

    # Load data
    val_raw = torch.load(data_dir / "val.pt", weights_only=False)
    correct_seq_len = config.seq_len + 1 - n_memory
    val_dataset = SpacingNextDataset(val_raw, seq_len=correct_seq_len)

    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

    return model, config, val_loader, device


@torch.no_grad()
def extract_predictions(model, val_loader, device, n_samples: int = 50000, context_len: int = 16):
    """
    Extract (features, predictions) pairs from model.

    Features: last K spacings before prediction
    Target: model's predicted mean (weighted by mixture weights)
    """
    console.print(f"[cyan]Extracting {n_samples:,} predictions...[/]")

    X_list = []  # Features: [s_{n-K}, ..., s_{n-1}]
    Y_pred_list = []  # Model predictions
    Y_true_list = []  # Actual next spacings

    total = 0

    for x, y in track(val_loader, description="Extracting..."):
        if total >= n_samples:
            break

        x, y = x.to(device), y.to(device)
        B, T = x.shape

        # Get model prediction
        pi, mu, sigma = model(x)

        # Weighted mean prediction
        pi = pi.squeeze(1)  # [B, K]
        mu = mu.squeeze(1)  # [B, K]
        pred_mean = (pi * mu).sum(dim=-1)  # [B]

        # Features: last context_len spacings
        features = x[:, -context_len:].cpu().numpy()  # [B, context_len]

        X_list.append(features)
        Y_pred_list.append(pred_mean.cpu().numpy())
        Y_true_list.append(y.cpu().numpy())

        total += B

    X = np.concatenate(X_list)[:n_samples]
    Y_pred = np.concatenate(Y_pred_list)[:n_samples]
    Y_true = np.concatenate(Y_true_list)[:n_samples]

    console.print(f"[green]Extracted {len(X):,} samples[/]")
    console.print(f"[dim]Features shape: {X.shape}[/]")
    console.print(f"[dim]Pred mean: {Y_pred.mean():.4f}, std: {Y_pred.std():.4f}[/]")
    console.print(f"[dim]True mean: {Y_true.mean():.4f}, std: {Y_true.std():.4f}[/]")

    return X, Y_pred, Y_true


def run_symbolic_regression(X, Y, max_features: int = 8):
    """
    Run PySR symbolic regression.

    Args:
        X: [N, context_len] features
        Y: [N] targets (model predictions)
        max_features: use only last K features to reduce complexity
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        console.print("[red]PySR not installed! Run: pip install pysr[/]")
        console.print("[yellow]Then run: python -c 'import pysr; pysr.install()'[/]")
        return None

    # Use only last max_features to reduce search space
    X_reduced = X[:, -max_features:]

    console.print(f"\n[bold cyan]═══ SYMBOLIC REGRESSION ═══[/]")
    console.print(f"[dim]Features: {X_reduced.shape[1]} (last spacings)[/]")
    console.print(f"[dim]Samples: {len(X_reduced):,}[/]")

    # Feature names
    feature_names = [f"s_{i}" for i in range(-max_features, 0)]

    # PySR config optimized for GUE-like operators (sinc, power, etc.)
    # sinc(x) = sin(pi*x)/(pi*x) is key for GUE pair correlation!
    # NOTE: Use ASCII "pi" not Unicode "π" for Julia compatibility
    import sympy as sp

    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/", "^"],  # Added power
        unary_operators=[
            "sin", "cos", "exp", "log", "sqrt", "abs",
            "sinc(x) = x == 0 ? 1.0 : sin(pi * x) / (pi * x)",  # GUE kernel!
        ],
        extra_sympy_mappings={
            "sinc": lambda x: sp.sinc(x),  # sympy has sinc built-in
        },
        extra_jax_mappings={},  # Disable JAX if not installed
        extra_torch_mappings={},  # Disable Torch mapping
        complexity_of_operators={
            "sin": 3, "cos": 3, "exp": 2, "log": 2, "sqrt": 2,
            "sinc": 4,  # Encourage sinc discovery (GUE!)
            "^": 2,     # Power operator
        },
        maxsize=30,  # Increased for complex GUE formulas
        populations=20,
        population_size=50,
        ncycles_per_iteration=500,
        weight_optimize=0.001,
        adaptive_parsimony_scaling=1000,
        parsimony=0.0025,  # Slightly lower to allow complexity
        early_stop_condition="stop_if(loss, complexity) = loss < 1e-6 && complexity < 12",
        timeout_in_seconds=900,  # 15 min for deeper search
        temp_equation_file=True,
        tempdir="/tmp/pysr_temp",
        delete_tempfiles=True,
        turbo=True,  # Speed optimization
        progress=True,
        verbosity=1,
    )

    console.print("[yellow]Starting symbolic regression (this may take 5-10 minutes)...[/]")
    model.fit(X_reduced, Y, variable_names=feature_names)

    return model


def analyze_results(pysr_model, X, Y_pred, Y_true):
    """Analyze and display results."""
    console.print("\n[bold cyan]═══ RESULTS ═══[/]")

    # Best equations
    console.print("\n[bold]Top Equations (by complexity):[/]")
    equations = pysr_model.equations_
    for i, row in equations.head(10).iterrows():
        console.print(f"  {row['complexity']:2d}: {row['equation']:<50} (loss={row['loss']:.6f})")

    # Best equation
    best = pysr_model.get_best()
    console.print(f"\n[bold green]Best Formula:[/]")
    console.print(f"  {best}")

    # Sympy version
    sympy_expr = pysr_model.sympy()
    console.print(f"\n[bold]Sympy:[/]")
    console.print(f"  {sympy_expr}")

    # Evaluate
    X_reduced = X[:, -len(pysr_model.feature_names_in_):]
    Y_symbolic = pysr_model.predict(X_reduced)

    # Correlation with model predictions
    corr_pred = np.corrcoef(Y_symbolic, Y_pred)[0, 1]
    console.print(f"\n[bold]Correlation with model predictions:[/] {corr_pred:.4f}")

    # Correlation with true values
    corr_true = np.corrcoef(Y_symbolic, Y_true)[0, 1]
    console.print(f"[bold]Correlation with true values:[/] {corr_true:.4f}")

    # MSE
    mse_pred = np.mean((Y_symbolic - Y_pred) ** 2)
    mse_true = np.mean((Y_symbolic - Y_true) ** 2)
    console.print(f"[bold]MSE vs model:[/] {mse_pred:.6f}")
    console.print(f"[bold]MSE vs true:[/] {mse_true:.6f}")

    return {
        'best_equation': str(best),
        'sympy': str(sympy_expr),
        'corr_with_model': corr_pred,
        'corr_with_true': corr_true,
        'mse_vs_model': mse_pred,
        'mse_vs_true': mse_true
    }


def main():
    parser = argparse.ArgumentParser(description="Symbolic Distillation")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/E4_s7_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/continuous_2M"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=50000, help="Samples for regression")
    parser.add_argument("--context-len", type=int, default=16, help="Context window for features")
    parser.add_argument("--max-features", type=int, default=8, help="Max features for PySR")
    args = parser.parse_args()

    console.print(f"[bold]Device: {args.device}[/]")

    model, config, val_loader, device = load_model_and_data(
        args.ckpt, args.data_dir, args.device
    )

    X, Y_pred, Y_true = extract_predictions(
        model, val_loader, device,
        n_samples=args.n_samples,
        context_len=args.context_len
    )

    pysr_model = run_symbolic_regression(X, Y_pred, max_features=args.max_features)

    if pysr_model is None:
        return

    results = analyze_results(pysr_model, X, Y_pred, Y_true)

    # Save
    output_file = Path("results/symbolic_distillation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/]")

    # Save equations CSV
    equations_file = Path("results/symbolic_equations.csv")
    pysr_model.equations_.to_csv(equations_file, index=False)
    console.print(f"[green]Equations saved to {equations_file}[/]")


if __name__ == "__main__":
    main()
