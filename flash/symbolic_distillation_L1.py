#!/usr/bin/env python3
"""
Symbolic Distillation from Flash Model Layer 1.

Extracts attention patterns from the critical Layer 1 and runs PySR
to find symbolic formulas for the GUE operator.

Layer 1 contains 8 heads, ALL critical:
  L1.H2: +312% NLL (main operator)
  L1.H3: +233%
  L1.H4: +120%
  L1.H7: +119%

Usage:
    python flash/symbolic_distillation_L1.py --ckpt out/mdn_memory_q3_flash/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from rich.console import Console
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parent))

from memory_mdn_flash import MemoryMDN, MemoryMDNConfig

console = Console()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_model_and_data(ckpt_path: Path, data_dir: Path, device: str):
    """Load Flash MDN model and data."""
    from torch.utils.data import DataLoader, TensorDataset

    console.print(f"[cyan]Loading checkpoint: {ckpt_path}[/]")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt['config']
    model = MemoryMDN(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    console.print(f"[green]MemoryMDN: {config.n_layer} layers x {config.n_head} heads[/]")

    val_data = torch.load(data_dir / "val.pt", weights_only=True)
    val_dataset = TensorDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    return model, config, val_loader, device


class AttentionHook:
    """Hook to capture attention weights from a layer."""

    def __init__(self):
        self.attention_weights = None
        self.q = None
        self.k = None
        self.v = None

    def hook_fn(self, module, input, output):
        """Capture Q, K before attention computation."""
        x = input[0]  # (B, L, C)
        B, L, C = x.shape

        # Get QKV
        qkv = module.c_attn(x)
        q, k, v = qkv.split(module.n_embd, dim=2)

        n_head = module.n_head
        head_dim = module.head_dim

        # Reshape: (B, L, H, D)
        q = q.view(B, L, n_head, head_dim)
        k = k.view(B, L, n_head, head_dim)

        # Transpose for attention: (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Compute attention scores (before softmax)
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)

        # Apply softmax
        attn = F.softmax(scores, dim=-1)

        self.attention_weights = attn.detach().cpu()
        self.q = q.detach().cpu()
        self.k = k.detach().cpu()


@torch.no_grad()
def extract_layer1_attention(model, val_loader, device, n_samples: int = 10000):
    """
    Extract attention patterns from Layer 1.

    Returns:
        distances: relative positions (|i-j|)
        attention_values: attention weights for each head
        predictions: model predictions (pi, mu, sigma)
    """
    console.print(f"[cyan]Extracting Layer 1 attention patterns...[/]")

    n_memory = model.config.n_memory
    n_head = model.config.n_head

    # Install hook on Layer 1 attention
    hook = AttentionHook()
    layer1_attn = model.blocks[1].attn
    handle = layer1_attn.register_forward_hook(hook.hook_fn)

    all_distances = []
    all_attention = {h: [] for h in range(n_head)}
    all_features = []
    all_predictions = []
    all_targets = []

    total = 0

    for batch in track(val_loader, description="Extracting..."):
        if total >= n_samples:
            break

        x = batch[0].to(device)
        B, T = x.shape

        # Forward pass (hook captures attention)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device=="cuda"):
            result = model(x, targets=x)

        # Get attention from hook (B, H, L, L) where L = K + T
        attn = hook.attention_weights  # (B, H, K+T, K+T)
        L = attn.shape[-1]
        K = n_memory

        # Focus on sequence-to-sequence attention (skip memory tokens)
        seq_attn = attn[:, :, K:, K:]  # (B, H, T, T)

        # Extract distance-based patterns
        for b in range(B):
            for i in range(T):
                for j in range(max(0, i-16), i):  # Look at last 16 positions
                    d = i - j  # Distance
                    all_distances.append(d)

                    for h in range(n_head):
                        all_attention[h].append(seq_attn[b, h, i, j].item())

        # Extract features and predictions
        pi = result['pi'][:, :-1].cpu()  # (B, T-1, K)
        mu = result['mu'][:, :-1].cpu()

        # Weighted mean prediction
        pred_mean = (pi * mu).sum(dim=-1)  # (B, T-1)

        # Features: input spacings
        features = x[:, :-1].cpu()  # (B, T-1)
        targets = x[:, 1:].cpu()    # (B, T-1)

        all_features.append(features.numpy())
        all_predictions.append(pred_mean.numpy())
        all_targets.append(targets.numpy())

        total += B

    handle.remove()

    # Convert to arrays
    distances = np.array(all_distances)
    attention = {h: np.array(all_attention[h]) for h in range(n_head)}
    features = np.concatenate(all_features).flatten()[:n_samples]
    predictions = np.concatenate(all_predictions).flatten()[:n_samples]
    targets = np.concatenate(all_targets).flatten()[:n_samples]

    console.print(f"[green]Extracted {len(distances):,} attention pairs[/]")
    console.print(f"[green]Extracted {len(features):,} predictions[/]")

    return distances, attention, features, predictions, targets


def analyze_attention_kernel(distances, attention, n_head):
    """Analyze attention as a function of distance."""
    console.print("\n[bold cyan]=== ATTENTION KERNEL ANALYSIS ===[/]")

    unique_d = np.unique(distances)

    for h in range(n_head):
        attn_h = attention[h]

        # Average attention by distance
        kernel = []
        for d in unique_d:
            mask = distances == d
            kernel.append(attn_h[mask].mean())
        kernel = np.array(kernel)

        # Find peak
        peak_d = unique_d[np.argmax(kernel)]
        peak_val = kernel.max()

        console.print(f"  H{h}: peak at d={peak_d}, max={peak_val:.4f}")

    return unique_d, {h: attention[h] for h in range(n_head)}


def run_symbolic_regression_on_predictions(features, predictions, targets, max_features: int = 8):
    """Run PySR to find formula for predictions."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        console.print("[red]PySR not installed! Run: pip install pysr[/]")
        console.print("[yellow]Then run: python -c 'import pysr; pysr.install()'[/]")
        return None

    console.print(f"\n[bold cyan]=== SYMBOLIC REGRESSION (Predictions) ===[/]")

    # Create feature matrix from sequence
    n = len(predictions)
    X = np.zeros((n - max_features, max_features))
    Y = predictions[max_features:]

    for i in range(n - max_features):
        X[i] = features[i:i+max_features]

    console.print(f"[dim]Features: {X.shape[1]} (last spacings)[/]")
    console.print(f"[dim]Samples: {len(X):,}[/]")

    feature_names = [f"s_{i}" for i in range(-max_features, 0)]

    import sympy as sp

    model = PySRRegressor(
        niterations=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "sin", "cos", "exp", "sqrt", "abs",
            "sinc(x) = x == 0 ? 1.0 : sin(3.14159 * x) / (3.14159 * x)",
        ],
        extra_sympy_mappings={
            "sinc": lambda x: sp.sinc(x),
        },
        complexity_of_operators={
            "sin": 3, "cos": 3, "exp": 2, "sqrt": 2,
            "sinc": 4,
        },
        maxsize=25,
        populations=15,
        population_size=40,
        ncycles_per_iteration=400,
        weight_optimize=0.001,
        parsimony=0.003,
        timeout_in_seconds=600,
        temp_equation_file=True,
        tempdir="/tmp/pysr_L1",
        delete_tempfiles=True,
        turbo=True,
        progress=True,
        verbosity=1,
    )

    console.print("[yellow]Starting symbolic regression (may take 5-10 min)...[/]")
    model.fit(X, Y, variable_names=feature_names)

    return model


def run_symbolic_regression_on_attention(distances, attention_h, head_idx: int):
    """Run PySR to find kernel formula K(d) for attention."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        console.print("[red]PySR not installed![/]")
        return None

    console.print(f"\n[bold cyan]=== SYMBOLIC REGRESSION (Attention H{head_idx}) ===[/]")

    # Subsample if too many points
    n_max = 50000
    if len(distances) > n_max:
        idx = np.random.choice(len(distances), n_max, replace=False)
        d_sub = distances[idx]
        a_sub = attention_h[idx]
    else:
        d_sub = distances
        a_sub = attention_h

    X = d_sub.reshape(-1, 1)
    Y = a_sub

    console.print(f"[dim]Samples: {len(X):,}[/]")
    console.print(f"[dim]Distance range: {d_sub.min()} to {d_sub.max()}[/]")

    import sympy as sp

    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=[
            "sin", "cos", "exp", "sqrt", "abs",
            "sinc(x) = x == 0 ? 1.0 : sin(3.14159 * x) / (3.14159 * x)",
        ],
        extra_sympy_mappings={
            "sinc": lambda x: sp.sinc(x),
        },
        complexity_of_operators={
            "sin": 3, "cos": 3, "exp": 2, "sqrt": 2,
            "sinc": 4, "^": 2,
        },
        maxsize=20,
        populations=10,
        population_size=30,
        ncycles_per_iteration=300,
        parsimony=0.005,
        timeout_in_seconds=300,
        temp_equation_file=True,
        tempdir=f"/tmp/pysr_attn_h{head_idx}",
        delete_tempfiles=True,
        turbo=True,
        progress=True,
        verbosity=1,
    )

    console.print(f"[yellow]Fitting attention kernel for H{head_idx}...[/]")
    model.fit(X, Y, variable_names=["d"])

    return model


def analyze_pysr_results(pysr_model, name: str):
    """Analyze and display PySR results."""
    if pysr_model is None:
        return None

    console.print(f"\n[bold green]=== RESULTS: {name} ===[/]")

    equations = pysr_model.equations_
    console.print("\n[bold]Top Equations:[/]")
    for i, row in equations.head(8).iterrows():
        console.print(f"  {row['complexity']:2d}: {row['equation']:<50} (loss={row['loss']:.6f})")

    best = pysr_model.get_best()
    console.print(f"\n[bold green]Best: {best}[/]")

    sympy_expr = pysr_model.sympy()
    console.print(f"[bold]Sympy: {sympy_expr}[/]")

    return {
        'name': name,
        'best_equation': str(best),
        'sympy': str(sympy_expr),
        'loss': float(equations.iloc[-1]['loss']),
    }


def main():
    parser = argparse.ArgumentParser(description="Symbolic Distillation from Layer 1")
    parser.add_argument("--ckpt", type=Path, default=Path("out/mdn_memory_q3_flash/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/continuous_residuals"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--target-head", type=int, default=2, help="Focus on this head (L1.H2 most critical)")
    args = parser.parse_args()

    console.print(f"[bold]Device: {args.device}[/]")

    if args.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[cyan]GPU: {gpu_name}[/]")

    model, config, val_loader, device = load_model_and_data(
        args.ckpt, args.data_dir, args.device
    )

    # Extract attention and predictions
    distances, attention, features, predictions, targets = extract_layer1_attention(
        model, val_loader, device, n_samples=args.n_samples
    )

    # Analyze attention kernel
    unique_d, attn_by_head = analyze_attention_kernel(distances, attention, config.n_head)

    all_results = []

    # 1. Symbolic regression on predictions
    pred_model = run_symbolic_regression_on_predictions(
        features, predictions, targets, max_features=args.max_features
    )
    if pred_model:
        res = analyze_pysr_results(pred_model, "Predictions")
        if res:
            all_results.append(res)
            # Save equations
            pred_model.equations_.to_csv("results/L1_prediction_equations.csv", index=False)

    # 2. Symbolic regression on attention kernel (focus on most critical head)
    attn_model = run_symbolic_regression_on_attention(
        distances, attention[args.target_head], args.target_head
    )
    if attn_model:
        res = analyze_pysr_results(attn_model, f"Attention H{args.target_head}")
        if res:
            all_results.append(res)
            attn_model.equations_.to_csv(f"results/L1_attention_H{args.target_head}_equations.csv", index=False)

    # Save summary
    output_file = Path("results/symbolic_L1_flash.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint': str(args.ckpt),
            'n_samples': args.n_samples,
            'target_head': args.target_head,
            'results': all_results,
        }, f, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/]")


if __name__ == "__main__":
    main()
