#!/usr/bin/env python3
"""
Extract Attention Patterns from Flash Model Layer 1.

This script extracts attention patterns without PySR dependency.
Focus on L1.H2 (the most critical head: +312% NLL when removed).

Usage:
    python flash/extract_L1_patterns.py --ckpt out/mdn_memory_q3_flash/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


class AttentionExtractor:
    """Extract attention weights from multiple layers."""

    def __init__(self, model, target_layers: list = [1]):
        self.model = model
        self.target_layers = target_layers
        self.attention_weights = {}
        self.hooks = []

        # Install hooks
        for layer_idx in target_layers:
            hook = self._make_hook(layer_idx)
            handle = model.blocks[layer_idx].attn.register_forward_hook(hook)
            self.hooks.append(handle)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            x = input[0]  # (B, L, C)
            B, L, C = x.shape

            # Get QKV
            qkv = module.c_attn(x)
            q, k, v = qkv.split(module.n_embd, dim=2)

            n_head = module.n_head
            head_dim = module.head_dim

            # Reshape: (B, L, H, D) -> (B, H, L, D)
            q = q.view(B, L, n_head, head_dim).transpose(1, 2)
            k = k.view(B, L, n_head, head_dim).transpose(1, 2)

            # Compute attention scores
            scale = 1.0 / (head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)

            # Apply softmax
            attn = F.softmax(scores, dim=-1)

            self.attention_weights[layer_idx] = attn.detach().cpu()

        return hook_fn

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def extract_patterns(model, val_loader, device, n_samples: int = 5000):
    """Extract attention patterns and predictions."""
    console.print(f"[cyan]Extracting Layer 1 patterns from {n_samples} samples...[/]")

    n_memory = model.config.n_memory
    n_head = model.config.n_head
    n_layer = model.config.n_layer

    # Extract from Layer 1 (critical) and Layer 0 (for comparison)
    extractor = AttentionExtractor(model, target_layers=[0, 1])

    # Storage
    all_distances = []
    attention_by_layer_head = {
        (l, h): [] for l in [0, 1] for h in range(n_head)
    }
    all_features = []
    all_predictions = []
    all_targets = []

    total = 0

    for batch in track(val_loader, description="Extracting..."):
        if total >= n_samples:
            break

        x = batch[0].to(device)
        B, T = x.shape

        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device=="cuda"):
            result = model(x, targets=x)

        # Process attention from each layer
        for layer_idx in [0, 1]:
            attn = extractor.attention_weights[layer_idx]  # (B, H, L, L)
            L = attn.shape[-1]
            K = n_memory

            # Focus on sequence-to-sequence attention
            seq_attn = attn[:, :, K:, K:]  # (B, H, T, T)

            # Extract distance-based patterns (sample subset)
            for b in range(min(B, 4)):  # Sample 4 per batch
                for i in range(16, T, 8):  # Sample every 8th position
                    for j in range(max(0, i-16), i):
                        d = i - j
                        if layer_idx == 1:  # Only store distances once
                            all_distances.append(d)

                        for h in range(n_head):
                            attention_by_layer_head[(layer_idx, h)].append(
                                seq_attn[b, h, i, j].item()
                            )

        # Extract predictions
        pi = result['pi'][:, :-1].cpu()
        mu = result['mu'][:, :-1].cpu()
        pred_mean = (pi * mu).sum(dim=-1)

        all_features.append(x[:, :-1].cpu().numpy())
        all_predictions.append(pred_mean.numpy())
        all_targets.append(x[:, 1:].cpu().numpy())

        total += B

    extractor.remove_hooks()

    # Convert
    distances = np.array(all_distances)
    features = np.concatenate(all_features).flatten()
    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()

    console.print(f"[green]Extracted {len(distances):,} attention pairs[/]")

    return distances, attention_by_layer_head, features, predictions, targets


def analyze_attention_kernel(distances, attention_by_layer_head, n_head: int):
    """Analyze attention as K(d) kernel for each head."""
    console.print("\n[bold cyan]=== ATTENTION KERNEL ANALYSIS ===[/]\n")

    unique_d = np.unique(distances)
    results = {}

    for layer_idx in [0, 1]:
        console.print(f"[bold]Layer {layer_idx}:[/]")

        for h in range(n_head):
            attn_h = np.array(attention_by_layer_head[(layer_idx, h)])

            # Average attention by distance
            kernel = []
            kernel_std = []
            for d in unique_d:
                mask = distances == d
                vals = attn_h[mask]
                kernel.append(vals.mean())
                kernel_std.append(vals.std())

            kernel = np.array(kernel)
            kernel_std = np.array(kernel_std)

            # Find characteristics
            peak_idx = np.argmax(kernel)
            peak_d = unique_d[peak_idx]
            peak_val = kernel[peak_idx]

            # Decay rate (fit exponential)
            if len(unique_d) > 3:
                from scipy.optimize import curve_fit
                try:
                    def exp_decay(x, a, b, c):
                        return a * np.exp(-b * x) + c

                    popt, _ = curve_fit(exp_decay, unique_d, kernel,
                                       p0=[0.5, 0.3, 0.01], maxfev=1000)
                    decay_rate = popt[1]
                except:
                    decay_rate = np.nan
            else:
                decay_rate = np.nan

            results[(layer_idx, h)] = {
                'kernel': kernel,
                'kernel_std': kernel_std,
                'peak_d': peak_d,
                'peak_val': peak_val,
                'decay_rate': decay_rate,
            }

            # Print summary
            if layer_idx == 1:  # Focus on Layer 1
                console.print(f"  H{h}: peak d={peak_d:2d}, val={peak_val:.4f}, decay={decay_rate:.3f}")

    return unique_d, results


def fit_sinc_kernel(distances, attention_h):
    """Try to fit sinc²(πd/λ) kernel (GUE signature)."""
    console.print("\n[bold cyan]=== SINC KERNEL FIT (GUE Test) ===[/]")

    unique_d = np.unique(distances)
    attn_h = np.array(attention_h)

    # Average by distance
    kernel = []
    for d in unique_d:
        mask = distances == d
        kernel.append(attn_h[mask].mean())
    kernel = np.array(kernel)

    # Normalize
    kernel = kernel / kernel.max()

    # Fit sinc²(π*d/λ)
    from scipy.optimize import curve_fit

    def sinc2(d, scale, offset, wavelength):
        x = np.pi * d / wavelength
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(x == 0, 1.0, np.sin(x) / x)
        return scale * s**2 + offset

    try:
        popt, pcov = curve_fit(sinc2, unique_d[unique_d > 0], kernel[unique_d > 0],
                               p0=[0.8, 0.1, 3.0], maxfev=2000)
        scale, offset, wavelength = popt

        # R² score
        y_pred = sinc2(unique_d[unique_d > 0], *popt)
        y_true = kernel[unique_d > 0]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        console.print(f"[green]sinc²(πd/{wavelength:.2f}) fit:[/]")
        console.print(f"  scale={scale:.4f}, offset={offset:.4f}, λ={wavelength:.3f}")
        console.print(f"  R² = {r2:.4f}")

        if abs(wavelength - 1.0) < 0.3:
            console.print(f"  [bold green]λ ≈ 1: Consistent with GUE sine kernel![/]")

        return {
            'scale': scale,
            'offset': offset,
            'wavelength': wavelength,
            'r2': r2,
        }

    except Exception as e:
        console.print(f"[red]Sinc fit failed: {e}[/]")
        return None


def analyze_predictions(features, predictions, targets):
    """Analyze prediction patterns."""
    console.print("\n[bold cyan]=== PREDICTION ANALYSIS ===[/]")

    # Correlation
    corr = np.corrcoef(predictions, targets)[0, 1]
    mse = np.mean((predictions - targets) ** 2)

    console.print(f"Prediction-Target correlation: {corr:.4f}")
    console.print(f"MSE: {mse:.6f}")

    # Look for patterns in residuals
    # Note: Flash uses RESIDUALS (s - 1.0), so mean ≈ 0, not 1
    n = len(features)
    if n > 100:
        # Create lagged features (residuals)
        r1 = features[3:]      # r_{n-1}
        r2 = features[2:-1]    # r_{n-2}
        r3 = features[1:-2]    # r_{n-3}
        y = targets[3:]        # r_n

        console.print(f"\n[bold]Residual statistics:[/]")
        console.print(f"  Mean: {features.mean():.4f} (should be ~0)")
        console.print(f"  Std:  {features.std():.4f}")

        # Test 1: Linear combination
        from sklearn.linear_model import LinearRegression
        X_lin = np.column_stack([r1, r2, r3])
        lr = LinearRegression().fit(X_lin, y)
        y_lin = lr.predict(X_lin)
        corr_lin = np.corrcoef(y_lin, y)[0, 1]

        console.print(f"\nLinear fit: r_n ≈ a*r₋₁ + b*r₋₂ + c*r₋₃")
        console.print(f"  Coefficients: [{lr.coef_[0]:.4f}, {lr.coef_[1]:.4f}, {lr.coef_[2]:.4f}]")
        console.print(f"  Intercept: {lr.intercept_:.4f}")
        console.print(f"  Correlation: {corr_lin:.4f}")

        # Test 2: GUE repulsion - negative correlation with previous
        corr_r1 = np.corrcoef(r1, y)[0, 1]
        corr_r2 = np.corrcoef(r2, y)[0, 1]
        corr_r3 = np.corrcoef(r3, y)[0, 1]

        console.print(f"\nLag correlations (GUE expects negative = repulsion):")
        console.print(f"  corr(r_n, r₋₁): {corr_r1:.4f}")
        console.print(f"  corr(r_n, r₋₂): {corr_r2:.4f}")
        console.print(f"  corr(r_n, r₋₃): {corr_r3:.4f}")

        # Test 3: Sum constraint (for raw spacings, sum of 3 ≈ π)
        # For residuals: sum of 3 residuals ≈ 0?
        sum_3 = r1 + r2 + r3
        console.print(f"\nSum of 3 residuals: mean={sum_3.mean():.4f}, std={sum_3.std():.4f}")

    return {
        'pred_target_corr': corr,
        'mse': mse,
    }


def plot_results(unique_d, kernel_results, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Attention kernels for Layer 1
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Layer 1 Attention Kernels K(d)", fontsize=14)

    for h in range(8):
        ax = axes[h // 4, h % 4]
        res = kernel_results.get((1, h))
        if res:
            ax.plot(unique_d, res['kernel'], 'b-', linewidth=2)
            ax.fill_between(unique_d,
                           res['kernel'] - res['kernel_std'],
                           res['kernel'] + res['kernel_std'],
                           alpha=0.3)
            ax.set_title(f"H{h} (peak d={res['peak_d']})")
            ax.set_xlabel("Distance d")
            ax.set_ylabel("Attention")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "L1_attention_kernels.png", dpi=150)
    console.print(f"[green]Saved: {output_dir}/L1_attention_kernels.png[/]")

    # Plot 2: Compare L0 vs L1 (most critical heads)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # L0 most critical (from E4 analysis: H3)
    ax = axes[0]
    for h in [0, 3, 5]:  # E4 critical heads
        res = kernel_results.get((0, h))
        if res:
            ax.plot(unique_d, res['kernel'], label=f"H{h}")
    ax.set_title("Layer 0 Attention")
    ax.set_xlabel("Distance d")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # L1 most critical (H2)
    ax = axes[1]
    for h in [2, 3, 4, 7]:  # Flash critical heads
        res = kernel_results.get((1, h))
        if res:
            ax.plot(unique_d, res['kernel'], label=f"H{h}")
    ax.set_title("Layer 1 Attention (Critical)")
    ax.set_xlabel("Distance d")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "L0_vs_L1_comparison.png", dpi=150)
    console.print(f"[green]Saved: {output_dir}/L0_vs_L1_comparison.png[/]")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Extract Layer 1 Patterns")
    parser.add_argument("--ckpt", type=Path, default=Path("out/mdn_memory_q3_flash/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/continuous_residuals"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/L1_analysis"))
    args = parser.parse_args()

    console.print(f"[bold]Device: {args.device}[/]")

    model, config, val_loader, device = load_model_and_data(
        args.ckpt, args.data_dir, args.device
    )

    # Extract patterns
    distances, attention_by_lh, features, predictions, targets = extract_patterns(
        model, val_loader, device, n_samples=args.n_samples
    )

    # Analyze attention kernels
    unique_d, kernel_results = analyze_attention_kernel(
        distances, attention_by_lh, config.n_head
    )

    # Try sinc kernel fit on most critical head (L1.H2)
    sinc_result = fit_sinc_kernel(distances, attention_by_lh[(1, 2)])

    # Analyze predictions
    pred_results = analyze_predictions(features, predictions, targets)

    # Create plots
    plot_results(unique_d, kernel_results, args.output_dir)

    # Save numerical results
    import json

    def to_python(obj):
        """Convert numpy types to Python types."""
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        return obj

    results = {
        'checkpoint': str(args.ckpt),
        'n_samples': args.n_samples,
        'sinc_fit': to_python(sinc_result),
        'prediction_analysis': to_python(pred_results),
        'kernel_summary': {
            f"L{l}_H{h}": {
                'peak_d': int(r['peak_d']),
                'peak_val': float(r['peak_val']),
                'decay_rate': float(r['decay_rate']) if not np.isnan(r['decay_rate']) else None,
            }
            for (l, h), r in kernel_results.items()
        }
    }

    output_file = args.output_dir / "analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/]")


if __name__ == "__main__":
    main()
