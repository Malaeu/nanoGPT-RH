#!/usr/bin/env python3
"""
extract_operator.py — Automated Operator Extraction from Attention Patterns

Extracts the learned kernel K(s_i, s_j) from trained SpacingMDN attention weights
and uses PySR to find symbolic formulas.

Usage:
    python scripts/extract_operator.py \
        --checkpoint checkpoints/E4_s7_best.pt \
        --data-dir data/continuous_2M \
        --output-dir results/operator_extraction \
        --n-samples 5000

What it does:
1. Load trained model and data
2. Extract attention weights from all layers/heads
3. Build kernel dataset: K(s_i, s_j, distance) -> attention weight
4. Run PySR symbolic regression
5. Compare with known GUE kernels (sine kernel, etc.)
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table

console = Console()


@dataclass
class KernelSample:
    """Single sample from attention kernel."""
    s_i: float       # Value at position i
    s_j: float       # Value at position j
    distance: int    # |i - j| (positional distance)
    K_ij: float      # Attention weight K(i, j)
    layer: int       # Which layer
    head: int        # Which head


def load_model_and_data(
    checkpoint_path: Path,
    data_dir: Path,
    device: torch.device
) -> Tuple[torch.nn.Module, torch.Tensor]:
    """Load trained model and validation data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from train_mdn_postfix import SpacingMDNPostfix
    from train_mdn import MDNConfig

    console.print(f"[cyan]Loading checkpoint: {checkpoint_path}[/]")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt['config']
    n_memory_slots = ckpt.get('n_memory_slots', 8)
    slot_id_mode = ckpt.get('slot_id_mode', 'fixed')
    content_mode = ckpt.get('content_mode', 'normal')
    use_aux_loss = ckpt.get('use_aux_loss', False)

    model = SpacingMDNPostfix(
        config,
        n_memory_slots=n_memory_slots,
        slot_id_mode=slot_id_mode,
        content_mode=content_mode,
        use_aux_loss=use_aux_loss
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    console.print(f"[green]Model loaded: {sum(p.numel() for p in model.parameters()):,} params[/]")

    # Load data
    val_path = data_dir / 'val.pt'
    console.print(f"[cyan]Loading data: {val_path}[/]")
    val_data = torch.load(val_path, weights_only=False)
    if isinstance(val_data, dict):
        val_data = val_data.get('data', val_data.get('spacings'))

    console.print(f"[green]Data shape: {val_data.shape}[/]")

    return model, val_data


def extract_attention_patterns(
    model: torch.nn.Module,
    data: torch.Tensor,
    device: torch.device,
    n_samples: int = 5000,
    batch_size: int = 64
) -> List[KernelSample]:
    """Extract attention patterns from model."""
    samples = []
    n_batches = min(n_samples // batch_size, len(data) // batch_size)

    console.print(f"[cyan]Extracting attention from {n_batches} batches...[/]")

    with torch.no_grad():
        for batch_idx in track(range(n_batches), description="Extracting"):
            # Get batch
            start = batch_idx * batch_size
            end = start + batch_size
            x = data[start:end, :-1].to(device)  # [B, T-1]

            # Forward with attention
            _, _, _, attentions = model(x, return_attention=True)
            # attentions: list of [B, n_heads, T+M, T+M] per layer

            # Extract samples from each layer and head
            for layer_idx, layer_attn in enumerate(attentions):
                # layer_attn: [B, n_heads, seq_len, seq_len]
                n_heads = layer_attn.shape[1]
                seq_len = layer_attn.shape[2]

                for head_idx in range(n_heads):
                    # Get attention weights for this head
                    attn = layer_attn[:, head_idx, :, :]  # [B, T, T]

                    # Sample random positions
                    for b in range(min(batch_size, 4)):  # Limit samples per batch
                        for i in range(0, seq_len - 1, 4):  # Skip some positions
                            for j in range(max(0, i - 20), i + 1):  # Look back up to 20
                                if j >= seq_len or i >= seq_len:
                                    continue

                                # Get values and attention weight
                                s_i = x[b, min(i, x.shape[1]-1)].item()
                                s_j = x[b, min(j, x.shape[1]-1)].item()
                                K_ij = attn[b, i, j].item()

                                samples.append(KernelSample(
                                    s_i=s_i,
                                    s_j=s_j,
                                    distance=abs(i - j),
                                    K_ij=K_ij,
                                    layer=layer_idx,
                                    head=head_idx
                                ))

            # Limit total samples
            if len(samples) >= n_samples * 10:
                break

    console.print(f"[green]Extracted {len(samples):,} kernel samples[/]")
    return samples


def build_kernel_dataset(
    samples: List[KernelSample],
    layer: Optional[int] = None,
    head: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Build X, y arrays for symbolic regression."""
    filtered = samples
    if layer is not None:
        filtered = [s for s in filtered if s.layer == layer]
    if head is not None:
        filtered = [s for s in filtered if s.head == head]

    X = np.array([[s.s_i, s.s_j, s.distance] for s in filtered])
    y = np.array([s.K_ij for s in filtered])

    return X, y


def run_pysr(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    max_complexity: int = 15,
    niterations: int = 50
) -> Dict:
    """Run PySR symbolic regression."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        console.print("[red]PySR not installed! Run: uv pip install pysr[/]")
        return {}

    console.print(f"[cyan]Running PySR on {len(X):,} samples...[/]")

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs"],
        constraints={
            "exp": 3,  # Limit exp complexity
            "log": 3,
        },
        maxsize=max_complexity,
        populations=15,
        population_size=50,
        ncycles_per_iteration=500,
        weight_optimize=0.01,
        adaptive_parsimony_scaling=1000.0,
        parsimony=0.001,
        progress=True,
        temp_equation_file=True,
        tempdir=str(output_dir / "pysr_temp"),
        random_state=42,
    )

    # Variable names for interpretability
    model.fit(X, y, variable_names=["s_i", "s_j", "d"])

    # Get best equations
    results = {
        "best_equation": str(model.get_best()),
        "best_loss": float(model.get_best().loss),
        "equations": []
    }

    for i, eq in enumerate(model.equations_):
        if hasattr(eq, 'equation'):
            results["equations"].append({
                "complexity": int(eq.complexity),
                "loss": float(eq.loss),
                "equation": str(eq.equation)
            })

    return results


def compare_with_gue(samples: List[KernelSample]) -> Dict:
    """Compare extracted kernel with known GUE kernels."""
    # Build dataset
    X, y = build_kernel_dataset(samples)
    s_i, s_j, d = X[:, 0], X[:, 1], X[:, 2]

    # Sine kernel: sin(π(s_i - s_j)) / (π(s_i - s_j))
    diff = s_i - s_j
    diff = np.where(np.abs(diff) < 1e-8, 1e-8, diff)  # Avoid division by zero
    sine_kernel = np.sin(np.pi * diff) / (np.pi * diff)
    sine_kernel = np.where(np.abs(diff) < 1e-8, 1.0, sine_kernel)

    # Distance-based decay: exp(-d/τ)
    tau_values = [5, 10, 20]
    decay_correlations = {}
    for tau in tau_values:
        exp_decay = np.exp(-d / tau)
        corr = np.corrcoef(y, exp_decay)[0, 1]
        decay_correlations[f"exp_decay_tau{tau}"] = float(corr)

    # Correlation with sine kernel
    sine_corr = np.corrcoef(y, sine_kernel)[0, 1]

    return {
        "sine_kernel_correlation": float(sine_corr) if not np.isnan(sine_corr) else 0.0,
        **decay_correlations,
        "mean_attention": float(np.mean(y)),
        "std_attention": float(np.std(y)),
    }


def visualize_kernel(
    samples: List[KernelSample],
    output_dir: Path,
    layer: int = 0,
    head: int = 0
):
    """Create visualization of extracted kernel."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed, skipping visualization[/]")
        return

    # Filter samples
    filtered = [s for s in samples if s.layer == layer and s.head == head]
    if not filtered:
        return

    X, y = build_kernel_dataset(filtered)
    s_i, s_j, d = X[:, 0], X[:, 1], X[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Attention vs distance
    ax = axes[0]
    ax.scatter(d, y, alpha=0.3, s=1)
    ax.set_xlabel("Distance |i - j|")
    ax.set_ylabel("Attention K(i, j)")
    ax.set_title(f"Attention vs Distance (Layer {layer}, Head {head})")

    # 2. Attention vs (s_i - s_j)
    ax = axes[1]
    diff = s_i - s_j
    ax.scatter(diff, y, alpha=0.3, s=1)
    ax.set_xlabel("s_i - s_j")
    ax.set_ylabel("Attention K(i, j)")
    ax.set_title("Attention vs Value Difference")

    # 3. 2D heatmap: K(s_i, s_j)
    ax = axes[2]
    # Bin the data
    bins = 30
    H, xedges, yedges = np.histogram2d(s_i, s_j, bins=bins, weights=y)
    counts, _, _ = np.histogram2d(s_i, s_j, bins=bins)
    H = np.divide(H, counts, where=counts > 0)

    im = ax.imshow(H.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='viridis')
    ax.set_xlabel("s_i")
    ax.set_ylabel("s_j")
    ax.set_title("Kernel K(s_i, s_j)")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / "kernel_visualization.png", dpi=150)
    plt.close()

    console.print(f"[green]Saved visualization to {output_dir}/kernel_visualization.png[/]")


def main():
    parser = argparse.ArgumentParser(description="Extract operator from attention patterns")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory with validation data")
    parser.add_argument("--output-dir", type=str, default="results/operator_extraction",
                       help="Output directory for results")
    parser.add_argument("--n-samples", type=int, default=5000,
                       help="Number of samples to extract")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for extraction")
    parser.add_argument("--run-pysr", action="store_true",
                       help="Run PySR symbolic regression (slow)")
    parser.add_argument("--pysr-iterations", type=int, default=50,
                       help="PySR iterations")
    parser.add_argument("--layer", type=int, default=None,
                       help="Specific layer to analyze (default: all)")
    parser.add_argument("--head", type=int, default=None,
                       help="Specific head to analyze (default: all)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load model and data
    model, data = load_model_and_data(
        Path(args.checkpoint),
        Path(args.data_dir),
        device
    )

    # Extract attention patterns
    samples = extract_attention_patterns(
        model, data, device,
        n_samples=args.n_samples,
        batch_size=args.batch_size
    )

    # Compare with GUE kernels
    console.print("\n[bold]Comparing with known GUE kernels...[/]")
    gue_comparison = compare_with_gue(samples)

    table = Table(title="GUE Kernel Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in gue_comparison.items():
        table.add_row(key, f"{value:.4f}")
    console.print(table)

    # Visualize
    visualize_kernel(
        samples, output_dir,
        layer=args.layer or 0,
        head=args.head or 0
    )

    # Run PySR if requested
    pysr_results = {}
    if args.run_pysr:
        X, y = build_kernel_dataset(samples, layer=args.layer, head=args.head)
        # Subsample for speed
        if len(X) > 10000:
            idx = np.random.choice(len(X), 10000, replace=False)
            X, y = X[idx], y[idx]

        pysr_results = run_pysr(
            X, y, output_dir,
            niterations=args.pysr_iterations
        )

        if pysr_results:
            console.print(f"\n[bold green]Best symbolic formula:[/]")
            console.print(f"  {pysr_results['best_equation']}")
            console.print(f"  Loss: {pysr_results['best_loss']:.6f}")

    # Save results
    results = {
        "checkpoint": str(args.checkpoint),
        "n_samples": len(samples),
        "gue_comparison": gue_comparison,
        "pysr_results": pysr_results,
    }

    results_path = output_dir / "extraction_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to {results_path}[/]")


if __name__ == "__main__":
    main()
