#!/usr/bin/env python3
"""
Masking Analysis for Flash MDN Model.

Progressive weight masking to find minimal core of knowledge.
Based on Masters (arXiv:2512.22238) approach.

Usage:
    python flash/masking_analysis_flash.py --ckpt out/mdn_memory_q3_flash/best.pt
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add flash to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_mdn_flash import MemoryMDN, MemoryMDNConfig
from mdn_flash import SpacingMDN, MDNConfig

console = Console()

# Flash-style optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def load_model_and_data(ckpt_path: Path, data_dir: Path, device: str):
    """Load Flash MDN model and residual data."""
    from torch.utils.data import DataLoader, TensorDataset

    console.print(f"[cyan]Loading checkpoint: {ckpt_path}[/]")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt['config']

    # Detect model type
    if isinstance(config, MemoryMDNConfig):
        model = MemoryMDN(config).to(device)
        model_type = "MemoryMDN"
    else:
        model = SpacingMDN(config).to(device)
        model_type = "SpacingMDN"

    model.load_state_dict(ckpt['model'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]{model_type}: {n_params:,} parameters[/]")

    # Load residual data
    console.print(f"[cyan]Loading data from {data_dir}[/]")
    val_data = torch.load(data_dir / "val.pt", weights_only=True)
    console.print(f"[green]Val data: {val_data.shape}[/]")

    # Create dataloader
    val_dataset = TensorDataset(val_data)
    batch_size = 256
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return model, config, val_loader, device


def mask_by_magnitude(model, ratio: float):
    """
    Mask weights with smallest magnitudes.
    Returns new model with masked weights.
    """
    masked_model = copy.deepcopy(model)

    if ratio == 0.0:
        return masked_model

    # Collect all weight magnitudes (only 2D+ weights)
    all_weights = []
    for name, param in masked_model.named_parameters():
        if param.dim() >= 2:
            all_weights.append(param.abs().flatten())

    all_weights = torch.cat(all_weights)

    # Find threshold
    threshold = torch.quantile(all_weights, ratio)

    # Apply mask
    total_masked = 0
    total_params = 0

    with torch.no_grad():
        for name, param in masked_model.named_parameters():
            if param.dim() >= 2:
                mask = param.abs() >= threshold
                param.data *= mask.float()
                total_masked += (~mask).sum().item()
                total_params += param.numel()

    return masked_model


@torch.no_grad()
def evaluate_nll(model, val_loader, device, max_batches: int = 0):
    """Evaluate NLL on residual data."""
    model.eval()
    total_nll = 0.0
    n_batches = 0

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for batch in val_loader:
        x = batch[0].to(device)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            result = model(x, targets=x)
            nll = result['nll']

        total_nll += nll.item()
        n_batches += 1

        if max_batches > 0 and n_batches >= max_batches:
            break

    return total_nll / n_batches


def run_masking_analysis(model, val_loader, device, ratios, max_batches: int = 0):
    """Run progressive masking analysis."""
    console.print("\n[bold cyan]═══ FLASH MASKING ANALYSIS ═══[/]\n")

    if max_batches > 0:
        console.print(f"[yellow]Using {max_batches} batches per eval[/]\n")

    results = []

    # Baseline
    baseline_nll = evaluate_nll(model, val_loader, device, max_batches)
    console.print(f"[green]Baseline NLL: {baseline_nll:.4f}[/]\n")

    for ratio in track(ratios, description="Testing mask ratios..."):
        masked_model = mask_by_magnitude(model, ratio)
        masked_model = masked_model.to(device)

        nll = evaluate_nll(masked_model, val_loader, device, max_batches)
        delta = nll - baseline_nll
        pct_change = (delta / abs(baseline_nll)) * 100

        results.append({
            'ratio': ratio,
            'active_pct': (1 - ratio) * 100,
            'nll': nll,
            'delta': delta,
            'pct_change': pct_change
        })

        # Status indicator
        if pct_change < -5:
            status = "[green]BETTER![/]"
        elif pct_change < 5:
            status = "[green]OK[/]"
        elif pct_change < 20:
            status = "[yellow]DEGRADED[/]"
        else:
            status = "[red]BROKEN[/]"

        console.print(f"  Mask {ratio*100:3.0f}%: NLL={nll:.4f} ({pct_change:+.1f}%) {status}")

        del masked_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return baseline_nll, results


def display_results(baseline_nll, results):
    """Display results in a nice table."""
    table = Table(title="Flash Masking Analysis Results")
    table.add_column("Mask %", justify="right")
    table.add_column("Active %", justify="right")
    table.add_column("NLL", justify="right")
    table.add_column("Δ NLL", justify="right")
    table.add_column("% Change", justify="right")
    table.add_column("Status", justify="center")

    best_idx = 0
    best_nll = results[0]['nll']

    for i, r in enumerate(results):
        if r['nll'] < best_nll:
            best_nll = r['nll']
            best_idx = i

        if r['pct_change'] < -5:
            status = "[bold green]BETTER[/]"
        elif r['pct_change'] < 5:
            status = "[green]OK[/]"
        elif r['pct_change'] < 20:
            status = "[yellow]DEGRADED[/]"
        else:
            status = "[red]BROKEN[/]"

        table.add_row(
            f"{r['ratio']*100:.0f}%",
            f"{r['active_pct']:.0f}%",
            f"{r['nll']:.4f}",
            f"{r['delta']:+.4f}",
            f"{r['pct_change']:+.1f}%",
            status
        )

    console.print(table)

    # Best result
    best = results[best_idx]
    console.print(f"\n[bold green]BEST: {best['ratio']*100:.0f}% masking[/]")
    console.print(f"  NLL: {best['nll']:.4f} ({best['pct_change']:+.1f}% vs baseline)")
    console.print(f"  Active: {best['active_pct']:.0f}% of weights")

    # Find critical point (where it breaks)
    for i, r in enumerate(results):
        if r['pct_change'] > 20:
            console.print(f"\n[bold red]CRITICAL POINT: {results[i-1]['ratio']*100:.0f}% masking[/]")
            console.print(f"  → {results[i-1]['active_pct']:.0f}% of weights contain the knowledge core")
            break


def main():
    parser = argparse.ArgumentParser(description="Masking Analysis for Flash MDN")
    parser.add_argument("--ckpt", type=Path, default=Path("out/mdn_memory_q3_flash/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/continuous_residuals"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ratios", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--max-batches", type=int, default=200, help="Max batches per eval (0=all)")
    args = parser.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]

    console.print(f"[bold]Device: {args.device}[/]")

    if args.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[cyan]GPU: {gpu_name} ({gpu_mem:.1f} GB)[/]")

    model, config, val_loader, device = load_model_and_data(
        args.ckpt, args.data_dir, args.device
    )

    baseline_nll, results = run_masking_analysis(
        model, val_loader, device, ratios, args.max_batches
    )

    display_results(baseline_nll, results)

    # Save results
    output_file = Path("results/masking_flash.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint': str(args.ckpt),
            'baseline_nll': baseline_nll,
            'results': results
        }, f, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/]")


if __name__ == "__main__":
    main()
