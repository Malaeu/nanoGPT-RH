#!/usr/bin/env python3
"""
Masking Analysis for Operator Extraction.

Progressive weight masking to find minimal core of knowledge.
Based on Masters (arXiv:2512.22238) approach.

Usage:
    python scripts/masking_analysis.py --ckpt checkpoints/E4_s7_best.pt
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

console = Console()

# === Flash-style optimizations (Ampere+ GPUs) ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def load_model_and_data(ckpt_path: Path, data_dir: Path, device: str):
    """Load E4 model and validation data using POSTFIX architecture."""
    from train_mdn_postfix import SpacingMDNPostfix, SpacingNextDataset
    from torch.utils.data import DataLoader

    console.print(f"[cyan]Loading checkpoint: {ckpt_path}[/]")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt['config']
    n_memory = ckpt.get('n_memory_slots', 8)
    slot_id_mode = ckpt.get('slot_id_mode', 'none')
    content_mode = ckpt.get('content_mode', 'normal')

    # Build POSTFIX model (includes memory_bank inside!)
    model = SpacingMDNPostfix(
        config,
        n_memory_slots=n_memory,
        slot_id_mode=slot_id_mode,
        content_mode=content_mode
    ).to(device)

    # Load all weights (strict=False to ignore aux_head if not present)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    # CRITICAL: Fix slot IDs during eval (permute_per_batch breaks inference!)
    if hasattr(model, 'memory_bank') and hasattr(model.memory_bank, 'eval_slot_id_mode'):
        model.memory_bank.eval_slot_id_mode = 'fixed'
        console.print(f"[yellow]Set eval_slot_id_mode='fixed' for stable inference[/]")

    # Load data using SAME dataset as training (sliding windows!)
    console.print(f"[cyan]Loading validation data...[/]")
    val_raw = torch.load(data_dir / "val.pt", weights_only=False)

    # CRITICAL FIX: config.seq_len ≠ dataset seq_len!
    # In training: config.seq_len = args.seq_len - 1 + n_memory_slots (for positional embeddings)
    # Dataset uses args.seq_len (input + target)
    # So: correct_seq_len = config.seq_len + 1 - n_memory
    correct_seq_len = config.seq_len + 1 - n_memory
    console.print(f"[yellow]Using correct seq_len={correct_seq_len} (config.seq_len={config.seq_len}, n_memory={n_memory})[/]")
    val_dataset = SpacingNextDataset(val_raw, seq_len=correct_seq_len)
    # Larger batch for GPU (512), num_workers=4 for faster loading
    batch_size = 512 if device == "cuda" else 256
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4 if device == "cuda" else 0)
    console.print(f"[green]Data loaded: {len(val_dataset):,} samples (sliding window)[/]")

    return model, config, val_loader, device


def mask_by_magnitude(model, ratio: float):
    """
    Mask weights with smallest magnitudes.
    Returns new model with masked weights.
    """
    masked_model = copy.deepcopy(model)

    if ratio == 0.0:
        return masked_model

    # Collect all weight magnitudes (only 2D+ weights, skip biases/norms)
    all_weights = []
    for name, param in masked_model.named_parameters():
        if param.dim() >= 2:  # Weight matrices only
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
    """
    Evaluate NLL on data using the same method as training.

    POSTFIX model predicts ONE next spacing per sequence (bottleneck design).
    Uses BFloat16 on CUDA for speed.

    Args:
        max_batches: Limit eval to N batches (0 = all)
    """
    from train_mdn_postfix import mdn_loss_1step

    model.eval()
    total_nll = 0.0
    n_batches = 0

    # BFloat16 autocast for CUDA (2-3x faster inference)
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            # POSTFIX model predicts ONE next spacing
            pi, mu, sigma = model(x)  # each [B, 1, K]

            # Use same loss function as training
            nll = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=0)

        total_nll += nll.item()
        n_batches += 1

        # Early stop if max_batches reached
        if max_batches > 0 and n_batches >= max_batches:
            break

    return total_nll / n_batches


def run_masking_analysis(model, val_loader, device, ratios, max_batches: int = 0):
    """Run progressive masking analysis."""
    console.print("\n[bold cyan]═══ MASKING ANALYSIS ═══[/]\n")

    if max_batches > 0:
        console.print(f"[yellow]Using {max_batches} batches per eval (~{max_batches * 512:,} samples)[/]\n")

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

        # Free memory
        del masked_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return baseline_nll, results


def display_results(baseline_nll, results):
    """Display results in a nice table."""
    table = Table(title="Masking Analysis Results")
    table.add_column("Mask %", justify="right")
    table.add_column("Active %", justify="right")
    table.add_column("NLL", justify="right")
    table.add_column("Δ NLL", justify="right")
    table.add_column("% Change", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status = ""
        if r['pct_change'] < 1:
            status = "[green]OK[/]"
        elif r['pct_change'] < 5:
            status = "[yellow]SLIGHT[/]"
        elif r['pct_change'] < 20:
            status = "[orange1]DEGRADED[/]"
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

    # Find critical point
    for i, r in enumerate(results):
        if r['pct_change'] > 10:
            console.print(f"\n[bold red]CRITICAL POINT: {results[i-1]['ratio']*100:.0f}% masking[/]")
            console.print(f"[yellow]→ {results[i-1]['active_pct']:.0f}% of weights contain ~{100 - results[i-1]['pct_change']:.0f}% of knowledge[/]")
            break


def main():
    parser = argparse.ArgumentParser(description="Masking Analysis for Operator Extraction")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/E4_s7_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/continuous_2M"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps")
    parser.add_argument("--ratios", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--max-batches", type=int, default=500, help="Max batches per eval (0=all)")
    args = parser.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]

    console.print(f"[bold]Device: {args.device}[/]")

    # GPU info
    if args.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[cyan]GPU: {gpu_name} ({gpu_mem:.1f} GB)[/]")
        console.print(f"[cyan]Optimizations: TF32 + BF16 + cuDNN benchmark[/]")

    model, config, val_loader, device = load_model_and_data(
        args.ckpt, args.data_dir, args.device
    )

    baseline_nll, results = run_masking_analysis(
        model, val_loader, device, ratios, args.max_batches
    )

    display_results(baseline_nll, results)

    # Save results
    output_file = Path("results/masking_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_file, 'w') as f:
        json.dump({
            'baseline_nll': baseline_nll,
            'results': results
        }, f, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/]")


if __name__ == "__main__":
    main()
