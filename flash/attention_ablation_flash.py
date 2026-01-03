#!/usr/bin/env python3
"""
Attention Head Ablation for Flash MDN Model.

Identifies which attention heads are critical for the model.

Usage:
    python flash/attention_ablation_flash.py --ckpt out/mdn_memory_q3_flash/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from memory_mdn_flash import MemoryMDN, MemoryMDNConfig
from mdn_flash import SpacingMDN, MDNConfig

console = Console()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_model_and_data(ckpt_path: Path, data_dir: Path, device: str):
    """Load Flash MDN model and data."""
    from torch.utils.data import DataLoader, TensorDataset

    console.print(f"[cyan]Loading checkpoint: {ckpt_path}[/]")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt['config']

    if isinstance(config, MemoryMDNConfig):
        model = MemoryMDN(config).to(device)
        model_type = "MemoryMDN"
    else:
        model = SpacingMDN(config).to(device)
        model_type = "SpacingMDN"

    model.load_state_dict(ckpt['model'])
    model.eval()

    console.print(f"[green]{model_type}: {config.n_layer} layers x {config.n_head} heads[/]")

    val_data = torch.load(data_dir / "val.pt", weights_only=True)
    val_dataset = TensorDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    return model, config, val_loader, device


@torch.no_grad()
def evaluate_nll(model, val_loader, device, max_batches: int = 100):
    """Evaluate NLL."""
    model.eval()
    total_nll = 0.0
    n_batches = 0

    for batch in val_loader:
        x = batch[0].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device=="cuda"):
            result = model(x, targets=x)
            nll = result['nll']
        total_nll += nll.item()
        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break

    return total_nll / n_batches


def ablate_head(model, layer_idx: int, head_idx: int):
    """Zero out a specific attention head's output projection."""
    # Get the block
    block = model.blocks[layer_idx]
    attn = block.attn

    # Get dimensions
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head

    # Zero out the output projection for this head
    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim

    # The output projection is c_proj (projects all heads back to embed dim)
    # We zero out the rows corresponding to this head
    with torch.no_grad():
        attn.c_proj.weight[:, start_idx:end_idx] = 0


def run_ablation(model, config, val_loader, device, max_batches: int):
    """Run ablation for all attention heads."""
    console.print("\n[bold cyan]═══ ATTENTION HEAD ABLATION ═══[/]\n")

    n_layer = config.n_layer
    n_head = config.n_head

    # Baseline
    baseline_nll = evaluate_nll(model, val_loader, device, max_batches)
    console.print(f"[green]Baseline NLL: {baseline_nll:.4f}[/]\n")

    results = []
    impact_matrix = np.zeros((n_layer, n_head))

    for layer in range(n_layer):
        for head in range(n_head):
            # Create fresh copy
            import copy
            test_model = copy.deepcopy(model)

            # Ablate the head
            ablate_head(test_model, layer, head)

            # Evaluate
            nll = evaluate_nll(test_model, val_loader, device, max_batches)
            delta = nll - baseline_nll
            pct_change = (delta / abs(baseline_nll)) * 100

            results.append({
                'layer': layer,
                'head': head,
                'nll': nll,
                'delta': delta,
                'pct_change': pct_change
            })

            impact_matrix[layer, head] = pct_change

            # Status
            if pct_change > 50:
                status = "[bold red]CRITICAL[/]"
            elif pct_change > 20:
                status = "[red]IMPORTANT[/]"
            elif pct_change > 5:
                status = "[yellow]USEFUL[/]"
            elif pct_change < -5:
                status = "[green]NOISE[/]"
            else:
                status = "[dim]negligible[/]"

            console.print(f"  L{layer}.H{head}: NLL={nll:.4f} ({pct_change:+.1f}%) {status}")

            del test_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return baseline_nll, results, impact_matrix


def display_results(baseline_nll, results, impact_matrix, config):
    """Display results."""
    n_layer = config.n_layer
    n_head = config.n_head

    # Impact heatmap (ASCII)
    console.print("\n[bold]Impact Heatmap (% change when head removed):[/]")
    console.print("")

    # Header
    header = "      " + "  ".join([f"H{h}" for h in range(n_head)])
    console.print(header)

    for layer in range(n_layer):
        row = f"L{layer}:  "
        for head in range(n_head):
            val = impact_matrix[layer, head]
            if val > 50:
                row += f"[bold red]{val:+4.0f}[/] "
            elif val > 20:
                row += f"[red]{val:+4.0f}[/] "
            elif val > 5:
                row += f"[yellow]{val:+4.0f}[/] "
            elif val < -5:
                row += f"[green]{val:+4.0f}[/] "
            else:
                row += f"[dim]{val:+4.0f}[/] "
        console.print(row)

    # Find critical and noise heads
    sorted_results = sorted(results, key=lambda x: x['pct_change'], reverse=True)

    console.print("\n[bold red]CRITICAL HEADS (removing hurts most):[/]")
    for r in sorted_results[:5]:
        console.print(f"  L{r['layer']}.H{r['head']}: {r['pct_change']:+.1f}%")

    console.print("\n[bold green]NOISE HEADS (removing helps):[/]")
    noise_heads = [r for r in sorted_results if r['pct_change'] < -5]
    for r in noise_heads[:5]:
        console.print(f"  L{r['layer']}.H{r['head']}: {r['pct_change']:+.1f}%")

    # Layer summary
    console.print("\n[bold]Layer Importance (avg impact):[/]")
    for layer in range(n_layer):
        avg_impact = np.mean(impact_matrix[layer, :])
        if avg_impact > 10:
            status = "[red]CRITICAL[/]"
        elif avg_impact > 0:
            status = "[yellow]useful[/]"
        else:
            status = "[green]noise[/]"
        console.print(f"  Layer {layer}: {avg_impact:+.1f}% avg {status}")


def main():
    parser = argparse.ArgumentParser(description="Attention Ablation for Flash MDN")
    parser.add_argument("--ckpt", type=Path, default=Path("out/mdn_memory_q3_flash/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/continuous_residuals"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()

    model, config, val_loader, device = load_model_and_data(
        args.ckpt, args.data_dir, args.device
    )

    baseline_nll, results, impact_matrix = run_ablation(
        model, config, val_loader, device, args.max_batches
    )

    display_results(baseline_nll, results, impact_matrix, config)

    # Save
    output_file = Path("results/ablation_flash.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint': str(args.ckpt),
            'baseline_nll': baseline_nll,
            'results': results,
            'impact_matrix': impact_matrix.tolist()
        }, f, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/]")


if __name__ == "__main__":
    main()
