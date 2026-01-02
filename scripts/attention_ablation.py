#!/usr/bin/env python3
"""
Attention Head Ablation Analysis.

Zero out each attention head and measure impact on NLL.
Finds critical heads that contain most of the learned structure.

Usage:
    python scripts/attention_ablation.py --ckpt checkpoints/E4_s7_best.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
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


class HeadAblationHook:
    """Hook to zero out specific attention head output."""

    def __init__(self, layer_idx: int, head_idx: int, n_head: int, head_dim: int):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.n_head = n_head
        self.head_dim = head_dim
        self.handle = None

    def hook_fn(self, module, input, output):
        """Zero out specific head in attention output."""
        # output shape: [B, T, n_embd] where n_embd = n_head * head_dim
        # Reshape to [B, T, n_head, head_dim], zero head, reshape back
        B, T, D = output.shape
        out = output.view(B, T, self.n_head, self.head_dim)
        out[:, :, self.head_idx, :] = 0.0
        return out.view(B, T, D)

    def register(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()


def load_model_and_data(ckpt_path: Path, data_dir: Path, device: str):
    """Load model and validation data."""
    from train_mdn_postfix import SpacingMDNPostfix, SpacingNextDataset
    from torch.utils.data import DataLoader

    console.print(f"[cyan]Loading checkpoint: {ckpt_path}[/]")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt['config']
    n_memory = ckpt.get('n_memory_slots', 8)
    slot_id_mode = ckpt.get('slot_id_mode', 'none')
    content_mode = ckpt.get('content_mode', 'normal')

    model = SpacingMDNPostfix(
        config,
        n_memory_slots=n_memory,
        slot_id_mode=slot_id_mode,
        content_mode=content_mode
    ).to(device)

    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    # Fix slot IDs during eval
    if hasattr(model, 'memory_bank') and hasattr(model.memory_bank, 'eval_slot_id_mode'):
        model.memory_bank.eval_slot_id_mode = 'fixed'

    # Load data with correct seq_len
    val_raw = torch.load(data_dir / "val.pt", weights_only=False)
    correct_seq_len = config.seq_len + 1 - n_memory
    val_dataset = SpacingNextDataset(val_raw, seq_len=correct_seq_len)

    batch_size = 512 if device == "cuda" else 256
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4 if device == "cuda" else 0)

    console.print(f"[green]Model: {config.n_layer} layers, {config.n_head} heads[/]")
    console.print(f"[green]Data: {len(val_dataset):,} samples[/]")

    return model, config, val_loader, device


@torch.no_grad()
def evaluate_nll(model, val_loader, device, max_batches: int = 100):
    """Evaluate NLL."""
    from train_mdn_postfix import mdn_loss_1step

    model.eval()
    total_nll = 0.0
    n_batches = 0

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            pi, mu, sigma = model(x)
            nll = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=0)

        total_nll += nll.item()
        n_batches += 1

        if max_batches > 0 and n_batches >= max_batches:
            break

    return total_nll / n_batches


def run_ablation_analysis(model, config, val_loader, device, max_batches: int = 100):
    """Run ablation for each attention head."""
    console.print("\n[bold cyan]═══ ATTENTION HEAD ABLATION ═══[/]\n")

    n_layer = config.n_layer
    n_head = config.n_head
    head_dim = config.n_embd // n_head

    # Baseline
    baseline_nll = evaluate_nll(model, val_loader, device, max_batches)
    console.print(f"[green]Baseline NLL: {baseline_nll:.4f}[/]\n")

    results = []

    # Get attention modules (they have c_attn attribute)
    attn_modules = []
    for block in model.blocks:
        attn_modules.append(block.attn)

    total_heads = n_layer * n_head

    for layer_idx in range(n_layer):
        for head_idx in track(range(n_head), description=f"Layer {layer_idx}"):
            # Create and register hook
            hook = HeadAblationHook(layer_idx, head_idx, n_head, head_dim)

            # Hook on the attention output (before c_proj)
            # We need to hook after the attention computation
            # The cleanest way: hook on c_proj input
            hook.register(attn_modules[layer_idx].c_proj)

            # Evaluate with ablated head
            ablated_nll = evaluate_nll(model, val_loader, device, max_batches)

            # Remove hook
            hook.remove()

            delta = ablated_nll - baseline_nll
            pct_change = (delta / abs(baseline_nll)) * 100

            results.append({
                'layer': layer_idx,
                'head': head_idx,
                'nll': ablated_nll,
                'delta': delta,
                'pct_change': pct_change
            })

    return baseline_nll, results


def display_results(baseline_nll, results, top_k: int = 10):
    """Display results sorted by impact."""
    # Sort by absolute delta (most impactful first)
    sorted_results = sorted(results, key=lambda x: abs(x['delta']), reverse=True)

    # Full table
    table = Table(title="All Attention Heads (sorted by impact)")
    table.add_column("Layer", justify="right")
    table.add_column("Head", justify="right")
    table.add_column("NLL", justify="right")
    table.add_column("Δ NLL", justify="right")
    table.add_column("% Change", justify="right")
    table.add_column("Status", justify="center")

    for r in sorted_results[:top_k]:
        if r['pct_change'] > 5:
            status = "[red]CRITICAL[/]"
        elif r['pct_change'] > 2:
            status = "[yellow]IMPORTANT[/]"
        elif r['pct_change'] > 0.5:
            status = "[blue]MINOR[/]"
        else:
            status = "[dim]negligible[/]"

        table.add_row(
            f"L{r['layer']}",
            f"H{r['head']}",
            f"{r['nll']:.4f}",
            f"{r['delta']:+.4f}",
            f"{r['pct_change']:+.1f}%",
            status
        )

    console.print(table)

    # Summary
    critical = [r for r in results if r['pct_change'] > 5]
    important = [r for r in results if 2 < r['pct_change'] <= 5]

    console.print(f"\n[bold]Summary:[/]")
    console.print(f"  [red]CRITICAL heads (>5%): {len(critical)}[/]")
    console.print(f"  [yellow]IMPORTANT heads (2-5%): {len(important)}[/]")
    console.print(f"  [dim]Total heads: {len(results)}[/]")

    if critical:
        console.print(f"\n[bold red]Critical heads contain core operator structure:[/]")
        for r in critical:
            console.print(f"  Layer {r['layer']}, Head {r['head']}: Δ={r['delta']:+.4f} ({r['pct_change']:+.1f}%)")

    # Heatmap summary
    console.print("\n[bold]Impact Heatmap (% change):[/]")
    n_layer = max(r['layer'] for r in results) + 1
    n_head = max(r['head'] for r in results) + 1

    header = "      " + " ".join([f"H{h}" for h in range(n_head)])
    console.print(header)

    for layer in range(n_layer):
        row = f"L{layer}:  "
        for head in range(n_head):
            r = next(x for x in results if x['layer'] == layer and x['head'] == head)
            pct = r['pct_change']
            if pct > 5:
                row += f"[red]{pct:4.1f}[/] "
            elif pct > 2:
                row += f"[yellow]{pct:4.1f}[/] "
            elif pct > 0.5:
                row += f"[blue]{pct:4.1f}[/] "
            else:
                row += f"[dim]{pct:4.1f}[/] "
        console.print(row)


def main():
    parser = argparse.ArgumentParser(description="Attention Head Ablation Analysis")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/E4_s7_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/continuous_2M"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps")
    parser.add_argument("--max-batches", type=int, default=100, help="Max batches per eval")
    parser.add_argument("--top-k", type=int, default=15, help="Show top K most impactful heads")
    args = parser.parse_args()

    console.print(f"[bold]Device: {args.device}[/]")

    if args.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[cyan]GPU: {gpu_name}[/]")

    model, config, val_loader, device = load_model_and_data(
        args.ckpt, args.data_dir, args.device
    )

    baseline_nll, results = run_ablation_analysis(
        model, config, val_loader, device, args.max_batches
    )

    display_results(baseline_nll, results, args.top_k)

    # Save results
    output_file = Path("results/attention_ablation.json")
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
