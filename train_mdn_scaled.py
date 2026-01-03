#!/usr/bin/env python3
"""
SCALED SpacingMDN Training - Full GPU Utilization

Optimizations:
- Mixed precision (FP16) for 2x memory efficiency
- Larger batch size (512)
- Bigger model (10L/12H/512E, ~20M params)
- More data workers (8)
- Pin memory for faster CPU->GPU transfer
- torch.compile for speed
"""

import argparse
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from model.mdn import SpacingMDN, MDNConfig

console = Console()

# Paths - use full data if available
DATA_DIR = Path("data/continuous_100M_full")
if not DATA_DIR.exists():
    DATA_DIR = Path("data/continuous_100M")
OUTPUT_DIR = Path("out/mdn_scaled")


def load_data(max_val: float = 10.0):
    """Load continuous spacing data with outlier clipping."""
    console.print(f"[cyan]Loading data from {DATA_DIR}...[/]")

    train_data = torch.load(DATA_DIR / "train.pt", weights_only=False)
    val_data = torch.load(DATA_DIR / "val.pt", weights_only=False)

    # Clip outliers
    train_data = torch.clamp(train_data, min=0.0, max=max_val)
    val_data = torch.clamp(val_data, min=0.0, max=max_val)

    console.print(f"[green]Train: {train_data.shape} ({train_data.numel()*4/1e9:.2f} GB)[/]")
    console.print(f"[green]Val: {val_data.shape}[/]")
    console.print(f"[dim]Clipped to [0, {max_val}], mean={train_data.mean():.4f}, std={train_data.std():.4f}[/]")

    return train_data, val_data


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    """Evaluate model on validation set."""
    model.eval()

    total_nll = 0.0
    total_entropy = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch in loader:
        x = batch[0].to(device)

        with autocast(enabled=use_amp):
            result = model(x, targets=x)

        total_nll += result['nll'].item()
        total_entropy += result['entropy'].item()

        pred_mean = torch.sum(result['pi'] * result['mu'], dim=-1)
        mae = torch.abs(pred_mean[:, :-1] - x[:, 1:]).mean().item()
        total_mae += mae

        n_batches += 1

    model.train()

    return {
        'nll': total_nll / n_batches,
        'entropy': total_entropy / n_batches,
        'mae': total_mae / n_batches,
    }


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(args):
    console.print(Panel.fit(
        "[bold magenta]SCALED SpacingMDN TRAINING[/]\n"
        "Full GPU utilization with mixed precision",
        title="🚀"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[cyan]GPU: {gpu_name} ({gpu_mem:.1f} GB)[/]")

    # Load data
    train_data, val_data = load_data(max_val=args.max_val)

    # Create datasets
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    config = MDNConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        seq_len=args.seq_len,
        dropout=args.dropout,
        n_components=args.n_components,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        entropy_reg=args.entropy_reg,
    )

    model = SpacingMDN(config).to(device)

    # Compile model for speed (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        console.print("[cyan]Compiling model with torch.compile...[/]")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Config: {config.n_layer}L/{config.n_head}H/{config.n_embd}E[/]")
    console.print(f"[green]Parameters: {n_params:,} ({n_params/1e6:.1f}M)[/]")
    console.print(f"[green]MDN: {config.n_components} components[/]")
    console.print(f"[green]Batch size: {args.batch_size}[/]")
    console.print(f"[green]Mixed precision: {args.amp}[/]")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True if torch.cuda.is_available() else False,
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.amp)

    # Training loop
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_nll = float('inf')
    step = 0

    console.print(f"\n[cyan]Training for {args.max_steps} steps...[/]\n")

    model.train()
    train_iter = iter(train_loader)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=args.max_steps)

        while step < args.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            x = batch[0].to(device, non_blocking=True)

            # Update LR
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward with mixed precision
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                result = model(x, targets=x)
                loss = result['loss']

            # Backward with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            progress.update(task, advance=1)

            # Eval
            if step % args.eval_interval == 0:
                val_metrics = evaluate(model, val_loader, device, use_amp=args.amp)

                # GPU memory
                if torch.cuda.is_available():
                    mem_used = torch.cuda.max_memory_allocated() / 1e9
                    mem_str = f", GPU={mem_used:.1f}GB"
                else:
                    mem_str = ""

                if val_metrics['nll'] < best_val_nll:
                    best_val_nll = val_metrics['nll']
                    # Save without compiled wrapper
                    state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
                    torch.save({
                        'model': state_dict,
                        'config': config,
                        'step': step,
                        'val_nll': val_metrics['nll'],
                    }, OUTPUT_DIR / 'best.pt')
                    marker = " [green]✓ BEST[/]"
                else:
                    marker = ""

                console.print(
                    f"Step {step}: "
                    f"NLL={val_metrics['nll']:.4f}, "
                    f"MAE={val_metrics['mae']:.4f}, "
                    f"H={val_metrics['entropy']:.2f}"
                    f"{mem_str}{marker}"
                )

            # Checkpoint
            if step % args.save_interval == 0:
                state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
                torch.save({
                    'model': state_dict,
                    'config': config,
                    'step': step,
                    'optimizer': optimizer.state_dict(),
                }, OUTPUT_DIR / f'ckpt_{step}.pt')

    # Final eval and save
    val_metrics = evaluate(model, val_loader, device, use_amp=args.amp)
    state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
    torch.save({
        'model': state_dict,
        'config': config,
        'step': step,
        'val_nll': val_metrics['nll'],
    }, OUTPUT_DIR / 'final.pt')

    # Summary
    console.print("\n")
    table = Table(title="[bold]Scaled Training Summary[/]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Final NLL", f"{val_metrics['nll']:.4f}")
    table.add_row("Best NLL", f"{best_val_nll:.4f}")
    table.add_row("Final MAE", f"{val_metrics['mae']:.4f}")
    table.add_row("Parameters", f"{n_params:,}")
    table.add_row("Steps", str(step))

    if torch.cuda.is_available():
        table.add_row("Peak GPU Memory", f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    console.print(table)
    console.print(f"\n[bold green]Saved to {OUTPUT_DIR}/[/]")


def main():
    parser = argparse.ArgumentParser(description="Scaled SpacingMDN Training")

    # Data
    parser.add_argument("--max-val", type=float, default=10.0)
    parser.add_argument("--seq-len", type=int, default=256)

    # Model - SCALED UP
    parser.add_argument("--n-layer", type=int, default=10)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # MDN
    parser.add_argument("--n-components", type=int, default=12)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--sigma-max", type=float, default=5.0)
    parser.add_argument("--entropy-reg", type=float, default=0.01)

    # Training - SCALED UP
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Optimization
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use mixed precision (default: True)")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--compile", action="store_true", default=True,
                        help="Use torch.compile (default: True)")
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    parser.add_argument("--num-workers", type=int, default=8)

    # Eval/Save
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=5000)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
