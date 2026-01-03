#!/usr/bin/env python3
"""
Train SpacingMDN on continuous spacing data.

Key differences from binned training:
- Uses NLL loss (negative log-likelihood of mixture)
- Outputs full distribution p(s|context)
- Tracks calibration and entropy
"""

import argparse
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from model.mdn import SpacingMDN, MDNConfig

console = Console()

# Default paths (overridden by args)
DEFAULT_DATA_DIR = Path("data/continuous_clean")
DEFAULT_OUTPUT_DIR = Path("out/mdn_clean_baseline")


def load_data(data_dir: Path, max_val: float = 10.0):
    """Load continuous spacing data with outlier clipping."""
    console.print(f"[cyan]Loading data from {data_dir}...[/]")

    train_data = torch.load(data_dir / "train.pt", weights_only=False)
    val_data = torch.load(data_dir / "val.pt", weights_only=False)

    # Clip outliers (important for stability)
    train_data = torch.clamp(train_data, min=0.0, max=max_val)
    val_data = torch.clamp(val_data, min=0.0, max=max_val)

    console.print(f"[green]Train: {train_data.shape}[/]")
    console.print(f"[green]Val: {val_data.shape}[/]")
    console.print(f"[dim]Clipped to [0, {max_val}][/]")

    # Stats after clipping
    console.print(f"[dim]Train mean={train_data.mean():.4f}, std={train_data.std():.4f}[/]")
    console.print(f"[dim]Val mean={val_data.mean():.4f}, std={val_data.std():.4f}[/]")

    return train_data, val_data


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()

    total_nll = 0.0
    total_entropy = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch in loader:
        x = batch[0].to(device)

        result = model(x, targets=x)
        total_nll += result['nll'].item()
        total_entropy += result['entropy'].item()

        # MAE using mixture mean
        pred_mean = torch.sum(result['pi'] * result['mu'], dim=-1)  # (B, T)
        mae = torch.abs(pred_mean[:, :-1] - x[:, 1:]).mean().item()
        total_mae += mae

        n_batches += 1

    model.train()

    avg_nll = total_nll / n_batches
    avg_entropy = total_entropy / n_batches
    avg_mae = total_mae / n_batches

    return {
        'nll': avg_nll,
        'entropy': avg_entropy,
        'mae': avg_mae,
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
        "[bold magenta]TRAINING SpacingMDN[/]\n"
        "Mixture Density Network for continuous spacing prediction",
        title="🎯"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Paths from args
    data_dir = Path(args.data_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data, val_data = load_data(data_dir, max_val=args.max_val)

    # Create datasets (input = target = spacings)
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4
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

    console.print(f"[green]Config: {config.n_layer}L/{config.n_head}H/{config.n_embd}E[/]")
    console.print(f"[green]MDN: {config.n_components} components[/]")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop

    best_val_nll = float('inf')
    train_losses = []
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

            x = batch[0].to(device)

            # Update LR
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward
            optimizer.zero_grad()
            result = model(x, targets=x)
            loss = result['loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_losses.append(loss.item())
            step += 1
            progress.update(task, advance=1)

            # Eval
            if step % args.eval_interval == 0:
                val_metrics = evaluate(model, val_loader, device)

                if val_metrics['nll'] < best_val_nll:
                    best_val_nll = val_metrics['nll']
                    torch.save({
                        'model': model.state_dict(),
                        'config': config,
                        'step': step,
                        'val_nll': val_metrics['nll'],
                    }, output_dir / 'best.pt')
                    marker = " [green]✓ NEW BEST[/]"
                else:
                    marker = ""

                console.print(
                    f"Step {step}: "
                    f"NLL={val_metrics['nll']:.4f}, "
                    f"MAE={val_metrics['mae']:.4f}, "
                    f"H={val_metrics['entropy']:.2f}"
                    f"{marker}"
                )

            # Checkpoint
            if step % args.save_interval == 0:
                torch.save({
                    'model': model.state_dict(),
                    'config': config,
                    'step': step,
                    'optimizer': optimizer.state_dict(),
                }, output_dir / f'ckpt_{step}.pt')

    # Final eval
    val_metrics = evaluate(model, val_loader, device)

    # Save final
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'step': step,
        'val_nll': val_metrics['nll'],
    }, output_dir / 'final.pt')

    # Summary
    console.print("\n")
    table = Table(title="[bold]Training Summary[/]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Final NLL", f"{val_metrics['nll']:.4f}")
    table.add_row("Best NLL", f"{best_val_nll:.4f}")
    table.add_row("Final MAE", f"{val_metrics['mae']:.4f}")
    table.add_row("Final Entropy", f"{val_metrics['entropy']:.2f}")
    table.add_row("Max Entropy", f"{math.log(config.n_components):.2f}")
    table.add_row("Steps", str(step))

    console.print(table)

    console.print(f"\n[bold green]Saved to {output_dir}/[/]")
    console.print(f"  best.pt: Best validation checkpoint")
    console.print(f"  final.pt: Final checkpoint")


def main():
    parser = argparse.ArgumentParser(description="Train SpacingMDN")

    # Paths
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Directory with train.pt/val.pt")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for checkpoints")

    # Data
    parser.add_argument("--max-val", type=float, default=10.0,
                        help="Maximum spacing value (clip outliers)")
    parser.add_argument("--seq-len", type=int, default=256)

    # Model
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # MDN
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--sigma-max", type=float, default=5.0)
    parser.add_argument("--entropy-reg", type=float, default=0.01)

    # Training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Eval/Save
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=2000)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
