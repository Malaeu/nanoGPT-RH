#!/usr/bin/env python3
"""
Train MemoryMDN with Q3 Memory Bank on clean spacing data.

Memory Map Q3 (8 invariant slots):
  M0 SIGN    - polarity of correction (arch - prime)
  M1 NORM    - scale/units calibration
  M2 TORUS   - translation invariance
  M3 SYMBOL  - kernel/PSF shape
  M4 FLOOR   - uncertainty floor
  M5 TOEPLITZ - discretization stability
  M6 PRIME-CAP - limit global correction
  M7 GOAL    - stability margin

Training protocol:
  - Memory learns slower (memory_lr_mult=0.1) to avoid early collapse
  - Diversity loss keeps slots orthogonal
  - Memory cap prevents any slot from dominating
  - Slot dropout ensures each slot is useful independently

Usage:
  python train_memory_q3.py --data-dir data/continuous_clean --out-dir out/mdn_memory_q3
"""

import argparse
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from model.memory_mdn import MemoryMDN, MemoryMDNConfig

console = Console()

# Defaults
DEFAULT_DATA_DIR = Path("data/continuous_clean")
DEFAULT_OUTPUT_DIR = Path("out/mdn_memory_q3")


def load_data(data_dir: Path, max_val: float = 10.0):
    """Load continuous spacing data."""
    console.print(f"[cyan]Loading data from {data_dir}...[/]")

    train_data = torch.load(data_dir / "train.pt", weights_only=False)
    val_data = torch.load(data_dir / "val.pt", weights_only=False)

    train_data = torch.clamp(train_data, min=0.0, max=max_val)
    val_data = torch.clamp(val_data, min=0.0, max=max_val)

    console.print(f"[green]Train: {train_data.shape}[/]")
    console.print(f"[green]Val: {val_data.shape}[/]")

    return train_data, val_data


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()

    total_nll = 0.0
    total_entropy = 0.0
    total_diversity = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch in loader:
        x = batch[0].to(device)
        result = model(x, targets=x)

        total_nll += result['nll'].item()
        total_entropy += result['entropy'].item()
        total_diversity += result['diversity_loss'].item()

        # MAE using mixture mean
        pred_mean = torch.sum(result['pi'] * result['mu'], dim=-1)
        mae = torch.abs(pred_mean[:, :-1] - x[:, 1:]).mean().item()
        total_mae += mae

        n_batches += 1

    model.train()

    return {
        'nll': total_nll / n_batches,
        'entropy': total_entropy / n_batches,
        'diversity': total_diversity / n_batches,
        'mae': total_mae / n_batches,
    }


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine LR schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(args):
    console.print(Panel.fit(
        "[bold magenta]TRAINING MemoryMDN[/]\n"
        "MDN with Q3 Memory Bank (8 invariant slots)\n"
        "[dim]M0:SIGN M1:NORM M2:TORUS M3:SYMBOL[/]\n"
        "[dim]M4:FLOOR M5:TOEPLITZ M6:PRIMECAP M7:GOAL[/]",
        title="Q3 Memory Bank"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data, val_data = load_data(data_dir, max_val=args.max_val)

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=args.batch_size,
        num_workers=4
    )

    # Create model
    config = MemoryMDNConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        seq_len=args.seq_len,
        dropout=args.dropout,
        n_components=args.n_components,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        entropy_reg=args.entropy_reg,
        # Memory Bank Q3 settings
        n_memory=args.n_memory,
        memory_dropout=args.memory_dropout,
        memory_cap=args.memory_cap,
        diversity_weight=args.diversity_weight,
        memory_lr_mult=args.memory_lr_mult,
    )

    model = MemoryMDN(config).to(device)

    # Optimizer with separate LR for memory (slower learning)
    param_groups = model.get_param_groups(base_lr=args.lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Resume logic
    start_step = 0
    best_val_nll = float('inf')
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            console.print(f"[cyan]Resuming from {resume_path}...[/]")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'step' in checkpoint:
                start_step = checkpoint['step']
            if 'val_nll' in checkpoint:
                best_val_nll = checkpoint['val_nll']
            console.print(f"[green]Resumed from step {start_step} with best_val_nll={best_val_nll:.4f}[/]")
        else:
            console.print(f"[red]Resume path {resume_path} not found. Starting from scratch.[/]")

    console.print(f"[green]Transformer: {config.n_layer}L/{config.n_head}H/{config.n_embd}E[/]")
    console.print(f"[green]MDN: K={config.n_components} components[/]")
    console.print(f"[yellow]Memory Bank Q3: {config.n_memory} slots[/]")
    console.print(f"[dim]  memory_dropout: {config.memory_dropout}[/]")
    console.print(f"[dim]  memory_cap: {config.memory_cap}[/]")
    console.print(f"[dim]  diversity_weight: {config.diversity_weight}[/]")
    console.print(f"[dim]  memory_lr_mult: {config.memory_lr_mult}x[/]")

    console.print(f"[dim]Main params LR: {args.lr}[/]")
    console.print(f"[dim]Memory params LR: {args.lr * config.memory_lr_mult}[/]")

    # Training loop
    step = start_step
    train_iter = iter(train_loader)

    console.print(f"\n[cyan]Training for {args.max_steps} steps...[/]\n")

    model.train()

    while step < args.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch[0].to(device)

        # Update LR (cosine schedule)
        # Memory group learns slower (identified by 'is_memory' flag)
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            if param_group.get('is_memory', False):
                param_group['lr'] = lr * config.memory_lr_mult
            else:
                param_group['lr'] = lr

        # Forward
        optimizer.zero_grad()
        result = model(x, targets=x)
        loss = result['loss']

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1

        # Eval
        if step % args.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device)
            mem_diag = model.get_memory_diagnostics()

            if val_metrics['nll'] < best_val_nll:
                best_val_nll = val_metrics['nll']
                torch.save({
                    'model': model.state_dict(),
                    'config': config,
                    'step': step,
                    'val_nll': val_metrics['nll'],
                }, output_dir / 'best.pt')
                marker = " [green]NEW BEST[/]"
            else:
                marker = ""

            console.print(
                f"Step {step}: "
                f"NLL={val_metrics['nll']:.4f}, "
                f"MAE={val_metrics['mae']:.4f}, "
                f"H={val_metrics['entropy']:.2f}, "
                f"div={val_metrics['diversity']:.4f}, "
                f"sim={mem_diag['mem_sim_mean']:.3f}"
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
    mem_diag = model.get_memory_diagnostics()

    # Save final
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'step': step,
        'val_nll': val_metrics['nll'],
    }, output_dir / 'final.pt')

    # Summary
    console.print("\n")
    table = Table(title="[bold]Q3 Memory Bank Training Summary[/]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Final NLL", f"{val_metrics['nll']:.4f}")
    table.add_row("Best NLL", f"{best_val_nll:.4f}")
    table.add_row("Final MAE", f"{val_metrics['mae']:.4f}")
    table.add_row("Final Entropy", f"{val_metrics['entropy']:.2f}")
    table.add_row("Max Entropy", f"{math.log(config.n_components):.2f}")
    table.add_row("Diversity Loss", f"{val_metrics['diversity']:.4f}")
    table.add_row("Memory Similarity", f"{mem_diag['mem_sim_mean']:.3f}")
    table.add_row("Steps", str(step))

    console.print(table)
    console.print(f"\n[bold green]Saved to {output_dir}/[/]")


def main():
    parser = argparse.ArgumentParser(description="Train MemoryMDN with Q3 Memory Bank")

    # Paths
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Directory with train.pt/val.pt")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Data
    parser.add_argument("--max-val", type=float, default=10.0,
                        help="Maximum spacing value (clip outliers)")
    parser.add_argument("--seq-len", type=int, default=256)

    # Model (Transformer)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # MDN
    parser.add_argument("--n-components", type=int, default=8,
                        help="Number of Gaussian mixture components")
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--sigma-max", type=float, default=5.0)
    parser.add_argument("--entropy-reg", type=float, default=0.005,
                        help="Entropy regularization weight")

    # Memory Bank Q3
    parser.add_argument("--n-memory", type=int, default=8,
                        help="Number of Q3 memory slots (M0-M7)")
    parser.add_argument("--memory-dropout", type=float, default=0.1,
                        help="Slot dropout probability")
    parser.add_argument("--memory-cap", type=float, default=2.0,
                        help="Max norm per memory vector")
    parser.add_argument("--diversity-weight", type=float, default=0.01,
                        help="Orthogonality regularization weight")
    parser.add_argument("--memory-lr-mult", type=float, default=0.1,
                        help="LR multiplier for memory (slower learning)")

    # Training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Eval/Save
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=5000)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
