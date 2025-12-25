#!/usr/bin/env python3
"""
Training script for SpacingGPT.
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

from model.gpt import SpacingGPT, GPTConfig

console = Console()


def load_data(data_dir: Path, batch_size: int):
    """Load preprocessed tensors."""
    train = torch.load(data_dir / "train.pt")
    val = torch.load(data_dir / "val.pt")
    meta = torch.load(data_dir / "meta.pt")

    console.print(f"[cyan]Loaded data from {data_dir}[/]")
    console.print(f"  Train: {train.shape}, Val: {val.shape}")
    console.print(f"  Vocab size: {meta['vocab_size']}")

    train_loader = DataLoader(
        TensorDataset(train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, meta


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute average loss and perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for (batch,) in loader:
        batch = batch.to(device)
        _, loss = model(batch, targets=batch)
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    perplexity = math.exp(avg_loss)
    model.train()
    return avg_loss, perplexity


def train(args):
    console.print(Panel.fit(
        "[bold magenta]SpacingGPT Training[/]",
        subtitle=f"device: {args.device}"
    ))

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    console.print(f"[green]Using device: {device}[/]")

    # Load data
    train_loader, val_loader, meta = load_data(Path(args.data_dir), args.batch_size)

    # Model config
    config = GPTConfig(
        vocab_size=meta["vocab_size"],
        seq_len=meta["seq_len"],
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    # Print config
    table = Table(title="Model Config")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for k, v in vars(config).items():
        table.add_row(k, str(v))
    console.print(table)

    # Create model
    model = SpacingGPT(config).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Learning rate scheduler (cosine with warmup)
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / args.warmup_steps
        if step > args.max_steps:
            return args.min_lr
        decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.lr - args.min_lr)

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    step = 0
    best_val_loss = float("inf")
    train_losses = []

    console.print(f"\n[bold]Starting training for {args.max_steps} steps...[/]\n")

    start_time = time.time()
    data_iter = iter(train_loader)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=args.max_steps)

        while step < args.max_steps:
            # Get batch
            try:
                (batch,) = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                (batch,) = next(data_iter)

            batch = batch.to(device)

            # Update learning rate
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward + backward
            optimizer.zero_grad()
            _, loss = model(batch, targets=batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_losses.append(loss.item())
            step += 1
            progress.update(task, advance=1)

            # Logging
            if step % args.log_interval == 0:
                avg_loss = sum(train_losses[-args.log_interval:]) / args.log_interval
                elapsed = time.time() - start_time
                console.print(
                    f"step {step:5d} | loss {avg_loss:.4f} | "
                    f"ppl {math.exp(avg_loss):.1f} | lr {lr:.2e} | "
                    f"{elapsed:.1f}s"
                )

            # Evaluation
            if step % args.eval_interval == 0:
                val_loss, val_ppl = evaluate(model, val_loader, device)
                console.print(
                    f"[yellow]>>> val loss {val_loss:.4f} | val ppl {val_ppl:.1f}[/]"
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model": model.state_dict(),
                        "config": config,
                        "step": step,
                        "val_loss": val_loss,
                    }, out_dir / "best.pt")
                    console.print(f"[green]✓ Saved best model (val_loss={val_loss:.4f})[/]")

            # Checkpoint
            if step % args.save_interval == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                }, out_dir / f"ckpt_{step}.pt")

    # Final evaluation
    console.print("\n[bold]Final evaluation...[/]")
    val_loss, val_ppl = evaluate(model, val_loader, device)
    train_loss = sum(train_losses[-100:]) / 100
    train_ppl = math.exp(train_loss)

    # Summary
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Final train loss", f"{train_loss:.4f}")
    table.add_row("Final train ppl", f"{train_ppl:.1f}")
    table.add_row("Final val loss", f"{val_loss:.4f}")
    table.add_row("Final val ppl", f"{val_ppl:.1f}")
    table.add_row("Best val loss", f"{best_val_loss:.4f}")
    table.add_row("Total time", f"{time.time() - start_time:.1f}s")
    console.print(table)

    # Save final
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "step": step,
        "val_loss": val_loss,
    }, out_dir / "final.pt")

    console.print(f"\n[bold green]✓ Done! Models saved to {out_dir}[/]")


def main():
    parser = argparse.ArgumentParser(description="Train SpacingGPT")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--out-dir", type=str, default="out", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")

    # Model
    parser.add_argument("--n-layer", type=int, default=4, help="Transformer layers")
    parser.add_argument("--n-head", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")

    # Training
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Min learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")

    # Logging
    parser.add_argument("--log-interval", type=int, default=100, help="Log interval")
    parser.add_argument("--eval-interval", type=int, default=500, help="Eval interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save interval")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
