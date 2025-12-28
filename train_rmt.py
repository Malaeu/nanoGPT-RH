#!/usr/bin/env python3
"""
RMT Training: Train SpacingGPT with Recurrent Memory.

Key differences from standard training:
1. Process sequences as multiple windows with memory carry-over
2. Memory consistency loss: encourage stable memory representations
3. Learnable EMA weight for memory update

Usage:
    python train_rmt.py --data-dir data --out-dir out_rmt --n-windows 4
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
from rich.panel import Panel

from model.gpt import RMTSpacingGPT, RMTConfig

console = Console()


def load_data(data_dir: Path, batch_size: int, n_windows: int = 4):
    """
    Load data and prepare for multi-window training.

    Returns sequences that can be split into n_windows.
    """
    train = torch.load(data_dir / "train.pt")
    val = torch.load(data_dir / "val.pt")
    meta = torch.load(data_dir / "meta.pt")

    seq_len = meta["seq_len"]

    console.print(f"[cyan]Loaded data from {data_dir}[/]")
    console.print(f"  Original: Train {train.shape}, Val {val.shape}")
    console.print(f"  Seq len: {seq_len}, Windows: {n_windows}")

    # For RMT training, we want longer sequences that span multiple windows
    # Each training sample is n_windows * seq_len tokens
    total_len = n_windows * seq_len

    # Reshape data into longer sequences
    def reshape_for_rmt(data, total_len):
        """Reshape data into sequences of total_len."""
        n_samples = data.shape[0]
        flat_len = data.shape[1]

        # Option 1: Concatenate consecutive samples
        # This preserves the sequential structure
        flat = data.reshape(-1)
        n_rmt_samples = len(flat) // total_len
        rmt_data = flat[:n_rmt_samples * total_len].reshape(n_rmt_samples, total_len)
        return rmt_data

    train_rmt = reshape_for_rmt(train, total_len)
    val_rmt = reshape_for_rmt(val, total_len)

    console.print(f"  RMT: Train {train_rmt.shape}, Val {val_rmt.shape}")

    train_loader = DataLoader(
        TensorDataset(train_rmt),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_rmt),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, meta


def split_into_windows(batch, seq_len, n_windows):
    """
    Split a batch into windows for RMT processing.

    Args:
        batch: (B, n_windows * seq_len)
        seq_len: window size
        n_windows: number of windows

    Returns:
        list of (B, seq_len) tensors
    """
    windows = []
    for i in range(n_windows):
        start = i * seq_len
        end = start + seq_len
        windows.append(batch[:, start:end])
    return windows


@torch.no_grad()
def evaluate(model, loader, device, seq_len, n_windows):
    """Compute average loss on dataset with memory carry-over."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for (batch,) in loader:
        batch = batch.to(device)
        windows = split_into_windows(batch, seq_len, n_windows)

        # Process with memory
        memory = model.init_memory(batch.size(0), device)
        for window in windows:
            _, loss, memory = model(window, targets=window, memory=memory, return_memory=True)
            if loss is not None:
                total_loss += loss.item()
                n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(avg_loss)
    model.train()
    return avg_loss, perplexity


def train(args):
    console.print(Panel.fit(
        "[bold magenta]RMT SpacingGPT Training[/]\n"
        f"Memory tokens: {args.n_mem_tokens}\n"
        f"Windows per sample: {args.n_windows}",
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
    train_loader, val_loader, meta = load_data(
        Path(args.data_dir), args.batch_size, args.n_windows
    )

    # Model config
    config = RMTConfig(
        vocab_size=meta["vocab_size"],
        seq_len=meta["seq_len"],
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        n_mem_tokens=args.n_mem_tokens,
        memory_alpha_init=args.memory_alpha,
    )

    # Print config
    table = Table(title="RMT Model Config")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for k, v in vars(config).items():
        table.add_row(k, str(v))
    console.print(table)

    # Create model
    model = RMTSpacingGPT(config).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Learning rate scheduler
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
    memory_alphas = []

    console.print(f"\n[bold]Starting RMT training for {args.max_steps} steps...[/]\n")

    start_time = time.time()
    data_iter = iter(train_loader)
    seq_len = meta["seq_len"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Training RMT...", total=args.max_steps)

        while step < args.max_steps:
            # Get batch
            try:
                (batch,) = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                (batch,) = next(data_iter)

            batch = batch.to(device)
            windows = split_into_windows(batch, seq_len, args.n_windows)

            # Update learning rate
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward with memory carry-over
            optimizer.zero_grad()

            memory = model.init_memory(batch.size(0), device)
            total_loss = 0.0
            memory_states = [memory.detach().clone()]

            for window in windows:
                _, loss, memory = model(
                    window, targets=window, memory=memory, return_memory=True
                )
                if loss is not None:
                    total_loss += loss
                memory_states.append(memory.detach().clone())

            # Average loss over windows
            avg_loss = total_loss / args.n_windows

            # Memory consistency loss (optional)
            # Encourage smooth memory transitions
            if args.memory_consistency > 0:
                mem_diff = 0.0
                for i in range(1, len(memory_states)):
                    mem_diff += torch.mean((memory_states[i] - memory_states[i-1])**2)
                mem_consistency_loss = mem_diff / (len(memory_states) - 1)
                avg_loss = avg_loss + args.memory_consistency * mem_consistency_loss

            # Backward
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_losses.append(avg_loss.item())
            memory_alphas.append(model.memory_alpha.item())
            step += 1
            progress.update(task, advance=1)

            # Logging
            if step % args.log_interval == 0:
                recent_loss = sum(train_losses[-args.log_interval:]) / args.log_interval
                recent_alpha = sum(memory_alphas[-args.log_interval:]) / args.log_interval
                elapsed = time.time() - start_time
                console.print(
                    f"step {step:5d} | loss {recent_loss:.4f} | "
                    f"ppl {math.exp(recent_loss):.1f} | α {recent_alpha:.3f} | "
                    f"lr {lr:.2e} | {elapsed:.1f}s"
                )

            # Evaluation
            if step % args.eval_interval == 0:
                val_loss, val_ppl = evaluate(
                    model, val_loader, device, seq_len, args.n_windows
                )
                console.print(
                    f"[yellow]>>> val loss {val_loss:.4f} | val ppl {val_ppl:.1f} | "
                    f"memory_α {model.memory_alpha.item():.4f}[/]"
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model": model.state_dict(),
                        "config": config,
                        "step": step,
                        "val_loss": val_loss,
                        "memory_alpha": model.memory_alpha.item(),
                    }, out_dir / "best_rmt.pt")
                    console.print(f"[green]✓ Saved best RMT model (val_loss={val_loss:.4f})[/]")

            # Checkpoint
            if step % args.save_interval == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                    "memory_alpha": model.memory_alpha.item(),
                }, out_dir / f"rmt_ckpt_{step}.pt")

    # Final evaluation
    console.print("\n[bold]Final evaluation...[/]")
    val_loss, val_ppl = evaluate(model, val_loader, device, seq_len, args.n_windows)
    train_loss = sum(train_losses[-100:]) / 100
    train_ppl = math.exp(train_loss)

    # Summary
    table = Table(title="RMT Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Final train loss", f"{train_loss:.4f}")
    table.add_row("Final train ppl", f"{train_ppl:.1f}")
    table.add_row("Final val loss", f"{val_loss:.4f}")
    table.add_row("Final val ppl", f"{val_ppl:.1f}")
    table.add_row("Best val loss", f"{best_val_loss:.4f}")
    table.add_row("Learned memory α", f"{model.memory_alpha.item():.4f}")
    table.add_row("Total time", f"{time.time() - start_time:.1f}s")
    console.print(table)

    # Save final
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "step": step,
        "val_loss": val_loss,
        "memory_alpha": model.memory_alpha.item(),
    }, out_dir / "final_rmt.pt")

    console.print(f"\n[bold green]✓ Done! RMT models saved to {out_dir}[/]")


def main():
    parser = argparse.ArgumentParser(description="Train RMT SpacingGPT")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--out-dir", type=str, default="out_rmt", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device")

    # Model
    parser.add_argument("--n-layer", type=int, default=4, help="Transformer layers")
    parser.add_argument("--n-head", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")

    # RMT-specific
    parser.add_argument("--n-mem-tokens", type=int, default=1, help="Memory tokens")
    parser.add_argument("--n-windows", type=int, default=4, help="Windows per sample")
    parser.add_argument("--memory-alpha", type=float, default=0.5, help="Initial memory EMA")
    parser.add_argument("--memory-consistency", type=float, default=0.01,
                        help="Memory consistency loss weight")

    # Training
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
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
