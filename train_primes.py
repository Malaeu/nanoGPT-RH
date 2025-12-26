#!/usr/bin/env python3
"""
Train SpacingGPT on prime gaps.

Quick training script to test if primes are more predictable than zeta zeros.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from rich.console import Console
from rich.progress import track
import time

from model.gpt import SpacingGPT, GPTConfig

console = Console()

# ============================================================
# CONFIG
# ============================================================
TRAIN_DATA = Path("data/train_primes.pt")
VAL_DATA = Path("data/val_primes.pt")
OUTPUT_DIR = Path("out_primes")

# Model config (same as zeta model for comparison)
MODEL_CONFIG = GPTConfig(
    vocab_size=256,
    seq_len=256,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.1,
    bias=False,
)

# Training config
BATCH_SIZE = 64
MAX_STEPS = 3000
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 500
EVAL_ITERS = 50


def load_data():
    """Load prime gap data."""
    train = torch.load(TRAIN_DATA)
    val = torch.load(VAL_DATA)
    return train, val


def get_batch(data, batch_size, device):
    """Get random batch from data."""
    n_seqs = len(data)
    idx = torch.randint(0, n_seqs, (batch_size,))
    x = data[idx, :-1].to(device)  # Input: all but last
    y = data[idx, 1:].to(device)   # Target: all but first
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    """Estimate loss on train and val sets."""
    model.eval()
    losses = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        total_loss = 0
        for _ in range(EVAL_ITERS):
            x, y = get_batch(data, BATCH_SIZE, device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
        losses[split] = total_loss / EVAL_ITERS

    model.train()
    return losses


@torch.no_grad()
def compute_accuracy(model, data, device, n_samples=1000):
    """Compute top-1 and top-5 accuracy."""
    model.eval()

    correct_1 = 0
    correct_5 = 0
    total = 0

    n_seqs = min(n_samples // 255, len(data))

    for seq_idx in range(n_seqs):
        seq = data[seq_idx].unsqueeze(0).to(device)
        logits, _ = model(seq[:, :-1])

        for pos in range(logits.size(1)):
            pred_logits = logits[0, pos]
            target = seq[0, pos + 1].item()

            # Top-1
            if torch.argmax(pred_logits).item() == target:
                correct_1 += 1

            # Top-5
            top5 = torch.topk(pred_logits, 5).indices.tolist()
            if target in top5:
                correct_5 += 1

            total += 1

    model.train()
    return correct_1 / total * 100, correct_5 / total * 100


def main():
    console.print("[bold magenta]═══ PRIME GAP TRAINING ═══[/]\n")

    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/]")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    console.print("[cyan]Loading data...[/]")
    train_data, val_data = load_data()
    console.print(f"[green]Train: {train_data.shape}, Val: {val_data.shape}[/]")

    # Create model
    console.print("[cyan]Creating model...[/]")
    model = SpacingGPT(MODEL_CONFIG).to(device)
    console.print(f"[green]Parameters: {sum(p.numel() for p in model.parameters()):,}[/]")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    console.print(f"\n[bold]Training for {MAX_STEPS} steps...[/]\n")

    model.train()
    start_time = time.time()

    for step in range(MAX_STEPS):
        # Get batch
        x, y = get_batch(train_data, BATCH_SIZE, device)

        # Forward
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss(model, train_data, val_data, device)
            ppl_train = torch.exp(torch.tensor(losses["train"])).item()
            ppl_val = torch.exp(torch.tensor(losses["val"])).item()

            top1, top5 = compute_accuracy(model, val_data, device)

            elapsed = time.time() - start_time
            console.print(
                f"Step {step:5d} | "
                f"Loss: {losses['train']:.4f}/{losses['val']:.4f} | "
                f"PPL: {ppl_train:.1f}/{ppl_val:.1f} | "
                f"Acc: {top1:.1f}%/{top5:.1f}% | "
                f"Time: {elapsed:.1f}s"
            )

    # Save model
    console.print("\n[cyan]Saving model...[/]")
    torch.save({
        "model": model.state_dict(),
        "config": MODEL_CONFIG,
        "step": MAX_STEPS,
    }, OUTPUT_DIR / "best_primes.pt")

    # Final evaluation
    console.print("\n[bold]Final Evaluation:[/]")
    losses = estimate_loss(model, train_data, val_data, device)
    top1, top5 = compute_accuracy(model, val_data, device, n_samples=5000)

    console.print(f"  Val Loss: {losses['val']:.4f}")
    console.print(f"  Val PPL:  {torch.exp(torch.tensor(losses['val'])).item():.2f}")
    console.print(f"  Top-1 Accuracy: {top1:.2f}%")
    console.print(f"  Top-5 Accuracy: {top5:.2f}%")

    # Compare with random baseline
    # With ~13 unique bins, random baseline is ~7.7% for top-1
    random_top1 = 100 / 13  # Approximate
    console.print(f"\n  Random baseline (13 bins): ~{random_top1:.1f}%")
    console.print(f"  Improvement: {top1 / random_top1:.1f}× better than random")

    console.print("\n[bold green]═══ TRAINING COMPLETE ═══[/]")


if __name__ == "__main__":
    main()
