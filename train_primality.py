#!/usr/bin/env python3
"""
🔮 PRIMALITY ORACLE: Train model to detect prime numbers

Instead of predicting gaps, we train a classifier:
Input:  Number N (as sequence of digits)
Output: 1 if prime, 0 if composite

If this works with high accuracy - we have a neural primality test!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import track
import time

console = Console()

# ============================================================
# CONFIG
# ============================================================
MAX_NUMBER = 1_000_000      # Train on numbers up to 1M
MAX_DIGITS = 7              # Max digits (1M = 7 digits)
TRAIN_SIZE = 200_000        # Training samples
VAL_SIZE = 20_000           # Validation samples
BATCH_SIZE = 256
MAX_STEPS = 5000
LEARNING_RATE = 1e-3

OUTPUT_DIR = Path("out_primality")


def sieve_primes(limit):
    """Generate set of primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return set(np.where(is_prime)[0])


def number_to_digits(n, max_len=MAX_DIGITS):
    """Convert number to digit sequence with padding."""
    digits = [int(d) for d in str(n)]
    # Pad with 10 (special padding token)
    padding = [10] * (max_len - len(digits))
    return padding + digits


def create_dataset(primes_set, n_samples, max_num):
    """Create balanced dataset of primes and composites."""
    # Get lists
    primes_list = [p for p in primes_set if p <= max_num and p >= 2]
    composites_list = [n for n in range(4, max_num + 1) if n not in primes_set]

    # Sample equally
    n_each = n_samples // 2

    sampled_primes = np.random.choice(primes_list, min(n_each, len(primes_list)), replace=False)
    sampled_composites = np.random.choice(composites_list, min(n_each, len(composites_list)), replace=False)

    # Create data
    X = []
    y = []

    for p in sampled_primes:
        X.append(number_to_digits(p))
        y.append(1)

    for c in sampled_composites:
        X.append(number_to_digits(c))
        y.append(0)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]

    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float)


class PrimalityTransformer(nn.Module):
    """
    Transformer for primality detection.

    Takes digit sequence, outputs probability of being prime.
    """
    def __init__(self, vocab_size=11, d_model=64, n_heads=4, n_layers=3, max_len=MAX_DIGITS):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)  # 0-9 + padding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len]
        B, T = x.shape

        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        # Embeddings
        tok_emb = self.embedding(x)
        pos_emb = self.pos_embedding(pos)
        h = tok_emb + pos_emb

        # Transformer
        h = self.transformer(h)

        # Pool (mean over sequence)
        h = h.mean(dim=1)

        # Classify
        logits = self.classifier(h).squeeze(-1)

        return logits


def train():
    console.print("[bold magenta]═══ 🔮 PRIMALITY ORACLE TRAINING ═══[/]\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/]")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate primes
    console.print(f"[cyan]Generating primes up to {MAX_NUMBER:,}...[/]")
    primes_set = sieve_primes(MAX_NUMBER)
    console.print(f"[green]Found {len(primes_set):,} primes[/]")

    # Create datasets
    console.print("[cyan]Creating datasets...[/]")
    X_train, y_train = create_dataset(primes_set, TRAIN_SIZE, int(MAX_NUMBER * 0.8))
    X_val, y_val = create_dataset(primes_set, VAL_SIZE, MAX_NUMBER)

    console.print(f"[green]Train: {len(X_train):,}, Val: {len(X_val):,}[/]")
    console.print(f"[green]Train primes: {y_train.sum().item():.0f}, composites: {(1-y_train).sum().item():.0f}[/]")

    # Create model
    console.print("[cyan]Creating model...[/]")
    model = PrimalityTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Parameters: {n_params:,}[/]")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Training loop
    console.print(f"\n[bold]Training for {MAX_STEPS} steps...[/]\n")

    model.train()
    start_time = time.time()

    for step in range(MAX_STEPS):
        # Get batch
        idx = torch.randint(0, len(X_train), (BATCH_SIZE,))
        x_batch = X_train[idx]
        y_batch = y_train[idx]

        # Forward
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % 500 == 0 or step == MAX_STEPS - 1:
            model.eval()
            with torch.no_grad():
                # Train accuracy
                train_logits = model(X_train[:5000])
                train_preds = (torch.sigmoid(train_logits) > 0.5).float()
                train_acc = (train_preds == y_train[:5000]).float().mean().item() * 100

                # Val accuracy
                val_logits = model(X_val)
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item() * 100

                # Precision/Recall for primes
                prime_mask = y_val == 1
                pred_prime = val_preds == 1

                true_pos = (pred_prime & prime_mask).sum().item()
                false_pos = (pred_prime & ~prime_mask).sum().item()
                false_neg = (~pred_prime & prime_mask).sum().item()

                precision = true_pos / (true_pos + false_pos + 1e-8) * 100
                recall = true_pos / (true_pos + false_neg + 1e-8) * 100

            elapsed = time.time() - start_time
            console.print(
                f"Step {step:5d} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {train_acc:.1f}%/{val_acc:.1f}% | "
                f"P/R: {precision:.1f}%/{recall:.1f}% | "
                f"Time: {elapsed:.1f}s"
            )
            model.train()

    # Save model
    console.print("\n[cyan]Saving model...[/]")
    torch.save({
        "model": model.state_dict(),
        "config": {
            "vocab_size": 11,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "max_len": MAX_DIGITS
        }
    }, OUTPUT_DIR / "primality_oracle.pt")

    # Final evaluation
    console.print("\n[bold]═══ FINAL EVALUATION ═══[/]")

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_preds = (torch.sigmoid(val_logits) > 0.5).float()
        val_acc = (val_preds == y_val).float().mean().item() * 100

        prime_mask = y_val == 1
        pred_prime = val_preds == 1

        true_pos = (pred_prime & prime_mask).sum().item()
        false_pos = (pred_prime & ~prime_mask).sum().item()
        false_neg = (~pred_prime & prime_mask).sum().item()
        true_neg = (~pred_prime & ~prime_mask).sum().item()

        precision = true_pos / (true_pos + false_pos + 1e-8) * 100
        recall = true_pos / (true_pos + false_neg + 1e-8) * 100
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

    console.print(f"  Accuracy:  {val_acc:.2f}%")
    console.print(f"  Precision: {precision:.2f}%")
    console.print(f"  Recall:    {recall:.2f}%")
    console.print(f"  F1 Score:  {f1:.2f}%")
    console.print(f"\n  Confusion Matrix:")
    console.print(f"    True Primes detected:    {true_pos:.0f}")
    console.print(f"    False Primes (errors):   {false_pos:.0f}")
    console.print(f"    Missed Primes:           {false_neg:.0f}")
    console.print(f"    True Composites:         {true_neg:.0f}")

    # Test on specific numbers
    console.print("\n[bold]═══ SPECIFIC NUMBER TESTS ═══[/]")
    test_numbers = [2, 3, 7, 11, 13, 17, 97, 100, 101, 1000, 1009, 9973, 10000, 104729]

    for n in test_numbers:
        digits = torch.tensor([number_to_digits(n)], dtype=torch.long, device=device)
        with torch.no_grad():
            logit = model(digits)
            prob = torch.sigmoid(logit).item()
            pred = "PRIME" if prob > 0.5 else "COMPOSITE"
            actual = "PRIME" if n in primes_set else "COMPOSITE"
            match = "✓" if pred == actual else "✗"

        console.print(f"  {n:>8}: {prob:.3f} → {pred:<10} (actual: {actual}) {match}")

    console.print("\n[bold green]═══ TRAINING COMPLETE ═══[/]")


if __name__ == "__main__":
    train()
