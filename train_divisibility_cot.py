#!/usr/bin/env python3
"""
🧠 CHAIN-OF-THOUGHT PRIMALITY: Learn Divisibility First

Step 1: Train model to check divisibility (N, d) → 0/1
Step 2: Use it to test all primes up to √N
Step 3: If no divisor found → PRIME

This is "neural trial division" - the model learns to compute N mod d.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import track
import time
import math

console = Console()

# ============================================================
# CONFIG
# ============================================================
MAX_NUMBER = 10_000       # Numbers to test
MAX_DIVISOR = 100         # Divisors to check (primes up to 100)
MAX_DIGITS = 5            # Max digits for number
BATCH_SIZE = 512
MAX_STEPS = 10000
LEARNING_RATE = 1e-3

OUTPUT_DIR = Path("out_cot")

# Small primes for trial division
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def number_to_digits(n, max_len):
    """Convert number to digit sequence."""
    digits = [int(d) for d in str(n)]
    padding = [10] * (max_len - len(digits))
    return padding + digits


def create_divisibility_dataset(n_samples, max_n, max_d):
    """
    Create dataset for divisibility checking.

    Each sample: (N, d, label) where label = 1 if d divides N, else 0
    """
    X_n = []  # Number digits
    X_d = []  # Divisor digits
    y = []    # 0 or 1

    # Balance: 50% divisible, 50% not
    n_each = n_samples // 2

    # Divisible pairs
    count = 0
    while count < n_each:
        d = np.random.choice(SMALL_PRIMES)
        if d > max_d:
            continue
        # Generate N that is divisible by d
        k = np.random.randint(1, max_n // d + 1)
        n = k * d
        if n > max_n or n < 2:
            continue

        X_n.append(number_to_digits(n, MAX_DIGITS))
        X_d.append(number_to_digits(d, 3))  # Divisors up to 3 digits
        y.append(1)
        count += 1

    # Non-divisible pairs
    count = 0
    while count < n_each:
        n = np.random.randint(2, max_n + 1)
        d = np.random.choice(SMALL_PRIMES)
        if d > max_d:
            continue
        if n % d == 0:  # Skip if actually divisible
            continue

        X_n.append(number_to_digits(n, MAX_DIGITS))
        X_d.append(number_to_digits(d, 3))
        y.append(0)
        count += 1

    # Shuffle
    indices = np.random.permutation(len(y))
    X_n = [X_n[i] for i in indices]
    X_d = [X_d[i] for i in indices]
    y = [y[i] for i in indices]

    return (torch.tensor(X_n, dtype=torch.long),
            torch.tensor(X_d, dtype=torch.long),
            torch.tensor(y, dtype=torch.float))


class DivisibilityChecker(nn.Module):
    """
    Model that checks if N is divisible by d.

    Input: (N as digits, d as digits)
    Output: probability that d divides N
    """
    def __init__(self, vocab_size=11, d_model=64, n_heads=4, n_layers=3):
        super().__init__()

        # Embeddings for N and d
        self.n_embedding = nn.Embedding(vocab_size, d_model)
        self.d_embedding = nn.Embedding(vocab_size, d_model)
        self.n_pos = nn.Embedding(MAX_DIGITS, d_model)
        self.d_pos = nn.Embedding(3, d_model)

        # Process N
        encoder_layer_n = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )
        self.transformer_n = nn.TransformerEncoder(encoder_layer_n, num_layers=n_layers)

        # Process d
        encoder_layer_d = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )
        self.transformer_d = nn.TransformerEncoder(encoder_layer_d, num_layers=2)

        # Combine and classify
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

    def forward(self, n_digits, d_digits):
        B = n_digits.shape[0]

        # Process N
        n_pos = torch.arange(n_digits.shape[1], device=n_digits.device).unsqueeze(0).expand(B, -1)
        h_n = self.n_embedding(n_digits) + self.n_pos(n_pos)
        h_n = self.transformer_n(h_n).mean(dim=1)

        # Process d
        d_pos = torch.arange(d_digits.shape[1], device=d_digits.device).unsqueeze(0).expand(B, -1)
        h_d = self.d_embedding(d_digits) + self.d_pos(d_pos)
        h_d = self.transformer_d(h_d).mean(dim=1)

        # Combine
        h = torch.cat([h_n, h_d], dim=1)
        logits = self.classifier(h).squeeze(-1)

        return logits


def sieve_primes(limit):
    """Generate set of primes."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return set(np.where(is_prime)[0])


def test_primality_cot(model, n, device, primes_set):
    """
    Test primality using chain-of-thought (trial division).

    Returns: (is_prime_prediction, reasoning_steps)
    """
    model.eval()
    sqrt_n = int(math.sqrt(n)) + 1

    reasoning = []

    for p in SMALL_PRIMES:
        if p > sqrt_n:
            break
        if p == n:
            continue

        # Check if p divides n
        n_digits = torch.tensor([number_to_digits(n, MAX_DIGITS)], dtype=torch.long, device=device)
        d_digits = torch.tensor([number_to_digits(p, 3)], dtype=torch.long, device=device)

        with torch.no_grad():
            logit = model(n_digits, d_digits)
            prob = torch.sigmoid(logit).item()

        divisible = prob > 0.5
        actual_divisible = (n % p == 0)

        reasoning.append({
            "divisor": p,
            "prob": prob,
            "predicted": divisible,
            "actual": actual_divisible
        })

        if divisible:
            # Found a divisor → composite
            return False, reasoning

    # No divisor found → prime
    return True, reasoning


def train():
    console.print("[bold magenta]═══ 🧠 CHAIN-OF-THOUGHT PRIMALITY ═══[/]\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/]")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Create datasets
    console.print("[cyan]Creating divisibility dataset...[/]")
    X_n_train, X_d_train, y_train = create_divisibility_dataset(100000, MAX_NUMBER, MAX_DIVISOR)
    X_n_val, X_d_val, y_val = create_divisibility_dataset(10000, MAX_NUMBER, MAX_DIVISOR)

    console.print(f"[green]Train: {len(y_train):,}, Val: {len(y_val):,}[/]")
    console.print(f"[green]Divisible: {y_train.sum().item():.0f}, Not: {(1-y_train).sum().item():.0f}[/]")

    # Create model
    console.print("[cyan]Creating DivisibilityChecker model...[/]")
    model = DivisibilityChecker().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Parameters: {n_params:,}[/]")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    X_n_train = X_n_train.to(device)
    X_d_train = X_d_train.to(device)
    y_train = y_train.to(device)
    X_n_val = X_n_val.to(device)
    X_d_val = X_d_val.to(device)
    y_val = y_val.to(device)

    console.print(f"\n[bold]Training divisibility checker for {MAX_STEPS} steps...[/]\n")

    model.train()
    start_time = time.time()

    for step in range(MAX_STEPS):
        idx = torch.randint(0, len(y_train), (BATCH_SIZE,))

        logits = model(X_n_train[idx], X_d_train[idx])
        loss = criterion(logits, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0 or step == MAX_STEPS - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_n_val, X_d_val)
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item() * 100

            elapsed = time.time() - start_time
            console.print(f"Step {step:5d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.1f}% | Time: {elapsed:.1f}s")
            model.train()

    # Save model
    torch.save({"model": model.state_dict()}, OUTPUT_DIR / "divisibility_checker.pt")

    # ============================================================
    # TEST PRIMALITY WITH CHAIN-OF-THOUGHT
    # ============================================================
    console.print("\n[bold]═══ PRIMALITY TEST (Chain-of-Thought) ═══[/]\n")

    primes_set = sieve_primes(MAX_NUMBER)

    test_numbers = [2, 3, 4, 7, 9, 11, 15, 17, 25, 29, 49, 51, 97, 100, 101,
                    121, 127, 143, 169, 289, 529, 841, 961, 1009, 2021, 9973]

    correct = 0
    total = 0

    for n in test_numbers:
        pred_prime, reasoning = test_primality_cot(model, n, device, primes_set)
        actual_prime = n in primes_set

        match = "✓" if pred_prime == actual_prime else "✗"
        if pred_prime == actual_prime:
            correct += 1
        total += 1

        pred_str = "PRIME" if pred_prime else "COMPOSITE"
        actual_str = "PRIME" if actual_prime else "COMPOSITE"

        # Show reasoning
        console.print(f"\n  {n}: {pred_str} (actual: {actual_str}) {match}")

        if len(reasoning) <= 5:
            for r in reasoning:
                status = "÷" if r["predicted"] else "✗"
                actual_status = "÷" if r["actual"] else "✗"
                console.print(f"    {n} mod {r['divisor']:2d} → {r['prob']:.2f} [{status}] (actual: [{actual_status}])")
        else:
            # Show abbreviated
            for r in reasoning[:3]:
                status = "÷" if r["predicted"] else "✗"
                console.print(f"    {n} mod {r['divisor']:2d} → {r['prob']:.2f} [{status}]")
            console.print(f"    ... ({len(reasoning)} steps total)")

    console.print(f"\n[bold]Chain-of-Thought Accuracy: {correct}/{total} = {correct/total*100:.1f}%[/]")

    console.print("\n[bold green]═══ COMPLETE ═══[/]")


if __name__ == "__main__":
    train()
