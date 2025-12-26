#!/usr/bin/env python3
"""
🔮 PRIMALITY ORACLE v2: With Modular Features

Key insight: All primes > 3 are of form 6k±1

We add explicit modular features:
- n mod 2 (even/odd)
- n mod 3
- n mod 6
- digit sum mod 3 (divisibility by 3)
- last digit (divisibility by 2, 5)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from rich.console import Console
import time

console = Console()

# ============================================================
# CONFIG
# ============================================================
MAX_NUMBER = 1_000_000
MAX_DIGITS = 7
TRAIN_SIZE = 200_000
VAL_SIZE = 20_000
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


def extract_features(n):
    """
    Extract features for number n:
    - digits (padded to MAX_DIGITS)
    - n mod 2
    - n mod 3
    - n mod 6
    - digit_sum mod 3 (divisibility test for 3)
    - digit_sum mod 9
    - last_digit
    - is_special (n == 2 or n == 3)
    """
    # Digits
    digits = [int(d) for d in str(n)]
    padding = [10] * (MAX_DIGITS - len(digits))
    padded_digits = padding + digits

    # Modular features
    mod_2 = n % 2
    mod_3 = n % 3
    mod_6 = n % 6

    # Digit sum (for divisibility by 3 and 9)
    digit_sum = sum(digits)
    dsum_mod_3 = digit_sum % 3
    dsum_mod_9 = digit_sum % 9

    # Last digit
    last_digit = digits[-1]

    # Special cases
    is_2 = 1 if n == 2 else 0
    is_3 = 1 if n == 3 else 0
    is_5 = 1 if n == 5 else 0

    # 6k±1 test (necessary but not sufficient for prime)
    is_6k_pm_1 = 1 if (n > 3 and (n % 6 == 1 or n % 6 == 5)) else 0

    mod_features = [mod_2, mod_3, mod_6, dsum_mod_3, dsum_mod_9,
                    last_digit, is_2, is_3, is_5, is_6k_pm_1]

    return padded_digits, mod_features


def create_dataset(primes_set, n_samples, max_num):
    """Create balanced dataset with modular features."""
    primes_list = [p for p in primes_set if 2 <= p <= max_num]
    composites_list = [n for n in range(4, max_num + 1) if n not in primes_set]

    n_each = n_samples // 2

    sampled_primes = np.random.choice(primes_list, min(n_each, len(primes_list)), replace=False)
    sampled_composites = np.random.choice(composites_list, min(n_each, len(composites_list)), replace=False)

    X_digits = []
    X_mod = []
    y = []

    for p in sampled_primes:
        digits, mod_feats = extract_features(p)
        X_digits.append(digits)
        X_mod.append(mod_feats)
        y.append(1)

    for c in sampled_composites:
        digits, mod_feats = extract_features(c)
        X_digits.append(digits)
        X_mod.append(mod_feats)
        y.append(0)

    # Shuffle
    indices = np.random.permutation(len(y))
    X_digits = [X_digits[i] for i in indices]
    X_mod = [X_mod[i] for i in indices]
    y = [y[i] for i in indices]

    return (torch.tensor(X_digits, dtype=torch.long),
            torch.tensor(X_mod, dtype=torch.float),
            torch.tensor(y, dtype=torch.float))


class PrimalityOracleV2(nn.Module):
    """
    Enhanced primality detector with modular features.
    """
    def __init__(self, vocab_size=11, d_model=64, n_heads=4, n_layers=3,
                 max_len=MAX_DIGITS, n_mod_features=10):
        super().__init__()

        # Digit processing (transformer)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Modular features processing
        self.mod_mlp = nn.Sequential(
            nn.Linear(n_mod_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, digits, mod_features):
        B, T = digits.shape

        # Process digits
        pos = torch.arange(T, device=digits.device).unsqueeze(0).expand(B, -1)
        h = self.embedding(digits) + self.pos_embedding(pos)
        h = self.transformer(h)
        h_digits = h.mean(dim=1)  # [B, d_model]

        # Process mod features
        h_mod = self.mod_mlp(mod_features)  # [B, 32]

        # Combine and classify
        h_combined = torch.cat([h_digits, h_mod], dim=1)
        logits = self.classifier(h_combined).squeeze(-1)

        return logits


def train():
    console.print("[bold magenta]═══ 🔮 PRIMALITY ORACLE v2 (with mod features) ═══[/]\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/]")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate primes
    console.print(f"[cyan]Generating primes up to {MAX_NUMBER:,}...[/]")
    primes_set = sieve_primes(MAX_NUMBER)
    console.print(f"[green]Found {len(primes_set):,} primes[/]")

    # Create datasets
    console.print("[cyan]Creating datasets with modular features...[/]")
    X_digits_train, X_mod_train, y_train = create_dataset(primes_set, TRAIN_SIZE, int(MAX_NUMBER * 0.8))
    X_digits_val, X_mod_val, y_val = create_dataset(primes_set, VAL_SIZE, MAX_NUMBER)

    console.print(f"[green]Train: {len(y_train):,}, Val: {len(y_val):,}[/]")

    # Show feature example
    console.print("\n[cyan]Feature example for n=97:[/]")
    digits, mod_feats = extract_features(97)
    console.print(f"  Digits: {digits}")
    console.print(f"  Mod features: {mod_feats}")
    console.print(f"  (mod2, mod3, mod6, dsum%3, dsum%9, last, is2, is3, is5, is6k±1)")

    # Create model
    console.print("\n[cyan]Creating model...[/]")
    model = PrimalityOracleV2().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Parameters: {n_params:,}[/]")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Move data
    X_digits_train = X_digits_train.to(device)
    X_mod_train = X_mod_train.to(device)
    y_train = y_train.to(device)
    X_digits_val = X_digits_val.to(device)
    X_mod_val = X_mod_val.to(device)
    y_val = y_val.to(device)

    # Training loop
    console.print(f"\n[bold]Training for {MAX_STEPS} steps...[/]\n")

    model.train()
    start_time = time.time()

    for step in range(MAX_STEPS):
        idx = torch.randint(0, len(y_train), (BATCH_SIZE,))

        logits = model(X_digits_train[idx], X_mod_train[idx])
        loss = criterion(logits, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == MAX_STEPS - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_digits_val, X_mod_val)
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item() * 100

                prime_mask = y_val == 1
                pred_prime = val_preds == 1

                tp = (pred_prime & prime_mask).sum().item()
                fp = (pred_prime & ~prime_mask).sum().item()
                fn = (~pred_prime & prime_mask).sum().item()

                precision = tp / (tp + fp + 1e-8) * 100
                recall = tp / (tp + fn + 1e-8) * 100

            elapsed = time.time() - start_time
            console.print(
                f"Step {step:5d} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {val_acc:.1f}% | "
                f"P/R: {precision:.1f}%/{recall:.1f}% | "
                f"Time: {elapsed:.1f}s"
            )
            model.train()

    # Final evaluation
    console.print("\n[bold]═══ FINAL EVALUATION ═══[/]")

    model.eval()
    with torch.no_grad():
        val_logits = model(X_digits_val, X_mod_val)
        val_preds = (torch.sigmoid(val_logits) > 0.5).float()
        val_acc = (val_preds == y_val).float().mean().item() * 100

        prime_mask = y_val == 1
        pred_prime = val_preds == 1

        tp = (pred_prime & prime_mask).sum().item()
        fp = (pred_prime & ~prime_mask).sum().item()
        fn = (~pred_prime & prime_mask).sum().item()
        tn = (~pred_prime & ~prime_mask).sum().item()

        precision = tp / (tp + fp + 1e-8) * 100
        recall = tp / (tp + fn + 1e-8) * 100
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

    console.print(f"  Accuracy:  {val_acc:.2f}%")
    console.print(f"  Precision: {precision:.2f}%")
    console.print(f"  Recall:    {recall:.2f}%")
    console.print(f"  F1 Score:  {f1:.2f}%")

    # Test specific numbers
    console.print("\n[bold]═══ SPECIFIC NUMBER TESTS ═══[/]")
    test_numbers = [2, 3, 4, 5, 6, 7, 9, 11, 15, 17, 21, 25, 29, 49, 97, 100, 101,
                    121, 143, 169, 289, 1009, 9973, 10007, 104729]

    for n in test_numbers:
        digits, mod_feats = extract_features(n)
        d_tensor = torch.tensor([digits], dtype=torch.long, device=device)
        m_tensor = torch.tensor([mod_feats], dtype=torch.float, device=device)

        with torch.no_grad():
            logit = model(d_tensor, m_tensor)
            prob = torch.sigmoid(logit).item()

        pred = "PRIME" if prob > 0.5 else "COMPOSITE"
        actual = "PRIME" if n in primes_set else "COMPOSITE"
        match = "✓" if pred == actual else "✗"

        # Show mod6 status
        mod6_status = "6k±1" if (n > 3 and n % 6 in [1, 5]) else "other"

        console.print(f"  {n:>8} ({mod6_status:>5}): {prob:.3f} → {pred:<10} (actual: {actual}) {match}")

    console.print("\n[bold green]═══ TRAINING COMPLETE ═══[/]")


if __name__ == "__main__":
    train()
