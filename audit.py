#!/usr/bin/env python3
"""
Audit tests to verify no data leakage or artifacts.
"""

import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.table import Table

from model.gpt import SpacingGPT

console = Console()


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SpacingGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def eval_ppl(model, data: torch.Tensor, device: torch.device,
             shift_targets: int = 0, reverse: bool = False) -> float:
    """
    Compute perplexity with optional modifications.

    Args:
        shift_targets: if > 0, randomly shift targets to break correspondence
        reverse: if True, reverse sequences
    """
    if reverse:
        data = data.flip(dims=[1])

    data = data.to(device)

    if shift_targets > 0:
        # Randomly roll targets to break input-output correspondence
        targets = data.clone()
        for i in range(len(targets)):
            shift = torch.randint(1, shift_targets + 1, (1,)).item()
            targets[i] = torch.roll(targets[i], shifts=shift)
    else:
        targets = data

    total_loss = 0.0
    n_batches = 0

    loader = DataLoader(TensorDataset(data, targets), batch_size=32, shuffle=False)

    for batch_x, batch_t in loader:
        batch_x = batch_x.to(device)
        batch_t = batch_t.to(device)

        logits, _ = model(batch_x)
        # Manual loss with potentially shifted targets
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, model.config.vocab_size),
            batch_t[:, 1:].contiguous().view(-1)
        )
        total_loss += loss.item()
        n_batches += 1

    return math.exp(total_loss / n_batches)


@torch.no_grad()
def eval_context_ablation(model, data: torch.Tensor, device: torch.device,
                          context_lengths: list) -> dict:
    """Evaluate ppl at different context lengths."""
    results = {}
    data = data.to(device)

    for ctx_len in context_lengths:
        if ctx_len >= data.shape[1]:
            continue

        # Truncate sequences to ctx_len
        truncated = data[:, :ctx_len]

        total_loss = 0.0
        n_batches = 0

        loader = DataLoader(TensorDataset(truncated), batch_size=32, shuffle=False)

        for (batch,) in loader:
            batch = batch.to(device)
            _, loss = model(batch, targets=batch)
            total_loss += loss.item()
            n_batches += 1

        results[ctx_len] = math.exp(total_loss / n_batches)

    return results


def main():
    console.print("[bold magenta]═══ AUDIT TESTS ═══[/]\n")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    console.print(f"[cyan]Device: {device}[/]\n")

    # Load model and data
    model = load_model(Path("out/best.pt"), device)
    val_data = torch.load("data/val.pt")

    console.print(f"Val data shape: {val_data.shape}\n")

    # Reference: entropy floor
    ENTROPY_FLOOR = 105.2

    results = Table(title="Audit Results")
    results.add_column("Test", style="cyan")
    results.add_column("PPL", style="green")
    results.add_column("vs Floor", style="yellow")
    results.add_column("Status", style="bold")

    # Test 1: Normal evaluation (baseline)
    console.print("[bold]Test 1: Normal evaluation[/]")
    ppl_normal = eval_ppl(model, val_data, device)
    console.print(f"  PPL: {ppl_normal:.1f}")
    status = "✓ BELOW" if ppl_normal < ENTROPY_FLOOR else "AT FLOOR"
    results.add_row("Normal", f"{ppl_normal:.1f}", f"{ppl_normal - ENTROPY_FLOOR:+.1f}", status)

    # Test 2: Shift test (break correspondence)
    console.print("\n[bold]Test 2: Shift test (targets shifted by 1-10)[/]")
    ppl_shift = eval_ppl(model, val_data, device, shift_targets=10)
    console.print(f"  PPL: {ppl_shift:.1f}")
    status = "✓ AT FLOOR" if ppl_shift >= ENTROPY_FLOOR * 0.95 else "⚠ SUSPICIOUS"
    results.add_row("Shift +1-10", f"{ppl_shift:.1f}", f"{ppl_shift - ENTROPY_FLOOR:+.1f}", status)

    # Test 3: Reverse test
    console.print("\n[bold]Test 3: Reverse test (sequences reversed)[/]")
    ppl_reverse = eval_ppl(model, val_data, device, reverse=True)
    console.print(f"  PPL: {ppl_reverse:.1f}")
    delta = ppl_reverse - ppl_normal
    status = "✓ CAUSAL" if delta > 5 else "≈ SYMMETRIC"
    results.add_row("Reversed", f"{ppl_reverse:.1f}", f"{delta:+.1f} vs normal", status)

    # Test 4: Context ablation
    console.print("\n[bold]Test 4: Context length ablation[/]")
    ctx_lengths = [16, 32, 64, 128, 256]
    ctx_results = eval_context_ablation(model, val_data, device, ctx_lengths)

    for ctx, ppl in ctx_results.items():
        console.print(f"  ctx={ctx}: PPL={ppl:.1f}")
        results.add_row(f"Context {ctx}", f"{ppl:.1f}", f"{ppl - ENTROPY_FLOOR:+.1f}", "")

    console.print()
    console.print(results)

    # Summary
    console.print("\n[bold]Summary:[/]")
    console.print(f"  Entropy floor: {ENTROPY_FLOOR}")
    console.print(f"  Normal PPL: {ppl_normal:.1f} ({ppl_normal/ENTROPY_FLOOR*100:.1f}% of floor)")
    console.print(f"  Shift test: {'PASSED' if ppl_shift >= ENTROPY_FLOOR * 0.95 else 'FAILED'}")
    console.print(f"  Reverse delta: {ppl_reverse - ppl_normal:+.1f} (causal signal)")

    # Context trend
    if ctx_results:
        trend = list(ctx_results.values())
        if all(trend[i] >= trend[i+1] for i in range(len(trend)-1)):
            console.print("  Context trend: PPL improves with more context ✓")
        else:
            console.print("  Context trend: Non-monotonic")


if __name__ == "__main__":
    main()
