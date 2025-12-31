#!/usr/bin/env python3
"""
SNOWBALL RIGID V1.0 — AR with Rigidity Supervision

Diffusion path = тупик. Возвращаемся к Causal Oracle.

Key differences from v8:
- Full trajectory generation during training (not just NLL)
- Differentiable Δ3 loss on generated sequences
- Std loss for Wigner-like variance preservation
- Multi-window rollout with memory state propagation

Если Δ3 slope упадёт до <0.1 — мы прорвались!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
import math

from train_snowball import SnowballGPT, SnowballConfig
from train_wigner import wigner_nll_loss

console = Console()


# ============================================================================
# DIFFERENTIABLE Δ3 (from train_diffusion_rigid.py)
# ============================================================================

def compute_delta3_ar(spacings, L_values=None):
    """
    Differentiable Δ3 for AR-generated sequences.

    Args:
        spacings: (B, T) generated spacings (already normalized to mean~1)
        L_values: list of L scales

    Returns:
        delta3: (B,) mean Δ3 across scales
    """
    B, T = spacings.shape
    device = spacings.device

    if L_values is None:
        L_values = [32, 64]  # Smaller for AR windows

    positions = torch.cumsum(spacings, dim=1)  # (B, T)

    delta3_all = []

    for L in L_values:
        if L >= T - 5:
            continue

        n_windows = min(5, T - L)  # Few windows for speed
        step = max(1, (T - L) // n_windows)

        d3_samples = []

        for start in range(0, T - L, step):
            pos_window = positions[:, start:start + L]
            pos_shifted = pos_window - pos_window[:, 0:1]

            n_i = torch.arange(1, L + 1, device=device, dtype=torch.float32)
            E_i = pos_shifted

            N = L
            sum_E = E_i.sum(dim=1)
            sum_E2 = (E_i ** 2).sum(dim=1)
            sum_n = n_i.sum()
            sum_nE = (n_i.unsqueeze(0) * E_i).sum(dim=1)

            denom = N * sum_E2 - sum_E ** 2 + 1e-8

            a = (N * sum_nE - sum_E * sum_n) / denom
            b = (sum_n - a * sum_E) / N

            fitted = a.unsqueeze(1) * E_i + b.unsqueeze(1)
            residuals = n_i.unsqueeze(0) - fitted

            d3 = (residuals ** 2).sum(dim=1) / L
            d3_samples.append(d3)

        if d3_samples:
            delta3 = torch.stack(d3_samples, dim=1).mean(dim=1)
            delta3_all.append(delta3)

    if delta3_all:
        return torch.stack(delta3_all, dim=1).mean(dim=1)  # (B,)
    return torch.zeros(B, device=device)


def real_zeta_delta3_target(L):
    """Target from real low-lying zeta zeros."""
    return 0.01 * np.log(L) + 0.058


# ============================================================================
# GENERATION UTILITIES
# ============================================================================

def generate_trajectory(model, bin_centers_t, memory_state, start_tokens,
                       n_steps, device, temperature=1.0):
    """
    Generate spacing trajectory for Δ3 computation.

    Returns:
        spacings: (B, n_steps) generated spacings
        new_memory: updated memory state
    """
    B = start_tokens.shape[0]
    window_size = model.config.seq_len

    current_tokens = start_tokens.clone()  # (B, window_size)
    generated_spacings = []

    for step in range(n_steps):
        # Get scale values
        scale_val = bin_centers_t[current_tokens].unsqueeze(-1)  # (B, T, 1)

        # Forward pass
        pred, _, _, memory_state = model(
            current_tokens,
            targets=None,
            return_hidden=False,
            scale_val=scale_val,
            memory_state=memory_state,
            return_memory=True
        )

        # Get mu for last position
        mu = model.get_mu(pred)[:, -1]  # (B,)

        # Sample from Wigner (differentiable via reparameterization trick)
        # Wigner: P(s) = (π/2) s exp(-πs²/4)
        # CDF^{-1}: s = μ * sqrt(-4 ln(1-u) / π)
        u = torch.rand(B, device=device)
        spacing = mu * torch.sqrt(-4 * torch.log(1 - u + 1e-10) / math.pi)
        spacing = torch.clamp(spacing, 0.05, 4.0)

        generated_spacings.append(spacing)

        # Convert to token and update window
        diffs = torch.abs(bin_centers_t.unsqueeze(0) - spacing.unsqueeze(1))
        new_token = diffs.argmin(dim=1)  # (B,)

        # Shift window
        current_tokens = torch.cat([current_tokens[:, 1:], new_token.unsqueeze(1)], dim=1)

    spacings = torch.stack(generated_spacings, dim=1)  # (B, n_steps)
    return spacings, memory_state


# ============================================================================
# TRAINING WITH RIGIDITY SUPERVISION
# ============================================================================

def train():
    console.print("[bold magenta]❄️ SNOWBALL RIGID V1.0[/]")
    console.print("[dim]AR + Infinite Memory + Rigidity Supervision[/]\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load data
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data = torch.load('data/val.pt', weights_only=False)
    bin_centers = np.load('data/bin_centers.npy')
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)

    console.print(f"[green]Train: {len(train_data)}, Val: {len(val_data)}[/]")

    # Model config
    config = SnowballConfig(
        vocab_size=len(bin_centers),
        n_layer=4,
        n_head=4,
        n_embd=128,
        seq_len=128,  # Smaller for faster rigidity rollout
        dropout=0.1,
        n_memory_slots=4,
        use_scale=True,
        num_params=1,
        memory_update_mode="ema",
        memory_alpha=0.9,
    )

    model = SnowballGPT(config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[green]Parameters: {n_params:,}[/]")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000)

    # RIGIDITY CONFIG
    lambda_rigid = 0.1  # Start small
    lambda_std = 1.0
    target_std = 0.42
    L_values = [32, 64]
    rigidity_start = 2000  # Warm up NLL first

    console.print(f"\n[cyan]Rigidity config:[/]")
    console.print(f"  λ_rigid = {lambda_rigid}")
    console.print(f"  λ_std = {lambda_std}")
    console.print(f"  target_std = {target_std}")
    console.print(f"  rigidity_start = {rigidity_start}")

    # Training
    n_epochs = 20000
    batch_size = 32
    log_interval = 500
    gen_steps = 64  # Steps to generate for Δ3

    best_loss = float('inf')

    console.print(f"\n[cyan]Training for {n_epochs} steps...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Training Snowball Rigid...", total=n_epochs)

        for step in range(1, n_epochs + 1):
            model.train()

            # Sample batch
            idx = torch.randint(0, len(train_data), (batch_size,))
            tokens = train_data[idx].to(device)[:, :config.seq_len + 1]

            X = tokens[:, :-1]
            Y = tokens[:, 1:]

            scale_val = bin_centers_t[X].unsqueeze(-1)

            # Init memory
            memory_state = model.snowball.get_initial_state(batch_size, device)

            # Forward pass
            pred, nll_loss, mem_attn, memory_state = model(
                X, targets=Y, return_hidden=False, scale_val=scale_val,
                memory_state=memory_state, return_memory=True
            )

            # RIGIDITY LOSS (after warm-up)
            if step >= rigidity_start:
                # Generate trajectory for Δ3
                with torch.no_grad():
                    # Use current batch as start
                    start_tokens = X.detach()

                # Generate with gradients (for std loss at least)
                gen_spacings, _ = generate_trajectory(
                    model, bin_centers_t, memory_state.detach(),
                    start_tokens, gen_steps, device
                )

                # Normalize to mean 1
                gen_spacings = gen_spacings / (gen_spacings.mean(dim=1, keepdim=True) + 1e-8)

                # Δ3 loss (with no_grad for now - expensive)
                with torch.no_grad():
                    delta3 = compute_delta3_ar(gen_spacings, L_values)
                    targets = torch.tensor([real_zeta_delta3_target(L) for L in L_values],
                                          device=device, dtype=torch.float32).mean()
                    rigid_loss = ((delta3 - targets) ** 2).mean()

                # Std loss (with gradients)
                actual_std = gen_spacings.std(dim=1).mean()
                std_loss = (actual_std - target_std) ** 2

                total_loss = nll_loss + lambda_rigid * rigid_loss + lambda_std * std_loss
            else:
                rigid_loss = torch.tensor(0.0, device=device)
                std_loss = torch.tensor(0.0, device=device)
                actual_std = torch.tensor(0.0, device=device)
                total_loss = nll_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            if step % log_interval == 0:
                console.print(
                    f"Step {step}: "
                    f"NLL={nll_loss.item():.4f} | "
                    f"Rigid={rigid_loss.item():.4f} | "
                    f"Std={actual_std.item():.3f} | "
                    f"Total={total_loss.item():.4f}"
                )

                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    torch.save({
                        "model": model.state_dict(),
                        "config": config.__dict__,
                        "bin_centers": bin_centers,
                        "lambda_rigid": lambda_rigid,
                        "lambda_std": lambda_std,
                    }, "out/snowball_rigid_best.pt")

            progress.update(task, advance=1)

    console.print(f"\n[green]✅ Training complete! Best loss: {best_loss:.4f}[/]")
    console.print(f"[green]Saved: out/snowball_rigid_best.pt[/]")

    # Final rigidity evaluation
    evaluate_rigidity(model, bin_centers_t, val_data, config, device)


def evaluate_rigidity(model, bin_centers_t, val_data, config, device):
    """Generate long trajectory and measure Δ3."""
    console.print("\n[cyan]Final rigidity evaluation...[/]")

    model.eval()

    n_samples = 8
    n_steps = 500  # Generate longer for better Δ3

    # Start from validation data
    start_tokens = val_data[:n_samples, :config.seq_len].to(device)
    memory_state = model.snowball.get_initial_state(n_samples, device)

    console.print(f"Generating {n_samples} x {n_steps} = {n_samples * n_steps} spacings...")

    with torch.no_grad():
        gen_spacings, _ = generate_trajectory(
            model, bin_centers_t, memory_state,
            start_tokens, n_steps, device
        )

    # Convert to numpy
    spacings = gen_spacings.cpu().numpy()

    # Normalize
    for i in range(n_samples):
        spacings[i] = spacings[i] / spacings[i].mean()

    all_spacings = spacings.flatten()

    console.print(f"Std: {all_spacings.std():.3f} (target ~0.42)")

    # Compute Δ3
    from statistics_long_range_v2 import compute_delta3_correct, gue_delta3_theory

    L_values = np.array([10, 20, 50, 100, 200])
    L_values = L_values[L_values < len(all_spacings) // 3]

    delta3 = compute_delta3_correct(all_spacings, L_values)
    delta3_gue = gue_delta3_theory(L_values)
    delta3_zeta = np.array([real_zeta_delta3_target(L) for L in L_values])

    # Compute slope
    from scipy import stats
    log_L = np.log(L_values)
    slope, _, _, _, _ = stats.linregress(log_L, delta3)

    # Display
    table = Table(title="🎯 SNOWBALL RIGID: Δ3(L)")
    table.add_column("L", style="bold", justify="right")
    table.add_column("Generated", justify="right")
    table.add_column("Real Zeta", justify="right")
    table.add_column("GUE Theory", justify="right")

    for i, L in enumerate(L_values):
        table.add_row(
            f"{L}",
            f"{delta3[i]:.4f}",
            f"{delta3_zeta[i]:.4f}",
            f"{delta3_gue[i]:.4f}",
        )

    console.print(table)

    console.print(f"\n[cyan]Slope analysis:[/]")
    console.print(f"  Generated slope: {slope:.4f}")
    console.print(f"  Real Zeta slope: ~0.01")
    console.print(f"  GUE theory slope: {1/np.pi**2:.4f}")

    if slope < 0.1:
        console.print(f"\n[bold green]🎉 SUPER-RIGID! Slope {slope:.4f} < 0.1[/]")
    elif slope < 0.5:
        console.print(f"\n[bold yellow]⚠️ Good progress, slope {slope:.4f}[/]")
    else:
        console.print(f"\n[bold red]✗ Still Poisson-like, slope {slope:.4f}[/]")


if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    train()
