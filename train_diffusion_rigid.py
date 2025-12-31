#!/usr/bin/env python3
"""
RIGIDITY-GUIDED DIFFUSION v1.0 — Форсим арифметическую жёсткость!

Ключевая идея:
- Diffusion генерит всю sequence целиком → можем считать Δ3
- Δ3 fully differentiable → градиенты толкают к flat/slow-log rigidity
- Если Δ3 станет ~0.11-0.15 constant → мы прорвали ceiling

Loss = MSE_denoise + λ * |Δ3(generated) - target|²

Если не поможет → архитектура не может represent arithmetic rigidity.
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

# Import from base diffusion
from train_diffusion import (
    ZetaDiffusionTransformer,
    DiffusionSchedule,
    cosine_beta_schedule,
)

console = Console()


# ============================================================================
# DIFFERENTIABLE Δ3 (SPECTRAL RIGIDITY)
# ============================================================================

def compute_delta3_differentiable(spacings, L_values=None):
    """
    Fully differentiable Δ3 computation.

    Args:
        spacings: (B, Seq) normalized spacings (mean ~1)
        L_values: list of L scales to compute Δ3 at

    Returns:
        delta3: (B, len(L_values)) Δ3 values at each scale
    """
    B, Seq = spacings.shape
    device = spacings.device

    if L_values is None:
        L_values = [32, 64, 128]  # smaller for 255 seq_len

    # Positions = cumulative sum of spacings
    positions = torch.cumsum(spacings, dim=1)  # (B, Seq)

    delta3_list = []

    for L in L_values:
        if L >= Seq - 10:
            # Not enough room for windows
            delta3_list.append(torch.zeros(B, device=device))
            continue

        n_windows = min(20, Seq - L - 1)  # Limit windows for speed
        step = max(1, (Seq - L - 1) // n_windows)

        d3_samples = []

        for start in range(0, Seq - L - 1, step):
            # Get window of positions
            pos_window = positions[:, start:start + L]  # (B, L)

            # Shift to start at 0
            pos_shifted = pos_window - pos_window[:, 0:1]  # (B, L)

            # Level counting: n(E) = number of levels up to E
            # For each position, n_i = i+1 (1-indexed)
            n_i = torch.arange(1, L + 1, device=device, dtype=torch.float32)  # (L,)
            E_i = pos_shifted  # (B, L)

            # Least squares fit: n = a*E + b
            # a = (N*sum(nE) - sum(n)*sum(E)) / (N*sum(E²) - sum(E)²)
            # b = (sum(n) - a*sum(E)) / N

            N = L
            sum_E = E_i.sum(dim=1)  # (B,)
            sum_E2 = (E_i ** 2).sum(dim=1)  # (B,)
            sum_n = n_i.sum()  # scalar
            sum_nE = (n_i.unsqueeze(0) * E_i).sum(dim=1)  # (B,)

            denom = N * sum_E2 - sum_E ** 2 + 1e-8  # (B,)

            a = (N * sum_nE - sum_E * sum_n) / denom  # (B,)
            b = (sum_n - a * sum_E) / N  # (B,)

            # Residuals
            fitted = a.unsqueeze(1) * E_i + b.unsqueeze(1)  # (B, L)
            residuals = n_i.unsqueeze(0) - fitted  # (B, L)

            # Δ3 = (1/L) * sum(residuals²)
            d3 = (residuals ** 2).sum(dim=1) / L  # (B,)
            d3_samples.append(d3)

        if d3_samples:
            delta3 = torch.stack(d3_samples, dim=1).mean(dim=1)  # (B,)
        else:
            delta3 = torch.zeros(B, device=device)

        delta3_list.append(delta3)

    return torch.stack(delta3_list, dim=1)  # (B, len(L_values))


def gue_delta3_target(L):
    """GUE theory: Δ3(L) = (1/π²) * (log(2πL) + γ - 5/4)"""
    return (1 / np.pi**2) * (np.log(2 * np.pi * L) + np.euler_gamma - 5/4)


def real_zeta_delta3_target(L):
    """
    Target from real low-lying zeta zeros.
    From debug_delta3_real_zeta.py: Δ3 ≈ 0.01 * log(L) + 0.058
    Super-rigid, almost constant!
    """
    return 0.01 * np.log(L) + 0.058


# ============================================================================
# TRAINING WITH RIGIDITY LOSS
# ============================================================================

def train_rigid():
    console.print("[bold magenta]🔥 RIGIDITY-GUIDED DIFFUSION v1.0[/]")
    console.print("[dim]Форсим super-rigidity через explicit Δ3 loss![/]\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load data
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data = torch.load('data/val.pt', weights_only=False)
    bin_centers = np.load('data/bin_centers.npy')
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)

    console.print(f"[green]Train sequences: {len(train_data)}[/]")
    console.print(f"[green]Val sequences: {len(val_data)}[/]")

    def tokens_to_spacings(tokens):
        """Convert token indices to normalized spacing sequences."""
        spacings = bin_centers_t[tokens.to(device)]
        spacings = spacings / (spacings.mean(dim=-1, keepdim=True) + 1e-8)
        return spacings.unsqueeze(-1)  # (B, L, 1)

    # Model config
    seq_len = 255
    dim = 128
    depth = 6
    n_heads = 8
    timesteps = 1000

    model = ZetaDiffusionTransformer(
        seq_len=seq_len,
        dim=dim,
        depth=depth,
        n_heads=n_heads,
        time_dim=128,
        dropout=0.1,
    ).to(device)

    schedule = DiffusionSchedule(timesteps=timesteps, schedule_type="cosine").to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[green]Parameters: {n_params:,}[/]")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30000)

    # RIGIDITY CONFIG
    lambda_rigid = 1.0  # Balanced (was 10.0, caused collapse)
    L_values = [32, 64, 128]  # Multi-scale

    # Targets from real zeta (super-rigid)
    targets = torch.tensor([
        real_zeta_delta3_target(L) for L in L_values
    ], device=device, dtype=torch.float32)

    console.print(f"\n[cyan]Rigidity config:[/]")
    console.print(f"  λ_rigid = {lambda_rigid}")
    console.print(f"  L scales = {L_values}")
    console.print(f"  Δ3 targets = {targets.cpu().numpy()}")

    # Training
    n_epochs = 30000
    batch_size = 32
    log_interval = 500
    rigid_start_epoch = 1000  # Warm up denoising first

    best_val_loss = float('inf')
    best_rigid_loss = float('inf')

    console.print(f"\n[cyan]Training for {n_epochs} steps...[/]")
    console.print(f"[yellow]Rigidity loss kicks in at epoch {rigid_start_epoch}[/]")

    history = {"mse": [], "rigid": [], "total": []}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Training Rigid Diffusion...", total=n_epochs)

        for step in range(1, n_epochs + 1):
            model.train()

            # Sample batch
            idx = torch.randint(0, len(train_data), (batch_size,))
            tokens = train_data[idx]
            x_0 = tokens_to_spacings(tokens[:, :-1])  # (B, 255, 1)

            # Sample timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device)

            # Forward diffusion
            noise = torch.randn_like(x_0)
            x_t = schedule.q_sample(x_0, t, noise)

            # Predict noise
            pred_noise = model(x_t, t)

            # MSE loss (denoising)
            mse_loss = F.mse_loss(pred_noise, noise)

            # RIGIDITY LOSS (after warm-up)
            if step >= rigid_start_epoch:
                # Approximate denoised sample (one-step prediction)
                # x_0_pred ≈ (x_t - sqrt(1-α) * pred_noise) / sqrt(α)
                sqrt_alpha = schedule.sqrt_alphas_cumprod[t]
                sqrt_one_minus_alpha = schedule.sqrt_one_minus_alphas_cumprod[t]

                while sqrt_alpha.dim() < x_t.dim():
                    sqrt_alpha = sqrt_alpha.unsqueeze(-1)
                    sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

                x_0_pred = (x_t - sqrt_one_minus_alpha * pred_noise) / (sqrt_alpha + 1e-8)

                # Ensure positive spacings
                x_0_pred = F.softplus(x_0_pred)  # Differentiable clamp to positive

                # Normalize to mean 1
                spacings_pred = x_0_pred.squeeze(-1)  # (B, 255)
                spacings_pred = spacings_pred / (spacings_pred.mean(dim=1, keepdim=True) + 1e-8)

                # Compute Δ3 at multiple scales
                delta3_pred = compute_delta3_differentiable(spacings_pred, L_values)  # (B, len(L))

                # Rigidity loss: push towards real zeta targets
                rigid_loss = ((delta3_pred - targets.unsqueeze(0)) ** 2).mean()

                # WIGNER VARIANCE LOSS: prevent collapse to constant spacings!
                # Wigner std ≈ 0.52, real zeta std ≈ 0.41
                target_std = 0.42  # Between Wigner and real zeta
                actual_std = spacings_pred.std(dim=1).mean()
                wigner_loss = (actual_std - target_std) ** 2

                # Balance: rigidity + variance preservation
                lambda_wigner = 5.0
                total_loss = mse_loss + lambda_rigid * rigid_loss + lambda_wigner * wigner_loss
            else:
                rigid_loss = torch.tensor(0.0, device=device)
                total_loss = mse_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            if step % log_interval == 0:
                model.eval()
                with torch.no_grad():
                    # Validation
                    val_idx = torch.randint(0, len(val_data), (64,))
                    val_tokens = val_data[val_idx]
                    val_x_0 = tokens_to_spacings(val_tokens[:, :-1])

                    val_t = torch.randint(0, timesteps, (64,), device=device)
                    val_noise = torch.randn_like(val_x_0)
                    val_x_t = schedule.q_sample(val_x_0, val_t, val_noise)
                    val_pred = model(val_x_t, val_t)
                    val_mse = F.mse_loss(val_pred, val_noise)

                    # Val rigidity
                    sqrt_alpha_v = schedule.sqrt_alphas_cumprod[val_t]
                    sqrt_one_minus_v = schedule.sqrt_one_minus_alphas_cumprod[val_t]
                    while sqrt_alpha_v.dim() < val_x_t.dim():
                        sqrt_alpha_v = sqrt_alpha_v.unsqueeze(-1)
                        sqrt_one_minus_v = sqrt_one_minus_v.unsqueeze(-1)

                    val_x0_pred = (val_x_t - sqrt_one_minus_v * val_pred) / (sqrt_alpha_v + 1e-8)
                    val_x0_pred = F.softplus(val_x0_pred)
                    val_spacings = val_x0_pred.squeeze(-1)
                    val_spacings = val_spacings / (val_spacings.mean(dim=1, keepdim=True) + 1e-8)

                    val_delta3 = compute_delta3_differentiable(val_spacings, L_values)
                    val_rigid = ((val_delta3 - targets.unsqueeze(0)) ** 2).mean()

                history["mse"].append(mse_loss.item())
                history["rigid"].append(rigid_loss.item())
                history["total"].append(total_loss.item())

                # Val std
                val_std = val_spacings.std(dim=1).mean().item()

                console.print(
                    f"Step {step}: "
                    f"MSE={mse_loss.item():.4f} | "
                    f"Rigid={rigid_loss.item():.4f} | "
                    f"Std={actual_std.item() if step >= rigid_start_epoch else 0:.3f} | "
                    f"Val_Δ3=[{val_delta3.mean(dim=0).cpu().numpy()}] | "
                    f"Val_std={val_std:.3f}"
                )

                # Save best
                combined_loss = val_mse.item() + lambda_rigid * val_rigid.item()
                if combined_loss < best_val_loss:
                    best_val_loss = combined_loss
                    best_rigid_loss = val_rigid.item()
                    torch.save({
                        "model": model.state_dict(),
                        "config": {
                            "seq_len": seq_len,
                            "dim": dim,
                            "depth": depth,
                            "n_heads": n_heads,
                            "timesteps": timesteps,
                        },
                        "lambda_rigid": lambda_rigid,
                        "L_values": L_values,
                        "targets": targets.cpu().numpy(),
                        "bin_centers": bin_centers,
                    }, "out/diffusion_rigid_best.pt")

            progress.update(task, advance=1)

    console.print(f"\n[green]✅ Training complete![/]")
    console.print(f"[green]Best combined loss: {best_val_loss:.4f}[/]")
    console.print(f"[green]Best rigid loss: {best_rigid_loss:.4f}[/]")
    console.print(f"[green]Saved: out/diffusion_rigid_best.pt[/]")

    # Final rigidity test
    console.print("\n[cyan]Final rigidity evaluation...[/]")
    evaluate_rigidity(model, schedule, device, bin_centers_t, val_data)


def evaluate_rigidity(model, schedule, device, bin_centers_t, val_data):
    """Generate samples and measure Δ3 at large scales."""
    model.eval()

    seq_len = 255
    n_samples = 16

    console.print(f"Generating {n_samples} samples for rigidity test...")

    with torch.no_grad():
        # Start from noise
        x = torch.randn(n_samples, seq_len, 1, device=device)

        # Full DDPM sampling (slow but accurate)
        for t_idx in reversed(range(schedule.timesteps)):
            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            x = schedule.p_sample(model, x, t)

        # Convert to spacings
        generated = x.squeeze(-1).cpu().numpy()
        generated = np.clip(generated, 0.01, 5.0)

        # Normalize each sequence
        for i in range(n_samples):
            generated[i] = generated[i] / generated[i].mean()

    # Concatenate for long-range analysis
    all_spacings = generated.flatten()
    console.print(f"Total spacings for analysis: {len(all_spacings)}")

    # Import proper Δ3 computation
    from statistics_long_range_v2 import compute_delta3_correct, gue_delta3_theory

    L_values = np.array([10, 20, 50, 100, 200, 500, 1000])
    L_values = L_values[L_values < len(all_spacings) // 3]

    delta3 = compute_delta3_correct(all_spacings, L_values)
    delta3_gue = gue_delta3_theory(L_values)
    delta3_zeta = np.array([real_zeta_delta3_target(L) for L in L_values])

    # Display
    table = Table(title="🎯 RIGIDITY TEST: Δ3(L)")
    table.add_column("L", style="bold", justify="right")
    table.add_column("Generated", justify="right")
    table.add_column("Real Zeta", justify="right")
    table.add_column("GUE Theory", justify="right")
    table.add_column("vs Zeta", justify="right")

    for i, L in enumerate(L_values):
        diff_zeta = abs(delta3[i] - delta3_zeta[i])
        color = "[green]" if diff_zeta < 0.05 else "[yellow]" if diff_zeta < 0.1 else "[red]"

        table.add_row(
            f"{L}",
            f"{delta3[i]:.4f}",
            f"{delta3_zeta[i]:.4f}",
            f"{delta3_gue[i]:.4f}",
            f"{color}{diff_zeta:.4f}[/]"
        )

    console.print(table)

    # Check slope
    from scipy import stats
    log_L = np.log(L_values)
    slope, intercept, r_value, _, _ = stats.linregress(log_L, delta3)

    console.print(f"\n[cyan]Rigidity slope analysis:[/]")
    console.print(f"  Generated slope: {slope:.4f}")
    console.print(f"  Real Zeta slope: ~0.01")
    console.print(f"  GUE theory slope: {1/np.pi**2:.4f}")

    if slope < 0.05:
        console.print(f"\n[bold green]🎉 SUPER-RIGID! Slope {slope:.4f} < 0.05[/]")
        console.print(f"[bold green]We broke the ceiling! Riemann would be proud![/]")
    elif slope < 0.15:
        console.print(f"\n[bold yellow]⚠️ GUE-like rigidity (slope ~{slope:.4f})[/]")
    else:
        console.print(f"\n[bold red]✗ Still Poisson-like (slope {slope:.4f})[/]")


if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    train_rigid()
