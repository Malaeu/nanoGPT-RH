#!/usr/bin/env python3
"""
RIGIDITY-GUIDED SAMPLING

Идея: На каждом шаге denoising добавляем градиент от Δ3 loss.
Толкаем генерацию в сторону rigid sequences.

Как classifier guidance, но вместо classifier — Δ3 objective.
"""

import torch
import torch.nn.functional as F
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track
from pathlib import Path

from train_diffusion import ZetaDiffusionTransformer, DiffusionSchedule
from train_diffusion_rigid import compute_delta3_differentiable, real_zeta_delta3_target

console = Console()


def guided_p_sample(model, schedule, x_t, t, guidance_scale=1.0, L_values=[32, 64, 128]):
    """
    Single reverse diffusion step WITH rigidity guidance.

    x_{t-1} = denoise(x_t) - guidance_scale * grad_x(Δ3_loss)
    """
    batch_size = x_t.shape[0]
    device = x_t.device

    # Enable gradients for guidance
    x_t_grad = x_t.clone().requires_grad_(True)

    # Predict noise
    with torch.no_grad():
        pred_noise = model(x_t_grad, t)

    # Compute x_0 prediction (for guidance)
    sqrt_alpha = schedule.sqrt_alphas_cumprod[t]
    sqrt_one_minus_alpha = schedule.sqrt_one_minus_alphas_cumprod[t]

    while sqrt_alpha.dim() < x_t.dim():
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

    # x_0 prediction
    x_0_pred = (x_t_grad - sqrt_one_minus_alpha * pred_noise) / (sqrt_alpha + 1e-8)
    x_0_pred = F.softplus(x_0_pred)  # Ensure positive

    # Normalize to mean 1
    spacings = x_0_pred.squeeze(-1)
    spacings = spacings / (spacings.mean(dim=1, keepdim=True) + 1e-8)

    # Compute Δ3 loss
    delta3 = compute_delta3_differentiable(spacings, L_values)
    targets = torch.tensor([real_zeta_delta3_target(L) for L in L_values],
                          device=device, dtype=torch.float32)

    delta3_loss = ((delta3 - targets.unsqueeze(0)) ** 2).sum()

    # Compute gradient w.r.t. x_t
    grad = torch.autograd.grad(delta3_loss, x_t_grad, retain_graph=False)[0]

    # Standard DDPM step (without grad)
    with torch.no_grad():
        beta = schedule.betas[t]
        sqrt_recip_alpha = schedule.sqrt_recip_alphas[t]
        sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[t]

        while beta.dim() < x_t.dim():
            beta = beta.unsqueeze(-1)
            sqrt_recip_alpha = sqrt_recip_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        # Mean prediction
        model_mean = sqrt_recip_alpha * (x_t - beta * pred_noise / sqrt_one_minus)

        # Apply guidance (subtract gradient to minimize Δ3 loss)
        model_mean = model_mean - guidance_scale * grad

        if t[0] == 0:
            return model_mean

        # Add noise
        posterior_var = schedule.posterior_variance[t]
        while posterior_var.dim() < x_t.dim():
            posterior_var = posterior_var.unsqueeze(-1)

        noise = torch.randn_like(x_t)
        x_prev = model_mean + torch.sqrt(posterior_var) * noise

    return x_prev


def sample_with_guidance(model, schedule, n_samples, seq_len, device,
                         guidance_scale=2.0, n_steps=100):
    """Generate samples with rigidity guidance."""

    console.print(f"[cyan]Sampling {n_samples} sequences with guidance_scale={guidance_scale}[/]")

    # Start from noise
    x = torch.randn(n_samples, seq_len, 1, device=device)

    # Subsample timesteps for speed
    timesteps = schedule.timesteps
    step_size = max(1, timesteps // n_steps)

    L_values = [32, 64, 128]

    for t_idx in track(range(timesteps - 1, -1, -step_size), description="Guided sampling..."):
        t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)

        # Apply guidance every few steps (not every step for speed)
        if t_idx % 10 == 0 and t_idx > 10:
            x = guided_p_sample(model, schedule, x, t, guidance_scale, L_values)
        else:
            # Standard DDPM step
            x = schedule.p_sample(model, x, t)

    return x


def main():
    console.print("[bold magenta]🎯 RIGIDITY-GUIDED SAMPLING[/]")
    console.print("[dim]Adding Δ3 gradient during denoising[/]\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"Device: {device}")

    # Load model (use the one trained with Wigner loss for better P(s))
    ckpt = torch.load("out/diffusion_rigid_best.pt", map_location=device, weights_only=False)
    config = ckpt["config"]

    model = ZetaDiffusionTransformer(
        seq_len=config["seq_len"],
        dim=config["dim"],
        depth=config["depth"],
        n_heads=config["n_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    schedule = DiffusionSchedule(timesteps=config["timesteps"], schedule_type="cosine").to(device)

    seq_len = config["seq_len"]
    n_samples = 8  # Reduced for speed

    # Test different guidance scales
    guidance_scales = [0.0, 5.0, 20.0]  # Fewer scales for quick test

    results = {}

    for gs in guidance_scales:
        console.print(f"\n[cyan]Testing guidance_scale = {gs}[/]")

        with torch.enable_grad():
            x = sample_with_guidance(model, schedule, n_samples, seq_len, device,
                                    guidance_scale=gs, n_steps=50)  # Faster

        # Convert to spacings
        generated = x.squeeze(-1).detach().cpu().numpy()
        generated = np.clip(generated, 0.01, 5.0)

        # Normalize
        for i in range(n_samples):
            generated[i] = generated[i] / generated[i].mean()

        all_spacings = generated.flatten()

        # Compute stats
        std = all_spacings.std()

        # Compute Δ3 on large scales
        from statistics_long_range_v2 import compute_delta3_correct, gue_delta3_theory

        L_values = np.array([10, 20, 50, 100, 200, 500])
        L_values = L_values[L_values < len(all_spacings) // 3]

        delta3 = compute_delta3_correct(all_spacings, L_values)

        # Compute slope
        from scipy import stats
        log_L = np.log(L_values)
        slope, _, _, _, _ = stats.linregress(log_L, delta3)

        results[gs] = {
            "std": std,
            "delta3": delta3,
            "slope": slope,
            "L_values": L_values,
        }

        console.print(f"  std={std:.3f}, slope={slope:.4f}")

    # Display comparison table
    console.print("\n")
    table = Table(title="🎯 GUIDANCE SCALE COMPARISON")
    table.add_column("Scale", style="bold", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Δ3 Slope", justify="right")
    table.add_column("Δ3(L=100)", justify="right")
    table.add_column("Verdict", justify="left")

    for gs in guidance_scales:
        r = results[gs]

        # Find Δ3 at L=100
        idx_100 = np.where(r["L_values"] == 100)[0]
        d3_100 = r["delta3"][idx_100[0]] if len(idx_100) > 0 else np.nan

        # Verdict
        if r["slope"] < 0.1:
            verdict = "[green]SUPER-RIGID![/]"
        elif r["slope"] < 0.5:
            verdict = "[yellow]Good[/]"
        elif r["slope"] < 2.0:
            verdict = "[yellow]GUE-like[/]"
        else:
            verdict = "[red]Poisson[/]"

        # Std check
        std_ok = 0.3 < r["std"] < 0.6
        std_str = f"[green]{r['std']:.3f}[/]" if std_ok else f"[red]{r['std']:.3f}[/]"

        table.add_row(
            f"{gs:.1f}",
            std_str,
            f"{r['slope']:.4f}",
            f"{d3_100:.4f}" if not np.isnan(d3_100) else "N/A",
            verdict,
        )

    console.print(table)

    # Best result
    best_gs = min(results.keys(), key=lambda gs: abs(results[gs]["slope"] - 0.01))
    console.print(f"\n[green]Best guidance_scale: {best_gs}[/]")
    console.print(f"[green]Achieved slope: {results[best_gs]['slope']:.4f} (target ~0.01)[/]")

    # Detailed Δ3 table for best
    console.print(f"\n[cyan]Detailed Δ3 for guidance_scale={best_gs}:[/]")
    best = results[best_gs]

    table2 = Table(title=f"Δ3(L) with guidance={best_gs}")
    table2.add_column("L", justify="right")
    table2.add_column("Generated", justify="right")
    table2.add_column("Real Zeta", justify="right")
    table2.add_column("GUE Theory", justify="right")

    gue_theory = gue_delta3_theory(best["L_values"])
    zeta_target = np.array([real_zeta_delta3_target(L) for L in best["L_values"]])

    for i, L in enumerate(best["L_values"]):
        table2.add_row(
            f"{L}",
            f"{best['delta3'][i]:.4f}",
            f"{zeta_target[i]:.4f}",
            f"{gue_theory[i]:.4f}",
        )

    console.print(table2)


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    main()
