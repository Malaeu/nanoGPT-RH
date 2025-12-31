#!/usr/bin/env python3
"""
DIFFUSION ZETA v1.0 — Global Rigidity via Denoising

Прощай, autoregressive bottleneck! Привет, joint distribution!

Ключевые отличия от AR:
- Full attention (не causal) → глобальные корреляции
- Denoising objective → учит восстанавливать всю траекторию
- Нет exposure bias → видит ground truth на train
- Rigidity "из коробки" → joint distribution естественно сохраняет Δ3 ~ log(L)

Architecture: 1D Diffusion Transformer (DiT-style)
Noise: Cosine schedule
Training: DDPM (predict noise)
Sampling: DDPM sampler, 50-100 steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import math

console = Console()


# ============================================================================
# NOISE SCHEDULE
# ============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from "Improved DDPM" paper.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSchedule(nn.Module):
    """Precomputed diffusion schedule quantities."""

    def __init__(self, timesteps=1000, schedule_type="cosine"):
        super().__init__()
        self.timesteps = timesteps

        if schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # For q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # For p(x_{t-1} | x_t)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar) * x_0, (1 - alpha_bar) * I)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # Expand dims for broadcasting
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def p_sample(self, model, x_t, t, clip_denoised=True):
        """Reverse diffusion step: p(x_{t-1} | x_t)"""
        # Predict noise
        pred_noise = model(x_t, t)

        # Compute x_{t-1}
        beta = self.betas[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        while beta.dim() < x_t.dim():
            beta = beta.unsqueeze(-1)
            sqrt_recip_alpha = sqrt_recip_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        # Mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alpha * (x_t - beta * pred_noise / sqrt_one_minus_alpha)

        if t[0] == 0:
            return model_mean

        # Add noise
        posterior_var = self.posterior_variance[t]
        while posterior_var.dim() < x_t.dim():
            posterior_var = posterior_var.unsqueeze(-1)

        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var) * noise


# ============================================================================
# TRANSFORMER BACKBONE (Full Attention)
# ============================================================================

class SinusoidalPositionEmb(nn.Module):
    """Sinusoidal embeddings for timestep t."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TransformerBlock(nn.Module):
    """Full attention transformer block (NOT causal!)."""

    def __init__(self, dim, n_heads, dropout=0.1, time_dim=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Time conditioning (FiLM-style)
        if time_dim:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, dim * 2),
                nn.GELU(),
            )
        else:
            self.time_mlp = None

    def forward(self, x, time_emb=None):
        # Time conditioning
        if self.time_mlp is not None and time_emb is not None:
            time_cond = self.time_mlp(time_emb)
            scale, shift = time_cond.chunk(2, dim=-1)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        else:
            scale, shift = 1, 0

        # Attention (full, not causal!)
        h = self.ln1(x)
        h = h * (1 + scale) + shift
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # MLP
        h = self.ln2(x)
        x = x + self.mlp(h)

        return x


class ZetaDiffusionTransformer(nn.Module):
    """
    1D Diffusion Transformer for spacing sequences.

    Key: Full attention (not causal) to capture global correlations.
    """

    def __init__(
        self,
        seq_len=512,
        dim=128,
        depth=6,
        n_heads=8,
        time_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim

        # Input embedding (spacing scalar → dim)
        self.input_proj = nn.Linear(1, dim)

        # Positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout, time_dim)
            for _ in range(depth)
        ])

        # Output projection
        self.ln_out = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, 1)

    def forward(self, x, t):
        """
        Args:
            x: (B, L, 1) noisy spacings
            t: (B,) timesteps

        Returns:
            noise: (B, L, 1) predicted noise
        """
        B, L, _ = x.shape

        # Embed input
        h = self.input_proj(x)  # (B, L, dim)

        # Add positional embedding
        h = h + self.pos_emb[:, :L, :]

        # Time embedding
        time_emb = self.time_mlp(t.float())  # (B, time_dim)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, time_emb)

        # Output
        h = self.ln_out(h)
        out = self.output_proj(h)  # (B, L, 1)

        return out


# ============================================================================
# TRAINING
# ============================================================================

def train():
    console.print("[bold magenta]🧨 DIFFUSION ZETA v1.0[/]")
    console.print("[dim]Global Rigidity via Denoising — Goodbye AR![/]\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load data
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data = torch.load('data/val.pt', weights_only=False)

    bin_centers = np.load('data/bin_centers.npy')
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)

    console.print(f"[green]Train sequences: {len(train_data)}[/]")
    console.print(f"[green]Val sequences: {len(val_data)}[/]")

    # Convert to continuous spacings
    def tokens_to_spacings(tokens):
        """Convert token indices to normalized spacing sequences."""
        spacings = bin_centers_t[tokens.to(device)]
        # Normalize per-sequence to mean 1
        spacings = spacings / (spacings.mean(dim=-1, keepdim=True) + 1e-8)
        return spacings.unsqueeze(-1)  # (B, L, 1)

    # Model config
    seq_len = 255  # train_data has shape (N, 256), so spacings are 255
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
    console.print(f"[green]Seq length: {seq_len}, Dim: {dim}, Depth: {depth}[/]")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000)

    # Training
    n_epochs = 20000
    batch_size = 32
    log_interval = 500

    best_val_loss = float('inf')

    console.print(f"\n[cyan]Training for {n_epochs} steps...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Training Diffusion...", total=n_epochs)

        for step in range(1, n_epochs + 1):
            model.train()

            # Sample batch
            idx = torch.randint(0, len(train_data), (batch_size,))
            tokens = train_data[idx]  # (B, 256)

            # Convert to spacings (tokens[:-1] → spacings)
            x_0 = tokens_to_spacings(tokens[:, :-1])  # (B, 255, 1)

            # Sample timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device)

            # Forward diffusion
            noise = torch.randn_like(x_0)
            x_t = schedule.q_sample(x_0, t, noise)

            # Predict noise
            pred_noise = model(x_t, t)

            # Loss
            loss = F.mse_loss(pred_noise, noise)

            # Backward
            optimizer.zero_grad()
            loss.backward()
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
                    val_loss = F.mse_loss(val_pred, val_noise)

                console.print(f"Step {step}: Loss={loss.item():.4f}, Val={val_loss.item():.4f}")

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save({
                        "model": model.state_dict(),
                        "config": {
                            "seq_len": seq_len,
                            "dim": dim,
                            "depth": depth,
                            "n_heads": n_heads,
                            "timesteps": timesteps,
                        },
                        "bin_centers": bin_centers,
                    }, "out/diffusion_zeta_best.pt")

            progress.update(task, advance=1)

    console.print(f"\n[green]✅ Training complete! Best val loss: {best_val_loss:.4f}[/]")
    console.print(f"[green]Saved: out/diffusion_zeta_best.pt[/]")

    # Quick generation test
    console.print("\n[cyan]Quick generation test...[/]")
    model.eval()

    with torch.no_grad():
        # Start from noise
        x = torch.randn(4, seq_len, 1, device=device)

        # Denoise
        for t_idx in reversed(range(timesteps)):
            t = torch.full((4,), t_idx, device=device, dtype=torch.long)
            x = schedule.p_sample(model, x, t)

        # Convert to spacings
        generated = x.squeeze(-1).cpu().numpy()
        generated = np.clip(generated, 0.01, 5.0)

        console.print(f"  Generated mean: {generated.mean():.3f}")
        console.print(f"  Generated std: {generated.std():.3f}")

    console.print("\n[green]Run test_diffusion_rigidity.py to check Δ3![/]")


if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    train()
