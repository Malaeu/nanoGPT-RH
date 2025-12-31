#!/usr/bin/env python3
"""
WIGNER SURMISE V6: Physics-Informed Likelihood

Вместо logistic mixture → используем ФИЗИЧЕСКИ ПРАВИЛЬНОЕ распределение:

    P(s) = (π s / 2) exp(-π s² / 4)   [Wigner Surmise / GUE]

Это точная аппроксимация spacing distribution для zeta zeros.

Почему это работает:
- Экспоненциальный cutoff на больших s → модель НЕ будет бояться хвостов
- Линейное подавление около 0 (level repulsion) → физика встроена
- Regression to the mean исчезнет — likelihood жёстко штрафует bias

Ожидаемый эффект:
- Bias на хвостах → ~0
- PC1 рухнет до 20-30%
- Residuals покажут НАСТОЯЩИЕ моды
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
import math

# For loading checkpoints
from train_memory_bank import MemoryBankConfig

console = Console()


# ============================================================================
# WIGNER SURMISE LIKELIHOOD (Physics-informed!)
# ============================================================================

def wigner_logprob(s, mu, log_sigma=None):
    """
    Log probability under scaled Wigner surmise.

    Standard Wigner: P(s) = (π s / 2) exp(-π s² / 4)

    Scaled version: P(s | μ, σ) with s_norm = s / μ
    This allows the model to predict the local mean.

    Args:
        s: (B, T) actual spacing values
        mu: (B, T) predicted mean spacing (location parameter)
        log_sigma: (B, T) optional log scale parameter

    Returns:
        log_prob: (B, T) log probability
    """
    # Normalize by predicted mean
    # If mu predicts the local mean correctly, s/mu should have mean ~1
    s_norm = s / (mu + 1e-8)  # Avoid division by zero

    # Optional scale parameter
    if log_sigma is not None:
        sigma = torch.exp(log_sigma.clamp(-3, 3))
        s_norm = s_norm / sigma

    # Wigner log probability: log(π s / 2) - π s² / 4
    # But we need to account for the Jacobian of normalization
    # log P(s | μ) = log P(s/μ) - log(μ)

    # Ensure s_norm > 0 for log
    s_norm = s_norm.clamp(min=1e-8)

    log_prob = (
        torch.log(torch.tensor(math.pi / 2))
        + torch.log(s_norm)
        - (math.pi / 4) * s_norm ** 2
        - torch.log(mu + 1e-8)  # Jacobian
    )

    return log_prob


def wigner_nll_loss(pred, target):
    """
    Negative log-likelihood under Wigner surmise.

    Args:
        pred: (B, T, 1) or (B, T, 2) predicted parameters
              If 1: just mu (mean)
              If 2: mu and log_sigma
        target: (B, T) actual spacing values

    Returns:
        nll: scalar
    """
    if pred.shape[-1] == 1:
        mu = pred[..., 0]
        log_sigma = None
    else:
        mu = pred[..., 0]
        log_sigma = pred[..., 1]

    # Ensure mu > 0 (spacings are positive)
    mu = F.softplus(mu) + 0.1  # Minimum 0.1 to avoid instability

    log_prob = wigner_logprob(target, mu, log_sigma)

    return -log_prob.mean()


# ============================================================================
# WIGNER GPT MODEL
# ============================================================================

class WignerConfig:
    """Config for WignerGPT (v6)"""
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    seq_len: int = 256
    dropout: float = 0.1
    n_memory_slots: int = 4
    use_scale: bool = True
    num_params: int = 1  # 1 = mu only, 2 = mu + sigma

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MemoryBank(nn.Module):
    """Learnable memory bank (same as before)."""
    def __init__(self, n_slots: int, dim: int):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim

        self.memory = nn.Parameter(torch.randn(n_slots, dim) * 0.02)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        q = self.query_proj(x.mean(dim=1))
        k = self.key_proj(self.memory)
        v = self.value_proj(self.memory)

        attn = torch.matmul(q, k.T) / math.sqrt(D)
        attn_weights = F.softmax(attn, dim=-1)

        mem_out = torch.matmul(attn_weights, v)
        mem_out = self.out_proj(mem_out)

        x_augmented = x + mem_out.unsqueeze(1)
        return x_augmented, attn_weights


class WignerGPT(nn.Module):
    """
    GPT with Wigner Surmise output.

    Physics-informed: output distribution matches GUE spacing statistics.
    """
    def __init__(self, config: WignerConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.seq_len, config.n_embd)

        # FiLM conditioning
        if config.use_scale:
            self.scale_ln = nn.LayerNorm(1)
            self.scale_proj = nn.Sequential(
                nn.Linear(1, config.n_embd * 4),
                nn.GELU(),
                nn.Linear(config.n_embd * 4, config.n_embd * 2),
                nn.Tanh(),
            )

        # Memory Bank
        self.memory_bank = MemoryBank(config.n_memory_slots, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.n_embd * 4,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.n_layer)
        ])

        # Output: Wigner parameters (mu, optional sigma)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_params, bias=True)

        # Initialize head bias to typical spacing mean (~1.0)
        with torch.no_grad():
            # softplus(0.5) ≈ 0.97, close to mean spacing
            self.head.bias.data[0] = 0.5
            if config.num_params > 1:
                self.head.bias.data[1] = 0.0  # log_sigma = 0 → sigma = 1

        # Causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.seq_len, config.seq_len), diagonal=1).bool()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and module is not self.head:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_hidden=False, scale_val=None):
        """
        Args:
            idx: (B, T) input token indices
            targets: (B, T) target spacing values (FLOAT!)
            scale_val: (B, T) scale values for FiLM conditioning

        Returns:
            pred: (B, T, num_params)
            loss: scalar NLL
            mem_attn: (B, n_slots)
        """
        B, T = idx.shape

        # Embeddings
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # FiLM conditioning
        if self.config.use_scale and scale_val is not None:
            if scale_val.dim() == 2:
                scale_val_cond = scale_val.unsqueeze(-1)
            else:
                scale_val_cond = scale_val

            scale_val_cond = torch.log1p(scale_val_cond)
            s = self.scale_ln(scale_val_cond)
            style = self.scale_proj(s)
            gamma, beta = style.chunk(2, dim=-1)

            DAMP_G = 0.2
            DAMP_B = 0.2
            x = x * (1.0 + DAMP_G * gamma) + DAMP_B * beta

        # Memory Bank
        x, mem_attn = self.memory_bank(x)

        # Transformer
        causal_mask = self.mask[:T, :T]
        for block in self.blocks:
            x = block(x, src_mask=causal_mask, is_causal=True)

        # Hidden states
        hidden_pre_ln = x
        x = self.ln_f(x)
        hidden_post_ln = x

        # Output parameters
        pred = self.head(x)  # (B, T, num_params)

        # Loss
        loss = None
        if targets is not None:
            loss = wigner_nll_loss(pred, targets)

        if return_hidden:
            return {
                "pred": pred,
                "loss": loss,
                "hidden_pre_ln": hidden_pre_ln,
                "hidden_post_ln": hidden_post_ln,
                "mem_attn": mem_attn,
            }

        return pred, loss, mem_attn

    def get_mu(self, pred):
        """Extract predicted mean from output."""
        raw_mu = pred[..., 0]
        return F.softplus(raw_mu) + 0.1

    def sample(self, pred, n_samples=1):
        """
        Sample from Wigner distribution with predicted parameters.

        Uses inverse CDF sampling.
        """
        mu = self.get_mu(pred)

        # Sample from standard Wigner using inverse CDF
        # CDF: F(s) = 1 - exp(-π s² / 4)
        # Inverse: s = sqrt(-4/π * log(1 - u))
        u = torch.rand(*mu.shape, n_samples, device=mu.device)
        u = u.clamp(1e-8, 1 - 1e-8)

        s_std = torch.sqrt(-4 / math.pi * torch.log(1 - u))

        # Scale by predicted mean
        samples = mu.unsqueeze(-1) * s_std

        if n_samples == 1:
            return samples.squeeze(-1)
        return samples

    def get_memory_vectors(self):
        return self.memory_bank.memory.detach().cpu().numpy()


# ============================================================================
# TRAINING
# ============================================================================

def train():
    console.print("[bold magenta]🌟 WIGNER SURMISE V6: Physics-Informed Likelihood[/]")
    console.print("[dim]P(s) = (π s / 2) exp(-π s² / 4) — GUE statistics embedded![/]\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load data
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data = torch.load('data/val.pt', weights_only=False)

    # Load bin_centers
    bin_centers = np.load('data/bin_centers.npy')
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)
    console.print(f"[green]Bin centers loaded: {len(bin_centers)}[/]")
    console.print(f"[green]Spacing range: [{bin_centers.min():.3f}, {bin_centers.max():.3f}][/]")

    # Check Wigner fit
    all_spacings = bin_centers[train_data.numpy().flatten()]
    wigner_mean = np.sqrt(math.pi) / 2  # ~0.886 for standard Wigner
    data_mean = all_spacings.mean()
    console.print(f"[cyan]Data mean spacing: {data_mean:.4f}[/]")
    console.print(f"[cyan]Wigner theoretical mean: {wigner_mean:.4f}[/]")

    # Config
    config = WignerConfig(
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        seq_len=256,
        n_memory_slots=4,
        dropout=0.1,
        use_scale=True,
        num_params=1,  # Just mu for now
    )

    console.print(f"[green]Output params: {config.num_params} (mu only)[/]")

    # Model
    model = WignerGPT(config).to(device)

    # Transfer from v5
    v5_ckpt_path = Path('out/continuous_v5_best.pt')
    if v5_ckpt_path.exists():
        console.print("[yellow]Loading v5 checkpoint for transfer...[/]")
        v5_ckpt = torch.load(v5_ckpt_path, map_location=device, weights_only=False)
        v5_state = v5_ckpt['model']
        model_state = model.state_dict()

        transferred = 0
        for name, param in v5_state.items():
            if name in model_state and 'head' not in name:
                if model_state[name].shape == param.shape:
                    model_state[name] = param
                    transferred += 1

        model.load_state_dict(model_state)
        console.print(f"[green]Transferred {transferred} layers from v5[/]")
    else:
        console.print("[yellow]No v5 checkpoint, training from scratch[/]")

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Parameters: {n_params:,}[/]\n")

    # Data loaders
    train_dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:])
    val_dataset = TensorDataset(val_data[:, :-1], val_data[:, 1:])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

    # Training
    n_steps = 10000
    step = 0
    best_val_loss = float('inf')

    model.train()
    train_iter = iter(train_loader)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Training Wigner v6...", total=n_steps)

        while step < n_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            scale_val = bin_centers_t[x]
            target_spacing = bin_centers_t[y]

            optimizer.zero_grad()
            pred, loss, mem_attn = model(x, targets=target_spacing, scale_val=scale_val)

            # Check for NaN
            if torch.isnan(loss):
                console.print("[red]NaN loss detected! Skipping batch.[/]")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            progress.update(task, advance=1)

            # Validation
            if step % 500 == 0:
                model.eval()
                val_losses = []
                mae_errors = []
                bias_small = []
                bias_large = []

                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        sv = bin_centers_t[x]
                        target = bin_centers_t[y]

                        pred, loss, _ = model(x, targets=target, scale_val=sv)
                        if not torch.isnan(loss):
                            val_losses.append(loss.item())

                        # MAE
                        mu = model.get_mu(pred)
                        error = (mu - target)
                        mae_errors.append(error.abs().mean().item())

                        # Bias analysis
                        small_mask = target < 0.5
                        large_mask = target > 1.5
                        if small_mask.any():
                            bias_small.append(error[small_mask].mean().item())
                        if large_mask.any():
                            bias_large.append(error[large_mask].mean().item())

                val_loss = np.mean(val_losses) if val_losses else float('inf')
                val_mae = np.mean(mae_errors)
                bias_s = np.mean(bias_small) if bias_small else 0
                bias_l = np.mean(bias_large) if bias_large else 0

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model': model.state_dict(),
                        'config': config.__dict__,
                        'bin_centers': bin_centers,
                        'step': step,
                        'val_loss': val_loss,
                        'val_mae': val_mae,
                    }, 'out/wigner_v6_best.pt')

                console.print(
                    f"[dim]Step {step}: NLL={val_loss:.4f}, MAE={val_mae:.4f}, "
                    f"bias_small={bias_s:+.3f}, bias_large={bias_l:+.3f}[/]"
                )
                model.train()

    # Final save
    torch.save({
        'model': model.state_dict(),
        'config': config.__dict__,
        'bin_centers': bin_centers,
        'step': step,
        'val_loss': val_loss,
        'val_mae': val_mae,
    }, 'out/wigner_v6_final.pt')

    console.print(f"\n[bold green]✅ Training complete![/]")
    console.print(f"[green]Best NLL: {best_val_loss:.4f}[/]")

    # Memory slots
    console.print("\n[bold]Memory Slot Usage:[/]")
    mem_attn_avg = mem_attn.mean(dim=0).detach().cpu().numpy()
    for i, w in enumerate(mem_attn_avg):
        bar = "█" * int(w * 40)
        console.print(f"  Slot {i}: {bar} ({w:.3f})")

    console.print(f"\n[cyan]Saved: out/wigner_v6_best.pt[/]")
    console.print(f"[cyan]Saved: out/wigner_v6_final.pt[/]")
    console.print("\n[yellow]Run mine_residuals_wigner.py to verify regression-to-mean is DEAD![/]")


if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    train()
