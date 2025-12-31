#!/usr/bin/env python3
"""
CONTINUOUS OUTPUT V5: Mixture of Logistics (убиваем Scale Blindness в корне)

Вместо softmax over 256 bins → предсказываем параметры смеси логистических распределений.
Модель работает напрямую с float spacing'ами → scale-bias исчезает.

Архитектура:
- Та же MemoryBankGPT с FiLM
- Но head выдаёт 3*K параметров: (log_weights, means, log_scales) для K компонент
- Loss = negative log-likelihood смеси логистиков

Ожидаемый эффект:
- corr(H1, spacing) → ~0.0-0.2 (вместо 0.753)
- PC1 → < 20% (чистые физические моды)
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

# Import for loading v4 checkpoints (pickle needs this class)
from train_memory_bank import MemoryBankConfig

console = Console()


# ============================================================================
# MIXTURE OF LOGISTICS LOSS (ядро v5)
# ============================================================================

def log_sum_exp(x, dim=-1):
    """Numerically stable log-sum-exp."""
    m = x.max(dim=dim, keepdim=True)[0]
    return m.squeeze(dim) + torch.log(torch.exp(x - m).sum(dim=dim))


def mix_logistic_logprob(target, log_weights, means, log_scales):
    """
    Log probability of target under mixture of logistics.

    Args:
        target: (B, T) actual spacing values (float)
        log_weights: (B, T, K) log mixture weights (after log_softmax)
        means: (B, T, K) mixture means
        log_scales: (B, T, K) log scales (clamped for stability)

    Returns:
        log_prob: (B, T) log probability for each position
    """
    # Expand target: (B, T, 1)
    target = target.unsqueeze(-1)

    # Clamp log_scales for numerical stability
    log_scales = log_scales.clamp(min=-7.0, max=7.0)

    # Compute centered value
    centered = target - means  # (B, T, K)
    inv_s = torch.exp(-log_scales)  # (B, T, K)

    # Logistic log-PDF:
    # log f(x|μ,s) = -(x-μ)/s - log(s) - 2*log(1 + exp(-(x-μ)/s))
    #              = -centered*inv_s - log_scales - 2*softplus(-centered*inv_s)
    z = centered * inv_s
    log_pdf = -z - log_scales - 2 * F.softplus(-z)  # (B, T, K)

    # Mixture: log Σ_k π_k * f_k(x)
    # = log Σ_k exp(log π_k + log f_k(x))
    log_prob = log_sum_exp(log_weights + log_pdf, dim=-1)  # (B, T)

    return log_prob


def discretized_mix_logistic_logprob(target, log_weights, means, log_scales, bin_width=0.02):
    """
    Log probability of target under DISCRETIZED mixture of logistics.

    Используется для сравнения с binned версией.
    P(x) = CDF(x + bin_width/2) - CDF(x - bin_width/2)

    Args:
        target: (B, T) actual spacing values (float)
        log_weights: (B, T, K) log mixture weights
        means: (B, T, K) mixture means
        log_scales: (B, T, K) log scales
        bin_width: width of discretization bin

    Returns:
        log_prob: (B, T) log probability
    """
    # Expand target: (B, T, 1)
    target = target.unsqueeze(-1)

    # Clamp for stability
    log_scales = log_scales.clamp(min=-7.0, max=7.0)

    # Compute CDF at bin edges
    inv_s = torch.exp(-log_scales)
    half_bin = bin_width / 2

    upper = (target + half_bin - means) * inv_s
    lower = (target - half_bin - means) * inv_s

    cdf_upper = torch.sigmoid(upper)
    cdf_lower = torch.sigmoid(lower)

    # Probability mass in bin
    prob_mass = cdf_upper - cdf_lower  # (B, T, K)
    prob_mass = prob_mass.clamp(min=1e-12)  # numerical stability

    # Log probability per component
    log_prob_component = torch.log(prob_mass)

    # Mixture
    log_prob = log_sum_exp(log_weights + log_prob_component, dim=-1)

    return log_prob


class MixLogisticLoss(nn.Module):
    """
    Negative log-likelihood loss for mixture of logistics.

    Supports both continuous and discretized versions.
    """
    def __init__(self, num_mixtures=10, discretized=False, bin_width=0.02):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.discretized = discretized
        self.bin_width = bin_width

    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, 3*K) model output
            target: (B, T) actual spacing values

        Returns:
            nll: scalar negative log-likelihood
        """
        B, T, _ = pred.shape
        K = self.num_mixtures

        # Split predictions
        log_weights = pred[..., :K]
        means = pred[..., K:2*K]
        log_scales = pred[..., 2*K:3*K]

        # Normalize weights
        log_weights = F.log_softmax(log_weights, dim=-1)

        # Compute log probability
        if self.discretized:
            log_prob = discretized_mix_logistic_logprob(
                target, log_weights, means, log_scales, self.bin_width
            )
        else:
            log_prob = mix_logistic_logprob(
                target, log_weights, means, log_scales
            )

        # Negative log-likelihood
        nll = -log_prob.mean()

        return nll


# ============================================================================
# CONTINUOUS OUTPUT MODEL
# ============================================================================

class ContinuousConfig:
    """Config for ContinuousGPT (v5)"""
    vocab_size: int = 256  # Still needed for input embedding
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    seq_len: int = 256
    dropout: float = 0.1
    n_memory_slots: int = 4
    use_scale: bool = True
    num_mixtures: int = 10  # NEW: number of logistic components

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MemoryBank(nn.Module):
    """Learnable memory bank with attention."""
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


class ContinuousGPT(nn.Module):
    """
    GPT with Continuous Output (Mixture of Logistics).

    Вместо softmax over 256 bins → предсказываем параметры смеси.
    Это убивает scale-bias в корне!
    """
    def __init__(self, config: ContinuousConfig):
        super().__init__()
        self.config = config

        # Input: token embedding (still discrete input)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.seq_len, config.n_embd)

        # FiLM conditioning (keep it - it helps!)
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

        # Output: CONTINUOUS (mixture params instead of logits!)
        self.ln_f = nn.LayerNorm(config.n_embd)
        K = config.num_mixtures
        self.head = nn.Linear(config.n_embd, 3 * K, bias=True)  # log_w, μ, log_s

        # Initialize means around typical spacing values (0.0 to 4.0)
        # This helps convergence
        self._init_mixture_head()

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

    def _init_mixture_head(self):
        """Initialize mixture head with good starting values."""
        K = self.config.num_mixtures

        # Weights: uniform
        nn.init.zeros_(self.head.weight[:K])
        if self.head.bias is not None:
            self.head.bias.data[:K] = 0.0  # equal log weights → equal probs

        # Means: spread across typical spacing range [0.2, 2.5]
        # Most spacings are around 1.0 (GUE mean)
        mean_init = torch.linspace(0.2, 2.5, K)
        if self.head.bias is not None:
            self.head.bias.data[K:2*K] = mean_init

        # Log scales: start at log(0.3) ≈ -1.2 (moderate spread)
        if self.head.bias is not None:
            self.head.bias.data[2*K:3*K] = -1.2

    def forward(self, idx, targets=None, return_hidden=False, scale_val=None):
        """
        Args:
            idx: (B, T) input token indices
            targets: (B, T) target SPACING VALUES (float, not token ids!)
            scale_val: (B, T) scale values for FiLM conditioning

        Returns:
            pred: (B, T, 3*K) mixture parameters
            loss: scalar NLL (if targets provided)
            mem_attn: (B, n_slots) memory attention
        """
        B, T = idx.shape

        # Embeddings
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # Damped FiLM conditioning
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

        # Transformer blocks
        causal_mask = self.mask[:T, :T]
        for block in self.blocks:
            x = block(x, src_mask=causal_mask, is_causal=True)

        # Hidden states for gradient analysis
        hidden_pre_ln = x
        x = self.ln_f(x)
        hidden_post_ln = x

        # Output: mixture parameters (NOT logits!)
        pred = self.head(x)  # (B, T, 3*K)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss_fn = MixLogisticLoss(
                num_mixtures=self.config.num_mixtures,
                discretized=False  # Pure continuous
            )
            loss = loss_fn(pred, targets)

        if return_hidden:
            return {
                "pred": pred,
                "loss": loss,
                "hidden_pre_ln": hidden_pre_ln,
                "hidden_post_ln": hidden_post_ln,
                "mem_attn": mem_attn,
            }

        return pred, loss, mem_attn

    def sample(self, pred, temperature=1.0):
        """
        Sample spacing values from predicted mixture.

        Args:
            pred: (B, T, 3*K) mixture parameters
            temperature: sampling temperature

        Returns:
            samples: (B, T) sampled spacing values
        """
        K = self.config.num_mixtures

        log_weights = pred[..., :K]
        means = pred[..., K:2*K]
        log_scales = pred[..., 2*K:3*K].clamp(min=-7.0, max=7.0)

        # Temperature scaling
        log_weights = log_weights / temperature

        # Sample component
        probs = F.softmax(log_weights, dim=-1)
        component = torch.multinomial(probs.view(-1, K), 1).view(*pred.shape[:-1])

        # Get params for selected component
        B, T = component.shape
        batch_idx = torch.arange(B, device=pred.device).view(-1, 1).expand(B, T)
        time_idx = torch.arange(T, device=pred.device).view(1, -1).expand(B, T)

        selected_mean = means[batch_idx, time_idx, component]
        selected_log_scale = log_scales[batch_idx, time_idx, component]

        # Sample from logistic: μ + s * logit(U) where U ~ Uniform(0,1)
        u = torch.rand_like(selected_mean).clamp(1e-5, 1 - 1e-5)
        samples = selected_mean + torch.exp(selected_log_scale) * (torch.log(u) - torch.log(1 - u))

        # Clamp to valid range (spacings are positive)
        samples = samples.clamp(min=0.01)

        return samples

    def get_expectation(self, pred):
        """
        Get expected spacing value from mixture.

        E[X] = Σ_k π_k * μ_k
        """
        K = self.config.num_mixtures

        log_weights = pred[..., :K]
        means = pred[..., K:2*K]

        probs = F.softmax(log_weights, dim=-1)
        expectation = (probs * means).sum(dim=-1)

        return expectation

    def get_memory_vectors(self):
        return self.memory_bank.memory.detach().cpu().numpy()


# ============================================================================
# TRAINING
# ============================================================================

def train():
    console.print("[bold magenta]🚀 CONTINUOUS OUTPUT V5: Mixture of Logistics[/]")
    console.print("[dim]Predicting float spacing directly → killing scale-bias at root![/]\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load data
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data = torch.load('data/val.pt', weights_only=False)

    # Load bin_centers for target conversion
    bin_centers = np.load('data/bin_centers.npy')
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)
    console.print(f"[green]Bin centers loaded: {len(bin_centers)} (for target conversion)[/]")
    console.print(f"[green]Spacing range: [{bin_centers.min():.3f}, {bin_centers.max():.3f}][/]")

    # Create config
    config = ContinuousConfig(
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        seq_len=256,
        n_memory_slots=4,
        dropout=0.1,
        use_scale=True,
        num_mixtures=10,  # 10 logistic components
    )

    console.print(f"[green]Mixture components: {config.num_mixtures}[/]")
    console.print(f"[green]Output params per token: {3 * config.num_mixtures}[/]")

    # Create model
    model = ContinuousGPT(config).to(device)

    # Try to initialize from v4 checkpoint (transfer learned representations)
    v4_ckpt_path = Path('out/memory_bank_v4_best.pt')
    if v4_ckpt_path.exists():
        console.print("[yellow]Loading v4 checkpoint for transfer learning...[/]")
        v4_ckpt = torch.load(v4_ckpt_path, map_location=device, weights_only=False)

        # Load matching weights (everything except head)
        v4_state = v4_ckpt['model']
        model_state = model.state_dict()

        transferred = 0
        for name, param in v4_state.items():
            if name in model_state and 'head' not in name:
                if model_state[name].shape == param.shape:
                    model_state[name] = param
                    transferred += 1

        model.load_state_dict(model_state)
        console.print(f"[green]Transferred {transferred} layers from v4[/]")
    else:
        console.print("[yellow]No v4 checkpoint found, training from scratch[/]")

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
        task = progress.add_task("[cyan]Training v5...", total=n_steps)

        while step < n_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            # Convert token IDs to actual spacing values
            scale_val = bin_centers_t[x]  # (B, T) for FiLM conditioning
            target_spacing = bin_centers_t[y]  # (B, T) target as FLOAT

            optimizer.zero_grad()
            pred, loss, mem_attn = model(x, targets=target_spacing, scale_val=scale_val)
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

                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        sv = bin_centers_t[x]
                        target = bin_centers_t[y]

                        pred, loss, _ = model(x, targets=target, scale_val=sv)
                        val_losses.append(loss.item())

                        # Compute MAE for interpretability
                        expected = model.get_expectation(pred)
                        mae = (expected - target).abs().mean().item()
                        mae_errors.append(mae)

                val_loss = np.mean(val_losses)
                val_mae = np.mean(mae_errors)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model': model.state_dict(),
                        'config': config.__dict__,
                        'bin_centers': bin_centers,
                        'step': step,
                        'val_loss': val_loss,
                        'val_mae': val_mae,
                    }, 'out/continuous_v5_best.pt')

                console.print(f"[dim]Step {step}: NLL={val_loss:.4f}, MAE={val_mae:.4f}[/]")
                model.train()

    # Final save
    torch.save({
        'model': model.state_dict(),
        'config': config.__dict__,
        'bin_centers': bin_centers,
        'step': step,
        'val_loss': val_loss,
        'val_mae': val_mae,
    }, 'out/continuous_v5_final.pt')

    console.print(f"\n[bold green]✅ Training complete![/]")
    console.print(f"[green]Best NLL: {best_val_loss:.4f}[/]")
    console.print(f"[green]Best MAE: {val_mae:.4f}[/]")

    # Memory attention distribution
    console.print("\n[bold]Memory Slot Usage (last batch):[/]")
    mem_attn_avg = mem_attn.mean(dim=0).detach().cpu().numpy()
    for i, w in enumerate(mem_attn_avg):
        bar = "█" * int(w * 40)
        console.print(f"  Slot {i}: {bar} ({w:.3f})")

    console.print(f"\n[cyan]Saved: out/continuous_v5_best.pt[/]")
    console.print(f"[cyan]Saved: out/continuous_v5_final.pt[/]")
    console.print("\n[yellow]Run mine_residuals_continuous.py to verify scale-bias is DEAD![/]")


if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    train()
