#!/usr/bin/env python3
"""
MULTI-SCALE SNOWBALL V9: Hierarchical Memory for Long-Range Rigidity

Проблема v8: single-scale EMA теряет long-range correlations (Δ3 → linear вместо log).

Решение v9: Multi-scale memory с разными timescales:
- Fast (α=0.9): локальный repulsion, short-term dynamics
- Medium (α=0.99): фаза на 100-500 шагов
- Slow (α=0.999): глобальная rigidity, long-term structure

+ Optional auxiliary loss на number variance (дифференцируемый proxy для Δ3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import math

from train_memory_bank import MemoryBankConfig
from train_wigner import wigner_logprob, wigner_nll_loss
from train_rope import (
    RotaryEmbedding, RoPEAttention, RoPETransformerBlock, MemoryBank
)

console = Console()


# ============================================================================
# MULTI-SCALE SNOWBALL MEMORY
# ============================================================================

class MultiScaleSnowball(nn.Module):
    """
    Hierarchical memory with multiple timescales.

    Each scale captures correlations at different ranges:
    - Fast: local repulsion (1-10 steps)
    - Medium: phase coherence (10-100 steps)
    - Slow: global rigidity (100-1000+ steps)

    Memory vectors are concatenated and projected to form the memory token.
    """

    def __init__(self, n_embd, n_scales=3, alphas=None):
        super().__init__()
        self.n_embd = n_embd
        self.n_scales = n_scales

        # Default alphas: fast, medium, slow
        if alphas is None:
            alphas = [0.9, 0.99, 0.999]
        self.alphas = alphas[:n_scales]

        # Per-scale dimension
        self.scale_dim = n_embd // n_scales

        # Learnable initial states for each scale
        self.init_memories = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, self.scale_dim) * 0.02)
            for _ in range(n_scales)
        ])

        # Scale-specific update networks (optional refinement)
        self.update_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.scale_dim * 2, self.scale_dim * 2),
                nn.GELU(),
                nn.Linear(self.scale_dim * 2, self.scale_dim),
            )
            for _ in range(n_scales)
        ])

        # Attention over scales (learn which scale matters for each position)
        self.scale_attention = nn.Sequential(
            nn.Linear(n_embd, n_scales),
            nn.Softmax(dim=-1),
        )

        # Project concatenated memories to memory token
        self.read_proj = nn.Linear(n_embd, n_embd)

    def get_initial_state(self, batch_size, device):
        """Get initial multi-scale memory state."""
        # Returns list of (B, 1, scale_dim) tensors
        return [
            mem.expand(batch_size, 1, -1).to(device)
            for mem in self.init_memories
        ]

    def prepare_input(self, x, memory_states):
        """Prepare memory token from multi-scale states."""
        # Concatenate all scales
        memory_concat = torch.cat(memory_states, dim=-1)  # (B, 1, n_embd)

        # Project to memory token
        memory_token = self.read_proj(memory_concat)  # (B, 1, n_embd)

        # Prepend to input
        return torch.cat([memory_token, x], dim=1)  # (B, T+1, n_embd)

    def update_state(self, memory_states, hidden_states):
        """Update each memory scale with its own dynamics."""
        B = hidden_states.shape[0]

        # Get window summary (last hidden state)
        window_summary = hidden_states[:, -1:, :]  # (B, 1, n_embd)

        # Split summary for each scale
        summary_splits = torch.split(window_summary, self.scale_dim, dim=-1)

        new_states = []
        for i, (mem_state, summary, alpha) in enumerate(
            zip(memory_states, summary_splits, self.alphas)
        ):
            # EMA update
            ema_update = alpha * mem_state + (1 - alpha) * summary

            # Optional MLP refinement
            combined = torch.cat([mem_state, summary], dim=-1)
            delta = self.update_nets[i](combined)
            new_mem = ema_update + 0.1 * delta  # Residual refinement

            new_states.append(new_mem)

        return new_states

    def get_memory_diagnostics(self, memory_states):
        """Get diagnostics for each scale."""
        norms = [mem.norm(dim=-1).mean().item() for mem in memory_states]
        return {f"scale_{i}_norm": norms[i] for i in range(len(norms))}


# ============================================================================
# SNOWBALL V9 MODEL
# ============================================================================

class SnowballV9Config:
    """Config for Multi-Scale Snowball GPT (v9)"""
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    seq_len: int = 256
    dropout: float = 0.1
    n_memory_slots: int = 4
    use_scale: bool = True
    num_params: int = 1
    # Multi-scale snowball
    n_scales: int = 3
    alphas: list = None  # [0.9, 0.99, 0.999]
    # Auxiliary loss
    use_rigidity_loss: bool = False
    rigidity_lambda: float = 0.01

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.alphas is None:
            self.alphas = [0.9, 0.99, 0.999]


class SnowballV9GPT(nn.Module):
    """
    GPT with Multi-Scale Snowball Memory.

    The hierarchical memory preserves both local dynamics and global rigidity.
    """

    def __init__(self, config: SnowballV9Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # FiLM conditioning
        if config.use_scale:
            self.scale_ln = nn.LayerNorm(1)
            self.scale_proj = nn.Sequential(
                nn.Linear(1, config.n_embd * 4),
                nn.GELU(),
                nn.Linear(config.n_embd * 4, config.n_embd * 2),
                nn.Tanh(),
            )

        # Multi-Scale Snowball Memory
        self.snowball = MultiScaleSnowball(
            config.n_embd,
            n_scales=config.n_scales,
            alphas=config.alphas
        )

        # Memory Bank
        self.memory_bank = MemoryBank(config.n_memory_slots, config.n_embd)

        # RoPE Transformer blocks
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(
                config.n_embd,
                config.n_head,
                config.dropout,
                max_position=config.seq_len + 1
            )
            for _ in range(config.n_layer)
        ])

        # Wigner output
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_params, bias=True)

        with torch.no_grad():
            self.head.bias.data[0] = 0.5

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.seq_len + 1, config.seq_len + 1), diagonal=1).bool()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_hidden=False, scale_val=None,
                memory_states=None, return_memory=False):
        """Forward pass with multi-scale memory."""
        B, T = idx.shape
        device = idx.device

        # Initialize memory if not provided
        if memory_states is None:
            memory_states = self.snowball.get_initial_state(B, device)

        # Token embedding
        x = self.tok_emb(idx)

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

        # Prepend memory token
        x = self.snowball.prepare_input(x, memory_states)

        # Memory Bank
        x, mem_attn = self.memory_bank(x)

        # Transformer blocks
        causal_mask = self.causal_mask[:T+1, :T+1]
        for block in self.blocks:
            x = block(x, causal_mask)

        # Save hidden for memory update
        hidden_states = x

        # Update memory state
        new_memory_states = self.snowball.update_state(memory_states, hidden_states)

        # Remove memory token position
        x = x[:, 1:, :]

        # Output
        hidden_pre_ln = x
        x = self.ln_f(x)
        hidden_post_ln = x

        pred = self.head(x)

        # Loss
        loss = None
        if targets is not None:
            loss = wigner_nll_loss(pred, targets)

        if return_hidden:
            result = {
                "pred": pred,
                "loss": loss,
                "hidden_pre_ln": hidden_pre_ln,
                "hidden_post_ln": hidden_post_ln,
                "mem_attn": mem_attn,
            }
            if return_memory:
                result["memory_states"] = new_memory_states
            return result

        if return_memory:
            return pred, loss, mem_attn, new_memory_states

        return pred, loss, mem_attn

    def get_mu(self, pred):
        raw_mu = pred[..., 0]
        return F.softplus(raw_mu) + 0.1


# ============================================================================
# AUXILIARY RIGIDITY LOSS
# ============================================================================

def compute_number_variance_loss(spacings, L=50):
    """
    Differentiable number variance proxy.

    Σ²(L) should be ~ (2/π²) log(L) for GUE.
    We penalize deviation from this target.
    """
    N = len(spacings)
    if N < L * 2:
        return torch.tensor(0.0, device=spacings.device)

    # Cumulative positions
    positions = torch.cumsum(spacings, dim=0)

    # Sample windows
    n_windows = min(50, N - L)
    variances = []

    for i in range(0, n_windows, max(1, n_windows // 10)):
        start = positions[i]
        end = start + L

        # Count levels in window (soft count)
        in_window = torch.sigmoid((positions - start) * 10) * torch.sigmoid((end - positions) * 10)
        count = in_window.sum()
        variances.append(count)

    if len(variances) < 2:
        return torch.tensor(0.0, device=spacings.device)

    variances = torch.stack(variances)
    sigma2 = torch.var(variances)

    # GUE target
    target = (2 / np.pi**2) * np.log(2 * np.pi * L) + 0.442

    # Penalize deviation
    loss = (sigma2 - target) ** 2

    return loss


# ============================================================================
# TRAINING
# ============================================================================

def train():
    console.print("[bold magenta]🌊 MULTI-SCALE SNOWBALL V9[/]")
    console.print("[dim]Hierarchical Memory for Long-Range Rigidity[/]\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load data
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data = torch.load('data/val.pt', weights_only=False)

    bin_centers = np.load('data/bin_centers.npy')
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)
    console.print(f"[green]Bin centers: {len(bin_centers)}[/]")

    # Config
    config = SnowballV9Config(
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_embd=120,  # Divisible by 3 scales, 4 heads, head_dim=30 (even for RoPE)
        seq_len=256,
        n_memory_slots=4,
        dropout=0.1,
        use_scale=True,
        num_params=1,
        n_scales=3,
        alphas=[0.9, 0.99, 0.999],
        use_rigidity_loss=True,
        rigidity_lambda=0.001,
    )

    console.print(f"[green]Scales: {config.n_scales}[/]")
    console.print(f"[green]Alphas: {config.alphas}[/]")
    console.print(f"[green]Rigidity loss: {config.use_rigidity_loss} (λ={config.rigidity_lambda})[/]")

    # Try loading v8 weights
    model = SnowballV9GPT(config).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[green]Parameters: {n_params:,}[/]")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15000)

    # Training loop
    n_epochs = 15000
    window_size = 128
    batch_size = 32
    log_interval = 500

    best_val_nll = float('inf')

    console.print(f"\n[cyan]Training for {n_epochs} steps...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Training v9...", total=n_epochs)

        for step in range(1, n_epochs + 1):
            model.train()

            # Sample batch
            idx = torch.randint(0, len(train_data), (batch_size,))
            seqs = train_data[idx].to(device)

            T = seqs.shape[1] - 1

            # Process windows with recurrent memory
            n_windows = T // window_size
            if n_windows == 0:
                n_windows = 1
                window_size_actual = T
            else:
                window_size_actual = window_size

            total_loss = 0
            total_rigidity_loss = 0

            # Initialize memory
            memory_states = model.snowball.get_initial_state(batch_size, device)

            for w in range(n_windows):
                start = w * window_size_actual
                end = min(start + window_size_actual, T)

                X = seqs[:, start:end]
                Y_idx = seqs[:, start+1:end+1]

                target_spacing = bin_centers_t[Y_idx]
                scale_val = bin_centers_t[X]

                # Forward
                pred, loss, mem_attn, memory_states = model(
                    X, targets=target_spacing, scale_val=scale_val,
                    memory_states=memory_states, return_memory=True
                )

                total_loss += loss

                # Auxiliary rigidity loss
                if config.use_rigidity_loss and w == n_windows - 1:
                    # Compute on last window predictions
                    mu = model.get_mu(pred)
                    # Flatten batch for rigidity computation
                    spacings = mu.view(-1)
                    rig_loss = compute_number_variance_loss(spacings, L=50)
                    total_rigidity_loss += rig_loss

            # Average loss
            total_loss = total_loss / n_windows

            if config.use_rigidity_loss:
                total_loss = total_loss + config.rigidity_lambda * total_rigidity_loss

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
                    val_idx = torch.randint(0, len(val_data), (64,))
                    val_seqs = val_data[val_idx].to(device)

                    # Single window eval
                    X_val = val_seqs[:, :window_size_actual]
                    Y_val_idx = val_seqs[:, 1:window_size_actual+1]
                    target_val = bin_centers_t[Y_val_idx]
                    scale_val_val = bin_centers_t[X_val]

                    memory_val = model.snowball.get_initial_state(64, device)
                    pred_val, val_loss, _, _ = model(
                        X_val, targets=target_val, scale_val=scale_val_val,
                        memory_states=memory_val, return_memory=True
                    )

                    mu_val = model.get_mu(pred_val)
                    mae = (mu_val - target_val).abs().mean().item()

                    # Memory diagnostics
                    diag = model.snowball.get_memory_diagnostics(memory_val)

                console.print(
                    f"Step {step}: NLL={val_loss.item():.4f}, MAE={mae:.4f}, "
                    f"Scales: {diag}"
                )

                if val_loss.item() < best_val_nll:
                    best_val_nll = val_loss.item()
                    torch.save({
                        "model": model.state_dict(),
                        "config": {
                            "vocab_size": config.vocab_size,
                            "n_layer": config.n_layer,
                            "n_head": config.n_head,
                            "n_embd": config.n_embd,
                            "seq_len": config.seq_len,
                            "n_memory_slots": config.n_memory_slots,
                            "dropout": config.dropout,
                            "use_scale": config.use_scale,
                            "num_params": config.num_params,
                            "n_scales": config.n_scales,
                            "alphas": config.alphas,
                            "use_rigidity_loss": config.use_rigidity_loss,
                            "rigidity_lambda": config.rigidity_lambda,
                        },
                        "bin_centers": bin_centers,
                    }, "out/snowball_v9_best.pt")

            progress.update(task, advance=1)

    console.print(f"\n[green]✅ Training complete! Best NLL: {best_val_nll:.4f}[/]")
    console.print(f"[green]Saved: out/snowball_v9_best.pt[/]")


if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    train()
