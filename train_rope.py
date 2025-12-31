#!/usr/bin/env python3
"""
WIGNER V7 + RoPE: Position-Aware Physics-Informed Model

Добавляем Rotary Position Embeddings (RoPE) чтобы убить position-bias.

Почему RoPE:
- Классические pos_emb плохо работают для ранних токенов (мало контекста)
- RoPE вращает Q/K по позиции → relative position attention нативно
- Ранние токены получают лучший фокус, hard tokens распределяются равномерно

Ожидаемый эффект:
- corr(H1, position) → ~0.0
- PC1 → 30-50%
- В residuals всплывёт настоящая физика
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

console = Console()


# ============================================================================
# ROTARY POSITION EMBEDDINGS (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from RoFormer/Llama."""

    def __init__(self, dim, max_position=512, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base

        # Precompute inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_position)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys."""
    # q, k: (batch, heads, seq, head_dim)
    # cos, sin: (seq, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================================
# CUSTOM ATTENTION WITH RoPE
# ============================================================================

class RoPEAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings."""

    def __init__(self, n_embd, n_head, dropout=0.1, max_position=512):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        # QKV projection
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_position=max_position)

        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        B, T, C = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask[:T, :T], float('-inf'))

        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))

        return out


class RoPETransformerBlock(nn.Module):
    """Transformer block with RoPE attention."""

    def __init__(self, n_embd, n_head, dropout=0.1, max_position=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = RoPEAttention(n_embd, n_head, dropout, max_position)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask=None):
        x = x + self.attn(self.ln1(x), causal_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================================
# MEMORY BANK (same as before)
# ============================================================================

class MemoryBank(nn.Module):
    def __init__(self, n_slots, dim):
        super().__init__()
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

        return x + mem_out.unsqueeze(1), attn_weights


# ============================================================================
# WIGNER + RoPE MODEL (V7)
# ============================================================================

class RoPEConfig:
    """Config for RoPE Wigner GPT (v7)"""
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    seq_len: int = 256
    dropout: float = 0.1
    n_memory_slots: int = 4
    use_scale: bool = True
    num_params: int = 1  # Wigner params

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class WignerRoPEGPT(nn.Module):
    """
    GPT with Wigner output + RoPE positional encoding.

    RoPE убивает position-bias, Wigner убивает scale-bias.
    """

    def __init__(self, config: RoPEConfig):
        super().__init__()
        self.config = config

        # Input: только token embedding (позиция через RoPE в attention!)
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

        # Memory Bank
        self.memory_bank = MemoryBank(config.n_memory_slots, config.n_embd)

        # RoPE Transformer blocks
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(
                config.n_embd,
                config.n_head,
                config.dropout,
                max_position=config.seq_len
            )
            for _ in range(config.n_layer)
        ])

        # Wigner output
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_params, bias=True)

        # Initialize
        with torch.no_grad():
            self.head.bias.data[0] = 0.5  # softplus(0.5) ≈ 0.97

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.seq_len, config.seq_len), diagonal=1).bool()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_hidden=False, scale_val=None):
        B, T = idx.shape

        # Token embedding (NO positional embedding - RoPE handles it!)
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

        # Memory Bank
        x, mem_attn = self.memory_bank(x)

        # RoPE Transformer blocks
        for block in self.blocks:
            x = block(x, self.causal_mask)

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
            return {
                "pred": pred,
                "loss": loss,
                "hidden_pre_ln": hidden_pre_ln,
                "hidden_post_ln": hidden_post_ln,
                "mem_attn": mem_attn,
            }

        return pred, loss, mem_attn

    def get_mu(self, pred):
        raw_mu = pred[..., 0]
        return F.softplus(raw_mu) + 0.1

    def get_memory_vectors(self):
        return self.memory_bank.memory.detach().cpu().numpy()


# ============================================================================
# TRAINING
# ============================================================================

def train():
    console.print("[bold magenta]🌀 WIGNER V7 + RoPE: Position-Aware Physics![/]")
    console.print("[dim]RoPE kills position-bias, Wigner kills scale-bias[/]\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load data
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data = torch.load('data/val.pt', weights_only=False)

    bin_centers = np.load('data/bin_centers.npy')
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)
    console.print(f"[green]Bin centers loaded: {len(bin_centers)}[/]")

    # Config
    config = RoPEConfig(
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        seq_len=256,
        n_memory_slots=4,
        dropout=0.1,
        use_scale=True,
        num_params=1,
    )

    console.print(f"[green]Using RoPE instead of learned pos_emb![/]")

    # Model
    model = WignerRoPEGPT(config).to(device)

    # Transfer from v6 (if available)
    v6_ckpt_path = Path('out/wigner_v6_best.pt')
    if v6_ckpt_path.exists():
        console.print("[yellow]Loading v6 checkpoint for transfer...[/]")
        v6_ckpt = torch.load(v6_ckpt_path, map_location=device, weights_only=False)
        v6_state = v6_ckpt['model']
        model_state = model.state_dict()

        transferred = 0
        skipped = []
        for name, param in v6_state.items():
            # Skip pos_emb (we use RoPE now) and incompatible layers
            if 'pos_emb' in name:
                skipped.append(name)
                continue
            if 'blocks' in name:
                # Skip old transformer blocks (different structure)
                skipped.append(name)
                continue

            if name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name] = param
                    transferred += 1

        model.load_state_dict(model_state)
        console.print(f"[green]Transferred {transferred} layers[/]")
        console.print(f"[dim]Skipped {len(skipped)} incompatible layers (pos_emb, old blocks)[/]")
    else:
        console.print("[yellow]No v6 checkpoint, training from scratch[/]")

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Parameters: {n_params:,}[/]\n")

    # Data loaders
    train_dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:])
    val_dataset = TensorDataset(val_data[:, :-1], val_data[:, 1:])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Optimizer (higher LR since training from scratch mostly)
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
        task = progress.add_task("[cyan]Training RoPE v7...", total=n_steps)

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

            if torch.isnan(loss):
                console.print("[red]NaN loss! Skipping.[/]")
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
                # Track hard tokens by position
                hard_pos_early = 0
                hard_pos_late = 0
                total_hard = 0

                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        sv = bin_centers_t[x]
                        target = bin_centers_t[y]

                        pred, loss, _ = model(x, targets=target, scale_val=sv)
                        if not torch.isnan(loss):
                            val_losses.append(loss.item())

                        mu = model.get_mu(pred)
                        error = (mu - target).abs()
                        mae_errors.append(error.mean().item())

                        # Check hard token positions
                        per_token_err = error
                        thr = per_token_err.quantile(0.95)
                        hard_mask = per_token_err > thr

                        # Count early vs late
                        early_mask = torch.arange(target.shape[1], device=device) < 50
                        late_mask = torch.arange(target.shape[1], device=device) >= 200

                        hard_pos_early += (hard_mask & early_mask).sum().item()
                        hard_pos_late += (hard_mask & late_mask).sum().item()
                        total_hard += hard_mask.sum().item()

                val_loss = np.mean(val_losses) if val_losses else float('inf')
                val_mae = np.mean(mae_errors)

                early_pct = hard_pos_early / max(total_hard, 1) * 100
                late_pct = hard_pos_late / max(total_hard, 1) * 100

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model': model.state_dict(),
                        'config': config.__dict__,
                        'bin_centers': bin_centers,
                        'step': step,
                        'val_loss': val_loss,
                        'val_mae': val_mae,
                    }, 'out/rope_v7_best.pt')

                console.print(
                    f"[dim]Step {step}: NLL={val_loss:.4f}, MAE={val_mae:.4f}, "
                    f"hard_early={early_pct:.1f}%, hard_late={late_pct:.1f}%[/]"
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
    }, 'out/rope_v7_final.pt')

    console.print(f"\n[bold green]✅ Training complete![/]")
    console.print(f"[green]Best NLL: {best_val_loss:.4f}[/]")

    console.print("\n[bold]Memory Slot Usage:[/]")
    mem_attn_avg = mem_attn.mean(dim=0).detach().cpu().numpy()
    for i, w in enumerate(mem_attn_avg):
        bar = "█" * int(w * 40)
        console.print(f"  Slot {i}: {bar} ({w:.3f})")

    console.print(f"\n[cyan]Saved: out/rope_v7_best.pt[/]")
    console.print(f"[cyan]Saved: out/rope_v7_final.pt[/]")
    console.print("\n[yellow]Run mine_residuals_rope.py to verify position-bias is DEAD![/]")


if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    train()
