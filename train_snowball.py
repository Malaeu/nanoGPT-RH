#!/usr/bin/env python3
"""
SNOWBALL MEMORY V8: Infinite Recurrent Context

Trifecta финальный босс:
- Wigner Surmise (physics-informed likelihood)
- RoPE (relative position attention)
- Snowball Memory (infinite recurrent context)

Memory Token — вектор который сжимает всю историю и подаётся в начало окна.
Это убивает early-token starvation: даже первый токен "видит" миллионы предыдущих.

Ожидаемый эффект:
- Position-bias → ~0 (early tokens имеют глобальный контекст)
- Long-range rigidity и phase drift становятся видимы
- PC1 падает, в residuals всплывает настоящая физика
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
    WignerRoPEGPT, RoPEConfig,
    RotaryEmbedding, RoPEAttention, RoPETransformerBlock, MemoryBank
)

console = Console()


# ============================================================================
# SNOWBALL MEMORY MODULE
# ============================================================================

class SnowballMemory(nn.Module):
    """
    Recurrent memory state that accumulates across windows.

    Each window:
    1. Memory token prepended to input
    2. Model processes [memory, x1, x2, ..., xT]
    3. Memory updated from hidden states

    This gives early tokens access to infinite past context.
    """

    def __init__(self, n_embd, update_mode="ema", alpha=0.9):
        super().__init__()
        self.n_embd = n_embd
        self.update_mode = update_mode
        self.alpha = alpha

        # Learnable initial memory state
        self.init_memory = nn.Parameter(torch.randn(1, 1, n_embd) * 0.02)

        # Memory update network (if not using simple EMA)
        if update_mode == "mlp":
            self.update_net = nn.Sequential(
                nn.Linear(n_embd * 2, n_embd * 4),
                nn.GELU(),
                nn.Linear(n_embd * 4, n_embd),
                nn.Tanh(),  # Bound updates
            )

        # Memory read projection (optional refinement)
        self.read_proj = nn.Linear(n_embd, n_embd)

    def get_initial_state(self, batch_size, device):
        """Get initial memory state for a batch."""
        return self.init_memory.expand(batch_size, 1, -1).to(device)

    def prepare_input(self, x, memory_state):
        """Prepend memory token to input sequence."""
        # x: (B, T, D), memory_state: (B, 1, D)
        memory_token = self.read_proj(memory_state)
        return torch.cat([memory_token, x], dim=1)  # (B, T+1, D)

    def update_state(self, memory_state, hidden_states):
        """Update memory from processed hidden states.

        Args:
            memory_state: (B, 1, D) current memory
            hidden_states: (B, T+1, D) hidden states from model
                           Note: position 0 is memory token

        Returns:
            new_memory: (B, 1, D) updated memory
        """
        B = memory_state.shape[0]

        # Use last hidden state as summary of current window
        window_summary = hidden_states[:, -1:, :]  # (B, 1, D)

        if self.update_mode == "ema":
            # Exponential moving average
            new_memory = self.alpha * memory_state + (1 - self.alpha) * window_summary

        elif self.update_mode == "mlp":
            # MLP-based update
            combined = torch.cat([memory_state, window_summary], dim=-1)  # (B, 1, 2*D)
            delta = self.update_net(combined)  # (B, 1, D)
            new_memory = memory_state + delta

        elif self.update_mode == "gate":
            # Gated update (GRU-style)
            combined = torch.cat([memory_state, window_summary], dim=-1)
            gate = torch.sigmoid(self.update_net(combined))
            new_memory = gate * memory_state + (1 - gate) * window_summary

        else:
            new_memory = window_summary  # Just use latest

        return new_memory


# ============================================================================
# SNOWBALL WRAPPER MODEL
# ============================================================================

class SnowballConfig:
    """Config for Snowball GPT (v8)"""
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    seq_len: int = 256
    dropout: float = 0.1
    n_memory_slots: int = 4
    use_scale: bool = True
    num_params: int = 1
    # Snowball specific
    memory_update_mode: str = "ema"
    memory_alpha: float = 0.9

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class SnowballGPT(nn.Module):
    """
    GPT with Wigner output + RoPE + Snowball Memory.

    The complete Trifecta for Causal Zeta Oracle.
    """

    def __init__(self, config: SnowballConfig):
        super().__init__()
        self.config = config

        # Token embedding (no positional - RoPE handles it)
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

        # Snowball Memory
        self.snowball = SnowballMemory(
            config.n_embd,
            update_mode=config.memory_update_mode,
            alpha=config.memory_alpha
        )

        # Memory Bank (additional learnable slots)
        self.memory_bank = MemoryBank(config.n_memory_slots, config.n_embd)

        # RoPE Transformer blocks (extended seq_len for memory token)
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(
                config.n_embd,
                config.n_head,
                config.dropout,
                max_position=config.seq_len + 1  # +1 for memory token
            )
            for _ in range(config.n_layer)
        ])

        # Wigner output
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_params, bias=True)

        with torch.no_grad():
            self.head.bias.data[0] = 0.5

        # Causal mask (extended for memory token)
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
                memory_state=None, return_memory=False):
        """
        Args:
            idx: (B, T) input token indices
            targets: (B, T) target spacing values (FLOAT!)
            scale_val: (B, T) scale values for FiLM
            memory_state: (B, 1, D) recurrent memory state
            return_memory: whether to return updated memory

        Returns:
            pred: (B, T, num_params)
            loss: scalar NLL
            mem_attn: (B, n_slots)
            new_memory: (B, 1, D) if return_memory
        """
        B, T = idx.shape
        device = idx.device

        # Initialize memory if not provided
        if memory_state is None:
            memory_state = self.snowball.get_initial_state(B, device)

        # Token embedding
        x = self.tok_emb(idx)  # (B, T, D)

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
        x = self.snowball.prepare_input(x, memory_state)  # (B, T+1, D)

        # Memory Bank
        x, mem_attn = self.memory_bank(x)

        # RoPE Transformer blocks
        causal_mask = self.causal_mask[:T+1, :T+1]
        for block in self.blocks:
            x = block(x, causal_mask)

        # Save hidden for memory update
        hidden_states = x

        # Update memory state
        new_memory = self.snowball.update_state(memory_state, hidden_states)

        # Remove memory token position from output
        x = x[:, 1:, :]  # (B, T, D) — back to original length

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
                result["memory_state"] = new_memory
            return result

        if return_memory:
            return pred, loss, mem_attn, new_memory

        return pred, loss, mem_attn

    def get_mu(self, pred):
        raw_mu = pred[..., 0]
        return F.softplus(raw_mu) + 0.1

    def get_memory_vectors(self):
        return self.memory_bank.memory.detach().cpu().numpy()


# ============================================================================
# TRAINING WITH RECURRENT WINDOWS
# ============================================================================

def train():
    console.print("[bold magenta]❄️ SNOWBALL V8: Infinite Recurrent Memory![/]")
    console.print("[dim]Trifecta: Wigner + RoPE + Snowball = Causal Zeta Oracle[/]\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load data
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data = torch.load('data/val.pt', weights_only=False)

    bin_centers = np.load('data/bin_centers.npy')
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)
    console.print(f"[green]Bin centers loaded: {len(bin_centers)}[/]")

    # Config
    config = SnowballConfig(
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        seq_len=256,
        n_memory_slots=4,
        dropout=0.1,
        use_scale=True,
        num_params=1,
        memory_update_mode="ema",
        memory_alpha=0.9,
    )

    console.print(f"[green]Memory update: {config.memory_update_mode} (alpha={config.memory_alpha})[/]")

    # Model
    model = SnowballGPT(config).to(device)

    # Transfer from v7 RoPE
    v7_ckpt_path = Path('out/rope_v7_best.pt')
    if v7_ckpt_path.exists():
        console.print("[yellow]Loading v7 RoPE checkpoint for transfer...[/]")
        v7_ckpt = torch.load(v7_ckpt_path, map_location=device, weights_only=False)
        v7_state = v7_ckpt['model']
        model_state = model.state_dict()

        transferred = 0
        for name, param in v7_state.items():
            # Skip snowball-specific layers
            if 'snowball' in name:
                continue
            if name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name] = param
                    transferred += 1

        model.load_state_dict(model_state)
        console.print(f"[green]Transferred {transferred} layers from v7[/]")
    else:
        console.print("[yellow]No v7 checkpoint, training from scratch[/]")

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Parameters: {n_params:,}[/]\n")

    # Data: train with recurrent windows
    # Use smaller window to process multiple windows per sequence
    window_size = 128  # Smaller to allow 2 windows in 255-token sequence
    n_windows_per_seq = 2

    train_dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:])
    val_dataset = TensorDataset(val_data[:, :-1], val_data[:, 1:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Smaller batch for memory
    val_loader = DataLoader(val_dataset, batch_size=32)

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
        task = progress.add_task("[cyan]Training Snowball v8...", total=n_steps)

        while step < n_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            B = x.shape[0]

            # Initialize memory
            memory_state = model.snowball.get_initial_state(B, device)

            # Process windows recurrently
            total_loss = 0
            T = x.shape[1]
            n_windows = min(n_windows_per_seq, T // window_size)

            for w in range(n_windows):
                start = w * window_size
                end = start + window_size
                if end > T:
                    break

                x_win = x[:, start:end]
                y_win = y[:, start:end]
                scale_win = bin_centers_t[x_win]
                target_win = bin_centers_t[y_win]

                pred, loss, _, memory_state = model(
                    x_win,
                    targets=target_win,
                    scale_val=scale_win,
                    memory_state=memory_state,
                    return_memory=True
                )

                if torch.isnan(loss):
                    continue

                total_loss += loss

                # Detach memory to prevent gradient explosion through time
                memory_state = memory_state.detach()

            if n_windows > 0 and total_loss > 0:
                avg_loss = total_loss / n_windows

                optimizer.zero_grad()
                avg_loss.backward()
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
                hard_early = 0
                hard_late = 0
                total_hard = 0

                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        B = x.shape[0]

                        # Single window validation with fresh memory
                        memory = model.snowball.get_initial_state(B, device)
                        sv = bin_centers_t[x]
                        target = bin_centers_t[y]

                        pred, loss, _, _ = model(
                            x, targets=target, scale_val=sv,
                            memory_state=memory, return_memory=True
                        )

                        if not torch.isnan(loss):
                            val_losses.append(loss.item())

                        mu = model.get_mu(pred)
                        error = (mu - target).abs()
                        mae_errors.append(error.mean().item())

                        # Hard token positions
                        thr = error.quantile(0.95)
                        hard_mask = error > thr
                        early_mask = torch.arange(target.shape[1], device=device) < 50
                        late_mask = torch.arange(target.shape[1], device=device) >= 200

                        hard_early += (hard_mask & early_mask).sum().item()
                        hard_late += (hard_mask & late_mask).sum().item()
                        total_hard += hard_mask.sum().item()

                val_loss = np.mean(val_losses) if val_losses else float('inf')
                val_mae = np.mean(mae_errors)
                early_pct = hard_early / max(total_hard, 1) * 100
                late_pct = hard_late / max(total_hard, 1) * 100

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model': model.state_dict(),
                        'config': config.__dict__,
                        'bin_centers': bin_centers,
                        'step': step,
                        'val_loss': val_loss,
                        'val_mae': val_mae,
                    }, 'out/snowball_v8_best.pt')

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
    }, 'out/snowball_v8_final.pt')

    console.print(f"\n[bold green]✅ Training complete![/]")
    console.print(f"[green]Best NLL: {best_val_loss:.4f}[/]")

    console.print(f"\n[cyan]Saved: out/snowball_v8_best.pt[/]")
    console.print(f"[cyan]Saved: out/snowball_v8_final.pt[/]")
    console.print("\n[yellow]Run mine_residuals_snowball.py to verify early-bias is DEAD![/]")


if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    train()
