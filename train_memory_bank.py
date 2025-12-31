#!/usr/bin/env python3
"""
MEMORY BANK TRAINING: Teaching AI to discover prime rhythms

Architecture: MemoryBankGPT
- 4 learnable memory slots (each 128-dim)
- Each slot can specialize on different "frequency" of the data
- After training, we probe the memory to find hidden rhythms

Hypothesis: If the model learns spectral structure, memory slots
will resonate with frequencies like 2π or 6.644 (the "Monster")
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

console = Console()


class MemoryBankConfig:
    """Config for MemoryBankGPT"""
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    seq_len: int = 256
    dropout: float = 0.1
    n_memory_slots: int = 4  # Number of learnable memory banks
    use_scale: bool = True   # Scale Injection to cure "calibration blindness"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MemoryBank(nn.Module):
    """
    Learnable memory bank with multiple slots.
    Each slot is a learnable vector that gets mixed into the input.
    """
    def __init__(self, n_slots: int, dim: int):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim

        # Learnable memory slots
        self.memory = nn.Parameter(torch.randn(n_slots, dim) * 0.02)

        # Attention over memory slots (query from input)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (batch, seq, dim)
        Returns: x augmented with memory, plus attention weights for analysis
        """
        B, T, D = x.shape

        # Query from input (use mean over sequence)
        q = self.query_proj(x.mean(dim=1))  # (B, D)

        # Keys and values from memory
        k = self.key_proj(self.memory)  # (n_slots, D)
        v = self.value_proj(self.memory)  # (n_slots, D)

        # Attention: which memory slots are relevant?
        attn = torch.matmul(q, k.T) / math.sqrt(D)  # (B, n_slots)
        attn_weights = F.softmax(attn, dim=-1)  # (B, n_slots)

        # Weighted sum of memory values
        mem_out = torch.matmul(attn_weights, v)  # (B, D)
        mem_out = self.out_proj(mem_out)

        # Add memory to each position
        x_augmented = x + mem_out.unsqueeze(1)

        return x_augmented, attn_weights


class MemoryBankGPT(nn.Module):
    """
    GPT with learnable Memory Bank.
    Memory slots can specialize on different spectral features.
    """
    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.seq_len, config.n_embd)

        # FiLM: Feature-wise Linear Modulation (cures "calibration blindness")
        # Output 2*n_embd: gamma (scale) + beta (shift)
        # x = x * (1 + gamma) + beta
        if config.use_scale:
            self.scale_ln = nn.LayerNorm(1)
            self.scale_proj = nn.Sequential(
                nn.Linear(1, config.n_embd * 4),   # Bigger capacity
                nn.GELU(),                         # Better than Tanh for deep
                nn.Linear(config.n_embd * 4, config.n_embd * 2),  # 2*D for gamma+beta
                nn.Tanh(),                         # Bound outputs to prevent explosion
            )

        # Memory Bank (BEFORE transformer layers)
        self.memory_bank = MemoryBank(config.n_memory_slots, config.n_embd)

        # Transformer blocks (simplified)
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

        # Output
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Causal mask
        self.register_buffer(
            "mask",
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

    def forward(self, idx, targets=None, return_hidden=False, scale_val=None, loss_weight=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # === Damped FiLM: Feature-wise Linear Modulation ===
        # Damped to prevent overcorrection (v3 inverted the bias!)
        if self.config.use_scale and scale_val is not None:
            if scale_val.dim() == 2:
                scale_val = scale_val.unsqueeze(-1)  # (B,T,1)

            # Log transform for better dynamic range (spacings vary 0.01 to 4+)
            scale_val = torch.log1p(scale_val)  # log(1 + s), stable for small s

            s = self.scale_ln(scale_val)        # (B,T,1)
            style = self.scale_proj(s)          # (B,T, 2*D)

            gamma, beta = style.chunk(2, dim=-1)  # Each (B,T,D)

            # Damped FiLM: gentle nudge instead of sledgehammer
            DAMP_G = 0.2  # Damping factor for gamma (scale)
            DAMP_B = 0.2  # Damping factor for beta (shift)
            x = x * (1.0 + DAMP_G * gamma) + DAMP_B * beta

        # Apply Memory Bank
        x, mem_attn = self.memory_bank(x)

        # Transformer blocks with causal mask
        causal_mask = self.mask[:T, :T]
        for block in self.blocks:
            x = block(x, src_mask=causal_mask, is_causal=True)

        # Hidden states for gradient analysis
        hidden_pre_ln = x
        x = self.ln_f(x)
        hidden_post_ln = x

        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                weight=loss_weight,  # Inverse frequency weighting for long-tail
            )

        if return_hidden:
            return {
                "logits": logits,
                "loss": loss,
                "hidden_pre_ln": hidden_pre_ln,
                "hidden_post_ln": hidden_post_ln,
                "mem_attn": mem_attn,
            }

        return logits, loss, mem_attn

    def get_memory_vectors(self):
        """Extract raw memory vectors for analysis"""
        return self.memory_bank.memory.detach().cpu().numpy()


def train():
    console.print("[bold magenta]🧠 MEMORY BANK TRAINING V4 (Damped FiLM + Reweighting)[/]")
    console.print("[dim]x = x*(1 + 0.2γ) + 0.2β + inverse freq loss weights[/]\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load data
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data = torch.load('data/val.pt', weights_only=False)

    # Load bin_centers for scale injection
    bin_centers = np.load('data/bin_centers.npy')
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)
    console.print(f"[green]Scale injection: loaded {len(bin_centers)} bin centers[/]")

    # Compute inverse frequency weights for loss (long-tail vocab fix)
    vocab_size = 256
    token_freq = torch.bincount(train_data.flatten(), minlength=vocab_size).float()
    token_weight = 1.0 / (token_freq + 1e-6)  # inverse frequency
    token_weight = token_weight / token_weight.mean()  # normalize to mean=1
    token_weight = token_weight.to(device)
    console.print(f"[green]Loss reweighting: inverse freq, range [{token_weight.min():.2f}, {token_weight.max():.2f}][/]")

    # Create config
    config = MemoryBankConfig(
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        seq_len=256,
        n_memory_slots=4,  # 4 slots to find patterns
        dropout=0.1
    )

    console.print(f"[green]Memory slots: {config.n_memory_slots}[/]")
    console.print(f"[green]Embedding dim: {config.n_embd}[/]")

    # Create model and load v3 checkpoint for fine-tuning
    model = MemoryBankGPT(config).to(device)

    # Fine-tune from v3 checkpoint (not from scratch!)
    v3_ckpt_path = Path('out/memory_bank_v3_best.pt')
    if v3_ckpt_path.exists():
        v3_ckpt = torch.load(v3_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(v3_ckpt['model'])
        console.print(f"[yellow]Fine-tuning from v3 checkpoint (val_loss={v3_ckpt['val_loss']:.4f})[/]")
    else:
        console.print("[red]WARNING: v3 checkpoint not found, training from scratch![/]")

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Parameters: {n_params:,}[/]\n")

    # Data loaders
    train_dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:])
    val_dataset = TensorDataset(val_data[:, :-1], val_data[:, 1:])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Optimizer (lower LR for fine-tuning v4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

    # Training loop (10000 steps for damped FiLM + reweighting to converge)
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
        task = progress.add_task("[cyan]Training...", total=n_steps)

        while step < n_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            scale_val = bin_centers_t[x]  # (B, T) lookup real spacing values

            optimizer.zero_grad()
            logits, loss, mem_attn = model(x, y, scale_val=scale_val, loss_weight=token_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            progress.update(task, advance=1)

            # Validation every 500 steps
            if step % 500 == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        sv = bin_centers_t[x]  # scale_val for validation
                        _, loss, _ = model(x, y, scale_val=sv)
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)
                val_ppl = np.exp(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model': model.state_dict(),
                        'config': config,
                        'bin_centers': bin_centers,  # save for inference
                        'step': step,
                        'val_loss': val_loss
                    }, 'out/memory_bank_v4_best.pt')

                console.print(f"[dim]Step {step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}[/]")
                model.train()

    # Final save
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'bin_centers': bin_centers,  # save for inference
        'step': step,
        'val_loss': val_loss
    }, 'out/memory_bank_v4_final.pt')

    console.print(f"\n[bold green]✅ Training complete![/]")
    console.print(f"[green]Best val loss: {best_val_loss:.4f}[/]")
    console.print(f"[green]Best val PPL: {np.exp(best_val_loss):.2f}[/]")

    # Show memory attention distribution
    console.print("\n[bold]Memory Slot Usage (last batch):[/]")
    mem_attn_avg = mem_attn.mean(dim=0).detach().cpu().numpy()
    for i, w in enumerate(mem_attn_avg):
        bar = "█" * int(w * 40)
        console.print(f"  Slot {i}: {bar} ({w:.3f})")

    console.print(f"\n[cyan]Saved: out/memory_bank_v4_final.pt[/]")
    console.print(f"[cyan]Saved: out/memory_bank_v4_best.pt[/]")
    console.print("\n[yellow]Run mine_residuals_v2.py to check if Damped FiLM + Reweighting killed Scale Blindness![/]")


if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    train()
