#!/usr/bin/env python3
"""
MEMORY BANK TRAINING ON 100M ZEROS
==================================

Training MemoryBankGPT on 100M Riemann zeta zeros to discover prime rhythms.
Larger dataset = better frequency detection.

Architecture: MemoryBankGPT with 8 learnable memory slots
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
from rich.panel import Panel
import math

console = Console()

# Paths for 100M data
DATA_DIR = Path("data/zeros_100M")
OUTPUT_DIR = Path("out/memory_bank_100M")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MemoryBankConfig:
    """Config for MemoryBankGPT - larger for 100M data"""
    vocab_size: int = 256
    n_layer: int = 6          # More layers
    n_head: int = 8           # More heads
    n_embd: int = 256         # Larger embedding
    seq_len: int = 256
    dropout: float = 0.1
    n_memory_slots: int = 8   # More memory slots for richer pattern discovery

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MemoryBank(nn.Module):
    """Learnable memory bank with multiple slots."""
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


class MemoryBankGPT(nn.Module):
    """GPT with learnable Memory Bank for 100M data."""
    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.seq_len, config.n_embd)
        self.memory_bank = MemoryBank(config.n_memory_slots, config.n_embd)

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

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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

    def forward(self, idx, targets=None, return_hidden=False):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        x, mem_attn = self.memory_bank(x)

        causal_mask = self.mask[:T, :T]
        for block in self.blocks:
            x = block(x, src_mask=causal_mask, is_causal=True)

        hidden_pre_ln = x
        x = self.ln_f(x)
        hidden_post_ln = x
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

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
        return self.memory_bank.memory.detach().cpu().numpy()


def train():
    console.print(Panel.fit(
        "[bold magenta]MEMORY BANK TRAINING ON 100M ZEROS[/]\n"
        "Discovering prime rhythms in 100M Riemann zeta zeros",
        title="🧠"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load 100M data
    console.print("[cyan]Loading 100M data...[/]")
    train_data = torch.load(DATA_DIR / 'train.pt', weights_only=False)
    val_data = torch.load(DATA_DIR / 'val.pt', weights_only=False)

    console.print(f"[green]Train: {train_data.shape}[/]")
    console.print(f"[green]Val: {val_data.shape}[/]")

    # Create config
    config = MemoryBankConfig(
        vocab_size=256,
        n_layer=6,
        n_head=8,
        n_embd=256,
        seq_len=256,
        n_memory_slots=8,
        dropout=0.1
    )

    console.print(f"[green]Memory slots: {config.n_memory_slots}[/]")
    console.print(f"[green]Architecture: {config.n_layer}L/{config.n_head}H/{config.n_embd}E[/]")

    model = MemoryBankGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Parameters: {n_params:,}[/]\n")

    # Data loaders
    train_dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:])
    val_dataset = TensorDataset(val_data[:, :-1], val_data[:, 1:])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

    # Training loop
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

            optimizer.zero_grad()
            logits, loss, mem_attn = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            progress.update(task, advance=1)

            if step % 1000 == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        if i >= 50:  # Sample validation
                            break
                        x, y = x.to(device), y.to(device)
                        _, loss, _ = model(x, y)
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)
                val_ppl = np.exp(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model': model.state_dict(),
                        'config': config,
                        'step': step,
                        'val_loss': val_loss
                    }, OUTPUT_DIR / 'best.pt')
                    console.print(f"[green]Step {step}: val_ppl={val_ppl:.2f} ✓ NEW BEST[/]")
                else:
                    console.print(f"[dim]Step {step}: val_ppl={val_ppl:.2f}[/]")
                model.train()

    # Final save
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'step': step,
        'val_loss': val_loss
    }, OUTPUT_DIR / 'final.pt')

    console.print(f"\n[bold green]Training complete![/]")
    console.print(f"[green]Best val loss: {best_val_loss:.4f}[/]")
    console.print(f"[green]Best val PPL: {np.exp(best_val_loss):.2f}[/]")

    # Memory attention distribution
    console.print("\n[bold]Memory Slot Usage (last batch):[/]")
    mem_attn_avg = mem_attn.mean(dim=0).detach().cpu().numpy()
    for i, w in enumerate(mem_attn_avg):
        bar = "█" * int(w * 40)
        console.print(f"  Slot {i}: {bar} ({w:.3f})")

    console.print(f"\n[cyan]Saved: {OUTPUT_DIR}/best.pt[/]")
    console.print(f"[cyan]Saved: {OUTPUT_DIR}/final.pt[/]")
    console.print("\n[yellow]Run probe_brain_100M.py to analyze memory frequencies![/]")


if __name__ == "__main__":
    train()
