#!/usr/bin/env python3
"""
TASK_SPEC_2M — SpacingMDN + Memory Bank v0 (as instrument)

Memory Bank подключается как ПРИБОР для измерения:
- Shortcut / концентрация в 1-2 слотах
- Symmetry / все слоты одинаковые
- Co-adaptation / сговор между слотами

БЕЗ slot-dropout, БЕЗ orthogonality на первом прогоне!
Цель: измерить, НЕ улучшить.

Architecture:
  - Base: SpacingMDN (6L/8H/256E, K=8)
  - Memory: M=8 learnable tokens (prefix)
  - Memory LR: 0.3x base lr
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
import numpy as np
import argparse

console = Console()

# Import base components from train_mdn
from train_mdn import (
    MDNConfig, CausalSelfAttention, MLP, Block, MDNHead, mdn_loss
)


# ============================================================================
# MEMORY BANK
# ============================================================================

class MemoryBank(nn.Module):
    """
    Learnable memory tokens that serve as global context.

    These tokens are prepended to the input sequence and attend to/from
    all other tokens. They capture global patterns and long-range dependencies.

    Instrumented to track:
    - Attention patterns to memory slots
    - Gradient norms per slot
    - Slot embeddings for similarity analysis
    """

    def __init__(self, n_slots: int, n_embd: int):
        super().__init__()
        self.n_slots = n_slots
        self.n_embd = n_embd

        # Learnable memory embeddings
        self.memory = nn.Parameter(torch.randn(n_slots, n_embd) * 0.02)

        # Track attention received by each slot (for diagnostics)
        self.register_buffer('attn_mass', torch.zeros(n_slots))
        self.register_buffer('attn_count', torch.tensor(0))

    def forward(self, batch_size: int, device: torch.device):
        """Return memory tokens expanded for batch."""
        return self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, D)

    def update_attn_stats(self, attn_weights):
        """
        Update attention statistics for diagnostics.

        Args:
            attn_weights: (B, n_heads, T, T) attention weights
                          First M columns are memory tokens
        """
        # Sum attention to memory slots across all positions and heads
        # attn[:, :, :, :M] = attention TO memory FROM all positions
        M = self.n_slots
        attn_to_memory = attn_weights[:, :, M:, :M]  # (B, H, T-M, M)
        mass_per_slot = attn_to_memory.sum(dim=(0, 1, 2))  # (M,)

        self.attn_mass += mass_per_slot.detach()
        self.attn_count += 1

    def get_attn_distribution(self):
        """Get normalized attention distribution across slots."""
        if self.attn_count == 0:
            return torch.ones(self.n_slots) / self.n_slots
        total = self.attn_mass.sum()
        if total == 0:
            return torch.ones(self.n_slots) / self.n_slots
        return self.attn_mass / total

    def reset_stats(self):
        """Reset attention statistics."""
        self.attn_mass.zero_()
        self.attn_count.zero_()

    def get_slot_similarity(self):
        """Compute cosine similarity matrix between memory slots."""
        # Normalize embeddings
        normed = F.normalize(self.memory, dim=-1)  # (M, D)
        # Cosine similarity
        sim = torch.mm(normed, normed.t())  # (M, M)
        return sim.detach()


# ============================================================================
# SPACING MDN WITH MEMORY
# ============================================================================

class SpacingMDNMemory(nn.Module):
    """
    SpacingMDN with Memory Bank prefix.

    Memory tokens are prepended to the sequence and participate in attention.
    This allows the model to learn global context across the entire sequence.
    """

    def __init__(self, config: MDNConfig, n_memory_slots: int = 8):
        super().__init__()
        self.config = config
        self.n_memory_slots = n_memory_slots

        # Memory bank
        self.memory_bank = MemoryBank(n_memory_slots, config.n_embd)

        # Continuous input projection
        self.input_proj = nn.Linear(1, config.n_embd)

        # Positional embeddings (extended for memory + sequence)
        max_len = config.seq_len + n_memory_slots
        self.wpe = nn.Embedding(max_len, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        # MDN head
        self.mdn_head = MDNHead(config.n_embd, config.n_components)

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        console.print(f"[green]SpacingMDN+Memory: {n_params/1e6:.2f}M parameters[/]")
        console.print(f"[dim]  Memory slots: {n_memory_slots}[/]")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_attention=False, collect_memory_stats=False):
        """
        Args:
            x: (B, T) continuous spacing values
            return_attention: return attention weights for all layers
            collect_memory_stats: update memory attention statistics

        Returns:
            pi, mu, sigma: MDN parameters for positions AFTER memory prefix
            attentions: list of attention weights if return_attention=True
        """
        B, T = x.size()
        device = x.device
        M = self.n_memory_slots

        # Get memory tokens
        memory = self.memory_bank(B, device)  # (B, M, D)

        # Project input values
        x = x.unsqueeze(-1)  # (B, T, 1)
        x = self.input_proj(x)  # (B, T, D)

        # Concatenate: [memory_tokens, input_tokens]
        x = torch.cat([memory, x], dim=1)  # (B, M+T, D)

        # Positional embeddings
        pos = torch.arange(0, M + T, dtype=torch.long, device=device)
        x = x + self.wpe(pos)
        x = self.drop(x)

        # Transformer blocks
        attentions = []
        for block in self.blocks:
            if return_attention or collect_memory_stats:
                x, attn = block(x, return_attention=True)
                attentions.append(attn)

                if collect_memory_stats:
                    self.memory_bank.update_attn_stats(attn)
            else:
                x = block(x)

        x = self.ln_f(x)

        # Remove memory positions for output (we only predict input positions)
        x = x[:, M:, :]  # (B, T, D)

        # MDN head
        pi, mu, sigma = self.mdn_head(x)

        if return_attention:
            return pi, mu, sigma, attentions
        return pi, mu, sigma

    def get_memory_param_groups(self, base_lr, memory_lr_mult=0.3):
        """
        Return parameter groups with different learning rates.
        Memory parameters get lower LR.
        """
        memory_params = list(self.memory_bank.parameters())
        other_params = [p for n, p in self.named_parameters()
                       if 'memory_bank' not in n]

        return [
            {'params': other_params, 'lr': base_lr},
            {'params': memory_params, 'lr': base_lr * memory_lr_mult},
        ]


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    console.print("[bold magenta]═══ TASK_SPEC_2M: SpacingMDN + Memory Bank v0 ═══[/]\n")
    console.print("[yellow]Mode: INSTRUMENT (measuring, not fixing)[/]")
    console.print("[dim]No slot-dropout, no orthogonality[/]\n")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load data
    data_dir = Path(args.data_dir)
    train_data = torch.load(data_dir / "train.pt", weights_only=False)
    val_data = torch.load(data_dir / "val.pt", weights_only=False)
    meta = torch.load(data_dir / "meta.pt", weights_only=False)

    console.print(f"[green]Train: {train_data.shape}[/]")
    console.print(f"[green]Val: {val_data.shape}[/]")

    # Config
    config = MDNConfig(
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_components=args.n_components,
        dropout=args.dropout,
    )

    # Model with memory
    model = SpacingMDNMemory(config, n_memory_slots=args.n_memory_slots).to(device)

    # torch.compile for faster training (PyTorch 2.0+)
    if args.compile and device.type == "cuda":
        console.print("[yellow]Compiling model with torch.compile()...[/]")
        model = torch.compile(model)

    # Get underlying model for accessing memory_bank (needed after compile)
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == "cuda" else None

    # Optimizer with separate LR for memory
    param_groups = base_model.get_memory_param_groups(args.lr, args.memory_lr_mult)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # LR scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        progress = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        return max(args.min_lr / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    console.print(f"\n[cyan]Training config:[/]")
    console.print(f"  max_steps={args.max_steps}")
    console.print(f"  batch_size={args.batch_size}")
    console.print(f"  lr={args.lr}, memory_lr={args.lr * args.memory_lr_mult}")
    console.print(f"  n_memory_slots={args.n_memory_slots}")

    best_val_nll = float('inf')
    train_losses = []
    val_nlls = []

    # For memory diagnostics
    grad_norms_per_slot = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Training MDN+Memory...", total=args.max_steps)

        for step in range(1, args.max_steps + 1):
            model.train()

            # Sample batch
            idx = torch.randint(0, len(train_data), (args.batch_size,))
            batch = train_data[idx].to(device)

            x = batch[:, :-1]
            y = batch[:, 1:]

            # Forward (collect memory stats periodically)
            collect_stats = (step % args.eval_interval == 0)

            # AMP autocast
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                pi, mu, sigma = model(x, collect_memory_stats=collect_stats)
                loss, nll, entropy = mdn_loss(pi, mu, sigma, y, args.entropy_reg)

            # Backward
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Collect gradient norms per memory slot
            if step % args.eval_interval == 0:
                # Get underlying model if compiled
                base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                mem_grads = base_model.memory_bank.memory.grad
                if mem_grads is not None:
                    slot_grad_norms = mem_grads.norm(dim=1).detach().cpu().numpy()
                    grad_norms_per_slot.append(slot_grad_norms)

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            train_losses.append(nll.item())

            # Eval & logging
            if step % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_idx = torch.randint(0, len(val_data), (min(512, len(val_data)),))
                    val_batch = val_data[val_idx].to(device)
                    val_x = val_batch[:, :-1]
                    val_y = val_batch[:, 1:]

                    val_pi, val_mu, val_sigma = model(val_x)
                    val_loss, val_nll, val_entropy = mdn_loss(val_pi, val_mu, val_sigma, val_y, args.entropy_reg)

                val_nlls.append(val_nll.item())

                # Memory attention distribution
                attn_dist = base_model.memory_bank.get_attn_distribution()
                attn_entropy = -(attn_dist * torch.log(attn_dist + 1e-10)).sum().item()
                max_attn_slot = attn_dist.argmax().item()

                console.print(
                    f"Step {step}: "
                    f"train_nll={nll.item():.4f} | "
                    f"val_nll={val_nll.item():.4f} | "
                    f"mem_entropy={attn_entropy:.2f} | "
                    f"top_slot={max_attn_slot}"
                )

                # Save best
                if val_nll.item() < best_val_nll:
                    best_val_nll = val_nll.item()
                    torch.save({
                        "model": model.state_dict(),
                        "config": config.__dict__,
                        "n_memory_slots": args.n_memory_slots,
                        "step": step,
                        "val_nll": val_nll.item(),
                        "meta": meta,
                    }, out_dir / "best.pt")

            # Save checkpoints
            if step % args.save_interval == 0:
                torch.save({
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "n_memory_slots": args.n_memory_slots,
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "meta": meta,
                    "grad_norms_per_slot": grad_norms_per_slot,
                }, out_dir / f"ckpt_{step}.pt")
                console.print(f"[dim]Saved ckpt_{step}.pt[/]")

            progress.update(task, advance=1)

    # Save final
    torch.save({
        "model": model.state_dict(),
        "config": config.__dict__,
        "n_memory_slots": args.n_memory_slots,
        "step": args.max_steps,
        "train_losses": train_losses,
        "val_nlls": val_nlls,
        "grad_norms_per_slot": grad_norms_per_slot,
        "meta": meta,
    }, out_dir / "final.pt")

    console.print(f"\n[green]═══ Training Complete ═══[/]")
    console.print(f"Best val NLL: {best_val_nll:.4f}")

    # ========================================================================
    # MEMORY DIAGNOSTICS
    # ========================================================================
    console.print("\n[bold cyan]═══ Memory Diagnostics ═══[/]")

    # Attention distribution
    attn_dist = base_model.memory_bank.get_attn_distribution()
    console.print("\n[cyan]Attention mass per slot:[/]")
    for i, mass in enumerate(attn_dist):
        bar = '█' * int(mass * 50)
        console.print(f"  Slot {i}: {mass:.3f} {bar}")

    # Slot similarity
    sim = base_model.memory_bank.get_slot_similarity()
    console.print("\n[cyan]Slot similarity matrix (diag=1):[/]")

    # Check for symmetry (all slots similar)
    off_diag = sim[~torch.eye(args.n_memory_slots, dtype=bool)].mean().item()
    console.print(f"  Mean off-diagonal similarity: {off_diag:.3f}")

    if off_diag > 0.9:
        console.print("[yellow]  ⚠ HIGH SYMMETRY: slots are nearly identical[/]")
    elif off_diag > 0.7:
        console.print("[yellow]  ⚠ Moderate symmetry[/]")
    else:
        console.print("[green]  ✓ Slots are differentiated[/]")

    # Check for shortcut (concentration in few slots)
    attn_entropy = -(attn_dist * torch.log(attn_dist + 1e-10)).sum().item()
    max_entropy = np.log(args.n_memory_slots)
    console.print(f"\n[cyan]Attention entropy: {attn_entropy:.2f} (max={max_entropy:.2f})[/]")

    if attn_entropy < max_entropy * 0.5:
        console.print("[yellow]  ⚠ SHORTCUT: attention concentrated in few slots[/]")
    else:
        console.print("[green]  ✓ Attention distributed across slots[/]")

    # Gradient norms
    if grad_norms_per_slot:
        avg_grad_norms = np.mean(grad_norms_per_slot, axis=0)
        console.print(f"\n[cyan]Average gradient norm per slot:[/]")
        for i, norm in enumerate(avg_grad_norms):
            bar = '█' * int(norm / avg_grad_norms.max() * 30)
            console.print(f"  Slot {i}: {norm:.4f} {bar}")

    # Save diagnostics
    diag = {
        "attn_dist": attn_dist.cpu().numpy(),
        "slot_similarity": sim.cpu().numpy(),
        "attn_entropy": attn_entropy,
        "off_diag_similarity": off_diag,
        "grad_norms_per_slot": grad_norms_per_slot,
    }
    torch.save(diag, out_dir / "memory_diagnostics.pt")
    console.print(f"\n[green]Saved memory_diagnostics.pt[/]")


def main():
    parser = argparse.ArgumentParser(description="SpacingMDN + Memory Bank Training")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/continuous_2M")

    # Model
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Memory
    parser.add_argument("--n-memory-slots", type=int, default=8)
    parser.add_argument("--memory-lr-mult", type=float, default=0.3)

    # Training
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=10000)  # Shorter for memory experiment
    parser.add_argument("--entropy-reg", type=float, default=0.005)

    # Checkpointing
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--out-dir", type=str, default="out/mdn_memory_v0")

    # Performance
    parser.add_argument("--use-amp", action="store_true", help="Use AMP for 2x memory savings")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for 1.5-2x speedup")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
