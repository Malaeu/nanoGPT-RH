#!/usr/bin/env python3
"""
train_mdn_postfix.py — POSTFIX Memory + Memory-Only Readout (Bottleneck)

KEY INSIGHT: PREFIX memory (E1/E2) is "blind" to data due to causal mask.
POSTFIX solves this: memory slots are AFTER data, so they CAN attend to all data.
Plus, readout only from memory creates a true BOTTLENECK.

Architecture changes vs PREFIX:
- Input sequence: [s₁, s₂, ..., s_T, M0, M1, ..., M7]  (data first, memory last)
- Causal mask => memory CAN attend to all data (they're to the left)
- Causal mask => data CANNOT attend to memory (they're to the right = "future")
- Readout uses ONLY memory hidden states (bottleneck!)
- MDN head predicts s_{T+1} (single next step), not per-position

Expected improvements after POSTFIX:
- Ablation Δ should INCREASE (memory now essential)
- Grad correlation should DECREASE (slots learn different things)
- Slots become TRUE "registers of window state"

Run:
  python train_mdn_postfix.py --data-dir data/continuous_2M --out-dir out/mdn_postfix_E3
"""

import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

# Enable TensorFloat32 for better H100/A100 performance
torch.set_float32_matmul_precision('high')

console = Console()

# Import base components from train_mdn (reuse existing blocks)
from train_mdn import MDNConfig, CausalSelfAttention, MLP, Block, MDNHead, mdn_loss


# ============================================================================
# POSTFIX MEMORY BANK
# ============================================================================

class MemoryBankPostfix(nn.Module):
    """
    Learnable memory slots for POSTFIX architecture.

    Unlike PREFIX, these slots are appended AFTER data tokens.
    In causal attention:
    - Memory CAN see all data (data is to the left)
    - Data CANNOT see memory (memory is to the right = "future")

    This makes memory a true "compression register" of the window.
    """

    def __init__(self, n_slots: int, n_embd: int, use_slot_id: bool = True):
        super().__init__()
        self.n_slots = n_slots
        self.n_embd = n_embd
        self.use_slot_id = use_slot_id

        # Learnable memory content
        self.memory = nn.Parameter(torch.randn(n_slots, n_embd) * 0.02)

        # Slot-ID embeddings (optional, but helps differentiation)
        if use_slot_id:
            self.slot_id = nn.Embedding(n_slots, n_embd)
            nn.init.orthogonal_(self.slot_id.weight)

        # Learnable readout weights for pooling memory outputs
        self.readout_weights = nn.Parameter(torch.zeros(n_slots))

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return memory tokens expanded for batch: [B, M, D]"""
        memory = self.memory  # [M, D]

        if self.use_slot_id:
            slot_ids = torch.arange(self.n_slots, device=device)
            memory = memory + self.slot_id(slot_ids)

        return memory.unsqueeze(0).expand(batch_size, -1, -1)  # [B, M, D]

    def get_readout_weights(self) -> torch.Tensor:
        """Get normalized readout weights for pooling."""
        return F.softmax(self.readout_weights, dim=0)

    def pool_memory_outputs(self, memory_hidden: torch.Tensor) -> torch.Tensor:
        """
        Pool memory outputs into single readout vector.

        Args:
            memory_hidden: [B, M, D] hidden states of memory positions
        Returns:
            readout: [B, D] pooled representation
        """
        w = self.get_readout_weights()  # [M]
        readout = (memory_hidden * w.view(1, -1, 1)).sum(dim=1)  # [B, D]
        return readout


# ============================================================================
# POSTFIX MDN MODEL
# ============================================================================

class SpacingMDNPostfix(nn.Module):
    """
    SpacingMDN with POSTFIX memory and memory-only readout.

    Key difference from PREFIX version:
    - Memory appended AFTER data: [data..., memory...]
    - Prediction comes ONLY from memory hidden states (bottleneck)
    - MDN predicts single next spacing s_{T+1}
    """

    def __init__(self, config: MDNConfig, n_memory_slots: int = 8,
                 use_slot_id: bool = True):
        super().__init__()
        self.config = config
        self.n_memory_slots = n_memory_slots
        self.use_slot_id = use_slot_id

        # Memory bank (POSTFIX)
        self.memory_bank = MemoryBankPostfix(n_memory_slots, config.n_embd, use_slot_id)

        # Input projection: spacing scalar -> embedding
        self.input_proj = nn.Linear(1, config.n_embd)

        # Positional embeddings (extended for data + memory)
        max_len = config.seq_len + n_memory_slots
        self.wpe = nn.Embedding(max_len, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        # MDN head for single-step prediction
        # Input: pooled memory [B, D], output: [B, 1, K] for pi/mu/sigma
        self.mdn_head = MDNHead(config.n_embd, config.n_components)

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        console.print(f"[green]SpacingMDN+POSTFIX: {n_params/1e6:.2f}M parameters[/]")
        console.print(f"[dim]  Memory slots: {n_memory_slots} (POSTFIX)[/]")
        console.print(f"[dim]  Readout: memory-only bottleneck[/]")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor,
                return_attention: bool = False,
                slot_off: Optional[int] = None) -> Tuple:
        """
        Args:
            x: [B, T] continuous spacing values (window)
            return_attention: return attention weights for analysis
            slot_off: if not None, zero out this memory slot (for ablation)

        Returns:
            pi, mu, sigma: [B, 1, K] MDN parameters for next spacing
            attentions: list of attention weights if return_attention=True
        """
        B, T = x.size()
        device = x.device
        M = self.n_memory_slots

        # Get memory tokens
        memory = self.memory_bank(B, device)  # [B, M, D]

        # Ablation: zero out specific slot
        if slot_off is not None:
            memory = memory.clone()
            memory[:, slot_off, :] = 0.0

        # Project input values
        x = x.unsqueeze(-1)  # [B, T, 1]
        x = self.input_proj(x)  # [B, T, D]

        # POSTFIX: [data tokens..., memory tokens...]
        # Memory is AFTER data, so memory CAN attend to data (causal OK)
        h = torch.cat([x, memory], dim=1)  # [B, T+M, D]

        # Positional embeddings
        pos = torch.arange(0, T + M, dtype=torch.long, device=device)
        h = h + self.wpe(pos)
        h = self.drop(h)

        # Transformer blocks
        attentions = []
        for block in self.blocks:
            if return_attention:
                h, attn = block(h, return_attention=True)
                attentions.append(attn)
            else:
                h = block(h)

        h = self.ln_f(h)

        # Extract memory hidden states (LAST M positions = memory)
        memory_hidden = h[:, -M:, :]  # [B, M, D]

        # Pool memory outputs (bottleneck!)
        readout = self.memory_bank.pool_memory_outputs(memory_hidden)  # [B, D]

        # MDN head predicts ONE next spacing
        readout = readout.unsqueeze(1)  # [B, 1, D]
        pi, mu, sigma = self.mdn_head(readout)  # each [B, 1, K]

        if return_attention:
            return pi, mu, sigma, attentions
        return pi, mu, sigma

    def get_memory_param_groups(self, base_lr: float, memory_lr_mult: float = 0.3):
        """Return parameter groups with different learning rates."""
        memory_params = list(self.memory_bank.parameters())
        other_params = [p for n, p in self.named_parameters()
                       if 'memory_bank' not in n]

        return [
            {'params': other_params, 'lr': base_lr},
            {'params': memory_params, 'lr': base_lr * memory_lr_mult},
        ]


# ============================================================================
# DATASET: ONE-STEP PREDICTION
# ============================================================================

class SpacingNextDataset(Dataset):
    """
    Dataset for next-spacing prediction.

    Each item:
        x: [T] spacing window (float)
        y: scalar = next spacing after window
    """

    def __init__(self, spacings: torch.Tensor, seq_len: int = 257):
        super().__init__()
        # Handle both [N, seq_len] and [N] formats
        if spacings.dim() == 2:
            # Pre-chunked data: flatten to 1D
            spacings = spacings.flatten()

        assert spacings.dim() == 1, f"Expected 1D tensor, got {spacings.dim()}D"

        self.spacings = spacings.float()
        self.seq_len = seq_len
        self.T = seq_len - 1  # input length (target is last element)
        self.n_samples = self.spacings.numel() - seq_len

        if self.n_samples <= 0:
            raise ValueError(f"Not enough data for seq_len={seq_len}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.spacings[idx : idx + self.seq_len]  # [seq_len]
        x = seq[:-1].contiguous()  # [T] = input window
        y = seq[-1].contiguous()   # scalar = target
        return x, y


# ============================================================================
# MDN LOSS FOR SINGLE-STEP
# ============================================================================

def mdn_loss_1step(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                   target: torch.Tensor, entropy_reg: float = 0.005,
                   eps: float = 1e-8) -> torch.Tensor:
    """
    MDN negative log-likelihood for single-step prediction.

    Args:
        pi, mu, sigma: [B, 1, K] MDN parameters
        target: [B] scalar targets
        entropy_reg: entropy regularization weight

    Returns:
        loss: scalar
    """
    # Squeeze sequence dim
    pi = pi.squeeze(1)      # [B, K]
    mu = mu.squeeze(1)      # [B, K]
    sigma = sigma.squeeze(1)  # [B, K]

    sigma = torch.clamp(sigma, min=eps)
    target = target.unsqueeze(-1)  # [B, 1]

    # Gaussian log-likelihood per component
    log_prob = -0.5 * (
        ((target - mu) / sigma) ** 2
        + 2.0 * torch.log(sigma)
        + math.log(2.0 * math.pi)
    )  # [B, K]

    # Mixture log-likelihood
    log_pi = torch.log(torch.clamp(pi, min=eps))
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # [B]

    nll = -log_mix.mean()

    # Entropy regularization (prevent mode collapse)
    if entropy_reg > 0:
        entropy = -(pi * torch.log(torch.clamp(pi, min=eps))).sum(dim=-1).mean()
        loss = nll - entropy_reg * entropy
    else:
        loss = nll

    return loss


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    console.print(f"[bold magenta]POSTFIX Memory Training (E3)[/]")
    console.print(f"[dim]Memory sees data, data doesn't see memory[/]\n")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    console.print(f"[cyan]Device: {device}[/]")

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(args.data_dir)
    train_data = torch.load(data_dir / 'train.pt', weights_only=False)
    val_data = torch.load(data_dir / 'val.pt', weights_only=False)

    # Handle different data formats
    if isinstance(train_data, dict):
        train_data = train_data.get('data', train_data.get('spacings'))
    if isinstance(val_data, dict):
        val_data = val_data.get('data', val_data.get('spacings'))

    console.print(f"[green]Train data: {train_data.shape}[/]")
    console.print(f"[green]Val data: {val_data.shape}[/]")

    # Create datasets (seq_len=257: 256 input + 1 target)
    train_dataset = SpacingNextDataset(train_data, seq_len=args.seq_len)
    val_dataset = SpacingNextDataset(val_data, seq_len=args.seq_len)

    console.print(f"[green]Train samples: {len(train_dataset):,}[/]")
    console.print(f"[green]Val samples: {len(val_dataset):,}[/]")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model config
    config = MDNConfig(
        seq_len=args.seq_len - 1 + args.n_memory_slots,  # T + M for mask size
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_components=args.n_components,
        dropout=args.dropout,
        bias=False
    )

    # Create model
    model = SpacingMDNPostfix(
        config,
        n_memory_slots=args.n_memory_slots,
        use_slot_id=args.use_slot_id
    ).to(device)

    # Optimizer with separate LR for memory
    if args.memory_lr_mult < 1.0:
        param_groups = model.get_memory_param_groups(args.lr, args.memory_lr_mult)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1
    )

    # AMP scaler for mixed precision
    scaler = GradScaler('cuda') if args.use_amp and device.type == 'cuda' else None

    # Training loop
    step = 0
    best_val_nll = float('inf')
    train_iter = iter(train_loader)

    # Log file
    log_file = open(out_dir / 'train.log', 'w')

    console.print(f"\n[bold]Starting training for {args.max_steps} steps...[/]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Training POSTFIX...", total=args.max_steps)

        start_time = time.time()

        while step < args.max_steps:
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with optional AMP
            if scaler is not None:
                with autocast('cuda'):
                    pi, mu, sigma = model(x)
                    loss = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=args.entropy_reg)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                pi, mu, sigma = model(x)
                loss = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=args.entropy_reg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            step += 1
            progress.update(task, advance=1)

            # Validation
            if step % args.eval_every == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        vpi, vmu, vsigma = model(vx)
                        vloss = mdn_loss_1step(vpi, vmu, vsigma, vy, entropy_reg=0)
                        val_losses.append(vloss.item())

                val_nll = np.mean(val_losses)

                # Save checkpoint
                ckpt = {
                    'model': model.state_dict(),
                    'config': config,
                    'n_memory_slots': args.n_memory_slots,
                    'use_slot_id': args.use_slot_id,
                    'step': step,
                    'val_nll': val_nll,
                    'architecture': 'POSTFIX'
                }

                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    torch.save(ckpt, out_dir / 'best.pt')

                # Log
                elapsed = time.time() - start_time
                msg = f"Step {step}: val_nll={val_nll:.4f} (best={best_val_nll:.4f}) elapsed={elapsed/60:.1f}m"
                console.print(f"[dim]{msg}[/]")
                log_file.write(msg + '\n')
                log_file.flush()

            # Save periodic checkpoints
            if step % args.save_every == 0:
                ckpt = {
                    'model': model.state_dict(),
                    'config': config,
                    'n_memory_slots': args.n_memory_slots,
                    'use_slot_id': args.use_slot_id,
                    'step': step,
                    'val_nll': val_nll if 'val_nll' in dir() else None,
                    'architecture': 'POSTFIX'
                }
                torch.save(ckpt, out_dir / f'ckpt_{step}.pt')

    # Final save
    total_time = time.time() - start_time
    total_steps = step
    avg_steps_per_sec = total_steps / total_time if total_time > 0 else 0
    samples_per_sec = avg_steps_per_sec * args.batch_size

    ckpt = {
        'model': model.state_dict(),
        'config': config,
        'n_memory_slots': args.n_memory_slots,
        'use_slot_id': args.use_slot_id,
        'step': step,
        'val_nll': best_val_nll,
        'architecture': 'POSTFIX'
    }
    torch.save(ckpt, out_dir / 'final.pt')

    # ═══════════════════════════════════════════════════════════════════
    # TRAINING COMPLETE - DETAILED SUMMARY
    # ═══════════════════════════════════════════════════════════════════

    console.print(f"\n[green]═══ Training Complete ═══[/]")
    console.print(f"Best val NLL: {best_val_nll:.4f}")

    # Timing Summary
    console.print(f"\n[bold cyan]⏱ Timing Summary:[/]")
    console.print(f"  Total time: {total_time/60:.1f} min ({total_time:.1f} sec)")
    console.print(f"  Steps/sec: {avg_steps_per_sec:.2f}")
    console.print(f"  Samples/sec: {samples_per_sec:.0f}")
    console.print(f"  Time per step: {1000*total_time/total_steps:.2f} ms")

    # Memory Readout Weights
    console.print(f"\n[bold cyan]═══ Memory Readout Weights ═══[/]")
    readout_w = model.memory_bank.get_readout_weights().detach().cpu().numpy()
    max_w = readout_w.max()
    for i, w in enumerate(readout_w):
        bar_len = int(w / max_w * 20) if max_w > 0 else 0
        bar = "█" * bar_len
        console.print(f"  Slot {i}: {w:.4f} {bar}")

    # Memory Slot Norms
    console.print(f"\n[bold cyan]═══ Memory Slot Norms ═══[/]")
    mem = model.memory_bank.memory.detach().cpu()
    if model.memory_bank.use_slot_id:
        slot_ids = torch.arange(args.n_memory_slots)
        mem = mem + model.memory_bank.slot_id(slot_ids).detach().cpu()
    norms = mem.norm(dim=1).numpy()
    max_norm = norms.max()
    for i, n in enumerate(norms):
        bar_len = int(n / max_norm * 20) if max_norm > 0 else 0
        bar = "█" * bar_len
        console.print(f"  Slot {i}: {n:.4f} {bar}")

    # Save memory diagnostics
    diagnostics = {
        'readout_weights': readout_w,
        'slot_norms': norms,
        'total_time': total_time,
        'steps_per_sec': avg_steps_per_sec,
        'samples_per_sec': samples_per_sec,
        'best_val_nll': best_val_nll,
        'total_steps': total_steps,
    }
    torch.save(diagnostics, out_dir / 'memory_diagnostics.pt')
    console.print(f"\nSaved memory_diagnostics.pt")

    # Write to log
    log_file.write(f"\n═══ Training Complete ═══\n")
    log_file.write(f"Best val NLL: {best_val_nll:.4f}\n")
    log_file.write(f"\n⏱ Timing Summary:\n")
    log_file.write(f"  Total time: {total_time/60:.1f} min ({total_time:.1f} sec)\n")
    log_file.write(f"  Steps/sec: {avg_steps_per_sec:.2f}\n")
    log_file.write(f"  Samples/sec: {samples_per_sec:.0f}\n")
    log_file.write(f"  Time per step: {1000*total_time/total_steps:.2f} ms\n")
    log_file.close()

    console.print(f"\n[green]Saved to: {out_dir}[/]")


def main():
    parser = argparse.ArgumentParser(description='Train SpacingMDN with POSTFIX memory')

    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory with train.pt and val.pt')
    parser.add_argument('--out-dir', type=str, required=True,
                       help='Output directory for checkpoints')

    # Model
    parser.add_argument('--n-layer', type=int, default=6)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=256)
    parser.add_argument('--n-components', type=int, default=8, help='MDN mixture components')
    parser.add_argument('--dropout', type=float, default=0.1)

    # Memory
    parser.add_argument('--n-memory-slots', type=int, default=8)
    parser.add_argument('--use-slot-id', action='store_true', default=True)
    parser.add_argument('--memory-lr-mult', type=float, default=0.3,
                       help='Memory LR multiplier (lower for stability)')

    # Training
    parser.add_argument('--seq-len', type=int, default=257,
                       help='Sequence length (256 input + 1 target)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--entropy-reg', type=float, default=0.005)
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--save-every', type=int, default=5000)

    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use-amp', action='store_true', default=True)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1337)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
