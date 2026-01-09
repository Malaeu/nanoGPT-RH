#!/usr/bin/env python3
"""
train_mdn_postfix.py — POSTFIX Memory + Memory-Only Readout (Bottleneck)

E3 → E4 UPGRADE:
- slot_id_mode: fixed | off | permute_per_batch (ID-detox)
- memory_content_mode: normal | zeroed (sanity tests)
- aux_loss: Q3-proxy supervision on memory hidden states
- early_stop: patience-based stopping

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
  # E3 mode (backward compatible)
  python train_mdn_postfix.py --data-dir data/continuous_2M --out-dir out/mdn_postfix_E3

  # E4 mode (ID-detox + aux-loss)
  python train_mdn_postfix.py --data-dir data/continuous_2M --out-dir out/mdn_postfix_E4 \\
      --slot-id-mode permute_per_batch --use-aux-loss --early-stop --patience 800
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

# Streaming data loading (GPU-direct, mmap, or legacy DataLoader)
from data_loading import load_data


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

    E4 additions:
    - slot_id_mode: 'fixed' (default), 'off' (no IDs), 'permute_per_batch' (shuffle IDs)
    - content_mode: 'normal' (default), 'zeroed' (content=0, ID-only test)
    """

    def __init__(self, n_slots: int, n_embd: int,
                 slot_id_mode: str = 'fixed',
                 content_mode: str = 'normal'):
        super().__init__()
        self.n_slots = n_slots
        self.n_embd = n_embd
        self.slot_id_mode = slot_id_mode
        self.content_mode = content_mode

        # E5.5: Separate eval mode for deterministic validation
        # FIX: Default to None = use same mode as training (no distribution mismatch!)
        self.eval_slot_id_mode = None  # None = use self.slot_id_mode
        self._eval_perm_seed = None  # For reproducible multi-perm eval

        # Learnable memory content
        self.memory = nn.Parameter(torch.randn(n_slots, n_embd) * 0.02)

        # Slot-ID embeddings (always create, but may not use)
        self.slot_id = nn.Embedding(n_slots, n_embd)
        nn.init.orthogonal_(self.slot_id.weight)

        # Learnable readout weights for pooling memory outputs
        self.readout_weights = nn.Parameter(torch.zeros(n_slots))

    def set_eval_perm_seed(self, seed: Optional[int]):
        """Set seed for reproducible permutation during eval."""
        self._eval_perm_seed = seed

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return memory tokens expanded for batch: [B, M, D]"""
        # Content: normal or zeroed
        if self.content_mode == 'zeroed':
            memory = torch.zeros(self.n_slots, self.n_embd, device=device)
        else:
            memory = self.memory  # [M, D]

        # E5.5: Use eval_slot_id_mode when not training (if explicitly set)
        effective_mode = self.slot_id_mode
        if not self.training and self.eval_slot_id_mode is not None:
            effective_mode = self.eval_slot_id_mode

        # Slot-ID: fixed, off, or permute_per_batch
        if effective_mode == 'fixed':
            slot_ids = torch.arange(self.n_slots, device=device)
            memory = memory + self.slot_id(slot_ids)
        elif effective_mode == 'permute_per_batch':
            # Use seeded RNG for reproducible permutations during eval
            if not self.training and self._eval_perm_seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self._eval_perm_seed)
                perm = torch.randperm(self.n_slots, device=device, generator=g)
            else:
                perm = torch.randperm(self.n_slots, device=device)
            memory = memory + self.slot_id(perm)
        # else: effective_mode == 'off' → no ID added

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

    E4 additions:
    - slot_id_mode: controls slot-ID embeddings
    - content_mode: controls memory content (for sanity tests)
    - aux_head: optional Q3-proxy supervision
    """

    def __init__(self, config: MDNConfig, n_memory_slots: int = 8,
                 slot_id_mode: str = 'fixed',
                 content_mode: str = 'normal',
                 use_aux_loss: bool = False):
        super().__init__()
        self.config = config
        self.n_memory_slots = n_memory_slots
        self.slot_id_mode = slot_id_mode
        self.content_mode = content_mode
        self.use_aux_loss = use_aux_loss

        # Memory bank (POSTFIX) with E4 modes
        self.memory_bank = MemoryBankPostfix(
            n_memory_slots, config.n_embd,
            slot_id_mode=slot_id_mode,
            content_mode=content_mode
        )

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

        # Aux head for Q3-proxy supervision (E4)
        # Predicts 8 statistics (M0..M7) from each memory slot
        if use_aux_loss:
            self.aux_head = nn.Linear(config.n_embd, 8)  # [M, D] → [M, 8]
            console.print(f"[yellow]  Aux loss: Q3-proxy supervision enabled[/]")
            # Note: aux_mode (mean/diagonal) is passed at training time, not stored in model
        else:
            self.aux_head = None

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        console.print(f"[green]SpacingMDN+POSTFIX: {n_params/1e6:.2f}M parameters[/]")
        console.print(f"[dim]  Memory slots: {n_memory_slots} (POSTFIX)[/]")
        console.print(f"[dim]  Readout: memory-only bottleneck[/]")
        console.print(f"[dim]  slot_id_mode: {slot_id_mode}[/]")
        console.print(f"[dim]  content_mode: {content_mode}[/]")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor,
                return_attention: bool = False,
                return_aux: bool = False,
                slot_off: Optional[int] = None) -> Tuple:
        """
        Args:
            x: [B, T] continuous spacing values (window)
            return_attention: return attention weights for analysis
            return_aux: return aux predictions from memory hidden states
            slot_off: if not None, zero out this memory slot (for ablation)

        Returns:
            pi, mu, sigma: [B, 1, K] MDN parameters for next spacing
            attentions: list of attention weights if return_attention=True
            aux_preds: [B, M, 8] aux predictions if return_aux=True
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

        # Aux predictions from memory hidden states
        aux_preds = None
        if return_aux and self.aux_head is not None:
            aux_preds = self.aux_head(memory_hidden)  # [B, M, 8]

        # Return based on what's requested
        if return_attention and return_aux:
            return pi, mu, sigma, attentions, aux_preds
        elif return_attention:
            return pi, mu, sigma, attentions
        elif return_aux:
            return pi, mu, sigma, aux_preds
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
# E5.5: MULTI-PERMUTATION EVAL
# ============================================================================

def eval_with_perm_average(model, val_batcher, device, n_perms: int = 1):
    """
    Evaluate model with optional multi-permutation averaging.

    Args:
        model: SpacingMDNPostfix model
        val_batcher: validation data batcher (iterable over (x, y) batches)
        device: torch device
        n_perms: number of permutations to average (1=fixed, >1=multi-perm)

    Returns:
        val_nll: mean NLL
        val_std: std of NLL across permutations (0 if n_perms=1)
    """
    model.eval()

    if n_perms == 1:
        # Single fixed eval (default, deterministic)
        # eval_slot_id_mode is already 'fixed' by default
        val_losses = []
        with torch.no_grad():
            for vx, vy in val_batcher:
                # Batcher already returns data on device (for GPU-direct)
                # For DataLoaderWrapper, it handles .to(device) internally
                vpi, vmu, vsigma = model(vx)
                vloss = mdn_loss_1step(vpi, vmu, vsigma, vy, entropy_reg=0)
                val_losses.append(vloss.item())
        return np.mean(val_losses), 0.0

    # Multi-perm eval: temporarily switch to permute mode with seeds
    original_mode = model.memory_bank.eval_slot_id_mode
    model.memory_bank.eval_slot_id_mode = 'permute_per_batch'
    all_nlls = []

    with torch.no_grad():
        for i in range(n_perms):
            model.memory_bank.set_eval_perm_seed(42 + i)  # Reproducible seed per run
            perm_nlls = []
            for vx, vy in val_batcher:
                vpi, vmu, vsigma = model(vx)
                vloss = mdn_loss_1step(vpi, vmu, vsigma, vy, entropy_reg=0)
                perm_nlls.append(vloss.item())
            all_nlls.append(np.mean(perm_nlls))

    # Reset to original mode
    model.memory_bank.eval_slot_id_mode = original_mode
    model.memory_bank.set_eval_perm_seed(None)

    return np.mean(all_nlls), np.std(all_nlls)


# ============================================================================
# Q3-PROXY TARGETS (E4)
# ============================================================================

def compute_q3_targets(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Q3-proxy statistics for aux supervision.

    Args:
        x: [B, T] spacing window

    Returns:
        targets: [B, 8] z-score normalized statistics (M0..M7)
    """
    B, T = x.size()
    device = x.device

    # M0: mean(x) - 1.0 (T0 normalization, should be ~0)
    m0 = x.mean(dim=1) - 1.0

    # M1: hist_entropy (A1' coverage) - entropy of histogram
    # Approximate with variance-based entropy
    std = x.std(dim=1) + 1e-8
    m1 = torch.log(std * np.sqrt(2 * np.pi * np.e))

    # M2: max|dx| (A2 Lipschitz bound)
    dx = x[:, 1:] - x[:, :-1]
    m2 = dx.abs().max(dim=1).values

    # M3: quantile 0.01 (A3 floor)
    m3 = torch.quantile(x, 0.01, dim=1)

    # M4: mean|d²x| (smoothness / curvature)
    d2x = dx[:, 1:] - dx[:, :-1]
    m4 = d2x.abs().mean(dim=1)

    # M5: half_window_divergence (Toeplitz symmetry)
    half = T // 2
    left_mean = x[:, :half].mean(dim=1)
    right_mean = x[:, half:].mean(dim=1)
    m5 = (left_mean - right_mean).abs()

    # M6: high_freq_energy (RKHS cap) - energy in high-freq components
    # Use second derivative as proxy for high-frequency
    m6 = (d2x ** 2).mean(dim=1)

    # M7: local_rigidity (Δ3-proxy) - variance of local variance
    # Window of 16 positions
    window = 16
    n_windows = T // window
    if n_windows > 1:
        x_reshaped = x[:, :n_windows * window].view(B, n_windows, window)
        local_vars = x_reshaped.var(dim=2)  # [B, n_windows]
        m7 = local_vars.var(dim=1)  # variance of local variances
    else:
        m7 = torch.zeros(B, device=device)

    # Stack all targets
    targets = torch.stack([m0, m1, m2, m3, m4, m5, m6, m7], dim=1)  # [B, 8]

    # Z-score normalize (per batch for stability)
    targets_mean = targets.mean(dim=0, keepdim=True)
    targets_std = targets.std(dim=0, keepdim=True) + 1e-8
    targets = (targets - targets_mean) / targets_std

    return targets


def aux_loss_q3(aux_preds: torch.Tensor, x: torch.Tensor, n_memory_slots: int = 8,
                aux_mode: str = 'mean') -> torch.Tensor:
    """
    Compute aux loss: MSE between slot predictions and Q3-proxy targets.

    Two modes:
    - 'mean' (E4): Each slot predicts ALL 8 targets, then average.
                   More gradient signal but encourages slot interchangeability.
    - 'diagonal' (E5-PR1): slot[i] predicts ONLY proxy[i].
                           Forces specialization: each slot owns one target.

    Args:
        aux_preds: [B, M, 8] predictions from memory hidden states
        x: [B, T] input spacing window
        n_memory_slots: number of memory slots (M)
        aux_mode: 'mean' or 'diagonal'

    Returns:
        loss: scalar MSE loss
    """
    targets = compute_q3_targets(x)  # [B, 8]

    if aux_mode == 'diagonal':
        # E5-PR1: slot[i] predicts proxy[i] only → forces specialization
        # aux_preds: [B, M, 8], we want aux_preds[:, i, i] for each i
        diag_preds = torch.stack([aux_preds[:, i, i] for i in range(n_memory_slots)], dim=1)  # [B, M]
        loss = F.mse_loss(diag_preds, targets)
    else:
        # E4 MODE (mean): Each slot should predict the same targets
        # Average predictions across slots
        aux_preds_mean = aux_preds.mean(dim=1)  # [B, 8]
        loss = F.mse_loss(aux_preds_mean, targets)

    return loss


def get_aux_weight(step: int) -> float:
    """
    Ramp schedule for aux weight.

    Schedule:
        step 0-500:     0 → 1e-3 (warmup)
        step 500-3000:  1e-3 → 1e-2 (ramp up)
        step 3000+:     1e-2 (hold)
    """
    if step < 500:
        # Linear warmup 0 → 1e-3
        return 1e-3 * (step / 500.0)
    elif step < 3000:
        # Linear ramp 1e-3 → 1e-2
        progress = (step - 500) / (3000 - 500)
        return 1e-3 + progress * (1e-2 - 1e-3)
    else:
        return 1e-2


# ============================================================================
# ORTHOGONALITY LOSS (E5 STEP 2)
# ============================================================================

def compute_ortho_loss(memory_bank: nn.Module) -> torch.Tensor:
    """
    Compute orthogonality loss on memory slot embeddings.

    Forces slots to be "different directions" in embedding space,
    breaking the symmetry/interchangeability that causes low Ablation Δ.

    Loss = ||G - I||_F^2 where G = normalized Gram matrix

    Args:
        memory_bank: MemoryBankPostfix module

    Returns:
        ortho_loss: scalar penalty
    """
    # Get memory content [M, D]
    mem = memory_bank.memory  # [M, D]

    # Normalize to unit vectors
    mem_norm = F.normalize(mem, p=2, dim=1)  # [M, D]

    # Gram matrix: cosine similarity between all pairs
    gram = mem_norm @ mem_norm.T  # [M, M]

    # Target: identity matrix (each slot orthogonal to others)
    M = gram.size(0)
    identity = torch.eye(M, device=gram.device)

    # Frobenius norm of (G - I)
    ortho_loss = ((gram - identity) ** 2).sum()

    return ortho_loss


def get_ortho_weight(step: int) -> float:
    """
    Ramp schedule for orthogonality loss weight.

    E5 STEP 2 Schedule:
        step 0-500:      0 → 1e-3 (warmup)
        step 500-5000:   1e-3 → 1e-2 (ramp up)
        step 5000+:      1e-2 (hold)
    """
    if step < 500:
        # Linear warmup 0 → 1e-3
        return 1e-3 * (step / 500.0)
    elif step < 5000:
        # Linear ramp 1e-3 → 1e-2
        progress = (step - 500) / (5000 - 500)
        return 1e-3 + progress * (1e-2 - 1e-3)
    else:
        return 1e-2


# ============================================================================
# ROLLOUT-SCORE (MASTER_SPEC §8.3)
# ============================================================================

def compute_quick_rollout_score(model, data, device, n_samples=200, batch_size=32):
    """
    Compute rollout-score for checkpoint selection.

    rollout_score = CRPS_1 + 0.5 * slope
    Lower is better.

    This is a "quick" version for use during training - uses fewer samples
    and shorter horizons than full eval.
    """
    model.eval()

    # Sample subset
    idx = torch.randint(0, len(data), (min(n_samples, len(data)),))
    subset = data[idx].to(device)

    with torch.no_grad():
        # Compute one-step CRPS approximation
        x = subset[:, :-1]
        y = subset[:, 1:]

        result = model(x)
        if isinstance(result, tuple):
            pi, mu, sigma = result[:3]
        else:
            pi, mu, sigma = result['pi'], result['mu'], result['sigma']

        # Simple CRPS approximation via MAE to mean prediction
        pred_mean = (pi * mu).sum(dim=-1)
        crps_approx = torch.abs(pred_mean - y).mean().item()

        # Quick rollout at h=50 and h=100
        errors = {}
        for h in [50, 100]:
            rollout_errors = []
            for i in range(0, len(subset), batch_size):
                batch = subset[i:i+batch_size]
                if batch.shape[1] < 128 + h:
                    continue

                context = batch[:, :128].clone()
                for step in range(h):
                    res = model(context)
                    if isinstance(res, tuple):
                        pi_t, mu_t, sigma_t = res[:3]
                    else:
                        pi_t, mu_t, sigma_t = res['pi'], res['mu'], res['sigma']
                    pred = (pi_t[:, -1, :] * mu_t[:, -1, :]).sum(dim=-1, keepdim=True)
                    context = torch.cat([context[:, 1:], pred], dim=1)

                true_val = batch[:, 128 + h - 1]
                pred_val = pred.squeeze(1)
                mae = torch.abs(pred_val - true_val)
                rollout_errors.append(mae.mean().item())

            if rollout_errors:
                errors[h] = np.mean(rollout_errors)

        # Compute slope
        if 50 in errors and 100 in errors:
            slope = (errors[100] - errors[50]) / 50
        else:
            slope = 1.0  # Penalty for failed rollout

        rollout_score = crps_approx + 0.5 * max(0, slope)

    model.train()
    return rollout_score, crps_approx, slope


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    # Determine experiment version
    is_e5 = args.use_ortho_loss  # E5 STEP 2: orthogonality loss
    is_e4 = args.slot_id_mode != 'fixed' or args.use_aux_loss or args.early_stop
    if is_e5:
        exp_name = "E5"
    elif is_e4:
        exp_name = "E4"
    else:
        exp_name = "E3"

    console.print(f"[bold magenta]POSTFIX Memory Training ({exp_name})[/]")
    console.print(f"[dim]Memory sees data, data doesn't see memory[/]")
    if is_e5:
        console.print(f"[yellow]E5 mode: slot_id_mode={args.slot_id_mode}, aux_loss={args.use_aux_loss}, aux_mode={args.aux_mode}, ortho_loss={args.use_ortho_loss}, early_stop={args.early_stop}[/]\n")
    elif is_e4:
        console.print(f"[yellow]E4 mode: slot_id_mode={args.slot_id_mode}, aux_loss={args.use_aux_loss}, aux_mode={args.aux_mode}, early_stop={args.early_stop}[/]\n")
    else:
        console.print("")

    # Device selection: cuda > mps > cpu
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device not in ['cuda', 'mps', 'cpu']:
        device = torch.device(args.device)  # custom device
    else:
        device = torch.device('cpu')
    console.print(f"[cyan]Device: {device}[/]")

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data with streaming strategy (GPU-direct, mmap, or legacy DataLoader)
    train_batcher, val_batcher, data_info = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=device,
        mode=args.data_mode,
        seq_len=args.seq_len,
        train_fraction=args.train_fraction,
        num_workers=args.num_workers
    )

    # Display data info
    console.print(f"[green]Data mode: {data_info['mode']}[/]")
    console.print(f"[green]Train samples: {data_info['train_samples']:,}[/]")
    console.print(f"[green]Val samples: {data_info['val_samples']:,}[/]")
    if 'gpu_memory_bytes' in data_info:
        gpu_mb = data_info['gpu_memory_bytes'] / 1e6
        console.print(f"[green]GPU data memory: {gpu_mb:.1f}MB[/]")
    if data_info.get('train_fraction_applied'):
        console.print(f"[yellow]Grokking mode: using {args.train_fraction*100:.0f}% of train data[/]")

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

    # Create model with E4 modes
    model = SpacingMDNPostfix(
        config,
        n_memory_slots=args.n_memory_slots,
        slot_id_mode=args.slot_id_mode,
        content_mode=args.content_mode,
        use_aux_loss=args.use_aux_loss
    ).to(device)

    # torch.compile() for 20-30% speedup (Ampere+ GPUs)
    if args.use_compile and device.type == 'cuda':
        try:
            # Check if GPU supports compilation (SM >= 8.0 for best results)
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                model = torch.compile(model, mode="reduce-overhead")
                console.print(f"[green]torch.compile() enabled (SM {capability[0]}.{capability[1]})[/]")
            else:
                console.print(f"[yellow]torch.compile() skipped: SM {capability[0]}.{capability[1]} < 8.0[/]")
        except Exception as e:
            console.print(f"[yellow]torch.compile() failed: {e}[/]")

    # Set eval_slot_id_mode if explicitly provided (else stays None = same as training)
    if args.eval_slot_id_mode is not None:
        model.memory_bank.eval_slot_id_mode = args.eval_slot_id_mode

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
    best_rollout_score = float('inf')  # For --select-by-rollout

    # Early stopping (E5: patience in STEPS, not eval iterations)
    best_step = 0
    early_stopped = False

    # Log file
    log_file = open(out_dir / 'train.log', 'w')

    max_steps_str = f"{args.max_steps}" if not args.early_stop else f"{args.max_steps} (early stop patience={args.patience})"
    console.print(f"\n[bold]Starting training for {max_steps_str} steps...[/]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Training POSTFIX...", total=args.max_steps)

        start_time = time.time()

        # E5.5: Separate timing for train vs eval
        train_time_total = 0.0
        eval_time_total = 0.0
        last_eval_step = 0
        last_eval_time = start_time

        # Grokking: track train_nll for curve analysis
        train_nll_sum = 0.0
        train_nll_count = 0

        while step < args.max_steps:
            # Get batch (streaming: no iterator, direct GPU sampling)
            x, y = train_batcher.get_batch()

            # E5.5: Time the training step
            train_step_start = time.time()

            model.train()

            # E5.5: Perm warmup - use fixed IDs during warmup period
            if args.perm_warmup_steps > 0 and args.slot_id_mode == 'permute_per_batch':
                if step < args.perm_warmup_steps:
                    model.memory_bank.slot_id_mode = 'fixed'
                else:
                    model.memory_bank.slot_id_mode = 'permute_per_batch'

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with optional AMP
            if scaler is not None:
                with autocast('cuda'):
                    if args.use_aux_loss:
                        pi, mu, sigma, aux_preds = model(x, return_aux=True)
                        mdn_loss_val = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=args.entropy_reg)
                        aux_weight = get_aux_weight(step)
                        aux_loss_val = aux_loss_q3(aux_preds, x, args.n_memory_slots, args.aux_mode)
                        loss = mdn_loss_val + aux_weight * aux_loss_val
                    else:
                        pi, mu, sigma = model(x)
                        mdn_loss_val = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=args.entropy_reg)
                        loss = mdn_loss_val

                    # E5 STEP 2: Orthogonality loss on memory slots
                    if args.use_ortho_loss:
                        ortho_weight = get_ortho_weight(step)
                        ortho_loss_val = compute_ortho_loss(model.memory_bank)
                        loss = loss + ortho_weight * ortho_loss_val

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.use_aux_loss:
                    pi, mu, sigma, aux_preds = model(x, return_aux=True)
                    mdn_loss_val = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=args.entropy_reg)
                    aux_weight = get_aux_weight(step)
                    aux_loss_val = aux_loss_q3(aux_preds, x, args.n_memory_slots, args.aux_mode)
                    loss = mdn_loss_val + aux_weight * aux_loss_val
                else:
                    pi, mu, sigma = model(x)
                    mdn_loss_val = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=args.entropy_reg)
                    loss = mdn_loss_val

                # E5 STEP 2: Orthogonality loss on memory slots
                if args.use_ortho_loss:
                    ortho_weight = get_ortho_weight(step)
                    ortho_loss_val = compute_ortho_loss(model.memory_bank)
                    loss = loss + ortho_weight * ortho_loss_val

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()

            # Grokking: accumulate train_nll
            train_nll_sum += mdn_loss_val.item()
            train_nll_count += 1

            # E5.5: Accumulate train time
            train_time_total += time.time() - train_step_start

            step += 1
            progress.update(task, advance=1)

            # Validation
            if step % args.eval_every == 0:
                # E5.5: Time the eval
                eval_start = time.time()

                # E5.5: Use eval_with_perm_average for optional multi-perm eval
                val_nll, val_std = eval_with_perm_average(
                    model, val_batcher, device, n_perms=args.eval_perm_average
                )

                # E5.5: Accumulate eval time
                eval_time_total += time.time() - eval_start

                # Save checkpoint with E4/E5 fields
                ckpt = {
                    'model': model.state_dict(),
                    'config': config,
                    'n_memory_slots': args.n_memory_slots,
                    'slot_id_mode': args.slot_id_mode,
                    'content_mode': args.content_mode,
                    'use_aux_loss': args.use_aux_loss,
                    'aux_mode': args.aux_mode,  # E5-PR1
                    'use_ortho_loss': args.use_ortho_loss,  # E5
                    'step': step,
                    'val_nll': val_nll,
                    'architecture': 'POSTFIX',
                    'experiment': exp_name
                }

                # ═══ Checkpoint selection logic ═══
                # Default: select by val_nll
                # With --select-by-rollout: select by rollout_score (at rollout_eval_freq intervals)

                saved_by_nll = False
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    if not args.select_by_rollout:
                        best_step = step
                        torch.save(ckpt, out_dir / 'best.pt')
                        saved_by_nll = True

                # Rollout-score based selection (Task 4 - Law-grade)
                if args.select_by_rollout and step % args.rollout_eval_freq == 0:
                    console.print(f"[cyan]Computing rollout score (H=100,200)...[/]")
                    rollout_score = compute_quick_rollout_score(
                        model, val_tensor, device,
                        n_samples=200, batch_size=32
                    )

                    if rollout_score < best_rollout_score:
                        best_rollout_score = rollout_score
                        best_step = step
                        ckpt['rollout_score'] = rollout_score
                        torch.save(ckpt, out_dir / 'best.pt')
                        console.print(f"[green]✓ New best rollout_score: {rollout_score:.4f}[/]")
                    else:
                        console.print(f"[dim]rollout_score: {rollout_score:.4f} (best={best_rollout_score:.4f})[/]")

                # Log with E5.5 separate timing
                elapsed = time.time() - start_time
                steps_since_best = step - best_step

                # Grokking: compute average train_nll since last eval
                avg_train_nll = train_nll_sum / train_nll_count if train_nll_count > 0 else 0.0

                # E5.5: Calculate train speed (excluding eval) and interval speed
                train_speed = step / train_time_total if train_time_total > 0 else 0
                interval_steps = step - last_eval_step
                interval_time = time.time() - last_eval_time
                interval_speed = interval_steps / interval_time if interval_time > 0 else 0

                msg = f"Step {step}: train_nll={avg_train_nll:.4f} | val_nll={val_nll:.4f} (best={best_val_nll:.4f}) | train:{train_speed:.1f} s/s | int:{interval_speed:.1f} s/s | elapsed={elapsed/60:.1f}m"
                if args.select_by_rollout:
                    msg += f" | rollout_best={best_rollout_score:.4f}"
                if args.early_stop:
                    msg += f" | stale={steps_since_best}/{args.patience}"
                console.print(f"[dim]{msg}[/]")
                log_file.write(msg + '\n')
                log_file.flush()

                # Grokking: reset train_nll accumulators
                train_nll_sum = 0.0
                train_nll_count = 0

                # E5.5: Update interval tracking
                last_eval_step = step
                last_eval_time = time.time()

                # Early stopping check (E5: patience in STEPS)
                if args.early_stop and steps_since_best >= args.patience:
                    console.print(f"[yellow]Early stopping triggered at step {step}[/]")
                    log_file.write(f"Early stopping triggered at step {step}\n")
                    early_stopped = True
                    break

            # Save periodic checkpoints
            if step % args.save_every == 0:
                ckpt = {
                    'model': model.state_dict(),
                    'config': config,
                    'n_memory_slots': args.n_memory_slots,
                    'slot_id_mode': args.slot_id_mode,
                    'content_mode': args.content_mode,
                    'use_aux_loss': args.use_aux_loss,
                    'aux_mode': args.aux_mode,  # E5-PR1
                    'use_ortho_loss': args.use_ortho_loss,  # E5
                    'step': step,
                    'val_nll': val_nll if 'val_nll' in dir() else None,
                    'architecture': 'POSTFIX',
                    'experiment': exp_name
                }
                torch.save(ckpt, out_dir / f'ckpt_{step}.pt')

            # Check early stopping (break after saving checkpoint)
            if early_stopped:
                break

    # Final save
    total_time = time.time() - start_time
    total_steps = step
    avg_steps_per_sec = total_steps / total_time if total_time > 0 else 0
    samples_per_sec = avg_steps_per_sec * args.batch_size

    ckpt = {
        'model': model.state_dict(),
        'config': config,
        'n_memory_slots': args.n_memory_slots,
        'slot_id_mode': args.slot_id_mode,
        'content_mode': args.content_mode,
        'use_aux_loss': args.use_aux_loss,
        'aux_mode': args.aux_mode,  # E5-PR1
        'use_ortho_loss': args.use_ortho_loss,  # E5
        'step': step,
        'val_nll': best_val_nll,
        'architecture': 'POSTFIX',
        'experiment': exp_name,
        'early_stopped': early_stopped
    }
    torch.save(ckpt, out_dir / 'final.pt')

    # ═══════════════════════════════════════════════════════════════════
    # TRAINING COMPLETE - DETAILED SUMMARY
    # ═══════════════════════════════════════════════════════════════════

    console.print(f"\n[green]═══ Training Complete ═══[/]")
    console.print(f"Best val NLL: {best_val_nll:.4f}")
    if args.select_by_rollout:
        console.print(f"Best rollout score: {best_rollout_score:.4f}")
        console.print(f"[dim](Checkpoint selected by rollout-score, not val_nll)[/]")

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
    if model.memory_bank.slot_id_mode == 'fixed':
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
    parser.add_argument('--train-fraction', type=float, default=1.0,
                       help='Fraction of train data to use (0.3 = 30%%, for grokking)')
    parser.add_argument('--data-mode', type=str, default='auto',
                       choices=['auto', 'gpu-direct', 'mmap', 'dataloader'],
                       help='Data loading strategy: auto (detect), gpu-direct (fastest), mmap (low RAM), dataloader (legacy)')

    # Model
    parser.add_argument('--n-layer', type=int, default=6)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=256)
    parser.add_argument('--n-components', type=int, default=8, help='MDN mixture components')
    parser.add_argument('--dropout', type=float, default=0.1)

    # Memory
    parser.add_argument('--n-memory-slots', type=int, default=8)
    parser.add_argument('--memory-lr-mult', type=float, default=0.3,
                       help='Memory LR multiplier (lower for stability)')

    # E4: Slot-ID and Content modes
    parser.add_argument('--slot-id-mode', type=str, default='fixed',
                       choices=['fixed', 'off', 'permute_per_batch'],
                       help='Slot-ID mode: fixed (E3), off, permute_per_batch (E4)')
    parser.add_argument('--content-mode', type=str, default='normal',
                       choices=['normal', 'zeroed'],
                       help='Memory content mode: normal, zeroed (ID-only test)')
    parser.add_argument('--eval-slot-id-mode', type=str, default=None,
                       choices=['fixed', 'off', 'permute_per_batch'],
                       help='Slot-ID mode during eval. Default=same as --slot-id-mode')

    # E4: Aux loss
    parser.add_argument('--use-aux-loss', action='store_true', default=False,
                       help='Enable Q3-proxy aux loss supervision')
    parser.add_argument('--aux-mode', type=str, default='mean',
                       choices=['mean', 'diagonal'],
                       help='Aux loss mode: mean (E4, all slots predict all), diagonal (E5-PR1, slot[i]→proxy[i])')

    # E4: Early stopping
    parser.add_argument('--early-stop', action='store_true', default=False,
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=800,
                       help='Early stopping patience (E5: in training STEPS)')

    # Rollout-score selection (MASTER_SPEC §8.3)
    parser.add_argument('--select-by-rollout', action='store_true', default=False,
                       help='Select best checkpoint by rollout-score instead of val_nll')
    parser.add_argument('--rollout-eval-freq', type=int, default=1000,
                       help='Evaluate rollout-score every N steps (expensive)')

    # E5: Orthogonality loss
    parser.add_argument('--use-ortho-loss', action='store_true', default=False,
                       help='Enable orthogonality loss on memory slots (E5 STEP 2)')

    # E5.5: Eval and warmup options
    parser.add_argument('--eval-perm-average', type=int, default=1,
                       help='Number of permutations to average during eval (1=fixed, >1=multi-perm)')
    parser.add_argument('--perm-warmup-steps', type=int, default=0,
                       help='Steps before enabling permute_per_batch (0=always on)')

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
    parser.add_argument('--use-compile', action='store_true', default=False,
                       help='Use torch.compile() for 20-30%% speedup (requires Ampere+ GPU)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1337)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
