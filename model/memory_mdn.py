#!/usr/bin/env python3
"""
Memory Bank MDN: Transformer + MDN with Q3 Memory Vectors.

Memory Map Q3 (8 slots):
  M0 SIGN    - polarity of correction (arch - prime)
  M1 NORM    - scale/units/coordinate calibration
  M2 TORUS   - translation invariance (only distances matter)
  M3 SYMBOL  - kernel/PSF shape (global fingerprint)
  M4 FLOOR   - uncertainty floor (no σ-collapse)
  M5 TOEPLITZ - discretization stability (seq_len robustness)
  M6 PRIME-CAP - limit global correction power
  M7 GOAL    - stability margin (rollout health)

Memory tokens are prepended to sequence and participate in attention,
but MDN output only uses sequence positions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from model.mdn import MDNHead, Block, MDNConfig


# Q3 Memory Vector names (for logging/debugging)
Q3_MEMORY_NAMES = [
    "M0_SIGN",      # Polarity: arch - prime
    "M1_NORM",      # Scale/normalization
    "M2_TORUS",     # Translation invariance
    "M3_SYMBOL",    # Kernel/PSF shape
    "M4_FLOOR",     # Uncertainty floor
    "M5_TOEPLITZ",  # Discretization stability
    "M6_PRIMECAP",  # Global correction limit
    "M7_GOAL",      # Stability margin
]


@dataclass
class MemoryMDNConfig(MDNConfig):
    """Configuration for Memory Bank MDN."""
    # Memory Bank settings
    n_memory: int = 8                    # Number of Q3 memory slots
    memory_dropout: float = 0.1          # Slot dropout probability
    memory_cap: float = 2.0              # Max norm per memory vector
    diversity_weight: float = 0.01       # Orthogonality regularization
    memory_lr_mult: float = 0.1          # LR multiplier for memory (slower learning)


class MemoryBank(nn.Module):
    """
    Q3 Memory Bank: learnable global tokens for invariants.

    Each slot stores one Q3 invariant as a learnable vector.
    Memory tokens are prepended to sequence for attention.
    """

    def __init__(self, n_memory: int, n_embd: int,
                 dropout: float = 0.1, cap: float = 2.0):
        super().__init__()
        self.n_memory = n_memory
        self.n_embd = n_embd
        self.cap = cap
        self.dropout_p = dropout

        # Learnable memory vectors (8 Q3 slots)
        self.memory = nn.Parameter(torch.randn(n_memory, n_embd) * 0.02)

        # Slot names for debugging
        self.slot_names = Q3_MEMORY_NAMES[:n_memory]

        # Dropout mask (applied per-batch during training)
        self.register_buffer('_dummy', torch.zeros(1))  # For device tracking

    def forward(self, x: torch.Tensor, apply_dropout: bool = True) -> torch.Tensor:
        """
        Prepend memory tokens to sequence.

        Args:
            x: (B, T, D) sequence embeddings
            apply_dropout: whether to apply slot dropout

        Returns:
            (B, K+T, D) with memory prepended
        """
        B, T, D = x.shape
        device = x.device

        # Get memory with norm cap
        memory = self._get_capped_memory()

        # Expand for batch: (K, D) -> (B, K, D)
        mem = memory.unsqueeze(0).expand(B, -1, -1)

        # Apply slot dropout during training
        if self.training and apply_dropout and self.dropout_p > 0:
            mask = torch.bernoulli(
                torch.ones(B, self.n_memory, 1, device=device) * (1 - self.dropout_p)
            )
            mem = mem * mask

        # Concatenate: [memory, sequence]
        return torch.cat([mem, x], dim=1)

    def _get_capped_memory(self) -> torch.Tensor:
        """Apply norm cap to memory vectors."""
        norms = self.memory.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.cap / norms, max=1.0)
        return self.memory * scale

    def diversity_loss(self) -> torch.Tensor:
        """
        Compute slot diversity loss (orthogonality regularization).

        Penalizes slots being too similar to each other.
        """
        memory = self._get_capped_memory()

        # Normalize for cosine similarity
        mem_norm = F.normalize(memory, dim=-1)

        # Gram matrix (cosine similarities)
        gram = mem_norm @ mem_norm.T  # (K, K)

        # Target: identity (each slot only similar to itself)
        target = torch.eye(self.n_memory, device=memory.device)

        # L2 loss on off-diagonal elements
        loss = ((gram - target) ** 2).mean()

        return loss

    def get_diagnostics(self) -> Dict[str, float]:
        """Get memory bank diagnostics for monitoring."""
        memory = self._get_capped_memory()

        # Per-slot norms
        norms = memory.norm(dim=-1)

        # Pairwise similarities
        mem_norm = F.normalize(memory, dim=-1)
        gram = mem_norm @ mem_norm.T
        off_diag = gram[~torch.eye(self.n_memory, dtype=bool, device=memory.device)]

        return {
            'mem_norm_mean': norms.mean().item(),
            'mem_norm_std': norms.std().item(),
            'mem_sim_mean': off_diag.mean().item(),
            'mem_sim_max': off_diag.max().item(),
        }


class CausalSelfAttentionWithMemory(nn.Module):
    """
    Causal self-attention that handles memory prefix.

    Memory tokens can attend to all memory tokens.
    Sequence tokens can attend to all memory + causal sequence.
    """

    def __init__(self, config: MemoryMDNConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_memory = config.n_memory
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K+T, D) with K memory tokens prepended

        Returns:
            (B, K+T, D) attended output
        """
        B, L, C = x.shape  # L = K + T
        K = self.n_memory
        T = L - K

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Build attention mask:
        # - Memory tokens (0:K) can see all memory tokens
        # - Sequence tokens (K:K+T) can see all memory + causal sequence
        mask = torch.ones(L, L, device=x.device, dtype=torch.bool)

        # Sequence part is causal
        seq_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        mask[K:, K:] = seq_mask

        # Apply mask
        att = att.masked_fill(~mask.view(1, 1, L, L), float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class BlockWithMemory(nn.Module):
    """Transformer block that handles memory prefix."""

    def __init__(self, config: MemoryMDNConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionWithMemory(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # MLP (same as base)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))

        # MLP
        h = self.ln_2(x)
        h = self.c_fc(h)
        h = self.gelu(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        x = x + h

        return x


class MemoryMDN(nn.Module):
    """
    Transformer + MDN with Q3 Memory Bank.

    Architecture:
        [Memory Bank] -> [M0...M7] (8 learnable tokens)
                              |
                              v
        [Spacings] -> [s1...sT] -> Concat -> [M0...M7, s1...sT]
                                                    |
                                                    v
                                          [Transformer Blocks]
                                                    |
                                                    v
                                          [MDN Head] (only on s1...sT)
    """

    def __init__(self, config: MemoryMDNConfig):
        super().__init__()
        self.config = config

        # Memory Bank (Q3 invariants)
        self.memory_bank = MemoryBank(
            n_memory=config.n_memory,
            n_embd=config.n_embd,
            dropout=config.memory_dropout,
            cap=config.memory_cap
        )

        # Input embedding
        self.input_proj = nn.Linear(1, config.n_embd)
        self.pos_emb = nn.Embedding(config.seq_len + config.n_memory, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks (memory-aware)
        self.blocks = nn.ModuleList([
            BlockWithMemory(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # MDN head
        self.mdn_head = MDNHead(
            config.n_embd,
            config.n_components,
            config.sigma_min,
            config.sigma_max
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        n_memory_params = sum(p.numel() for p in self.memory_bank.parameters())
        print(f"MemoryMDN: {n_params:,} parameters ({n_memory_params:,} in memory bank)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        spacings: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            spacings: (B, T) input spacings
            targets: (B, T) target spacings for loss

        Returns:
            Dictionary with pi, mu, sigma, loss, entropy, diversity_loss
        """
        B, T = spacings.shape
        K = self.config.n_memory
        device = spacings.device

        # Embed spacings
        x = spacings.unsqueeze(-1)  # (B, T, 1)
        x = self.input_proj(x)       # (B, T, n_embd)

        # Prepend memory tokens
        x = self.memory_bank(x)      # (B, K+T, n_embd)

        # Add positional embeddings
        pos = torch.arange(0, K + T, dtype=torch.long, device=device)
        x = x + self.pos_emb(pos)
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # MDN head (only on sequence positions, not memory)
        x_seq = x[:, K:]  # (B, T, n_embd)
        pi, mu, sigma = self.mdn_head(x_seq)

        result = {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
        }

        # Entropy for monitoring
        entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=-1).mean()
        result['entropy'] = entropy

        # Diversity loss for memory
        diversity_loss = self.memory_bank.diversity_loss()
        result['diversity_loss'] = diversity_loss

        # Compute NLL if targets provided
        if targets is not None:
            pi_pred = pi[:, :-1]
            mu_pred = mu[:, :-1]
            sigma_pred = sigma[:, :-1]
            target_vals = targets[:, 1:]

            nll = self._mdn_loss(pi_pred, mu_pred, sigma_pred, target_vals)

            # Total loss: NLL - entropy_reg * entropy + diversity_weight * diversity
            loss = (
                nll
                - self.config.entropy_reg * entropy
                + self.config.diversity_weight * diversity_loss
            )

            result['loss'] = loss
            result['nll'] = nll

        return result

    def _mdn_loss(self, pi, mu, sigma, target):
        """Negative log-likelihood for Gaussian mixture."""
        target = target.unsqueeze(-1)
        var = sigma ** 2
        log_probs = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((target - mu) ** 2) / var
        )
        log_pi = torch.log(pi + 1e-10)
        log_mixture = torch.logsumexp(log_pi + log_probs, dim=-1)
        return -log_mixture.mean()

    @torch.no_grad()
    def predict_mean(self, spacings: torch.Tensor) -> torch.Tensor:
        """Predict next spacing as mixture mean."""
        result = self.forward(spacings)
        pi = result['pi'][:, -1]
        mu = result['mu'][:, -1]
        return torch.sum(pi * mu, dim=-1)

    @torch.no_grad()
    def sample(self, spacings: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample next spacing from predicted mixture."""
        result = self.forward(spacings)
        pi = result['pi'][:, -1]
        mu = result['mu'][:, -1]
        sigma = result['sigma'][:, -1]

        if temperature != 1.0:
            pi = F.softmax(torch.log(pi + 1e-10) / temperature, dim=-1)

        comp_idx = torch.multinomial(pi, 1).squeeze(-1)
        mu_sel = mu[torch.arange(len(comp_idx)), comp_idx]
        sigma_sel = sigma[torch.arange(len(comp_idx)), comp_idx]

        sample = mu_sel + sigma_sel * torch.randn_like(mu_sel)
        return torch.clamp(sample, min=0.0, max=10.0)

    def get_memory_diagnostics(self) -> Dict[str, float]:
        """Get memory bank diagnostics."""
        return self.memory_bank.get_diagnostics()

    def get_param_groups(self, base_lr: float) -> List[Dict]:
        """
        Get parameter groups with different LR for memory.

        Memory learns slower (memory_lr_mult) to avoid early collapse.
        Returns groups with 'is_memory' flag for safe identification.
        """
        memory_params = list(self.memory_bank.parameters())
        memory_ids = {id(p) for p in memory_params}

        other_params = [p for p in self.parameters() if id(p) not in memory_ids]

        return [
            {'params': other_params, 'lr': base_lr, 'is_memory': False},
            {'params': memory_params, 'lr': base_lr * self.config.memory_lr_mult, 'is_memory': True},
        ]


def test_memory_mdn():
    """Quick test of MemoryMDN."""
    config = MemoryMDNConfig(
        n_layer=4,
        n_head=4,
        n_embd=128,
        seq_len=256,
        n_components=8,
        n_memory=8,
        memory_dropout=0.1,
        memory_cap=2.0,
        diversity_weight=0.01,
    )

    model = MemoryMDN(config)

    # Fake batch
    B, T = 4, 256
    spacings = torch.abs(torch.randn(B, T)) + 0.5

    # Forward
    result = model(spacings, targets=spacings)

    print(f"pi shape: {result['pi'].shape}")
    print(f"mu shape: {result['mu'].shape}")
    print(f"NLL: {result['nll'].item():.4f}")
    print(f"Entropy: {result['entropy'].item():.4f}")
    print(f"Diversity loss: {result['diversity_loss'].item():.6f}")

    # Memory diagnostics
    diag = model.get_memory_diagnostics()
    print(f"Memory diagnostics: {diag}")

    # Test inference
    pred = model.predict_mean(spacings)
    print(f"Mean prediction: {pred[0].item():.4f}")

    sample = model.sample(spacings)
    print(f"Sample: {sample[0].item():.4f}")

    # Test param groups
    groups = model.get_param_groups(base_lr=3e-4)
    print(f"Param groups: {len(groups[0]['params'])} main, {len(groups[1]['params'])} memory")

    print("\nTest passed!")


if __name__ == "__main__":
    test_memory_mdn()
