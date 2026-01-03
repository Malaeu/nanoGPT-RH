#!/usr/bin/env python3
"""
Mixture Density Network (MDN) for Continuous Spacing Prediction.

Instead of discrete bins, we model p(s_{n+1} | context) as a Gaussian mixture:

    p(s | x) = sum_{k=1}^K  pi_k(x) * N(s; mu_k(x), sigma_k(x))

This gives us:
- Continuous predictions (no discretization error)
- Full uncertainty quantification
- Proper density estimation for GUE-like distributions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class MDNConfig:
    """Configuration for SpacingMDN model."""
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    seq_len: int = 256
    dropout: float = 0.1
    bias: bool = False

    # MDN specific
    n_components: int = 8          # Number of Gaussian components
    sigma_min: float = 1e-4        # Minimum sigma (prevents collapse)
    sigma_max: float = 10.0        # Maximum sigma (prevents explosion)
    entropy_reg: float = 0.01      # Entropy regularization weight


class MDNHead(nn.Module):
    """
    Mixture Density Network head.

    Outputs parameters for K Gaussian components:
    - pi: mixture weights (K values, sum to 1)
    - mu: component means (K values)
    - sigma: component stds (K values, > 0)
    """

    def __init__(self, n_embd: int, n_components: int = 8,
                 sigma_min: float = 1e-4, sigma_max: float = 10.0):
        super().__init__()
        self.n_components = n_components
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Project to 3K parameters: pi_logits, mu, log_sigma
        self.proj = nn.Linear(n_embd, 3 * n_components)

        # Initialize mu with spread to avoid collapse
        # Spacings are typically in [0, 4], mean ~1
        with torch.no_grad():
            # Spread initial means across [0.2, 2.0]
            mu_init = torch.linspace(0.2, 2.0, n_components)
            self.proj.bias[n_components:2*n_components] = mu_init

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Hidden states (B, T, n_embd)

        Returns:
            pi: Mixture weights (B, T, K)
            mu: Component means (B, T, K)
            sigma: Component stds (B, T, K)
        """
        out = self.proj(x)  # (B, T, 3K)

        pi_logits, mu, log_sigma = out.chunk(3, dim=-1)

        # Mixture weights via softmax
        pi = F.softmax(pi_logits, dim=-1)

        # Means can be any value (will learn appropriate range)
        # mu stays as-is

        # Sigmas must be positive, bounded
        sigma = torch.sigmoid(log_sigma) * (self.sigma_max - self.sigma_min) + self.sigma_min

        return pi, mu, sigma


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: MDNConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len))
            .view(1, 1, config.seq_len, config.seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if T <= self.mask.size(2):
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        else:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask.view(1, 1, T, T), float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: MDNConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config: MDNConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SpacingMDN(nn.Module):
    """
    Transformer with MDN head for continuous spacing prediction.

    Input: sequence of continuous spacings (B, T)
    Output: MDN parameters for p(s_{t+1} | s_1, ..., s_t)
    """

    def __init__(self, config: MDNConfig):
        super().__init__()
        self.config = config

        # Input embedding: continuous -> n_embd
        # Using a linear projection instead of discrete embedding
        self.input_proj = nn.Linear(1, config.n_embd)
        self.pos_emb = nn.Embedding(config.seq_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
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
        print(f"SpacingMDN: {n_params:,} parameters")

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
            spacings: Input spacings (B, T), float values
            targets: Target spacings for loss (B, T), float values
                     Usually targets = spacings shifted by 1

        Returns:
            Dictionary with:
            - pi: mixture weights (B, T, K)
            - mu: component means (B, T, K)
            - sigma: component stds (B, T, K)
            - loss: NLL loss if targets provided
            - entropy: entropy of mixture weights
        """
        B, T = spacings.shape
        device = spacings.device

        # Embed continuous spacings
        x = spacings.unsqueeze(-1)  # (B, T, 1)
        x = self.input_proj(x)       # (B, T, n_embd)

        # Add positional embedding
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = x + self.pos_emb(pos)
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # MDN head
        pi, mu, sigma = self.mdn_head(x)

        result = {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
        }

        # Compute entropy for monitoring/regularization
        entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=-1).mean()
        result['entropy'] = entropy

        # Compute loss if targets provided
        if targets is not None:
            # Shift: predict t+1 from t
            # pi[:, :-1] predicts targets[:, 1:]
            pi_pred = pi[:, :-1]
            mu_pred = mu[:, :-1]
            sigma_pred = sigma[:, :-1]
            target_vals = targets[:, 1:]

            nll = self._mdn_loss(pi_pred, mu_pred, sigma_pred, target_vals)

            # Add entropy regularization
            loss = nll - self.config.entropy_reg * entropy

            result['loss'] = loss
            result['nll'] = nll

        return result

    def _mdn_loss(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative log-likelihood for Gaussian mixture.

        Args:
            pi: (B, T, K) mixture weights
            mu: (B, T, K) component means
            sigma: (B, T, K) component stds
            target: (B, T) target values

        Returns:
            Scalar NLL loss
        """
        target = target.unsqueeze(-1)  # (B, T, 1)

        # Gaussian log-probability for each component
        # log N(x; mu, sigma) = -0.5*log(2*pi) - log(sigma) - 0.5*((x-mu)/sigma)^2
        var = sigma ** 2
        log_probs = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((target - mu) ** 2) / var
        )  # (B, T, K)

        # Log-sum-exp for mixture
        log_pi = torch.log(pi + 1e-10)
        log_mixture = torch.logsumexp(log_pi + log_probs, dim=-1)  # (B, T)

        # Mean NLL
        nll = -log_mixture.mean()

        return nll

    @torch.no_grad()
    def predict_mean(self, spacings: torch.Tensor) -> torch.Tensor:
        """
        Predict next spacing as mixture mean.

        Args:
            spacings: (B, T) input spacings

        Returns:
            (B,) predicted next spacing (mixture mean)
        """
        result = self.forward(spacings)
        pi = result['pi'][:, -1]      # (B, K)
        mu = result['mu'][:, -1]      # (B, K)

        # Weighted mean
        pred = torch.sum(pi * mu, dim=-1)  # (B,)
        return pred

    @torch.no_grad()
    def predict_mode(self, spacings: torch.Tensor) -> torch.Tensor:
        """
        Predict next spacing as mode of dominant component.

        Args:
            spacings: (B, T) input spacings

        Returns:
            (B,) predicted next spacing (mode)
        """
        result = self.forward(spacings)
        pi = result['pi'][:, -1]      # (B, K)
        mu = result['mu'][:, -1]      # (B, K)

        # Take mean of component with highest weight
        max_idx = pi.argmax(dim=-1)   # (B,)
        pred = mu[torch.arange(len(max_idx)), max_idx]
        return pred

    @torch.no_grad()
    def sample(self, spacings: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample next spacing from predicted mixture.

        Args:
            spacings: (B, T) input spacings
            temperature: Temperature for mixture weights (1.0 = normal)

        Returns:
            (B,) sampled next spacing
        """
        result = self.forward(spacings)
        pi = result['pi'][:, -1]        # (B, K)
        mu = result['mu'][:, -1]        # (B, K)
        sigma = result['sigma'][:, -1]  # (B, K)

        # Apply temperature to mixture weights
        if temperature != 1.0:
            pi = F.softmax(torch.log(pi + 1e-10) / temperature, dim=-1)

        # Sample component
        comp_idx = torch.multinomial(pi, 1).squeeze(-1)  # (B,)

        # Sample from selected Gaussian
        mu_selected = mu[torch.arange(len(comp_idx)), comp_idx]
        sigma_selected = sigma[torch.arange(len(comp_idx)), comp_idx]

        # Reparameterization: x = mu + sigma * eps
        eps = torch.randn_like(mu_selected)
        sample = mu_selected + sigma_selected * eps

        # Clip to valid range [0, max_spacing]
        sample = torch.clamp(sample, min=0.0, max=10.0)

        return sample

    @torch.no_grad()
    def get_distribution(self, spacings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get full distribution parameters for analysis.

        Args:
            spacings: (B, T) input spacings

        Returns:
            Dictionary with pi, mu, sigma for last position
        """
        result = self.forward(spacings)
        return {
            'pi': result['pi'][:, -1],
            'mu': result['mu'][:, -1],
            'sigma': result['sigma'][:, -1],
        }


def test_mdn():
    """Quick test of SpacingMDN."""
    config = MDNConfig(
        n_layer=4,
        n_head=4,
        n_embd=128,
        seq_len=256,
        n_components=8,
    )

    model = SpacingMDN(config)

    # Fake batch of spacings
    B, T = 4, 256
    spacings = torch.abs(torch.randn(B, T)) + 0.5  # Positive spacings around 1

    # Forward pass
    result = model(spacings, targets=spacings)

    print(f"pi shape: {result['pi'].shape}")
    print(f"mu shape: {result['mu'].shape}")
    print(f"sigma shape: {result['sigma'].shape}")
    print(f"NLL: {result['nll'].item():.4f}")
    print(f"Entropy: {result['entropy'].item():.4f}")

    # Test inference
    pred_mean = model.predict_mean(spacings)
    pred_mode = model.predict_mode(spacings)
    pred_sample = model.sample(spacings)

    print(f"Mean prediction: {pred_mean[0].item():.4f}")
    print(f"Mode prediction: {pred_mode[0].item():.4f}")
    print(f"Sample: {pred_sample[0].item():.4f}")

    print("\nTest passed!")


if __name__ == "__main__":
    test_mdn()
