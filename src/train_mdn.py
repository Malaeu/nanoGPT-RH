#!/usr/bin/env python3
"""
TASK_SPEC_2M — SpacingMDN Baseline Training

Architecture:
  - Transformer: 6L/8H/256E
  - MDN head: K=8 Gaussian mixture
  - Loss: stable logsumexp (log_softmax(pi_logits) + log_sum_exp)

Training:
  - batch=256
  - lr=3e-4, warmup=1500
  - entropy_reg=0.005
  - max_steps=20000
  - save: ckpt_5000/10000/15000/final + best.pt
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Enable TensorFloat32 for better H100/A100 performance (~10-15% speedup)
torch.set_float32_matmul_precision('high')
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
import numpy as np
import argparse

console = Console()

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class MDNConfig:
    """SpacingMDN configuration per TASK_SPEC_2M."""
    seq_len: int = 256          # context length
    n_layer: int = 6            # transformer layers
    n_head: int = 8             # attention heads
    n_embd: int = 256           # embedding dimension
    n_components: int = 8       # MDN mixture components
    dropout: float = 0.1
    bias: bool = False


# ============================================================================
# TRANSFORMER COMPONENTS
# ============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: MDNConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len))
            .view(1, 1, config.seq_len, config.seq_len)
        )

    def forward(self, x, return_attention=False):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Use FlashAttention via scaled_dot_product_attention (PyTorch 2.0+)
        # This is 2-4x faster and more memory efficient
        if not return_attention:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
            att_weights = None
        else:
            # Fallback to manual attention when we need attention weights
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if T <= self.mask.size(2):
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            else:
                causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                att = att.masked_fill(~causal_mask.view(1, 1, T, T), float('-inf'))
            att = F.softmax(att, dim=-1)
            att_weights = att
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        if return_attention:
            return y, att_weights
        return y


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: MDNConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config: MDNConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


# ============================================================================
# MDN HEAD
# ============================================================================

class MDNHead(nn.Module):
    """
    Mixture Density Network head for continuous spacing prediction.

    Outputs K Gaussian components:
      - pi: mixture weights (sum=1)
      - mu: means
      - sigma: standard deviations (positive)
    """

    def __init__(self, n_embd: int, n_components: int = 8):
        super().__init__()
        self.n_components = n_components

        # Output: pi (K) + mu (K) + log_sigma (K)
        self.head = nn.Linear(n_embd, 3 * n_components)

    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd) transformer output

        Returns:
            pi: (B, T, K) mixture weights (softmaxed)
            mu: (B, T, K) means
            sigma: (B, T, K) std devs (positive)
        """
        out = self.head(x)  # (B, T, 3K)

        pi_logits, mu, log_sigma = out.chunk(3, dim=-1)

        # Mixture weights via softmax
        pi = F.softmax(pi_logits, dim=-1)  # (B, T, K)

        # Mu can be any value (spacings are positive but centered around 1)
        # Use softplus to ensure mu > 0 for positive spacings
        mu = F.softplus(mu)  # (B, T, K)

        # Sigma must be positive
        sigma = F.softplus(log_sigma) + 1e-4  # (B, T, K), min sigma for stability

        return pi, mu, sigma


# ============================================================================
# FULL MODEL: SpacingMDN
# ============================================================================

class SpacingMDN(nn.Module):
    """
    SpacingMDN: Transformer + MDN head for continuous spacing prediction.

    Input: (B, T) continuous spacing values
    Output: MDN parameters (pi, mu, sigma) for next spacing distribution
    """

    def __init__(self, config: MDNConfig):
        super().__init__()
        self.config = config

        # Continuous input projection (single value -> n_embd)
        self.input_proj = nn.Linear(1, config.n_embd)

        # Positional embeddings
        self.wpe = nn.Embedding(config.seq_len, config.n_embd)

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
        console.print(f"[green]SpacingMDN: {n_params/1e6:.2f}M parameters[/]")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, T) continuous spacing values

        Returns:
            pi, mu, sigma: MDN parameters (B, T, K each)
            attentions: list of attention weights if return_attention=True
        """
        B, T = x.size()
        device = x.device

        # Project continuous values to embeddings
        x = x.unsqueeze(-1)  # (B, T, 1)
        x = self.input_proj(x)  # (B, T, n_embd)

        # Add positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = x + self.wpe(pos)
        x = self.drop(x)

        # Transformer blocks
        attentions = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attentions.append(attn)
            else:
                x = block(x)

        x = self.ln_f(x)

        # MDN head
        pi, mu, sigma = self.mdn_head(x)

        if return_attention:
            return pi, mu, sigma, attentions
        return pi, mu, sigma


# ============================================================================
# MDN LOSS (STABLE)
# ============================================================================

def mdn_loss(pi, mu, sigma, target, entropy_reg=0.005):
    """
    Compute negative log-likelihood for mixture of Gaussians.

    Loss = -log( sum_k pi_k * N(target | mu_k, sigma_k) )

    Uses logsumexp for numerical stability.

    Args:
        pi: (B, T, K) mixture weights
        mu: (B, T, K) means
        sigma: (B, T, K) std devs
        target: (B, T) target values
        entropy_reg: entropy regularization coefficient

    Returns:
        nll: scalar negative log-likelihood
    """
    target = target.unsqueeze(-1)  # (B, T, 1)

    # Log of Gaussian PDF: log N(x|mu,sigma) = -0.5*log(2*pi) - log(sigma) - 0.5*((x-mu)/sigma)^2
    log_prob = -0.5 * math.log(2 * math.pi) - torch.log(sigma) - 0.5 * ((target - mu) / sigma) ** 2
    # (B, T, K)

    # Log mixture weights
    log_pi = torch.log(pi + 1e-10)  # (B, T, K)

    # Log-sum-exp for numerical stability: log(sum_k pi_k * p_k) = logsumexp(log_pi + log_prob)
    log_likelihood = torch.logsumexp(log_pi + log_prob, dim=-1)  # (B, T)

    # Negative log-likelihood
    nll = -log_likelihood.mean()

    # Entropy regularization (encourage diverse mixture weights)
    entropy = -(pi * torch.log(pi + 1e-10)).sum(dim=-1).mean()
    loss = nll - entropy_reg * entropy

    return loss, nll, entropy


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    console.print("[bold magenta]═══ TASK_SPEC_2M: SpacingMDN Training ═══[/]\n")

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
    console.print(f"[dim]Meta: mean={meta['spacing_mean']:.4f}, std={meta['spacing_std']:.4f}[/]")

    # Config
    config = MDNConfig(
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_components=args.n_components,
        dropout=args.dropout,
    )

    # Model
    model = SpacingMDN(config).to(device)

    # torch.compile for faster training (PyTorch 2.0+)
    if args.compile and device.type == "cuda":
        console.print("[yellow]Compiling model with torch.compile()...[/]")
        model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # LR scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        # Cosine decay
        progress = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        return max(args.min_lr / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler (for CUDA)
    use_amp = device.type == "cuda" and args.use_amp
    scaler = GradScaler('cuda') if use_amp else None

    # Output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    console.print(f"\n[cyan]Training for {args.max_steps} steps...[/]")
    console.print(f"  batch_size={args.batch_size}")
    console.print(f"  lr={args.lr}, warmup={args.warmup_steps}")
    console.print(f"  entropy_reg={args.entropy_reg}")
    console.print(f"  save_interval={args.save_interval}")

    best_val_nll = float('inf')
    train_losses = []
    val_nlls = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Training SpacingMDN...", total=args.max_steps)

        # Timing
        train_start_time = time.time()
        last_log_time = train_start_time
        last_log_step = 0

        for step in range(1, args.max_steps + 1):
            model.train()

            # Sample batch
            idx = torch.randint(0, len(train_data), (args.batch_size,))
            batch = train_data[idx].to(device)  # (B, T)

            # Input: all but last, Target: all but first
            x = batch[:, :-1]  # (B, T-1)
            y = batch[:, 1:]   # (B, T-1)

            # Forward
            if use_amp:
                with autocast(device_type='cuda'):
                    pi, mu, sigma = model(x)
                    loss, nll, entropy = mdn_loss(pi, mu, sigma, y, args.entropy_reg)
            else:
                pi, mu, sigma = model(x)
                loss, nll, entropy = mdn_loss(pi, mu, sigma, y, args.entropy_reg)

            # Backward
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            train_losses.append(nll.item())

            # Eval & logging
            if step % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    # Val NLL
                    val_idx = torch.randint(0, len(val_data), (min(512, len(val_data)),))
                    val_batch = val_data[val_idx].to(device)
                    val_x = val_batch[:, :-1]
                    val_y = val_batch[:, 1:]

                    val_pi, val_mu, val_sigma = model(val_x)
                    val_loss, val_nll, val_entropy = mdn_loss(val_pi, val_mu, val_sigma, val_y, args.entropy_reg)

                val_nlls.append(val_nll.item())

                # Timing stats
                now = time.time()
                elapsed_total = now - train_start_time
                elapsed_interval = now - last_log_time
                steps_interval = step - last_log_step
                steps_per_sec = steps_interval / elapsed_interval if elapsed_interval > 0 else 0
                remaining_steps = args.max_steps - step
                eta_sec = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

                # Format time nicely
                if elapsed_total < 60:
                    time_str = f"{elapsed_total:.0f}s"
                else:
                    time_str = f"{elapsed_total/60:.1f}m"

                if elapsed_interval < 60:
                    delta_str = f"{elapsed_interval:.0f}s"
                else:
                    delta_str = f"{elapsed_interval/60:.1f}m"

                # Update for next interval
                last_log_time = now
                last_log_step = step

                console.print(
                    f"Step {step}: "
                    f"train_nll={nll.item():.4f} | "
                    f"val_nll={val_nll.item():.4f} | "
                    f"entropy={entropy.item():.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"[cyan]{steps_per_sec:.1f} steps/s[/cyan] | "
                    f"Time: {time_str} | Δ: {delta_str} | "
                    f"ETA: {eta_sec/60:.1f}m"
                )

                # Save best
                if val_nll.item() < best_val_nll:
                    best_val_nll = val_nll.item()
                    torch.save({
                        "model": model.state_dict(),
                        "config": config.__dict__,
                        "step": step,
                        "val_nll": val_nll.item(),
                        "meta": meta,
                    }, out_dir / "best.pt")

            # Save checkpoints
            if step % args.save_interval == 0:
                torch.save({
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "meta": meta,
                }, out_dir / f"ckpt_{step}.pt")
                console.print(f"[dim]Saved ckpt_{step}.pt[/]")

            progress.update(task, advance=1)

    # Save final
    torch.save({
        "model": model.state_dict(),
        "config": config.__dict__,
        "step": args.max_steps,
        "val_nll": val_nlls[-1] if val_nlls else None,
        "train_losses": train_losses,
        "val_nlls": val_nlls,
        "meta": meta,
    }, out_dir / "final.pt")

    # Final timing stats
    total_time = time.time() - train_start_time
    total_steps = args.max_steps
    avg_steps_per_sec = total_steps / total_time if total_time > 0 else 0
    samples_per_sec = avg_steps_per_sec * args.batch_size

    console.print(f"\n[green]═══ Training Complete ═══[/]")
    console.print(f"Best val NLL: {best_val_nll:.4f}")
    console.print(f"Saved: {out_dir}/")
    console.print(f"\n[bold cyan]⏱ Timing Summary:[/]")
    console.print(f"  Total time: {total_time/60:.1f} min ({total_time:.1f} sec)")
    console.print(f"  Steps/sec: {avg_steps_per_sec:.2f}")
    console.print(f"  Samples/sec: {samples_per_sec:.0f}")
    console.print(f"  Time per step: {1000*total_time/total_steps:.2f} ms")

    # Summary table
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", f"SpacingMDN {config.n_layer}L/{config.n_head}H/{config.n_embd}E")
    table.add_row("MDN Components", str(config.n_components))
    table.add_row("Total Steps", str(args.max_steps))
    table.add_row("Best Val NLL", f"{best_val_nll:.4f}")
    table.add_row("Final Train NLL", f"{train_losses[-1]:.4f}" if train_losses else "N/A")
    table.add_row("Checkpoints", f"ckpt_5000, ckpt_10000, ckpt_15000, final, best")

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="SpacingMDN Training")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/continuous_2M")

    # Model
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--entropy-reg", type=float, default=0.005)

    # Checkpointing
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--out-dir", type=str, default="out/mdn_baseline")

    # AMP
    parser.add_argument("--use-amp", action="store_true")

    # Optimization
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for 1.5-2x speedup")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
