#!/usr/bin/env python3
"""
Kernel Check: Compare empirical μ(d) with theoretical candidates.

This is the "Honest Math-Check":
- Extract attention logits (before softmax)
- Compute μ(d) = E[L_ij | |i-j| = d]
- Compare with sinc, gauss, sinc*gauss
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from scipy.optimize import curve_fit

from model.gpt import SpacingGPT, GPTConfig

console = Console()


def load_model(ckpt_path: Path, device: str):
    """Load trained model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpacingGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["config"]


def extract_attention_logits(model, x, config, layer: int = 0):
    """
    Extract attention logits (before softmax) for a specific layer.
    Returns: (B, n_head, T, T) tensor
    """
    device = x.device
    B, T = x.size()

    # Embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(x)
    pos_emb = model.transformer.wpe(pos)
    h = model.transformer.drop(tok_emb + pos_emb)

    # Forward through blocks until target layer
    for layer_idx, block in enumerate(model.transformer.h):
        if layer_idx == layer:
            # Extract Q, K manually
            ln_out = block.ln_1(h)
            B, T, C = ln_out.size()
            qkv = block.attn.c_attn(ln_out)
            q, k, v = qkv.split(C, dim=2)

            n_head = config.n_head
            head_dim = C // n_head

            q = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)

            # Raw attention logits (no softmax!)
            scale = 1.0 / math.sqrt(head_dim)
            logits = (q @ k.transpose(-2, -1)) * scale

            return logits  # (B, n_head, T, T)

        h = block(h)

    raise ValueError(f"Layer {layer} not found")


@torch.no_grad()
def compute_mu_d(model, data, config, layer=0, head=0, d_max=128, batch_size=64):
    """
    Compute μ(d) = E[L_ij | |i-j| = d] using index distance.
    """
    device = next(model.parameters()).device

    # Take batch
    idx = torch.randperm(len(data))[:batch_size]
    x = data[idx].to(device)

    # Extract logits
    logits = extract_attention_logits(model, x, config, layer=layer)
    logits = logits[:, head]  # (B, T, T)

    B, T, _ = logits.shape

    # Collect by distance
    sums = torch.zeros(d_max + 1, device=device)
    counts = torch.zeros(d_max + 1, device=device)

    for d in range(1, min(d_max + 1, T)):
        # Extract diagonal at offset -d (causal: looking back)
        diag = torch.diagonal(logits, offset=-d, dim1=-2, dim2=-1)
        values = diag.flatten()
        finite_mask = torch.isfinite(values)

        sums[d] = values[finite_mask].sum()
        counts[d] = finite_mask.sum()

    mu = (sums / counts.clamp_min(1)).cpu().numpy()
    cnt = counts.cpu().numpy()

    return np.arange(len(mu)), mu, cnt


def theoretical_candidates(d, params=None):
    """
    Generate theoretical kernel candidates:
    1. sinc(πd) = sin(πd)/(πd)
    2. exp(-a*d^2) - Gaussian envelope
    3. sinc(πd) * exp(-a*d^2) - damped sinc
    4. cos(ωd) * exp(-γd) - damped cosine (what PySR found)
    """
    eps = 1e-9

    # Standard sinc (period = 2)
    sinc = np.where(d > eps, np.sin(np.pi * d) / (np.pi * d), 1.0)

    # Scaled sinc to match our period ~18
    # sin(2π/18 * d) / (2π/18 * d) ≈ sin(0.35d) / (0.35d)
    omega = 2 * np.pi / 18
    scaled_sinc = np.where(d > eps, np.sin(omega * d) / (omega * d), 1.0)

    # Gaussian
    a = 0.01
    gauss = np.exp(-a * d**2)

    # Damped sinc
    damped_sinc = scaled_sinc * gauss

    # PySR formula: cos(0.357d - 2.05) * exp(-0.0024d)
    pysr = np.cos(0.357 * d - 2.05) * np.exp(-0.0024 * d)

    return {
        'sinc(πd)': sinc,
        'scaled_sinc(ωd)': scaled_sinc,
        'gauss': gauss,
        'scaled_sinc × gauss': damped_sinc,
        'PySR formula': pysr
    }


def fit_and_compare(d, mu, candidates):
    """Fit candidates to empirical data and compute R²."""
    results = {}
    mask = (d >= 1) & np.isfinite(mu)
    d_fit = d[mask]
    mu_fit = mu[mask]

    for name, f in candidates.items():
        f_fit = f[mask]

        # Affine fit: y = a*f + b
        A = np.vstack([f_fit, np.ones(len(f_fit))]).T
        try:
            coef, residuals, rank, s = np.linalg.lstsq(A, mu_fit, rcond=None)
            fitted = coef[0] * f + coef[1]

            # R² score
            ss_res = np.sum((mu_fit - (coef[0] * f_fit + coef[1]))**2)
            ss_tot = np.sum((mu_fit - np.mean(mu_fit))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            results[name] = {
                'fitted': fitted,
                'r2': r2,
                'coef': coef
            }
        except:
            results[name] = {'fitted': f, 'r2': 0, 'coef': [1, 0]}

    return results


def main():
    console.print("[bold magenta]═══ KERNEL CHECK: Honest Math Test ═══[/]\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/]")

    # Load model
    model, config = load_model(Path("out/best.pt"), device)
    console.print(f"[green]Model: {config.n_layer} layers, {config.n_head} heads[/]")

    # Load validation data
    val_data = torch.load("data/val.pt")
    console.print(f"[green]Data: {val_data.shape}[/]\n")

    # Test all heads in layer 0
    console.print("[bold]Extracting μ(d) for all heads in Layer 0...[/]")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    best_r2 = 0
    best_head = 0

    for head in range(config.n_head):
        d, mu, cnt = compute_mu_d(model, val_data, config, layer=0, head=head, d_max=127)

        # Get candidates
        candidates = theoretical_candidates(d)

        # Fit and compare
        fits = fit_and_compare(d, mu, candidates)

        # Find best fit
        best_fit_name = max(fits, key=lambda x: fits[x]['r2'])
        head_best_r2 = fits[best_fit_name]['r2']

        if head_best_r2 > best_r2:
            best_r2 = head_best_r2
            best_head = head

        console.print(f"  Head {head}: Best fit = {best_fit_name} (R² = {head_best_r2:.3f})")

        # Plot
        ax = axes[head // 2, head % 2]
        ax.plot(d[1:], mu[1:], 'b-', lw=2, label='Empirical μ(d)')

        # Plot top 3 fits
        sorted_fits = sorted(fits.items(), key=lambda x: -x[1]['r2'])[:3]
        colors = ['red', 'green', 'orange']
        for (name, fit), color in zip(sorted_fits, colors):
            ax.plot(d, fit['fitted'], '--', color=color, lw=1.5,
                   label=f'{name} (R²={fit["r2"]:.3f})')

        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('d = |i-j|')
        ax.set_ylabel('μ(d)')
        ax.set_title(f'Layer 0, Head {head}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 100)

    plt.suptitle('Kernel Check: Empirical vs Theoretical', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kernel_check.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]✓ Saved kernel_check.png[/]")

    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]VERDICT:[/]")
    console.print("="*60)

    if best_r2 > 0.8:
        console.print(f"[green]✅ PASS: Strong match with theoretical kernel (R² = {best_r2:.3f})[/]")
    elif best_r2 > 0.5:
        console.print(f"[yellow]🟡 PARTIAL: Moderate match (R² = {best_r2:.3f})[/]")
    else:
        console.print(f"[red]❌ FAIL: Weak match (R² = {best_r2:.3f})[/]")

    console.print(f"\nBest head: {best_head}")
    console.print("="*60)


if __name__ == "__main__":
    main()
