#!/usr/bin/env python3
"""
DEPRECATED - DO NOT USE

BUG: This file extracts POST-SOFTMAX attention weights, NOT raw logits!
     The softmax weights are always positive and sum to 1, which is NOT
     the "interaction energy" kernel we want.

USE INSTEAD: extract_kernel.py or experiments_100M.py
     These correctly extract pre-softmax attention logits via manual Q@K^T.

---------------------------------------------------------------------------

🔬 KERNEL EXTRACTION IN UNFOLDED COORDINATES

Переводим attention kernel μ(d) из индексного расстояния
в unfolded distance для корректного сравнения с sine-kernel.

d_idx → d_unf = Σ s_k (сумма spacing'ов между позициями)

Sine-kernel prediction: K(d) = sin(πd)/(πd)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
from scipy.optimize import curve_fit

from model.gpt import SpacingGPT

console = Console()

CKPT_PATH = Path("out/best.pt")
ZEROS_PATH = Path("zeros/zeros2M.txt")
DATA_DIR = Path("data")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = SpacingGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model, config


def unfold_val(gamma):
    """Unfold gamma values."""
    gamma = np.asarray(gamma)
    return (gamma / (2 * np.pi)) * np.log(gamma / (2 * np.pi * np.e))


def extract_attention_with_unfolded_distance(n_samples=1000):
    """
    Extract attention weights with UNFOLDED distance.

    For each attention weight A[i,j], compute:
    - d_idx = |i - j| (index distance)
    - d_unf = Σ_{k=min(i,j)}^{max(i,j)-1} s_k (unfolded distance)
    """
    console.print("\n[bold magenta]═══ 🔬 KERNEL IN UNFOLDED COORDINATES ═══[/]\n")

    model, config = load_model()
    console.print(f"[green]Model loaded: seq_len={config.seq_len}[/]")

    # Load data
    zeros = np.loadtxt(ZEROS_PATH)
    meta = torch.load(DATA_DIR / "meta.pt", weights_only=False)
    bin_edges = np.array(meta["bin_edges"])

    # Compute spacings
    unfolded = unfold_val(zeros)
    all_gaps = np.diff(unfolded)

    # Sample random windows
    np.random.seed(42)
    start_indices = np.random.randint(0, len(all_gaps) - config.seq_len - 1, n_samples)

    attention_data = []  # (d_idx, d_unf, attn_weight, head_idx, layer_idx)

    console.print(f"[cyan]Extracting attention from {n_samples} samples...[/]")

    with torch.no_grad():
        for idx, start in enumerate(start_indices):
            if idx % 200 == 0:
                console.print(f"  Processing {idx}/{n_samples}...")

            # Get spacing sequence
            spacings = all_gaps[start:start + config.seq_len]

            # Convert to bins
            bins = np.digitize(spacings, bin_edges) - 1
            bins = np.clip(bins, 0, config.vocab_size - 1)

            x = torch.tensor(bins, dtype=torch.long).unsqueeze(0).to(DEVICE)

            # Forward with attention
            _, _, attentions = model(x, return_attention=True)

            # attentions: list of [batch, n_heads, seq, seq]
            for layer_idx, attn in enumerate(attentions):
                attn_np = attn[0].cpu().numpy()  # [n_heads, seq, seq]

                for head_idx in range(attn_np.shape[0]):
                    head_attn = attn_np[head_idx]  # [seq, seq]

                    # Sample some pairs (not all - too many)
                    n_pairs = 20
                    for _ in range(n_pairs):
                        i = np.random.randint(10, config.seq_len)
                        j = np.random.randint(0, i)

                        d_idx = i - j

                        # Unfolded distance = sum of spacings between j and i
                        d_unf = np.sum(spacings[j:i])

                        attn_weight = head_attn[i, j]

                        attention_data.append({
                            'd_idx': d_idx,
                            'd_unf': d_unf,
                            'attn': attn_weight,
                            'head': head_idx,
                            'layer': layer_idx
                        })

    console.print(f"[green]Collected {len(attention_data)} attention samples[/]")

    return attention_data


def analyze_kernel(attention_data):
    """Analyze kernel in both coordinate systems."""

    d_idx = np.array([d['d_idx'] for d in attention_data])
    d_unf = np.array([d['d_unf'] for d in attention_data])
    attn = np.array([d['attn'] for d in attention_data])

    console.print("\n[bold]═══ COORDINATE COMPARISON ═══[/]")
    console.print(f"d_idx range: [{d_idx.min():.1f}, {d_idx.max():.1f}]")
    console.print(f"d_unf range: [{d_unf.min():.2f}, {d_unf.max():.2f}]")
    console.print(f"Mean spacing check: d_unf/d_idx = {d_unf.mean()/d_idx.mean():.4f} (should be ~1)")

    # Bin by index distance
    unique_d_idx = np.unique(d_idx)
    mu_idx = []
    mu_idx_std = []
    for d in unique_d_idx:
        mask = d_idx == d
        mu_idx.append(attn[mask].mean())
        mu_idx_std.append(attn[mask].std())
    mu_idx = np.array(mu_idx)
    mu_idx_std = np.array(mu_idx_std)

    # Bin by unfolded distance
    d_unf_bins = np.linspace(d_unf.min(), min(d_unf.max(), 100), 50)
    mu_unf = []
    mu_unf_std = []
    d_unf_centers = []
    for i in range(len(d_unf_bins) - 1):
        mask = (d_unf >= d_unf_bins[i]) & (d_unf < d_unf_bins[i+1])
        if mask.sum() > 10:
            mu_unf.append(attn[mask].mean())
            mu_unf_std.append(attn[mask].std())
            d_unf_centers.append((d_unf_bins[i] + d_unf_bins[i+1]) / 2)
    mu_unf = np.array(mu_unf)
    mu_unf_std = np.array(mu_unf_std)
    d_unf_centers = np.array(d_unf_centers)

    # Fit models
    # 1. Damped cosine (what we found before)
    def damped_cos(d, A, omega, phi, gamma, c):
        return A * np.cos(omega * d + phi) * np.exp(-gamma * d) + c

    # 2. Sine kernel (GUE prediction)
    def sine_kernel(d, A, alpha, c):
        # K(d) = A * sinc(alpha * d) + c
        d_safe = np.maximum(d, 0.01)
        return A * np.sinc(alpha * d_safe) + c

    # Fit on index distance
    try:
        popt_cos_idx, _ = curve_fit(
            damped_cos, unique_d_idx, mu_idx,
            p0=[0.1, 0.35, -2.0, 0.002, 0],
            maxfev=5000
        )
        console.print(f"\n[cyan]Damped cosine fit (d_idx):[/]")
        console.print(f"  A={popt_cos_idx[0]:.4f}, ω={popt_cos_idx[1]:.4f}, φ={popt_cos_idx[2]:.4f}")
        console.print(f"  γ={popt_cos_idx[3]:.6f}, c={popt_cos_idx[4]:.4f}")
        console.print(f"  Period = 2π/ω = {2*np.pi/popt_cos_idx[1]:.2f}")
    except:
        popt_cos_idx = None
        console.print("[yellow]Could not fit damped cosine to d_idx[/]")

    # Fit on unfolded distance
    try:
        popt_cos_unf, _ = curve_fit(
            damped_cos, d_unf_centers, mu_unf,
            p0=[0.1, 0.35, -2.0, 0.002, 0],
            maxfev=5000
        )
        console.print(f"\n[cyan]Damped cosine fit (d_unf):[/]")
        console.print(f"  A={popt_cos_unf[0]:.4f}, ω={popt_cos_unf[1]:.4f}, φ={popt_cos_unf[2]:.4f}")
        console.print(f"  γ={popt_cos_unf[3]:.6f}, c={popt_cos_unf[4]:.4f}")
        console.print(f"  Period = 2π/ω = {2*np.pi/popt_cos_unf[1]:.2f}")
    except:
        popt_cos_unf = None
        console.print("[yellow]Could not fit damped cosine to d_unf[/]")

    # Try sine kernel fit
    try:
        popt_sine, _ = curve_fit(
            sine_kernel, d_unf_centers, mu_unf,
            p0=[0.1, 0.1, 0],
            maxfev=5000
        )
        console.print(f"\n[cyan]Sine kernel fit (d_unf):[/]")
        console.print(f"  A={popt_sine[0]:.4f}, α={popt_sine[1]:.4f}, c={popt_sine[2]:.4f}")

        # Compute R² for sine kernel
        y_pred = sine_kernel(d_unf_centers, *popt_sine)
        ss_res = np.sum((mu_unf - y_pred)**2)
        ss_tot = np.sum((mu_unf - mu_unf.mean())**2)
        r2_sine = 1 - ss_res/ss_tot
        console.print(f"  R² = {r2_sine:.4f}")
    except:
        popt_sine = None
        console.print("[yellow]Could not fit sine kernel[/]")

    return {
        'd_idx': unique_d_idx,
        'mu_idx': mu_idx,
        'mu_idx_std': mu_idx_std,
        'd_unf': d_unf_centers,
        'mu_unf': mu_unf,
        'mu_unf_std': mu_unf_std,
        'popt_cos_idx': popt_cos_idx,
        'popt_cos_unf': popt_cos_unf,
        'popt_sine': popt_sine,
    }


def visualize_kernels(results, save_path="kernel_unfolded.png"):
    """Visualize kernel in both coordinate systems."""
    console.print("\n[cyan]Creating visualization...[/]")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top Left: Kernel vs index distance
    ax = axes[0, 0]
    ax.errorbar(results['d_idx'], results['mu_idx'],
                yerr=results['mu_idx_std']/3,
                fmt='b.', alpha=0.5, markersize=3, label='Data')
    if results['popt_cos_idx'] is not None:
        d_fit = np.linspace(1, results['d_idx'].max(), 200)
        def damped_cos(d, A, omega, phi, gamma, c):
            return A * np.cos(omega * d + phi) * np.exp(-gamma * d) + c
        ax.plot(d_fit, damped_cos(d_fit, *results['popt_cos_idx']),
                'r-', linewidth=2, label='Damped cosine fit')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Index distance d_idx')
    ax.set_ylabel('Mean attention μ(d)')
    ax.set_title('Kernel vs Index Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top Right: Kernel vs unfolded distance
    ax = axes[0, 1]
    ax.errorbar(results['d_unf'], results['mu_unf'],
                yerr=results['mu_unf_std']/3,
                fmt='g.', alpha=0.5, markersize=5, label='Data')

    d_fit = np.linspace(0.5, results['d_unf'].max(), 200)

    if results['popt_cos_unf'] is not None:
        def damped_cos(d, A, omega, phi, gamma, c):
            return A * np.cos(omega * d + phi) * np.exp(-gamma * d) + c
        ax.plot(d_fit, damped_cos(d_fit, *results['popt_cos_unf']),
                'r-', linewidth=2, label='Damped cosine')

    if results['popt_sine'] is not None:
        def sine_kernel(d, A, alpha, c):
            d_safe = np.maximum(d, 0.01)
            return A * np.sinc(alpha * d_safe) + c
        ax.plot(d_fit, sine_kernel(d_fit, *results['popt_sine']),
                'b--', linewidth=2, label='Sine kernel')

    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Unfolded distance d_unf')
    ax.set_ylabel('Mean attention μ(d)')
    ax.set_title('Kernel vs Unfolded Distance (correct metric)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom Left: Compare coordinate systems
    ax = axes[1, 0]
    # Normalize both for comparison
    mu_idx_norm = (results['mu_idx'] - results['mu_idx'].mean()) / results['mu_idx'].std()
    mu_unf_norm = (results['mu_unf'] - results['mu_unf'].mean()) / results['mu_unf'].std()

    ax.plot(results['d_idx'][:len(mu_unf_norm)], mu_idx_norm[:len(mu_unf_norm)],
            'b-', linewidth=2, label='Index distance', alpha=0.7)
    ax.plot(results['d_unf'], mu_unf_norm,
            'r-', linewidth=2, label='Unfolded distance', alpha=0.7)
    ax.axhline(0, color='k', linestyle=':')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Normalized μ(d)')
    ax.set_title('Index vs Unfolded (normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom Right: Theoretical comparison
    ax = axes[1, 1]
    d = np.linspace(0.5, 50, 200)

    # GUE sine kernel
    sine_gue = np.sinc(d / np.pi)  # sin(πd)/(πd) = sinc(d) in numpy convention

    # Our fitted kernel (unfolded)
    if results['popt_cos_unf'] is not None:
        def damped_cos(d, A, omega, phi, gamma, c):
            return A * np.cos(omega * d + phi) * np.exp(-gamma * d) + c
        our_kernel = damped_cos(d, *results['popt_cos_unf'])
        our_kernel_norm = (our_kernel - our_kernel.mean()) / our_kernel.std()
        ax.plot(d, our_kernel_norm, 'r-', linewidth=2, label='Neural kernel')

    sine_norm = (sine_gue - sine_gue.mean()) / sine_gue.std()
    ax.plot(d, sine_norm, 'b--', linewidth=2, label='GUE sine kernel')

    ax.axhline(0, color='k', linestyle=':')
    ax.set_xlabel('Unfolded distance')
    ax.set_ylabel('Normalized kernel')
    ax.set_title('Neural vs GUE Sine Kernel')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Attention Kernel: Index vs Unfolded Coordinates',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    console.print(f"[green]✅ Saved to {save_path}[/]")
    plt.close()


def main():
    attention_data = extract_attention_with_unfolded_distance(n_samples=500)
    results = analyze_kernel(attention_data)
    visualize_kernels(results)

    # Summary
    console.print("\n")
    table = Table(title="🔬 KERNEL ANALYSIS SUMMARY", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Index (d_idx)", style="yellow")
    table.add_column("Unfolded (d_unf)", style="green")

    if results['popt_cos_idx'] is not None:
        period_idx = 2 * np.pi / results['popt_cos_idx'][1]
    else:
        period_idx = "N/A"

    if results['popt_cos_unf'] is not None:
        period_unf = 2 * np.pi / results['popt_cos_unf'][1]
    else:
        period_unf = "N/A"

    table.add_row("Period (2π/ω)", f"{period_idx:.2f}" if isinstance(period_idx, float) else period_idx,
                  f"{period_unf:.2f}" if isinstance(period_unf, float) else period_unf)

    if results['popt_cos_idx'] is not None:
        table.add_row("Decay (γ)", f"{results['popt_cos_idx'][3]:.6f}",
                      f"{results['popt_cos_unf'][3]:.6f}" if results['popt_cos_unf'] is not None else "N/A")

    console.print(table)

    console.print("\n[bold]═══ KEY INSIGHT ═══[/]")
    console.print("Index distance ≈ Unfolded distance when mean spacing ≈ 1")
    console.print("But unfolded is the CORRECT metric for comparison with sine-kernel")
    console.print("\n[bold green]═══ COMPLETE ═══[/]")


if __name__ == "__main__":
    main()
