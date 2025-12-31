#!/usr/bin/env python3
"""
ROPE V7 RESIDUAL ANALYSIS

Проверяем убил ли RoPE position-bias:
- corr(H1, position) → ~0.0?
- Hard tokens равномерно по окну?
- PC1 < 50%?
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt

from train_rope import WignerRoPEGPT, RoPEConfig
from train_wigner import wigner_logprob

console = Console()


def mine_residuals_rope(
    ckpt_path="out/rope_v7_best.pt",
    n_seq=256,
    quantile=0.95,
    which_hidden="pre",
):
    console.print(Panel.fit(
        "[bold cyan]🌀 RoPE V7 RESIDUAL ANALYSIS[/]\n"
        "Is position-bias DEAD?",
        title="DISCOVERY MODE"
    ))

    device = torch.device("cpu")

    # Load model
    console.print("Loading RoPE v7 model...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = RoPEConfig(**ckpt["config"])
    model = WignerRoPEGPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load data
    console.print("Loading data...")
    val = torch.load("data/val.pt", weights_only=False)[:n_seq]
    X = val[:, :-1].to(device)
    Y = val[:, 1:].to(device)

    bin_centers = ckpt.get("bin_centers", np.load("data/bin_centers.npy"))
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)

    scale_val = bin_centers_t[X]
    target_spacing = bin_centers_t[Y]

    console.print(f"  Sequences: {X.shape[0]}")
    console.print(f"  Seq length: {X.shape[1]}")

    # Forward
    console.print("\nForward pass...")
    out = model(X, targets=None, return_hidden=True, scale_val=scale_val)
    pred = out["pred"]

    h = out["hidden_pre_ln"] if which_hidden == "pre" else out["hidden_post_ln"]
    h.retain_grad()

    B, T, D = h.shape

    # NLL
    console.print("Computing per-token NLL...")
    mu = model.get_mu(pred)
    log_prob = wigner_logprob(target_spacing, mu)
    nll = -log_prob

    # Hard tokens
    thr = torch.quantile(nll.detach(), quantile)
    hard_mask = nll.detach() > thr
    n_hard = int(hard_mask.sum().item())

    console.print(f"  Quantile: {quantile:.2f}")
    console.print(f"  Hard tokens: {n_hard} ({n_hard / (B * T) * 100:.1f}%)")

    # Backward
    console.print("\nBackward pass...")
    hard_loss = nll[hard_mask].mean()
    hard_loss.backward()

    g = h.grad.detach()
    g_vec = g[hard_mask].cpu().numpy()

    # PCA
    console.print("\nPCA on gradients...")
    pca = PCA(n_components=min(10, D))
    comps = pca.fit_transform(g_vec)
    exp = pca.explained_variance_ratio_

    H1 = comps[:, 0]

    # Metadata
    idx = torch.nonzero(hard_mask, as_tuple=False).cpu().numpy()
    batch_idx = idx[:, 0].astype(np.float64)
    t_pos = idx[:, 1].astype(np.float64)
    real_spacing = target_spacing.detach()[hard_mask].cpu().numpy().astype(np.float64)

    # Correlations
    corr_t = np.corrcoef(H1, t_pos)[0, 1] if len(H1) > 1 else 0
    corr_spacing = np.corrcoef(H1, real_spacing)[0, 1] if len(H1) > 1 else 0
    corr_batch = np.corrcoef(H1, batch_idx)[0, 1] if len(H1) > 1 else 0

    # Bias
    pred_mu = mu.detach()[hard_mask].cpu().numpy()
    error = pred_mu - real_spacing
    small_mask = real_spacing < 0.5
    large_mask = real_spacing > 1.5
    bias_small = error[small_mask].mean() if small_mask.sum() > 0 else 0
    bias_large = error[large_mask].mean() if large_mask.sum() > 0 else 0

    # Hard token position distribution
    pos_bins = [0, 50, 100, 150, 200, 255]
    from scipy.stats import binned_statistic
    hard_count, _, _ = binned_statistic(t_pos, np.ones_like(t_pos), statistic='sum', bins=pos_bins)
    hard_pct = hard_count / len(t_pos) * 100

    # Results
    console.print("\n")
    table = Table(title="🌀 RoPE V7 DISCOVERY RESULTS")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation")

    pc1_pct = exp[0] * 100
    pc2_pct = exp[1] * 100 if len(exp) > 1 else 0

    def pct_verdict(p):
        if p > 60:
            return "[red]High[/]"
        elif p > 40:
            return "[yellow]Moderate[/]"
        else:
            return "[green]Low (good!)[/]"

    def corr_verdict(c):
        ac = abs(c)
        if ac > 0.3:
            return "[red]Strong[/]"
        elif ac > 0.2:
            return "[yellow]Medium[/]"
        elif ac > 0.1:
            return "[cyan]Weak[/]"
        else:
            return "[green]~Zero (goal!)[/]"

    table.add_row("PC1 explained", f"{pc1_pct:.1f}%", pct_verdict(pc1_pct))
    table.add_row("PC2 explained", f"{pc2_pct:.1f}%", "")
    table.add_row("", "", "")
    table.add_row("corr(H1, position)", f"{corr_t:.3f}", corr_verdict(corr_t))
    table.add_row("corr(H1, spacing)", f"{corr_spacing:.3f}", corr_verdict(corr_spacing))
    table.add_row("corr(H1, batch)", f"{corr_batch:.3f}", corr_verdict(corr_batch))
    table.add_row("", "", "")
    table.add_row("Bias (s < 0.5)", f"{bias_small:+.4f}", "")
    table.add_row("Bias (s > 1.5)", f"{bias_large:+.4f}", "")

    console.print(table)

    # Hard token distribution
    console.print("\n[bold]Hard Token Distribution by Position:[/]")
    uniform_pct = 100 / (len(pos_bins) - 1)
    for i in range(len(pos_bins) - 1):
        bar = "█" * int(hard_pct[i] / 2)
        is_uniform = "✓" if abs(hard_pct[i] - uniform_pct) < 5 else "⚠" if abs(hard_pct[i] - uniform_pct) < 10 else "✗"
        console.print(f"  [{pos_bins[i]:3d}-{pos_bins[i+1]:3d}): {bar} {hard_pct[i]:.1f}% {is_uniform}")

    # Evolution comparison
    console.print("\n")
    comp = Table(title="📊 EVOLUTION: v4 → v5 → v6 → v7")
    comp.add_column("Metric", style="bold")
    comp.add_column("v4", justify="center")
    comp.add_column("v5", justify="center")
    comp.add_column("v6", justify="center")
    comp.add_column("v7 (RoPE)", justify="center")

    comp.add_row("corr(spacing)", "0.75", "-0.86", "-0.33", f"{corr_spacing:.2f}")
    comp.add_row("corr(position)", "0.05", "0.05", "-0.36", f"{corr_t:.2f}")
    comp.add_row("PC1", "54%", "78%", "98%", f"{pc1_pct:.0f}%")

    console.print(comp)

    # Interpretation
    console.print("\n")
    if abs(corr_t) < 0.15 and abs(corr_spacing) < 0.25:
        interp = (
            "[bold green]🎉 BOTH BIASES ELIMINATED![/]\n"
            "RoPE killed position-bias, Wigner killed scale-bias.\n"
            "Residuals now show REAL physics!"
        )
    elif abs(corr_t) < 0.2:
        interp = (
            "[bold yellow]⚠️ POSITION-BIAS REDUCED![/]\n"
            f"corr(position) from -0.36 to {corr_t:.3f}\n"
            "RoPE helped but scale correlation remains."
        )
    else:
        interp = (
            "[bold red]🔴 POSITION-BIAS PERSISTS[/]\n"
            f"corr(position) = {corr_t:.3f}\n"
            "RoPE didn't fully fix it."
        )

    console.print(Panel.fit(interp, title="💡 DIAGNOSIS", border_style="cyan"))

    # Save
    np.savez(
        "reports/residual_rope_v7.npz",
        H1=H1,
        H2=comps[:, 1] if comps.shape[1] > 1 else np.zeros_like(H1),
        pos=t_pos,
        batch=batch_idx,
        spacing=real_spacing,
        explained=exp,
    )
    console.print(f"\n[green]Saved: reports/residual_rope_v7.npz[/]")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.bar(range(1, len(exp) + 1), exp * 100)
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('PC')
    ax.set_ylabel('Variance %')
    ax.set_title('PCA (RoPE v7)')

    ax = axes[0, 1]
    ax.scatter(t_pos, H1, alpha=0.3, s=10, c='blue')
    ax.set_xlabel('Position')
    ax.set_ylabel('H1')
    ax.set_title(f'H1 vs Position (corr={corr_t:.3f})')

    ax = axes[1, 0]
    ax.scatter(real_spacing, H1, alpha=0.3, s=10, c='green')
    ax.set_xlabel('Spacing')
    ax.set_ylabel('H1')
    ax.set_title(f'H1 vs Spacing (corr={corr_spacing:.3f})')

    ax = axes[1, 1]
    ax.bar([(pos_bins[i] + pos_bins[i+1])/2 for i in range(len(pos_bins)-1)],
           hard_pct, width=40, alpha=0.7)
    ax.axhline(y=uniform_pct, color='r', linestyle='--', label=f'Uniform ({uniform_pct:.1f}%)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Hard tokens %')
    ax.set_title('Hard Token Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig('reports/discovery_rope_v7.png', dpi=150, bbox_inches='tight')
    console.print(f"[green]Plot saved: reports/discovery_rope_v7.png[/]")

    return {
        "pc1_explained": pc1_pct,
        "corr_spacing": corr_spacing,
        "corr_position": corr_t,
        "bias_small": bias_small,
        "bias_large": bias_large,
    }


if __name__ == "__main__":
    from pathlib import Path
    Path("reports").mkdir(exist_ok=True)
    mine_residuals_rope()
