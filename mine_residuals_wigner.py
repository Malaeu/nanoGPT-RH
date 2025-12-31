#!/usr/bin/env python3
"""
WIGNER V6 RESIDUAL ANALYSIS

Проверяем:
1. Regression to the mean УБИТА?
2. Bias на хвостах → 0?
3. PC1 упал < 30%?
4. corr(H1, spacing) → 0?

Если да → residuals покажут НАСТОЯЩУЮ физику!
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
import math

from train_wigner import WignerGPT, WignerConfig, wigner_logprob

console = Console()


def mine_residuals_wigner(
    ckpt_path="out/wigner_v6_best.pt",
    n_seq=256,
    quantile=0.95,
    which_hidden="pre",
):
    console.print(Panel.fit(
        "[bold cyan]🌟 WIGNER V6 RESIDUAL ANALYSIS[/]\n"
        "Physics-informed likelihood: Is regression-to-mean DEAD?",
        title="DISCOVERY MODE"
    ))

    device = torch.device("cpu")

    # Load model
    console.print("Loading Wigner v6 model...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config_dict = ckpt["config"]
    config = WignerConfig(**config_dict)

    model = WignerGPT(config).to(device)
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

    # Compute per-token NLL
    console.print("Computing per-token NLL...")
    mu = model.get_mu(pred)
    log_prob = wigner_logprob(target_spacing, mu)
    nll = -log_prob

    # Hard tokens
    thr = torch.quantile(nll.detach(), quantile)
    hard_mask = nll.detach() > thr
    n_hard = int(hard_mask.sum().item())

    console.print(f"  Quantile: {quantile:.2f}")
    console.print(f"  NLL threshold: {thr.item():.4f}")
    console.print(f"  Hard tokens: {n_hard} ({n_hard / (B * T) * 100:.1f}%)")

    # Backward
    console.print("\nBackward pass...")
    hard_loss = nll[hard_mask].mean()
    hard_loss.backward()

    g = h.grad.detach()
    g_vec = g[hard_mask].cpu().numpy()

    console.print(f"  Gradient vectors: {g_vec.shape}")

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

    # Bias analysis
    pred_mu = mu.detach()[hard_mask].cpu().numpy()
    error = pred_mu - real_spacing

    small_mask = real_spacing < 0.5
    large_mask = real_spacing > 1.5

    bias_small = error[small_mask].mean() if small_mask.sum() > 0 else 0
    bias_large = error[large_mask].mean() if large_mask.sum() > 0 else 0

    # Results
    console.print("\n")
    table = Table(title="🌟 WIGNER V6 DISCOVERY RESULTS")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation")

    pc1_pct = exp[0] * 100
    pc2_pct = exp[1] * 100 if len(exp) > 1 else 0

    def pct_verdict(p):
        if p > 40:
            return "[red]High[/]"
        elif p > 25:
            return "[yellow]Moderate[/]"
        else:
            return "[green]Low (good!)[/]"

    def corr_verdict(c):
        ac = abs(c)
        if ac > 0.5:
            return "[red]Strong[/]"
        elif ac > 0.3:
            return "[yellow]Medium[/]"
        elif ac > 0.2:
            return "[cyan]Weak[/]"
        else:
            return "[green]~Zero (goal!)[/]"

    def bias_verdict(b):
        ab = abs(b)
        if ab > 0.2:
            return "[red]Large[/]"
        elif ab > 0.1:
            return "[yellow]Moderate[/]"
        else:
            return "[green]Small (goal!)[/]"

    table.add_row("PC1 explained", f"{pc1_pct:.1f}%", pct_verdict(pc1_pct))
    table.add_row("PC2 explained", f"{pc2_pct:.1f}%", "")
    table.add_row("", "", "")
    table.add_row("corr(H1, position)", f"{corr_t:.3f}", corr_verdict(corr_t))
    table.add_row("corr(H1, spacing)", f"{corr_spacing:.3f}", corr_verdict(corr_spacing))
    table.add_row("corr(H1, batch)", f"{corr_batch:.3f}", corr_verdict(corr_batch))
    table.add_row("", "", "")
    table.add_row("Bias (s < 0.5)", f"{bias_small:+.4f}", bias_verdict(bias_small))
    table.add_row("Bias (s > 1.5)", f"{bias_large:+.4f}", bias_verdict(bias_large))

    console.print(table)

    # Comparison
    console.print("\n")
    comp = Table(title="📊 EVOLUTION: v4 → v5 → v6")
    comp.add_column("Metric", style="bold")
    comp.add_column("v4 (binned)", justify="center")
    comp.add_column("v5 (logistic)", justify="center")
    comp.add_column("v6 (Wigner)", justify="center")

    # Historical values
    v4_corr = 0.753
    v5_corr = -0.860
    v4_bias_s = 0.38
    v5_bias_s = 0.38  # approximately
    v4_bias_l = -0.27
    v5_bias_l = -0.27

    comp.add_row("corr(H1, spacing)", f"{v4_corr:.3f}", f"{v5_corr:.3f}", f"{corr_spacing:.3f}")
    comp.add_row("Bias (small s)", f"+{v4_bias_s:.2f}", f"+{v5_bias_s:.2f}", f"{bias_small:+.2f}")
    comp.add_row("Bias (large s)", f"{v4_bias_l:.2f}", f"{v5_bias_l:.2f}", f"{bias_large:+.2f}")

    console.print(comp)

    # Interpretation
    console.print("\n")
    if abs(corr_spacing) < 0.3 and abs(bias_small) < 0.15 and abs(bias_large) < 0.15:
        interp = (
            "[bold green]🎉 REGRESSION TO MEAN ELIMINATED![/]\n"
            "Physics-informed likelihood worked!\n"
            "Residuals now show REAL structure (if any)."
        )
    elif abs(bias_small) < 0.2 and abs(bias_large) < 0.2:
        interp = (
            "[bold yellow]⚠️ BIAS REDUCED![/]\n"
            f"Small s bias: {v5_bias_s:.2f} → {bias_small:+.2f}\n"
            f"Large s bias: {v5_bias_l:.2f} → {bias_large:+.2f}\n"
            "But correlation still present. May need more training."
        )
    else:
        interp = (
            "[bold red]🔴 BIAS PERSISTS[/]\n"
            "Wigner likelihood didn't fix regression-to-mean.\n"
            "Check model architecture or training."
        )

    console.print(Panel.fit(interp, title="💡 DIAGNOSIS", border_style="cyan"))

    # Save
    np.savez(
        "reports/residual_wigner_v6.npz",
        H1=H1,
        H2=comps[:, 1] if comps.shape[1] > 1 else np.zeros_like(H1),
        pos=t_pos,
        batch=batch_idx,
        spacing=real_spacing,
        explained=exp,
        pca_components=pca.components_,
        bias_small=bias_small,
        bias_large=bias_large,
    )
    console.print(f"\n[green]Saved: reports/residual_wigner_v6.npz[/]")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. PCA
    ax = axes[0, 0]
    ax.bar(range(1, len(exp) + 1), exp * 100)
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='30% target')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA on Gradients (Wigner v6)')
    ax.legend()

    # 2. H1 vs spacing
    ax = axes[0, 1]
    ax.scatter(real_spacing, H1, alpha=0.3, s=10, c='purple')
    ax.set_xlabel('Target spacing')
    ax.set_ylabel('H1')
    ax.set_title(f'H1 vs Spacing (corr={corr_spacing:.3f})')

    # 3. Prediction error by bin
    ax = axes[1, 0]
    bins = [0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
    bin_centers_plot = [0.15, 0.45, 0.8, 1.25, 1.75, 2.5]
    from scipy.stats import binned_statistic
    mean_err, _, _ = binned_statistic(real_spacing, error, statistic='mean', bins=bins)
    ax.bar(bin_centers_plot, mean_err, width=0.2, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='±0.1 target')
    ax.axhline(y=-0.1, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('Target spacing bin')
    ax.set_ylabel('Mean prediction error')
    ax.set_title('Bias by Spacing (goal: flat at 0)')
    ax.legend()

    # 4. Error distribution
    ax = axes[1, 1]
    ax.hist(error, bins=50, density=True, alpha=0.7, color='green')
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel('Prediction error (pred - target)')
    ax.set_ylabel('Density')
    ax.set_title(f'Error Distribution (mean={error.mean():.4f})')

    plt.tight_layout()
    plt.savefig('reports/discovery_wigner_v6.png', dpi=150, bbox_inches='tight')
    console.print(f"[green]Plot saved: reports/discovery_wigner_v6.png[/]")

    return {
        "pc1_explained": pc1_pct,
        "corr_spacing": corr_spacing,
        "bias_small": bias_small,
        "bias_large": bias_large,
    }


if __name__ == "__main__":
    from pathlib import Path
    Path("reports").mkdir(exist_ok=True)
    mine_residuals_wigner()
