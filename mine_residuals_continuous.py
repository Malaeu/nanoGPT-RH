#!/usr/bin/env python3
"""
DISCOVERY MODE for Continuous v5: Verify Scale Blindness is DEAD

Адаптация mine_residuals для continuous output модели.
Теперь loss = NLL вместо cross-entropy, но PCA анализ тот же.

Ожидания:
- corr(H1, spacing) → ~0.0-0.2 (не 0.753!)
- PC1 → < 20% (остаётся только настоящая физика)
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt

from train_continuous import ContinuousGPT, ContinuousConfig, MixLogisticLoss

console = Console()


def mine_residuals_continuous(
    ckpt_path="out/continuous_v5_best.pt",
    n_seq=256,
    quantile=0.95,
    which_hidden="pre",
):
    console.print(Panel.fit(
        "[bold cyan]🔍 CONTINUOUS V5 RESIDUAL ANALYSIS[/]\n"
        "Verifying scale-bias is DEAD with mixture output",
        title="DISCOVERY MODE"
    ))

    device = torch.device("cpu")  # CPU for gradient analysis

    # Load model
    console.print("Loading continuous v5 model...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Handle config
    config_dict = ckpt["config"]
    if isinstance(config_dict, dict):
        config = ContinuousConfig(**config_dict)
    else:
        config = ContinuousConfig(**config_dict.__dict__)

    model = ContinuousGPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load data
    console.print("Loading data...")
    val = torch.load("data/val.pt", weights_only=False)[:n_seq]
    X = val[:, :-1].to(device)
    Y = val[:, 1:].to(device)

    # Load bin_centers
    bin_centers = ckpt.get("bin_centers", np.load("data/bin_centers.npy"))
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)

    # Convert to float spacing values
    scale_val = bin_centers_t[X]  # (B, T) for conditioning
    target_spacing = bin_centers_t[Y]  # (B, T) float targets

    console.print(f"  Sequences: {X.shape[0]}")
    console.print(f"  Seq length: {X.shape[1]}")
    console.print(f"  Total tokens: {X.shape[0] * X.shape[1]}")

    # Forward with hidden states
    console.print("\nForward pass with hidden states...")
    out = model(X, targets=None, return_hidden=True, scale_val=scale_val)
    pred = out["pred"]  # (B, T, 3*K)

    h = out["hidden_pre_ln"] if which_hidden == "pre" else out["hidden_post_ln"]
    h.retain_grad()

    B, T, D = h.shape
    K = config.num_mixtures

    console.print(f"  Hidden shape: {h.shape}")
    console.print(f"  Mixture components: {K}")
    console.print(f"  Using: hidden_{which_hidden}_ln")

    # Compute per-token NLL
    console.print("\nComputing per-token NLL...")
    loss_fn = MixLogisticLoss(num_mixtures=K, discretized=False)

    # Per-token loss
    log_weights = pred[..., :K]
    means = pred[..., K:2*K]
    log_scales = pred[..., 2*K:3*K].clamp(min=-7.0, max=7.0)
    log_weights = F.log_softmax(log_weights, dim=-1)

    # Log prob per token
    from train_continuous import mix_logistic_logprob
    log_prob = mix_logistic_logprob(target_spacing, log_weights, means, log_scales)
    nll = -log_prob  # (B, T)

    # Find hard tokens (top quantile loss)
    thr = torch.quantile(nll.detach(), quantile)
    hard_mask = nll.detach() > thr
    n_hard = int(hard_mask.sum().item())

    console.print(f"  Quantile: {quantile:.2f}")
    console.print(f"  NLL threshold: {thr.item():.4f}")
    console.print(f"  Hard tokens: {n_hard} ({n_hard / (B * T) * 100:.1f}%)")

    # Backward on hard tokens
    console.print("\nBackward pass on hard tokens...")
    hard_loss = nll[hard_mask].mean()
    hard_loss.backward()

    # Get gradients
    g = h.grad.detach()  # (B, T, D)
    g_vec = g[hard_mask].cpu().numpy()  # (N_hard, D)

    console.print(f"  Gradient vectors: {g_vec.shape}")

    # PCA on gradients
    console.print("\nPCA on gradient directions...")
    pca = PCA(n_components=min(10, D))
    comps = pca.fit_transform(g_vec)
    exp = pca.explained_variance_ratio_

    # Extract H1
    H1 = comps[:, 0]

    # Get metadata for correlation
    idx = torch.nonzero(hard_mask, as_tuple=False).cpu().numpy()
    batch_idx = idx[:, 0].astype(np.float64)
    t_pos = idx[:, 1].astype(np.float64)

    # Real spacing values for hard tokens
    real_spacing = target_spacing.detach()[hard_mask].cpu().numpy().astype(np.float64)

    # Correlations
    corr_t = np.corrcoef(H1, t_pos)[0, 1] if len(H1) > 1 else 0
    corr_spacing = np.corrcoef(H1, real_spacing)[0, 1] if len(H1) > 1 else 0
    corr_batch = np.corrcoef(H1, batch_idx)[0, 1] if len(H1) > 1 else 0

    # Results table
    console.print("\n")
    table = Table(title="🔬 CONTINUOUS V5 DISCOVERY RESULTS")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation")

    # Variance explained
    pc1_pct = exp[0] * 100
    pc2_pct = exp[1] * 100 if len(exp) > 1 else 0
    pc3_pct = exp[2] * 100 if len(exp) > 2 else 0

    def pct_verdict(p):
        if p > 25:
            return "[red]High (systematic)[/]"
        elif p > 15:
            return "[yellow]Moderate[/]"
        else:
            return "[green]Low (good!)[/]"

    table.add_row("PC1 explained", f"{pc1_pct:.1f}%", pct_verdict(pc1_pct))
    table.add_row("PC2 explained", f"{pc2_pct:.1f}%", "")
    table.add_row("PC3 explained", f"{pc3_pct:.1f}%", "")
    table.add_row("", "", "")

    # Correlations
    def corr_verdict(c):
        ac = abs(c)
        if ac > 0.5:
            return "[red]Strong (bad!)[/]"
        elif ac > 0.3:
            return "[yellow]Medium[/]"
        elif ac > 0.2:
            return "[cyan]Weak[/]"
        else:
            return "[green]~Zero (goal!)[/]"

    table.add_row("corr(H1, position)", f"{corr_t:.3f}", corr_verdict(corr_t))
    table.add_row("corr(H1, spacing)", f"{corr_spacing:.3f}", corr_verdict(corr_spacing))
    table.add_row("corr(H1, batch)", f"{corr_batch:.3f}", corr_verdict(corr_batch))

    console.print(table)

    # Comparison with v4
    console.print("\n")
    comparison = Table(title="📊 COMPARISON: v4 (binned) vs v5 (continuous)")
    comparison.add_column("Metric", style="bold")
    comparison.add_column("v4 (binned)", justify="center")
    comparison.add_column("v5 (continuous)", justify="center")
    comparison.add_column("Change")

    v4_corr = 0.753  # From v4 results
    change = abs(corr_spacing) - abs(v4_corr)
    change_str = f"[green]{change:+.3f}[/]" if change < 0 else f"[red]{change:+.3f}[/]"

    comparison.add_row("corr(H1, spacing)", f"{v4_corr:.3f}", f"{corr_spacing:.3f}", change_str)

    console.print(comparison)

    # Interpretation
    console.print("\n")
    if abs(corr_spacing) < 0.2:
        interp = (
            "[bold green]🎉 SCALE BLINDNESS ELIMINATED![/]\n"
            "corr(H1, spacing) near zero → model no longer struggles with scale.\n"
            "Continuous output killed the discretization bottleneck!"
        )
    elif abs(corr_spacing) < 0.4:
        interp = (
            "[bold yellow]⚠️ SIGNIFICANT IMPROVEMENT![/]\n"
            f"corr dropped from {v4_corr:.3f} to {corr_spacing:.3f}.\n"
            "Scale-bias weakened but not eliminated. Try more training."
        )
    else:
        interp = (
            "[bold red]🔴 SCALE-BIAS PERSISTS[/]\n"
            f"corr = {corr_spacing:.3f} still strong.\n"
            "Continuous output alone didn't fix it. Investigate further."
        )

    console.print(Panel.fit(interp, title="💡 DIAGNOSIS", border_style="cyan"))

    # Save results
    np.savez(
        "reports/residual_continuous_v5.npz",
        H1=H1,
        H2=comps[:, 1] if comps.shape[1] > 1 else np.zeros_like(H1),
        H3=comps[:, 2] if comps.shape[1] > 2 else np.zeros_like(H1),
        pos=t_pos,
        batch=batch_idx,
        spacing=real_spacing,
        explained=exp,
        pca_components=pca.components_,
    )
    console.print(f"\n[green]Saved: reports/residual_continuous_v5.npz[/]")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Variance explained
    ax = axes[0, 0]
    ax.bar(range(1, len(exp) + 1), exp * 100)
    ax.axhline(y=25, color='r', linestyle='--', alpha=0.5, label='25% threshold')
    ax.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='15% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA on Gradient Directions (v5 Continuous)')
    ax.legend()

    # 2. H1 vs position
    ax = axes[0, 1]
    ax.scatter(t_pos, H1, alpha=0.3, s=10)
    ax.set_xlabel('Position in window')
    ax.set_ylabel('H1 (dominant gradient direction)')
    ax.set_title(f'H1 vs Position (corr={corr_t:.3f})')

    # 3. H1 vs REAL SPACING (the key plot!)
    ax = axes[1, 0]
    ax.scatter(real_spacing, H1, alpha=0.3, s=10, c='green')
    ax.set_xlabel('Target spacing (float)')
    ax.set_ylabel('H1')
    ax.set_title(f'H1 vs Spacing (corr={corr_spacing:.3f}) ← KEY METRIC')

    # Add trend line
    if len(real_spacing) > 10:
        z = np.polyfit(real_spacing, H1, 1)
        p = np.poly1d(z)
        x_line = np.linspace(real_spacing.min(), real_spacing.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend (slope={z[0]:.3f})')
        ax.legend()

    # 4. H1 distribution
    ax = axes[1, 1]
    ax.hist(H1, bins=50, density=True, alpha=0.7, color='purple')
    ax.set_xlabel('H1 value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of H1')

    plt.tight_layout()
    plt.savefig('reports/discovery_continuous_v5.png', dpi=150, bbox_inches='tight')
    console.print(f"[green]Plot saved: reports/discovery_continuous_v5.png[/]")

    return {
        "pc1_explained": pc1_pct,
        "corr_position": corr_t,
        "corr_spacing": corr_spacing,
        "n_hard": n_hard,
        "improvement_from_v4": v4_corr - abs(corr_spacing),
    }


if __name__ == "__main__":
    from pathlib import Path
    Path("reports").mkdir(exist_ok=True)
    mine_residuals_continuous()
