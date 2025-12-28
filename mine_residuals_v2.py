#!/usr/bin/env python3
"""
DISCOVERY MODE STEP 1: Mine Residuals

This script finds WHERE the model fails and WHAT direction it wanted to go.

Method:
1. Find "hard" tokens (high loss, top 5%)
2. Compute gradient of loss w.r.t. hidden state
3. PCA on gradients → find dominant "blindness direction" H1
4. Analyze: what does H1 correlate with?

If PC1 > 25% → there's a systematic blindness → candidate for new variable
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt

from train_memory_bank import MemoryBankGPT, MemoryBankConfig

console = Console()


def mine_residuals_v2(
    ckpt_path="out/memory_bank_best.pt",
    n_seq=256,
    quantile=0.95,
    which_hidden="pre",  # "pre" or "post" ln_f
):
    console.print(Panel.fit(
        "[bold cyan]🔍 DISCOVERY MODE: Mining Residuals[/]\n"
        "Finding systematic blindness in the model",
        title="STEP 1: DETECT"
    ))

    device = torch.device("cpu")  # CPU for gradient analysis

    # Load model
    console.print("Loading model...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Handle config (might be object or dict)
    if hasattr(ckpt["config"], '__dict__'):
        config = MemoryBankConfig(**ckpt["config"].__dict__)
    else:
        config = MemoryBankConfig(**ckpt["config"])

    model = MemoryBankGPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Note: DON'T freeze weights - we need gradient flow through hidden
    # We just won't call optimizer.step()

    # Load validation data
    console.print("Loading data...")
    val = torch.load("data/val.pt", weights_only=False)[:n_seq]
    X = val[:, :-1].to(device)
    Y = val[:, 1:].to(device)

    console.print(f"  Sequences: {X.shape[0]}")
    console.print(f"  Seq length: {X.shape[1]}")
    console.print(f"  Total tokens: {X.shape[0] * X.shape[1]}")

    # Forward with hidden states
    console.print("\nForward pass with hidden states...")
    out = model(X, targets=None, return_hidden=True)
    logits = out["logits"]  # (B, T, V)

    h = out["hidden_pre_ln"] if which_hidden == "pre" else out["hidden_post_ln"]
    h.retain_grad()  # Key: allow gradient on intermediate tensor

    B, T, V = logits.shape
    D = h.shape[-1]

    console.print(f"  Hidden shape: {h.shape}")
    console.print(f"  Using: hidden_{which_hidden}_ln")

    # Compute per-token NLL
    console.print("\nComputing per-token loss...")
    nll = F.cross_entropy(
        logits.reshape(B * T, V),
        Y.reshape(B * T),
        reduction="none",
    ).reshape(B, T)

    # Find hard tokens (top quantile loss)
    thr = torch.quantile(nll.detach(), quantile)
    hard_mask = nll.detach() > thr
    n_hard = int(hard_mask.sum().item())

    console.print(f"  Quantile: {quantile:.2f}")
    console.print(f"  Threshold: {thr.item():.4f}")
    console.print(f"  Hard tokens: {n_hard} ({n_hard / (B * T) * 100:.1f}%)")

    # Backward on hard tokens only
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

    # Extract H1 (dominant blindness direction)
    H1 = comps[:, 0]

    # Get metadata for correlation analysis
    idx = torch.nonzero(hard_mask, as_tuple=False).cpu().numpy()
    batch_idx = idx[:, 0].astype(np.float64)
    t_pos = idx[:, 1].astype(np.float64)  # Position in window
    y_true = Y.detach()[hard_mask].cpu().numpy().astype(np.float64)

    # Correlations
    corr_t = np.corrcoef(H1, t_pos)[0, 1] if len(H1) > 1 else 0
    corr_y = np.corrcoef(H1, y_true)[0, 1] if len(H1) > 1 else 0
    corr_batch = np.corrcoef(H1, batch_idx)[0, 1] if len(H1) > 1 else 0

    # Results table
    console.print("\n")
    table = Table(title="🔬 DISCOVERY RESULTS (Gradient PCA)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation")

    # Variance explained
    pc1_pct = exp[0] * 100
    pc2_pct = exp[1] * 100 if len(exp) > 1 else 0
    pc3_pct = exp[2] * 100 if len(exp) > 2 else 0

    verdict_pc1 = "[green]Strong signal![/]" if pc1_pct > 25 else "[yellow]Weak[/]" if pc1_pct > 15 else "[red]Noise[/]"

    table.add_row("PC1 explained", f"{pc1_pct:.1f}%", verdict_pc1)
    table.add_row("PC2 explained", f"{pc2_pct:.1f}%", "")
    table.add_row("PC3 explained", f"{pc3_pct:.1f}%", "")
    table.add_row("", "", "")

    # Correlations
    def corr_verdict(c):
        ac = abs(c)
        if ac > 0.5:
            return "[green]Strong[/]"
        elif ac > 0.3:
            return "[yellow]Medium[/]"
        else:
            return "[dim]Weak[/]"

    table.add_row("corr(H1, position)", f"{corr_t:.3f}", corr_verdict(corr_t))
    table.add_row("corr(H1, token-id)", f"{corr_y:.3f}", corr_verdict(corr_y))
    table.add_row("corr(H1, batch)", f"{corr_batch:.3f}", corr_verdict(corr_batch))

    console.print(table)

    # Interpretation
    console.print("\n")
    if pc1_pct > 25:
        if abs(corr_t) > 0.3:
            interp = (
                "[bold yellow]FINDING: Position-dependent blindness![/]\n"
                "Model struggles more at certain positions in window.\n"
                "Candidate: Add positional phase/mode variable."
            )
        elif abs(corr_y) > 0.3:
            interp = (
                "[bold yellow]FINDING: Token-dependent blindness![/]\n"
                "Model struggles with certain spacing values.\n"
                "Candidate: Add calibration/scaling variable."
            )
        else:
            interp = (
                "[bold green]FINDING: Systematic blindness detected![/]\n"
                "H1 doesn't correlate with obvious variables.\n"
                "This could be a NEW hidden structure!"
            )
    else:
        interp = (
            "[dim]No strong systematic blindness detected.[/]\n"
            "Errors appear random, not structured.\n"
            "Model may already capture main patterns."
        )

    console.print(Panel.fit(interp, title="💡 INTERPRETATION", border_style="cyan"))

    # Save results
    np.savez(
        "reports/residual_gradients_v2.npz",
        H1=H1,
        H2=comps[:, 1] if comps.shape[1] > 1 else np.zeros_like(H1),
        H3=comps[:, 2] if comps.shape[1] > 2 else np.zeros_like(H1),
        pos=t_pos,
        batch=batch_idx,
        y_true=y_true,
        explained=exp,
        pca_components=pca.components_,
    )
    console.print(f"\n[green]Saved: reports/residual_gradients_v2.npz[/]")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Variance explained
    ax = axes[0, 0]
    ax.bar(range(1, len(exp) + 1), exp * 100)
    ax.axhline(y=25, color='r', linestyle='--', alpha=0.5, label='25% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA on Gradient Directions')
    ax.legend()

    # 2. H1 vs position
    ax = axes[0, 1]
    ax.scatter(t_pos, H1, alpha=0.3, s=10)
    ax.set_xlabel('Position in window')
    ax.set_ylabel('H1 (dominant gradient direction)')
    ax.set_title(f'H1 vs Position (corr={corr_t:.3f})')

    # 3. H1 vs token value
    ax = axes[1, 0]
    ax.scatter(y_true, H1, alpha=0.3, s=10)
    ax.set_xlabel('True token (spacing bin)')
    ax.set_ylabel('H1')
    ax.set_title(f'H1 vs Token (corr={corr_y:.3f})')

    # 4. H1 distribution
    ax = axes[1, 1]
    ax.hist(H1, bins=50, density=True, alpha=0.7)
    ax.set_xlabel('H1 value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of H1')

    plt.tight_layout()
    plt.savefig('reports/discovery_residuals.png', dpi=150, bbox_inches='tight')
    console.print(f"[green]Plot saved: reports/discovery_residuals.png[/]")

    return {
        "pc1_explained": pc1_pct,
        "corr_position": corr_t,
        "corr_token": corr_y,
        "n_hard": n_hard,
    }


if __name__ == "__main__":
    mine_residuals_v2()
