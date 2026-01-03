#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    🔬 100M ZEROS EXPERIMENTS SUITE
═══════════════════════════════════════════════════════════════════════════════

Running all analysis experiments on the 100M zeros model:
1. Kernel Extraction - attention patterns as function of distance
2. PySR Symbolic Regression - find analytic formula for kernel
3. SFF Analysis - spectral form factor comparison
4. Attention Visualization - heatmaps of learned patterns
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import warnings
warnings.filterwarnings('ignore')

from model.gpt import SpacingGPT, GPTConfig

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
CKPT_PATH = Path("out/spacing_100M/best.pt")
DATA_DIR = Path("data/zeros_100M")
ZEROS_PATH = Path("data/raw/zeros_100M.txt")
OUTPUT_DIR = Path("reports/100M")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def load_model():
    """Load trained 100M model."""
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = SpacingGPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def load_meta():
    """Load metadata (bin edges, etc)."""
    return torch.load(DATA_DIR / "meta.pt", weights_only=False)


def unfold_val(gamma):
    """Unfolding: γ → u(γ) with mean spacing = 1."""
    gamma = np.asarray(gamma)
    return (gamma / (2 * np.pi)) * np.log(gamma / (2 * np.pi * np.e))


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: KERNEL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def extract_attention_logits(model, x, config):
    """
    Extract RAW attention logits (before softmax) for each layer/head.
    Returns dict: layer_idx -> (B, n_head, T, T) logits tensor
    """
    device = x.device
    B, T = x.size()

    # Embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(x)
    pos_emb = model.transformer.wpe(pos)
    h = model.transformer.drop(tok_emb + pos_emb)

    all_logits = {}

    for layer_idx, block in enumerate(model.transformer.h):
        # Layer norm before attention
        ln_out = block.ln_1(h)

        # Extract Q, K, V manually
        B, T, C = ln_out.size()
        qkv = block.attn.c_attn(ln_out)
        q, k, v = qkv.split(C, dim=2)

        # Reshape for multi-head
        n_head = config.n_head
        head_dim = C // n_head

        q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)

        # RAW ATTENTION LOGITS (interaction energy) - NO SOFTMAX!
        scale = 1.0 / math.sqrt(head_dim)
        att_logits = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)

        all_logits[layer_idx] = att_logits

        # Continue forward pass
        h = block(h)

    return all_logits


def compute_distance_statistics(att_logits: torch.Tensor, max_dist: int = 128):
    """Compute mean attention logit as function of distance d = |i-j|."""
    B, n_head, T, _ = att_logits.shape
    results = {}

    for head in range(n_head):
        head_logits = att_logits[:, head, :, :]  # (B, T, T)

        dists = []
        means = []
        stds = []

        for d in range(1, min(max_dist, T)):
            diag = torch.diagonal(head_logits, offset=-d, dim1=-2, dim2=-1)
            values = diag.flatten().cpu().numpy()

            if len(values) > 0:
                dists.append(d)
                means.append(np.mean(values))
                stds.append(np.std(values))

        results[head] = (np.array(dists), np.array(means), np.array(stds))

    return results


def find_physics_head(all_stats: dict, config):
    """Find the head with most interesting structure (oscillations, etc)."""
    scores = []

    for layer_idx, layer_stats in all_stats.items():
        for head_idx, (dists, means, stds) in layer_stats.items():
            if len(means) < 10:
                continue

            # Non-monotonicity
            diff = np.diff(means)
            sign_changes = np.sum(np.abs(np.diff(np.sign(diff))) > 0)

            # Variance
            variance = np.var(means)

            # Level repulsion
            repulsion = 0
            if len(means) > 5:
                if means[0] < means[3]:
                    repulsion = means[3] - means[0]

            score = sign_changes * 0.5 + variance * 100 + repulsion * 10

            scores.append({
                "layer": layer_idx,
                "head": head_idx,
                "sign_changes": sign_changes,
                "variance": variance,
                "repulsion": repulsion,
                "score": score,
            })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def run_kernel_extraction():
    """Extract and analyze attention kernels from 100M model."""
    console.print(Panel.fit(
        "[bold cyan]EXPERIMENT 1: KERNEL EXTRACTION[/]\n"
        "Analyzing attention patterns as f(distance)",
        title="🔬"
    ))

    model, config = load_model()
    console.print(f"[green]Model: {config.n_layer}L/{config.n_head}H/{config.n_embd}E[/]")

    # Load validation data
    val_data = torch.load(DATA_DIR / "val.pt", weights_only=False)
    console.print(f"[green]Val data: {val_data.shape}[/]")

    # Take batch for analysis
    batch_size = 128
    batch = val_data[:batch_size].to(DEVICE)

    # Extract attention logits
    console.print("[cyan]Extracting attention logits...[/]")
    with torch.no_grad():
        all_logits = extract_attention_logits(model, batch, config)

    # Compute distance statistics
    console.print("[cyan]Computing distance statistics...[/]")
    all_stats = {}
    for layer_idx, logits in all_logits.items():
        all_stats[layer_idx] = compute_distance_statistics(logits)

    # Find physics head
    head_scores = find_physics_head(all_stats, config)

    # Display results
    table = Table(title="Top Physics Head Candidates (100M Model)")
    table.add_column("Rank", style="cyan")
    table.add_column("Layer", style="green")
    table.add_column("Head", style="green")
    table.add_column("Oscillations", style="yellow")
    table.add_column("Variance", style="yellow")
    table.add_column("Score", style="bold")

    for i, s in enumerate(head_scores[:10]):
        table.add_row(
            str(i + 1),
            str(s["layer"]),
            str(s["head"]),
            str(s["sign_changes"]),
            f"{s['variance']:.6f}",
            f"{s['score']:.2f}"
        )
    console.print(table)

    # Plot all heads
    fig, axes = plt.subplots(config.n_layer, config.n_head,
                              figsize=(4*config.n_head, 3*config.n_layer))

    for layer_idx in range(config.n_layer):
        for head_idx in range(config.n_head):
            ax = axes[layer_idx, head_idx] if config.n_layer > 1 else axes[head_idx]
            dists, means, stds = all_stats[layer_idx][head_idx]

            ax.plot(dists, means, 'b-', linewidth=1.5)
            ax.fill_between(dists, means - stds, means + stds, alpha=0.2)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

            is_top = any(s["layer"] == layer_idx and s["head"] == head_idx
                        for s in head_scores[:3])
            color = 'red' if is_top else 'black'
            ax.set_title(f"L{layer_idx} H{head_idx}", color=color,
                        fontweight='bold' if is_top else 'normal', fontsize=10)

            if layer_idx == config.n_layer - 1:
                ax.set_xlabel("Distance")
            if head_idx == 0:
                ax.set_ylabel("Logit")

    plt.suptitle("100M Model: Attention Kernel Signatures", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kernel_signatures_100M.png", dpi=150)
    console.print(f"[green]✓ Saved {OUTPUT_DIR}/kernel_signatures_100M.png[/]")

    # Detailed plot for top head
    if head_scores:
        top = head_scores[0]
        dists, means, stds = all_stats[top["layer"]][top["head"]]

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(dists, means, 'b-', linewidth=2, label='Mean logit')
        ax2.fill_between(dists, means - stds, means + stds, alpha=0.3, label='±1 std')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_xlabel("Distance |i-j|", fontsize=14)
        ax2.set_ylabel("Attention Logit (Interaction Energy)", fontsize=14)
        ax2.set_title(f"100M Model - Physics Head: L{top['layer']} H{top['head']}", fontsize=16)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "physics_head_100M.png", dpi=150)
        console.print(f"[green]✓ Saved {OUTPUT_DIR}/physics_head_100M.png[/]")

        # Save data for PySR
        np.savez(OUTPUT_DIR / "kernel_data_100M.npz",
                 distances=dists,
                 means=means,
                 stds=stds,
                 layer=top["layer"],
                 head=top["head"])
        console.print(f"[green]✓ Saved {OUTPUT_DIR}/kernel_data_100M.npz[/]")

    return all_stats, head_scores


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: PySR SYMBOLIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
def run_pysr_extraction():
    """Run PySR on extracted kernel data."""
    console.print(Panel.fit(
        "[bold cyan]EXPERIMENT 2: PySR SYMBOLIC REGRESSION[/]\n"
        "Finding analytic formula for attention kernel",
        title="🧬"
    ))

    # Load kernel data
    data_path = OUTPUT_DIR / "kernel_data_100M.npz"
    if not data_path.exists():
        console.print("[yellow]Kernel data not found, running extraction first...[/]")
        run_kernel_extraction()

    data = np.load(data_path)
    X = data['distances']
    y = data['means']

    console.print(f"[green]Loaded {len(X)} data points[/]")
    console.print(f"[dim]Distance range: [{X.min():.1f}, {X.max():.1f}][/]")
    console.print(f"[dim]Mean range: [{y.min():.4f}, {y.max():.4f}][/]")

    try:
        from pysr import PySRRegressor
    except ImportError:
        console.print("[red]PySR not installed! Run: uv pip install pysr[/]")
        console.print("[yellow]Skipping PySR experiment...[/]")
        return None

    # Configure PySR
    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "square"],
        populations=15,
        population_size=50,
        maxsize=20,
        loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="best",
        progress=True,
        verbosity=1,
        random_state=42,
        deterministic=False,
        procs=0,
        multithreading=False,
    )

    console.print("[cyan]Running PySR (this may take a few minutes)...[/]")
    X_2d = X.reshape(-1, 1)
    model.fit(X_2d, y)

    # Results
    console.print("\n[bold]Pareto Front:[/]")
    equations = model.equations_

    table = Table(show_header=True)
    table.add_column("Complexity", style="cyan")
    table.add_column("Loss", style="green")
    table.add_column("Equation", style="yellow")

    for _, row in equations.iterrows():
        table.add_row(
            str(int(row['complexity'])),
            f"{row['loss']:.6f}",
            str(row['equation'])
        )
    console.print(table)

    best_eq = model.sympy()
    console.print(f"\n[bold green]Best equation: μ(d) = {best_eq}[/]")

    # Predictions
    y_pred = model.predict(X_2d)
    mse = np.mean((y - y_pred)**2)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

    console.print(f"[cyan]MSE: {mse:.8f}[/]")
    console.print(f"[cyan]R²: {r2:.4f}[/]")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(X, y, alpha=0.5, s=20, label='Data', color='blue')
    sort_idx = np.argsort(X)
    ax.plot(X[sort_idx], y_pred[sort_idx], 'r-', linewidth=2, label='PySR fit')
    ax.set_xlabel('Distance d')
    ax.set_ylabel('Mean attention μ(d)')
    ax.set_title(f'PySR Fit (R²={r2:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    residuals = y - y_pred
    ax.scatter(X, residuals, alpha=0.5, s=20, color='purple')
    ax.axhline(0, color='k', linestyle='-')
    ax.set_xlabel('Distance d')
    ax.set_ylabel('Residual')
    ax.set_title(f'Residuals (std={residuals.std():.6f})')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"100M Model PySR: μ(d) = {best_eq}", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pysr_100M.png", dpi=150)
    console.print(f"[green]✓ Saved {OUTPUT_DIR}/pysr_100M.png[/]")

    return model, best_eq


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: SPECTRAL FORM FACTOR
# ═══════════════════════════════════════════════════════════════════════════════
def compute_sff(unfolded_positions, tau_values):
    """
    Compute Spectral Form Factor.
    K(τ) = |∑_n exp(2πi·u_n·τ)|² / N
    """
    N = len(unfolded_positions)
    u = np.array(unfolded_positions)
    u_centered = u - np.mean(u)

    sff = []
    for tau in tau_values:
        phases = 2 * np.pi * u_centered * tau
        sum_exp = np.sum(np.exp(1j * phases))
        K_tau = np.abs(sum_exp)**2 / N
        sff.append(K_tau)

    return np.array(sff)


def gue_sff_theory(tau):
    """
    GUE theoretical SFF (connected part) in Heisenberg units.
    K(τ) = τ    for τ < 1  (ramp)
    K(τ) = 1    for τ >= 1 (plateau)
    Reference: Haake, "Quantum Signatures of Chaos", Chapter 10
    """
    tau = np.array(tau)
    K = np.where(tau < 1, tau, np.ones_like(tau))
    return K


def run_sff_analysis(n_zeros=10000):
    """Compute SFF on 100M zeros sample."""
    console.print(Panel.fit(
        "[bold cyan]EXPERIMENT 3: SPECTRAL FORM FACTOR[/]\n"
        f"Analyzing {n_zeros} zeros from 100M dataset",
        title="📊"
    ))

    # Load zeros
    console.print("[cyan]Loading zeros (this may take a moment)...[/]")

    # Read only first n_zeros + offset lines
    zeros = []
    offset = 50_000_000  # Start from middle of 100M
    count = 0

    with open(ZEROS_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if count >= n_zeros:
                break
            zeros.append(float(line.strip()))
            count += 1

    zeros = np.array(zeros)
    console.print(f"[green]Loaded {len(zeros)} zeros[/]")
    console.print(f"[dim]Range: [{zeros[0]:.2f}, {zeros[-1]:.2f}][/]")

    # Unfold
    unfolded = unfold_val(zeros)
    console.print(f"[dim]Unfolded range: [{unfolded[0]:.2f}, {unfolded[-1]:.2f}][/]")

    # Compute SFF
    tau_values = np.linspace(0.01, 2.0, 100)

    console.print("[cyan]Computing SFF...[/]")
    sff_true = compute_sff(unfolded, tau_values)
    sff_gue = gue_sff_theory(tau_values)

    # Metrics
    ramp_mask = tau_values < 1.0
    mse_true_gue = np.mean((sff_true[ramp_mask] - sff_gue[ramp_mask])**2)

    # Poisson comparison
    poisson_mse = np.mean((np.ones_like(sff_gue[ramp_mask]) - sff_gue[ramp_mask])**2)

    # Results
    table = Table(title="SFF Analysis Results (100M Dataset)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Interpretation", style="dim")

    table.add_row("True vs GUE MSE", f"{mse_true_gue:.6f}", "How GUE-like are zeros")
    table.add_row("Poisson vs GUE MSE", f"{poisson_mse:.4f}", "Reference (random)")
    table.add_row("Improvement ratio", f"{poisson_mse/mse_true_gue:.1f}x", "True vs random")

    console.print(table)

    if mse_true_gue < poisson_mse / 10:
        console.print("[bold green]✅ Zeros show strong GUE correlations![/]")
    else:
        console.print("[yellow]⚠️ SFF deviates from pure GUE[/]")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(tau_values, sff_true, 'b-', linewidth=2, label='100M Zeros')
    ax.plot(tau_values, sff_gue, 'g--', linewidth=2, label='GUE Theory', alpha=0.7)
    ax.axhline(1.0, color='gray', linestyle=':', label='Poisson')
    ax.axvline(1.0, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('τ')
    ax.set_ylabel('K(τ)')
    ax.set_title('Spectral Form Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(tau_values[ramp_mask], sff_true[ramp_mask], 'b-', linewidth=2, label='100M Zeros')
    ax.plot(tau_values[ramp_mask], sff_gue[ramp_mask], 'g--', linewidth=2, label='GUE: K=2τ')
    ax.set_xlabel('τ')
    ax.set_ylabel('K(τ)')
    ax.set_title(f'Ramp Region (MSE={mse_true_gue:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"SFF Analysis: 100M Riemann Zeros (sample of {n_zeros})", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sff_100M.png", dpi=150)
    console.print(f"[green]✓ Saved {OUTPUT_DIR}/sff_100M.png[/]")

    return {
        'tau': tau_values,
        'sff_true': sff_true,
        'sff_gue': sff_gue,
        'mse': mse_true_gue
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: ATTENTION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def run_attention_visualization():
    """Visualize attention patterns from 100M model."""
    console.print(Panel.fit(
        "[bold cyan]EXPERIMENT 4: ATTENTION VISUALIZATION[/]\n"
        "Heatmaps of learned attention patterns",
        title="🎨"
    ))

    model, config = load_model()
    val_data = torch.load(DATA_DIR / "val.pt", weights_only=False)

    # Take single sequence for visualization
    sample = val_data[0:1].to(DEVICE)

    console.print("[cyan]Extracting attention weights...[/]")
    with torch.no_grad():
        _, _, attentions = model(sample, return_attention=True)

    # Create heatmap grid
    fig, axes = plt.subplots(config.n_layer, config.n_head,
                              figsize=(3*config.n_head, 3*config.n_layer))

    for layer_idx, attn in enumerate(attentions):
        attn_np = attn[0].cpu().numpy()  # (n_head, T, T)

        for head_idx in range(config.n_head):
            ax = axes[layer_idx, head_idx] if config.n_layer > 1 else axes[head_idx]

            # Show only first 64 positions for clarity
            im = ax.imshow(attn_np[head_idx, :64, :64], cmap='viridis', aspect='auto')
            ax.set_title(f"L{layer_idx} H{head_idx}", fontsize=9)

            if layer_idx == config.n_layer - 1:
                ax.set_xlabel("Key pos")
            if head_idx == 0:
                ax.set_ylabel("Query pos")

    plt.suptitle("100M Model: Attention Patterns (first 64 positions)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attention_heatmaps_100M.png", dpi=150)
    console.print(f"[green]✓ Saved {OUTPUT_DIR}/attention_heatmaps_100M.png[/]")

    # Average attention by distance
    console.print("[cyan]Computing average attention by distance...[/]")

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, config.n_layer))

    for layer_idx, attn in enumerate(attentions):
        attn_np = attn[0].cpu().numpy().mean(axis=0)  # Average over heads
        T = attn_np.shape[0]

        dist_means = []
        for d in range(1, min(64, T)):
            diag = np.diagonal(attn_np, offset=-d)
            dist_means.append(np.mean(diag))

        ax2.plot(range(1, len(dist_means)+1), dist_means,
                 color=colors[layer_idx], linewidth=2,
                 label=f'Layer {layer_idx}', alpha=0.8)

    ax2.set_xlabel("Distance |i-j|", fontsize=12)
    ax2.set_ylabel("Mean Attention Weight", fontsize=12)
    ax2.set_title("100M Model: Attention vs Distance (averaged over heads)", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attention_distance_100M.png", dpi=150)
    console.print(f"[green]✓ Saved {OUTPUT_DIR}/attention_distance_100M.png[/]")

    return attentions


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    console.print(Panel.fit(
        "[bold magenta]═══ 100M ZEROS EXPERIMENTS SUITE ═══[/]\n"
        f"Model: {CKPT_PATH}\n"
        f"Device: {DEVICE}",
        title="🔬"
    ))

    # Check files exist
    if not CKPT_PATH.exists():
        console.print(f"[red]Model not found: {CKPT_PATH}[/]")
        return
    if not (DATA_DIR / "val.pt").exists():
        console.print(f"[red]Data not found: {DATA_DIR}[/]")
        return

    results = {}

    # Run experiments
    console.print("\n" + "="*60 + "\n")
    results['kernel'] = run_kernel_extraction()

    console.print("\n" + "="*60 + "\n")
    results['attention'] = run_attention_visualization()

    console.print("\n" + "="*60 + "\n")
    results['sff'] = run_sff_analysis(n_zeros=10000)

    console.print("\n" + "="*60 + "\n")
    results['pysr'] = run_pysr_extraction()

    # Summary
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]═══ ALL EXPERIMENTS COMPLETE ═══[/]\n\n"
        f"Output directory: {OUTPUT_DIR}\n\n"
        "Files created:\n"
        "  - kernel_signatures_100M.png\n"
        "  - physics_head_100M.png\n"
        "  - kernel_data_100M.npz\n"
        "  - attention_heatmaps_100M.png\n"
        "  - attention_distance_100M.png\n"
        "  - sff_100M.png\n"
        "  - pysr_100M.png",
        title="✅"
    ))

    return results


if __name__ == "__main__":
    main()
