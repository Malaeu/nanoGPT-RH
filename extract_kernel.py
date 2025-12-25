#!/usr/bin/env python3
"""
Operation "Open Brain" - Extract attention kernels from trained SpacingGPT.
Looking for "The Physics Head" with sine-kernel signatures.
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table

from model.gpt import SpacingGPT, GPTConfig

console = Console()

# --- CONFIG ---
CKPT_PATH = Path("out/best.pt")
DATA_PATH = Path("data/val.pt")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64  # More samples for better statistics
MAX_DIST = 128   # Look at distances up to this
# --------------


def load_model(ckpt_path: Path, device: str):
    """Load trained model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpacingGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["config"]


def extract_attention_logits(model, x, config):
    """
    Manually extract attention logits (before softmax) for each layer/head.
    Returns dict: layer_idx -> head_idx -> (B, T, T) logits tensor
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
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)

        # RAW ATTENTION LOGITS (interaction energy) - NO SOFTMAX!
        scale = 1.0 / math.sqrt(head_dim)
        att_logits = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)

        all_logits[layer_idx] = att_logits

        # Continue forward pass
        h = block(h)

    return all_logits


def compute_distance_statistics(att_logits: torch.Tensor, max_dist: int):
    """
    Compute mean attention logit as function of distance d = |i-j|.

    Args:
        att_logits: (B, nh, T, T) tensor
        max_dist: maximum distance to consider

    Returns:
        dict: head_idx -> (distances, means, stds)
    """
    B, n_head, T, _ = att_logits.shape
    results = {}

    for head in range(n_head):
        head_logits = att_logits[:, head, :, :]  # (B, T, T)

        dists = []
        means = []
        stds = []

        for d in range(1, min(max_dist, T)):
            # Extract diagonal at offset -d (causal: looking back)
            diag = torch.diagonal(head_logits, offset=-d, dim1=-2, dim2=-1)
            values = diag.flatten().cpu().numpy()

            if len(values) > 0:
                dists.append(d)
                means.append(np.mean(values))
                stds.append(np.std(values))

        results[head] = (np.array(dists), np.array(means), np.array(stds))

    return results


def find_physics_head(all_stats: dict, config):
    """
    Analyze all heads to find the one with most interesting structure.
    Looking for: oscillations, non-monotonic behavior, level repulsion.
    """
    scores = []

    for layer_idx, layer_stats in all_stats.items():
        for head_idx, (dists, means, stds) in layer_stats.items():
            if len(means) < 10:
                continue

            # Score 1: Non-monotonicity (sign changes in derivative)
            diff = np.diff(means)
            sign_changes = np.sum(np.abs(np.diff(np.sign(diff))) > 0)

            # Score 2: Variance of the kernel (structure vs flat)
            variance = np.var(means)

            # Score 3: Level repulsion signature (dip at d=1, rise after)
            repulsion = 0
            if len(means) > 5:
                if means[0] < means[3]:  # First value lower than d=4
                    repulsion = means[3] - means[0]

            # Combined score
            score = sign_changes * 0.5 + variance * 100 + repulsion * 10

            scores.append({
                "layer": layer_idx,
                "head": head_idx,
                "sign_changes": sign_changes,
                "variance": variance,
                "repulsion": repulsion,
                "score": score,
            })

    # Sort by score
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def main():
    console.print("[bold magenta]═══ Operation 'Open Brain' ═══[/]\n")
    console.print(f"[cyan]Device: {DEVICE}[/]")

    # Load model
    model, config = load_model(CKPT_PATH, DEVICE)
    console.print(f"[green]Model loaded: {config.n_layer} layers, {config.n_head} heads[/]")

    # Load validation data
    val_data = torch.load(DATA_PATH)
    console.print(f"[green]Data loaded: {val_data.shape}[/]")

    # Take batch for analysis
    batch = val_data[:BATCH_SIZE].to(DEVICE)
    console.print(f"[cyan]Analyzing {BATCH_SIZE} sequences...[/]\n")

    # Extract attention logits
    with torch.no_grad():
        all_logits = extract_attention_logits(model, batch, config)

    # Compute distance statistics for each layer/head
    console.print("[bold]Computing distance statistics...[/]")
    all_stats = {}
    for layer_idx, logits in all_logits.items():
        all_stats[layer_idx] = compute_distance_statistics(logits, MAX_DIST)

    # Find physics head
    console.print("\n[bold]Analyzing heads for physics signatures...[/]")
    head_scores = find_physics_head(all_stats, config)

    # Display top candidates
    table = Table(title="Top Physics Head Candidates")
    table.add_column("Rank", style="cyan")
    table.add_column("Layer", style="green")
    table.add_column("Head", style="green")
    table.add_column("Oscillations", style="yellow")
    table.add_column("Variance", style="yellow")
    table.add_column("Repulsion", style="yellow")
    table.add_column("Score", style="bold")

    for i, s in enumerate(head_scores[:8]):
        table.add_row(
            str(i + 1),
            str(s["layer"]),
            str(s["head"]),
            str(s["sign_changes"]),
            f"{s['variance']:.4f}",
            f"{s['repulsion']:.2f}",
            f"{s['score']:.2f}"
        )
    console.print(table)

    # Plot all heads
    console.print("\n[bold]Plotting kernel signatures...[/]")

    fig, axes = plt.subplots(config.n_layer, config.n_head, figsize=(4*config.n_head, 3*config.n_layer))

    for layer_idx in range(config.n_layer):
        for head_idx in range(config.n_head):
            ax = axes[layer_idx, head_idx] if config.n_layer > 1 else axes[head_idx]

            dists, means, stds = all_stats[layer_idx][head_idx]

            # Plot mean with std band
            ax.plot(dists, means, 'b-', linewidth=2, label='mean')
            ax.fill_between(dists, means - stds, means + stds, alpha=0.2)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

            # Mark if this is a top candidate
            is_top = any(s["layer"] == layer_idx and s["head"] == head_idx
                        for s in head_scores[:3])
            color = 'red' if is_top else 'black'
            ax.set_title(f"L{layer_idx} H{head_idx}", color=color, fontweight='bold' if is_top else 'normal')

            if layer_idx == config.n_layer - 1:
                ax.set_xlabel("Distance |i-j|")
            if head_idx == 0:
                ax.set_ylabel("Interaction Energy")

    plt.tight_layout()
    plt.savefig("kernel_signatures.png", dpi=150)
    console.print("[green]✓ Saved kernel_signatures.png[/]")

    # Save detailed plot for top candidate
    if head_scores:
        top = head_scores[0]
        dists, means, stds = all_stats[top["layer"]][top["head"]]

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(dists, means, 'b-', linewidth=2)
        ax2.fill_between(dists, means - stds, means + stds, alpha=0.3)
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_xlabel("Distance |i-j|", fontsize=14)
        ax2.set_ylabel("Attention Logit (Interaction Energy)", fontsize=14)
        ax2.set_title(f"Physics Head: Layer {top['layer']}, Head {top['head']}", fontsize=16)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("physics_head_kernel.png", dpi=150)
        console.print("[green]✓ Saved physics_head_kernel.png[/]")

        # Save data for PySR
        np.savez("kernel_data.npz",
                 distances=dists,
                 means=means,
                 stds=stds,
                 layer=top["layer"],
                 head=top["head"])
        console.print("[green]✓ Saved kernel_data.npz for PySR[/]")

    console.print("\n[bold green]✓ Extraction complete![/]")


if __name__ == "__main__":
    main()
