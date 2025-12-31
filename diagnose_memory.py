#!/usr/bin/env python3
"""
TASK_SPEC_2M — Memory Bank Diagnostics

Detailed measurements of three phenomena:

A) SHORTCUT / Concentration
   - Attention mass per slot (which slots are used)
   - Ablation: remove slot i → Δ(NLL), Δ(bias), Δ(Err@100)
   - Gradient norm per slot (who drives learning)

B) SYMMETRY / Redundancy
   - Cosine similarity matrix between slots
   - Permutation test: shuffle slots → metrics unchanged?
   - Entropy of importance distribution

C) CO-ADAPTATION / Entanglement
   - CCA/correlation between slots
   - If remove slot i, how do others change?
   - Transfer: does co-adaptation hurt val_tail?

Output:
  reports/2M/memory_diagnostics.md
  reports/2M/memory_heatmap.png
  reports/2M/memory_ablation.png
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
import matplotlib.pyplot as plt
from scipy import stats
import argparse

console = Console()

from train_mdn_memory import SpacingMDNMemory, MDNConfig
from train_mdn import mdn_loss


def mdn_mean(pi, mu, sigma):
    """Expected value of MDN."""
    return (pi * mu).sum(dim=-1)


# ============================================================================
# A) SHORTCUT DIAGNOSTICS
# ============================================================================

def measure_attention_concentration(model, data, device, n_batches=20):
    """
    Measure attention mass to each memory slot.

    Shortcut = attention concentrated in 1-2 slots.
    """
    model.eval()
    model.memory_bank.reset_stats()

    batch_size = 128

    for i in range(n_batches):
        idx = torch.randint(0, len(data), (batch_size,))
        batch = data[idx].to(device)
        x = batch[:, :-1]

        with torch.no_grad():
            model(x, collect_memory_stats=True)

    attn_dist = model.memory_bank.get_attn_distribution()

    # Entropy (lower = more concentrated)
    entropy = -(attn_dist * torch.log(attn_dist + 1e-10)).sum().item()
    max_entropy = np.log(model.n_memory_slots)

    return {
        "attn_dist": attn_dist.cpu().numpy(),
        "entropy": entropy,
        "max_entropy": max_entropy,
        "concentration_ratio": (1 - entropy / max_entropy),  # 0 = uniform, 1 = single slot
        "top_slot": attn_dist.argmax().item(),
        "top_slot_mass": attn_dist.max().item(),
    }


def ablation_study(model, data, device, n_eval=500):
    """
    Measure impact of removing each memory slot.

    For each slot i:
      1. Zero out slot i embedding
      2. Measure Δ(NLL), Δ(bias)
    """
    model.eval()

    # Baseline metrics (all slots active)
    idx = torch.randint(0, len(data), (n_eval,))
    batch = data[idx].to(device)
    x = batch[:, :-1]
    y = batch[:, 1:]

    with torch.no_grad():
        pi, mu, sigma = model(x)
        _, base_nll, _ = mdn_loss(pi, mu, sigma, y, 0.0)
        base_bias = (mdn_mean(pi, mu, sigma) - y).mean().item()

    base_nll = base_nll.item()

    # Ablate each slot
    n_slots = model.n_memory_slots
    ablation_results = []

    original_memory = model.memory_bank.memory.data.clone()

    for slot_idx in range(n_slots):
        # Zero out slot
        model.memory_bank.memory.data[slot_idx] = 0

        with torch.no_grad():
            pi, mu, sigma = model(x)
            _, abl_nll, _ = mdn_loss(pi, mu, sigma, y, 0.0)
            abl_bias = (mdn_mean(pi, mu, sigma) - y).mean().item()

        delta_nll = abl_nll.item() - base_nll
        delta_bias = abs(abl_bias) - abs(base_bias)

        ablation_results.append({
            "slot": slot_idx,
            "delta_nll": delta_nll,
            "delta_bias": delta_bias,
            "importance": delta_nll,  # Higher = more important
        })

        # Restore
        model.memory_bank.memory.data = original_memory.clone()

    return {
        "base_nll": base_nll,
        "base_bias": base_bias,
        "ablations": ablation_results,
    }


# ============================================================================
# B) SYMMETRY DIAGNOSTICS
# ============================================================================

def measure_slot_similarity(model):
    """
    Compute cosine similarity between memory slots.

    High similarity = slots are redundant (symmetry).
    """
    sim = model.memory_bank.get_slot_similarity()

    n_slots = model.n_memory_slots

    # Off-diagonal mean (excluding self-similarity)
    mask = ~torch.eye(n_slots, dtype=bool, device=sim.device)
    off_diag = sim[mask]

    return {
        "similarity_matrix": sim.cpu().numpy(),
        "mean_similarity": off_diag.mean().item(),
        "max_similarity": off_diag.max().item(),
        "min_similarity": off_diag.min().item(),
    }


def permutation_test(model, data, device, n_perms=5, n_eval=500):
    """
    Test if shuffling memory slots changes metrics.

    If permutation invariant → slots are interchangeable (bad differentiation).
    """
    model.eval()

    idx = torch.randint(0, len(data), (n_eval,))
    batch = data[idx].to(device)
    x = batch[:, :-1]
    y = batch[:, 1:]

    # Baseline
    with torch.no_grad():
        pi, mu, sigma = model(x)
        _, base_nll, _ = mdn_loss(pi, mu, sigma, y, 0.0)
    base_nll = base_nll.item()

    # Permutation tests
    original_memory = model.memory_bank.memory.data.clone()
    perm_nlls = []

    for _ in range(n_perms):
        perm = torch.randperm(model.n_memory_slots)
        model.memory_bank.memory.data = original_memory[perm]

        with torch.no_grad():
            pi, mu, sigma = model(x)
            _, perm_nll, _ = mdn_loss(pi, mu, sigma, y, 0.0)
        perm_nlls.append(perm_nll.item())

        model.memory_bank.memory.data = original_memory.clone()

    # If permutation doesn't change NLL much → slots are symmetric
    nll_variance = np.var(perm_nlls)

    return {
        "base_nll": base_nll,
        "perm_nlls": perm_nlls,
        "nll_variance": nll_variance,
        "is_permutation_invariant": nll_variance < 0.001,
    }


# ============================================================================
# C) CO-ADAPTATION DIAGNOSTICS
# ============================================================================

def measure_slot_correlations(model):
    """
    Measure correlation/dependence between slot embeddings.

    High correlation = co-adaptation (slots learned together).
    """
    memory = model.memory_bank.memory.detach().cpu().numpy()  # (M, D)

    # Pearson correlation between slot vectors
    corr_matrix = np.corrcoef(memory)

    n_slots = model.n_memory_slots
    mask = ~np.eye(n_slots, dtype=bool)
    off_diag = corr_matrix[mask]

    return {
        "correlation_matrix": corr_matrix,
        "mean_correlation": np.mean(np.abs(off_diag)),
        "max_correlation": np.max(np.abs(off_diag)),
    }


def measure_slot_independence(model, data, device, n_eval=500):
    """
    Test if removing one slot affects the "role" of others.

    High co-adaptation = removing slot i dramatically changes attention to others.
    """
    model.eval()

    idx = torch.randint(0, len(data), (n_eval,))
    batch = data[idx].to(device)
    x = batch[:, :-1]

    # Baseline attention distribution
    model.memory_bank.reset_stats()
    with torch.no_grad():
        model(x, collect_memory_stats=True)
    base_attn = model.memory_bank.get_attn_distribution().cpu().numpy()

    # For each slot: remove it and measure how attention redistributes
    n_slots = model.n_memory_slots
    original_memory = model.memory_bank.memory.data.clone()

    redistribution_scores = []

    for slot_idx in range(n_slots):
        model.memory_bank.memory.data[slot_idx] = 0
        model.memory_bank.reset_stats()

        with torch.no_grad():
            model(x, collect_memory_stats=True)

        abl_attn = model.memory_bank.get_attn_distribution().cpu().numpy()

        # How much did attention to OTHER slots change?
        mask = np.ones(n_slots, dtype=bool)
        mask[slot_idx] = False

        # Renormalize ablated attention (exclude removed slot)
        abl_attn_others = abl_attn[mask]
        abl_attn_others = abl_attn_others / (abl_attn_others.sum() + 1e-10)

        base_attn_others = base_attn[mask]
        base_attn_others = base_attn_others / (base_attn_others.sum() + 1e-10)

        # KL divergence as measure of redistribution
        kl_div = np.sum(abl_attn_others * np.log((abl_attn_others + 1e-10) / (base_attn_others + 1e-10)))
        redistribution_scores.append(kl_div)

        model.memory_bank.memory.data = original_memory.clone()

    return {
        "redistribution_per_slot": redistribution_scores,
        "mean_redistribution": np.mean(redistribution_scores),
        "max_redistribution": np.max(redistribution_scores),
        "has_co_adaptation": np.max(redistribution_scores) > 0.5,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Memory Bank Diagnostics")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="reports/2M")
    args = parser.parse_args()

    console.print("[bold magenta]═══ Memory Bank Diagnostics ═══[/]\n")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    console.print(f"[cyan]Loading {args.ckpt}[/]")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    config = MDNConfig(**ckpt["config"])
    n_slots = ckpt.get("n_memory_slots", 8)

    model = SpacingMDNMemory(config, n_memory_slots=n_slots).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load data
    data_dir = Path(args.data_dir)
    val_data = torch.load(data_dir / "val.pt", weights_only=False)
    console.print(f"[green]Val data: {val_data.shape}[/]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # A) SHORTCUT DIAGNOSTICS
    # ========================================================================
    console.print("\n[bold cyan]═══ A) SHORTCUT / Concentration ═══[/]")

    attn_conc = measure_attention_concentration(model, val_data, device)
    console.print(f"Attention entropy: {attn_conc['entropy']:.2f} / {attn_conc['max_entropy']:.2f}")
    console.print(f"Concentration ratio: {attn_conc['concentration_ratio']:.2f}")
    console.print(f"Top slot: {attn_conc['top_slot']} (mass={attn_conc['top_slot_mass']:.2f})")

    if attn_conc['concentration_ratio'] > 0.5:
        console.print("[red]⚠ HIGH SHORTCUT: model relies on few slots[/]")
    else:
        console.print("[green]✓ Attention reasonably distributed[/]")

    ablation = ablation_study(model, val_data, device)
    console.print("\n[cyan]Ablation study (Δ NLL when removing slot):[/]")
    for res in ablation['ablations']:
        bar = '█' * int(max(0, res['delta_nll']) * 100)
        console.print(f"  Slot {res['slot']}: Δ={res['delta_nll']:+.4f} {bar}")

    # ========================================================================
    # B) SYMMETRY DIAGNOSTICS
    # ========================================================================
    console.print("\n[bold cyan]═══ B) SYMMETRY / Redundancy ═══[/]")

    similarity = measure_slot_similarity(model)
    console.print(f"Mean slot similarity: {similarity['mean_similarity']:.3f}")
    console.print(f"Max slot similarity: {similarity['max_similarity']:.3f}")

    if similarity['mean_similarity'] > 0.8:
        console.print("[red]⚠ HIGH SYMMETRY: slots nearly identical[/]")
    elif similarity['mean_similarity'] > 0.5:
        console.print("[yellow]⚠ Moderate symmetry[/]")
    else:
        console.print("[green]✓ Slots differentiated[/]")

    perm_test = permutation_test(model, val_data, device)
    console.print(f"\nPermutation test NLL variance: {perm_test['nll_variance']:.6f}")
    if perm_test['is_permutation_invariant']:
        console.print("[red]⚠ Slots are permutation invariant (interchangeable)[/]")
    else:
        console.print("[green]✓ Slots have distinct roles[/]")

    # ========================================================================
    # C) CO-ADAPTATION DIAGNOSTICS
    # ========================================================================
    console.print("\n[bold cyan]═══ C) CO-ADAPTATION / Entanglement ═══[/]")

    correlations = measure_slot_correlations(model)
    console.print(f"Mean absolute correlation: {correlations['mean_correlation']:.3f}")

    independence = measure_slot_independence(model, val_data, device)
    console.print(f"Mean redistribution on ablation: {independence['mean_redistribution']:.3f}")

    if independence['has_co_adaptation']:
        console.print("[yellow]⚠ Co-adaptation detected: slots are entangled[/]")
    else:
        console.print("[green]✓ Slots are relatively independent[/]")

    # ========================================================================
    # SUMMARY & PLOTS
    # ========================================================================
    console.print("\n[bold cyan]═══ Summary ═══[/]")

    summary_table = Table(title="Memory Bank Diagnostics")
    summary_table.add_column("Phenomenon", style="cyan")
    summary_table.add_column("Metric", style="white")
    summary_table.add_column("Value", style="green")
    summary_table.add_column("Status", style="bold")

    summary_table.add_row(
        "Shortcut",
        "Concentration ratio",
        f"{attn_conc['concentration_ratio']:.2f}",
        "⚠" if attn_conc['concentration_ratio'] > 0.5 else "✓"
    )
    summary_table.add_row(
        "Symmetry",
        "Mean similarity",
        f"{similarity['mean_similarity']:.3f}",
        "⚠" if similarity['mean_similarity'] > 0.5 else "✓"
    )
    summary_table.add_row(
        "Co-adaptation",
        "Mean redistribution",
        f"{independence['mean_redistribution']:.3f}",
        "⚠" if independence['has_co_adaptation'] else "✓"
    )

    console.print(summary_table)

    # ========================================================================
    # PLOTS
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # A) Attention distribution
    ax = axes[0, 0]
    ax.bar(range(n_slots), attn_conc['attn_dist'], color='steelblue', alpha=0.7)
    ax.axhline(y=1/n_slots, color='r', linestyle='--', label=f'Uniform ({1/n_slots:.2f})')
    ax.set_xlabel('Slot')
    ax.set_ylabel('Attention Mass')
    ax.set_title(f'Attention Distribution (entropy={attn_conc["entropy"]:.2f})')
    ax.legend()

    # B) Ablation importance
    ax = axes[0, 1]
    importances = [r['delta_nll'] for r in ablation['ablations']]
    colors = ['red' if imp > 0.01 else 'steelblue' for imp in importances]
    ax.bar(range(n_slots), importances, color=colors, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel('Slot')
    ax.set_ylabel('Δ NLL on removal')
    ax.set_title('Slot Importance (Ablation)')

    # C) Similarity heatmap
    ax = axes[1, 0]
    im = ax.imshow(similarity['similarity_matrix'], cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Slot')
    ax.set_ylabel('Slot')
    ax.set_title(f'Slot Similarity (mean={similarity["mean_similarity"]:.2f})')
    plt.colorbar(im, ax=ax)

    # D) Redistribution on ablation
    ax = axes[1, 1]
    ax.bar(range(n_slots), independence['redistribution_per_slot'], color='steelblue', alpha=0.7)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Co-adaptation threshold')
    ax.set_xlabel('Removed Slot')
    ax.set_ylabel('Attention Redistribution (KL)')
    ax.set_title('Co-adaptation Test')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "memory_diagnostics.png", dpi=150)
    plt.close()

    console.print(f"\n[green]Saved memory_diagnostics.png[/]")

    # ========================================================================
    # REPORT
    # ========================================================================
    report = f"""# Memory Bank Diagnostics Report

## Model Info
- Checkpoint: `{args.ckpt}`
- Memory slots: {n_slots}

## A) Shortcut / Concentration
| Metric | Value | Status |
|--------|-------|--------|
| Attention entropy | {attn_conc['entropy']:.2f} / {attn_conc['max_entropy']:.2f} | {'⚠' if attn_conc['concentration_ratio'] > 0.5 else '✓'} |
| Concentration ratio | {attn_conc['concentration_ratio']:.2f} | {'HIGH' if attn_conc['concentration_ratio'] > 0.5 else 'OK'} |
| Top slot | {attn_conc['top_slot']} (mass={attn_conc['top_slot_mass']:.2f}) | |

### Ablation Study
| Slot | Δ NLL | Interpretation |
|------|-------|---------------|
"""
    for res in ablation['ablations']:
        status = "CRITICAL" if res['delta_nll'] > 0.05 else "important" if res['delta_nll'] > 0.01 else "minor"
        report += f"| {res['slot']} | {res['delta_nll']:+.4f} | {status} |\n"

    report += f"""
## B) Symmetry / Redundancy
| Metric | Value | Status |
|--------|-------|--------|
| Mean slot similarity | {similarity['mean_similarity']:.3f} | {'⚠ HIGH' if similarity['mean_similarity'] > 0.5 else '✓'} |
| Max slot similarity | {similarity['max_similarity']:.3f} | |
| Permutation invariant | {'YES ⚠' if perm_test['is_permutation_invariant'] else 'NO ✓'} | |

## C) Co-adaptation / Entanglement
| Metric | Value | Status |
|--------|-------|--------|
| Mean slot correlation | {correlations['mean_correlation']:.3f} | |
| Mean redistribution | {independence['mean_redistribution']:.3f} | {'⚠' if independence['has_co_adaptation'] else '✓'} |
| Co-adaptation detected | {'YES ⚠' if independence['has_co_adaptation'] else 'NO ✓'} | |

## Summary
- **Shortcut**: {'⚠ Detected - model relies on few slots' if attn_conc['concentration_ratio'] > 0.5 else '✓ OK'}
- **Symmetry**: {'⚠ High - slots are redundant' if similarity['mean_similarity'] > 0.5 else '✓ OK'}
- **Co-adaptation**: {'⚠ Detected - slots are entangled' if independence['has_co_adaptation'] else '✓ OK'}
"""

    with open(output_dir / "memory_diagnostics.md", "w") as f:
        f.write(report)

    console.print(f"[green]Saved memory_diagnostics.md[/]")
    console.print("\n[bold green]═══ Diagnostics Complete ═══[/]")


if __name__ == "__main__":
    main()
