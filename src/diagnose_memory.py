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
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import track
import matplotlib.pyplot as plt
from scipy import stats
import argparse

console = Console()

from train_mdn_memory import SpacingMDNMemory, MDNConfig
from train_mdn import mdn_loss


# ============================================================================
# UNIVERSAL HELPERS (from alternative version)
# ============================================================================

def get_memory_tensor(model: torch.nn.Module) -> Optional[torch.Tensor]:
    """
    Try to locate the learnable memory slots tensor.
    Common patterns: model.memory_bank.memory / .slots / .mem
    """
    mb = getattr(model, "memory_bank", None)
    if mb is None:
        return None
    for name in ["memory", "slots", "mem"]:
        if hasattr(mb, name):
            t = getattr(mb, name)
            if isinstance(t, torch.Tensor):
                return t
            if hasattr(t, "data") and isinstance(t.data, torch.Tensor):
                return t
    return None


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append a single JSON line to file."""
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return str(o)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_default) + "\n")


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
        pi, mu, sigma, *_ = model(x)  # Ignore aux outputs if present
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
            pi, mu, sigma, *_ = model(x)  # Ignore aux outputs if present
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
        pi, mu, sigma, *_ = model(x)  # Ignore aux outputs if present
        _, base_nll, _ = mdn_loss(pi, mu, sigma, y, 0.0)
    base_nll = base_nll.item()

    # Permutation tests
    original_memory = model.memory_bank.memory.data.clone()
    perm_nlls = []

    for _ in range(n_perms):
        perm = torch.randperm(model.n_memory_slots)
        model.memory_bank.memory.data = original_memory[perm]

        with torch.no_grad():
            pi, mu, sigma, *_ = model(x)  # Ignore aux outputs if present
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
# D) GRADIENT CORRELATION (NEW)
# ============================================================================

def measure_gradient_correlation(model, data, device, n_batches=10):
    """
    Measure gradient correlation between memory slots.

    High correlation = slots are learning together (co-adaptation through gradients).
    This is the REAL co-adaptation signal: if removing one slot would
    change what the others learn, their gradients are correlated.

    Returns per-slot gradient norms AND pairwise correlations.
    """
    from train_mdn import mdn_loss

    model.train()  # Need gradients
    batch_size = 64

    # Collect gradients for each slot over multiple batches
    n_slots = model.n_memory_slots
    slot_grads = [[] for _ in range(n_slots)]

    for i in range(n_batches):
        idx = torch.randint(0, len(data), (batch_size,))
        batch = data[idx].to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        # Forward pass
        pi, mu, sigma, *_ = model(x)
        loss, nll, _ = mdn_loss(pi, mu, sigma, y, 0.0)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Collect memory slot gradients
        mem_grad = model.memory_bank.memory.grad  # (M, D)
        for slot_idx in range(n_slots):
            slot_grads[slot_idx].append(mem_grad[slot_idx].detach().cpu().clone())

    # Stack gradients: (n_batches, D) per slot
    slot_grads = [torch.stack(grads) for grads in slot_grads]

    # Compute gradient norms per slot
    grad_norms = [g.norm(dim=-1).mean().item() for g in slot_grads]

    # Compute pairwise gradient correlation
    # Flatten gradients: (n_batches * D,) per slot
    flat_grads = [g.flatten().numpy() for g in slot_grads]

    corr_matrix = np.zeros((n_slots, n_slots))
    for i in range(n_slots):
        for j in range(n_slots):
            corr_matrix[i, j] = np.corrcoef(flat_grads[i], flat_grads[j])[0, 1]

    # Off-diagonal mean
    mask = ~np.eye(n_slots, dtype=bool)
    off_diag = corr_matrix[mask]

    model.eval()  # Back to eval

    return {
        "grad_norms": grad_norms,
        "grad_correlation_matrix": corr_matrix,
        "mean_grad_correlation": np.mean(np.abs(off_diag)),
        "max_grad_correlation": np.max(np.abs(off_diag)),
        "grad_leader": np.argmax(grad_norms),  # Slot with biggest gradients
    }


# ============================================================================
# E) TRAIN VS ROLLOUT STABILITY (NEW)
# ============================================================================

def compare_train_vs_rollout(model, data, device, rollout_steps=50, n_eval=200):
    """
    Compare slot usage during normal forward vs rollout prediction.

    If slot usage changes dramatically during rollout, the memory bank
    is overfitting to train-time patterns and may not generalize.

    Returns KL divergence between train and rollout slot distributions.
    """
    model.eval()

    # === Train-time slot usage ===
    model.memory_bank.reset_stats()

    idx = torch.randint(0, len(data), (n_eval,))
    batch = data[idx].to(device)
    x = batch[:, :-1]

    with torch.no_grad():
        model(x, collect_memory_stats=True)

    train_attn = model.memory_bank.get_attn_distribution().cpu().numpy()

    # === Rollout slot usage ===
    model.memory_bank.reset_stats()

    # Start from first context_len tokens, predict the rest
    context_len = 128
    seq = batch[:, :context_len].clone()  # (B, context_len)

    with torch.no_grad():
        for step in range(rollout_steps):
            # Predict next spacing
            pi, mu, sigma, *_ = model(seq, collect_memory_stats=True)

            # Sample from last position
            pi_last = pi[:, -1, :]
            mu_last = mu[:, -1, :]
            sigma_last = sigma[:, -1, :]

            # Pick highest weight component mean
            comp_idx = pi_last.argmax(dim=-1)
            next_spacing = mu_last.gather(1, comp_idx.unsqueeze(-1)).squeeze(-1)

            # Append prediction
            seq = torch.cat([seq, next_spacing.unsqueeze(-1)], dim=-1)

    rollout_attn = model.memory_bank.get_attn_distribution().cpu().numpy()

    # KL divergence: how much does distribution shift?
    epsilon = 1e-10
    kl_div = np.sum(train_attn * np.log((train_attn + epsilon) / (rollout_attn + epsilon)))

    # Which slots changed most?
    slot_shift = np.abs(train_attn - rollout_attn)

    return {
        "train_attn": train_attn,
        "rollout_attn": rollout_attn,
        "kl_divergence": kl_div,
        "slot_shift": slot_shift,
        "max_shift_slot": np.argmax(slot_shift),
        "max_shift_amount": np.max(slot_shift),
        "is_stable": kl_div < 0.1,  # Low KL = stable
    }


# ============================================================================
# F) SLOT EFFECT NORM (NEW)
# ============================================================================

def measure_slot_effect_norm(model, data, device, n_eval=100):
    """
    Measure how much each slot ACTUALLY affects the hidden state.

    For each slot, compare hidden state WITH vs WITHOUT that slot.
    L2 norm of difference = how much that slot matters for representations.

    This is different from attention mass: a slot can receive attention
    but have negligible effect if its values are small.
    """
    model.eval()

    n_slots = model.n_memory_slots
    slot_effects = []

    idx = torch.randint(0, len(data), (n_eval,))
    batch = data[idx].to(device)
    x = batch[:, :-1]

    # Get baseline hidden states (all slots active)
    with torch.no_grad():
        # We need to modify forward to return hidden states
        # For now, use a proxy: measure output change
        pi_base, mu_base, sigma_base, *_ = model(x)
        pred_base = (pi_base * mu_base).sum(dim=-1)  # (B, T)

    original_memory = model.memory_bank.memory.data.clone()

    for slot_idx in range(n_slots):
        # Zero out this slot
        model.memory_bank.memory.data[slot_idx] = 0

        with torch.no_grad():
            pi, mu, sigma, *_ = model(x)
            pred = (pi * mu).sum(dim=-1)

        # L2 distance in prediction space
        effect = (pred_base - pred).pow(2).mean().sqrt().item()
        slot_effects.append(effect)

        # Restore
        model.memory_bank.memory.data = original_memory.clone()

    # Normalize to percentages
    total_effect = sum(slot_effects) + 1e-10
    slot_effect_pct = [e / total_effect for e in slot_effects]

    return {
        "slot_effects": slot_effects,
        "slot_effect_pct": slot_effect_pct,
        "dominant_slot": np.argmax(slot_effects),
        "effect_entropy": -sum(p * np.log(p + 1e-10) for p in slot_effect_pct),
        "max_effect": max(slot_effects),
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
    use_slot_id = ckpt.get("use_slot_id", False)  # Default False for v0 compat
    use_aux_loss = ckpt.get("use_aux_loss", False)

    model = SpacingMDNMemory(
        config,
        n_memory_slots=n_slots,
        use_slot_id=use_slot_id,
        use_aux_loss=use_aux_loss
    ).to(device)

    # Handle torch.compile() checkpoint (strips _orig_mod. prefix)
    state_dict = ckpt["model"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    if use_slot_id:
        console.print("[cyan]  Slot-ID embeddings: ENABLED[/]")
    if use_aux_loss:
        console.print(f"[cyan]  Aux loss: ENABLED (weight={ckpt.get('aux_loss_weight', 0)})[/]")

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
    # D) GRADIENT CORRELATION (NEW)
    # ========================================================================
    console.print("\n[bold cyan]═══ D) GRADIENT CORRELATION (learning together?) ═══[/]")

    grad_corr = measure_gradient_correlation(model, val_data, device)
    console.print(f"Mean gradient correlation: {grad_corr['mean_grad_correlation']:.3f}")
    console.print(f"Max gradient correlation: {grad_corr['max_grad_correlation']:.3f}")
    console.print(f"Gradient leader slot: {grad_corr['grad_leader']} (norm={grad_corr['grad_norms'][grad_corr['grad_leader']]:.4f})")

    console.print("\n[dim]Gradient norms per slot:[/]")
    for i, norm in enumerate(grad_corr['grad_norms']):
        bar = '█' * int(norm * 1000)
        console.print(f"  Slot {i}: {norm:.4f} {bar}")

    if grad_corr['mean_grad_correlation'] > 0.7:
        console.print("[yellow]⚠ High gradient correlation: slots learning same thing[/]")
    else:
        console.print("[green]✓ Slots learning independently[/]")

    # ========================================================================
    # E) TRAIN VS ROLLOUT STABILITY (NEW)
    # ========================================================================
    console.print("\n[bold cyan]═══ E) TRAIN vs ROLLOUT STABILITY ═══[/]")

    rollout_cmp = compare_train_vs_rollout(model, val_data, device)
    console.print(f"KL divergence (train→rollout): {rollout_cmp['kl_divergence']:.4f}")
    console.print(f"Max shift slot: {rollout_cmp['max_shift_slot']} (Δ={rollout_cmp['max_shift_amount']:.3f})")

    console.print("\n[dim]Slot attention comparison:[/]")
    for i in range(n_slots):
        train_pct = rollout_cmp['train_attn'][i] * 100
        roll_pct = rollout_cmp['rollout_attn'][i] * 100
        shift = rollout_cmp['slot_shift'][i] * 100
        arrow = "↑" if roll_pct > train_pct else "↓" if roll_pct < train_pct else "="
        console.print(f"  Slot {i}: train={train_pct:.1f}% → rollout={roll_pct:.1f}% {arrow}")

    if rollout_cmp['is_stable']:
        console.print("[green]✓ Slot usage stable during rollout[/]")
    else:
        console.print("[yellow]⚠ Slot usage shifts during rollout - potential overfitting[/]")

    # ========================================================================
    # F) SLOT EFFECT NORM (NEW)
    # ========================================================================
    console.print("\n[bold cyan]═══ F) SLOT EFFECT NORM (real impact) ═══[/]")

    slot_effect = measure_slot_effect_norm(model, val_data, device)
    console.print(f"Effect entropy: {slot_effect['effect_entropy']:.2f}")
    console.print(f"Dominant slot: {slot_effect['dominant_slot']} (effect={slot_effect['max_effect']:.4f})")

    console.print("\n[dim]Slot effect distribution (% of total impact):[/]")
    for i, pct in enumerate(slot_effect['slot_effect_pct']):
        bar = '█' * int(pct * 50)
        console.print(f"  Slot {i}: {pct*100:.1f}% {bar}")

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
    summary_table.add_row(
        "Grad Correlation",
        "Mean grad corr",
        f"{grad_corr['mean_grad_correlation']:.3f}",
        "⚠" if grad_corr['mean_grad_correlation'] > 0.7 else "✓"
    )
    summary_table.add_row(
        "Rollout Stability",
        "KL divergence",
        f"{rollout_cmp['kl_divergence']:.4f}",
        "⚠" if not rollout_cmp['is_stable'] else "✓"
    )
    summary_table.add_row(
        "Effect Concentration",
        "Dominant slot %",
        f"{slot_effect['slot_effect_pct'][slot_effect['dominant_slot']]*100:.1f}%",
        "⚠" if slot_effect['slot_effect_pct'][slot_effect['dominant_slot']] > 0.5 else "✓"
    )

    console.print(summary_table)

    # ========================================================================
    # PLOTS
    # ========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # A) Attention distribution
    ax = axes[0, 0]
    ax.bar(range(n_slots), attn_conc['attn_dist'], color='steelblue', alpha=0.7)
    ax.axhline(y=1/n_slots, color='r', linestyle='--', label=f'Uniform ({1/n_slots:.2f})')
    ax.set_xlabel('Slot')
    ax.set_ylabel('Attention Mass')
    ax.set_title(f'A) Attention Distribution (entropy={attn_conc["entropy"]:.2f})')
    ax.legend()

    # B) Ablation importance
    ax = axes[0, 1]
    importances = [r['delta_nll'] for r in ablation['ablations']]
    colors = ['red' if imp > 0.01 else 'steelblue' for imp in importances]
    ax.bar(range(n_slots), importances, color=colors, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel('Slot')
    ax.set_ylabel('Δ NLL on removal')
    ax.set_title('B) Slot Importance (Ablation)')

    # C) Similarity heatmap
    ax = axes[1, 0]
    im = ax.imshow(similarity['similarity_matrix'], cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Slot')
    ax.set_ylabel('Slot')
    ax.set_title(f'C) Slot Similarity (mean={similarity["mean_similarity"]:.2f})')
    plt.colorbar(im, ax=ax)

    # D) Gradient correlation heatmap
    ax = axes[1, 1]
    im2 = ax.imshow(grad_corr['grad_correlation_matrix'], cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Slot')
    ax.set_ylabel('Slot')
    ax.set_title(f'D) Gradient Correlation (mean={grad_corr["mean_grad_correlation"]:.2f})')
    plt.colorbar(im2, ax=ax)

    # E) Train vs Rollout comparison
    ax = axes[2, 0]
    x_pos = np.arange(n_slots)
    width = 0.35
    ax.bar(x_pos - width/2, rollout_cmp['train_attn'], width, label='Train', color='steelblue', alpha=0.7)
    ax.bar(x_pos + width/2, rollout_cmp['rollout_attn'], width, label='Rollout', color='coral', alpha=0.7)
    ax.set_xlabel('Slot')
    ax.set_ylabel('Attention Mass')
    ax.set_title(f'E) Train vs Rollout (KL={rollout_cmp["kl_divergence"]:.4f})')
    ax.legend()
    ax.set_xticks(x_pos)

    # F) Slot Effect Norm
    ax = axes[2, 1]
    colors = ['red' if pct > 0.3 else 'steelblue' for pct in slot_effect['slot_effect_pct']]
    ax.bar(range(n_slots), slot_effect['slot_effect_pct'], color=colors, alpha=0.7)
    ax.axhline(y=1/n_slots, color='r', linestyle='--', label=f'Uniform ({1/n_slots:.2f})')
    ax.set_xlabel('Slot')
    ax.set_ylabel('Effect Share')
    ax.set_title(f'F) Slot Effect Norm (dominant={slot_effect["dominant_slot"]})')
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

## D) Gradient Correlation (learning together?)
| Metric | Value | Status |
|--------|-------|--------|
| Mean grad correlation | {grad_corr['mean_grad_correlation']:.3f} | {'⚠' if grad_corr['mean_grad_correlation'] > 0.7 else '✓'} |
| Max grad correlation | {grad_corr['max_grad_correlation']:.3f} | |
| Gradient leader | Slot {grad_corr['grad_leader']} (norm={grad_corr['grad_norms'][grad_corr['grad_leader']]:.4f}) | |

## E) Train vs Rollout Stability
| Metric | Value | Status |
|--------|-------|--------|
| KL divergence | {rollout_cmp['kl_divergence']:.4f} | {'⚠' if not rollout_cmp['is_stable'] else '✓'} |
| Max shift slot | Slot {rollout_cmp['max_shift_slot']} (Δ={rollout_cmp['max_shift_amount']:.3f}) | |
| Stable on rollout | {'NO ⚠' if not rollout_cmp['is_stable'] else 'YES ✓'} | |

## F) Slot Effect Norm (real impact)
| Metric | Value | Status |
|--------|-------|--------|
| Effect entropy | {slot_effect['effect_entropy']:.2f} | |
| Dominant slot | Slot {slot_effect['dominant_slot']} ({slot_effect['slot_effect_pct'][slot_effect['dominant_slot']]*100:.1f}%) | {'⚠' if slot_effect['slot_effect_pct'][slot_effect['dominant_slot']] > 0.5 else '✓'} |

## Summary
- **Shortcut**: {'⚠ Detected - model relies on few slots' if attn_conc['concentration_ratio'] > 0.5 else '✓ OK'}
- **Symmetry**: {'⚠ High - slots are redundant' if similarity['mean_similarity'] > 0.5 else '✓ OK'}
- **Co-adaptation**: {'⚠ Detected - slots are entangled' if independence['has_co_adaptation'] else '✓ OK'}
- **Grad Correlation**: {'⚠ Slots learning together' if grad_corr['mean_grad_correlation'] > 0.7 else '✓ Independent learning'}
- **Rollout Stability**: {'⚠ Slot usage shifts during rollout' if not rollout_cmp['is_stable'] else '✓ Stable'}
- **Effect Concentration**: {'⚠ One slot dominates' if slot_effect['slot_effect_pct'][slot_effect['dominant_slot']] > 0.5 else '✓ Distributed'}
"""

    with open(output_dir / "memory_diagnostics.md", "w") as f:
        f.write(report)

    console.print(f"[green]Saved memory_diagnostics.md[/]")

    # ========================================================================
    # JSONL LOGGING (for experiment tracking)
    # ========================================================================
    jsonl_row = {
        "ckpt": str(args.ckpt),
        "data_dir": str(args.data_dir),
        "n_slots": n_slots,
        # A) Shortcut
        "attn_entropy": attn_conc['entropy'],
        "concentration_ratio": attn_conc['concentration_ratio'],
        "top_slot": attn_conc['top_slot'],
        "top_slot_mass": attn_conc['top_slot_mass'],
        # B) Symmetry
        "mean_similarity": similarity['mean_similarity'],
        "max_similarity": similarity['max_similarity'],
        "permutation_invariant": perm_test['is_permutation_invariant'],
        # C) Co-adaptation
        "mean_correlation": correlations['mean_correlation'],
        "mean_redistribution": independence['mean_redistribution'],
        "has_co_adaptation": independence['has_co_adaptation'],
        # D) Gradient correlation
        "mean_grad_correlation": grad_corr['mean_grad_correlation'],
        "max_grad_correlation": grad_corr['max_grad_correlation'],
        "grad_leader": grad_corr['grad_leader'],
        "grad_norms": grad_corr['grad_norms'],
        # E) Train vs Rollout
        "rollout_kl_divergence": rollout_cmp['kl_divergence'],
        "rollout_is_stable": rollout_cmp['is_stable'],
        "rollout_max_shift_slot": rollout_cmp['max_shift_slot'],
        # F) Slot effect
        "effect_entropy": slot_effect['effect_entropy'],
        "dominant_slot": slot_effect['dominant_slot'],
        "max_effect_pct": slot_effect['slot_effect_pct'][slot_effect['dominant_slot']],
        "slot_effects": slot_effect['slot_effects'],
        # Ablation
        "base_nll": ablation['base_nll'],
        "ablation_deltas": [r['delta_nll'] for r in ablation['ablations']],
    }

    jsonl_path = output_dir / "memory_diagnostics.jsonl"
    write_jsonl(jsonl_path, jsonl_row)
    console.print(f"[green]Appended to {jsonl_path}[/]")

    console.print("\n[bold green]═══ Diagnostics Complete ═══[/]")


if __name__ == "__main__":
    main()
