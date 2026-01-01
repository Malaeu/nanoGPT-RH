#!/usr/bin/env python3
"""
diagnose_memory_postfix.py — Diagnostics for POSTFIX Memory Architecture (E3)

Key differences from PREFIX:
- Memory tokens are AFTER data: [s1..sT, M0..M7]
- Memory CAN see data (causal mask allows left-attention)
- Readout is ONLY from memory (bottleneck)
- Single-step prediction: predict s_{T+1} from window

Diagnostics:
A) Ablation: remove slot i → Δ(NLL)
B) Slot similarity (cosine between embeddings)
C) Gradient correlation (are slots learning together?)
D) Readout weight distribution (which slots matter for prediction?)
E) Slot effect norm (how much each slot affects output)

Usage:
  python diagnose_memory_postfix.py --ckpt out/mdn_postfix_E3_s1337/best.pt --data-dir data/continuous_2M
"""

import math
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Any, Dict
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import argparse

console = Console()

from train_mdn_postfix import SpacingMDNPostfix, MDNConfig, mdn_loss_1step


# ============================================================================
# HELPERS
# ============================================================================

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


def mdn_mean_1step(pi, mu, sigma):
    """Expected value of MDN for single-step prediction."""
    # pi, mu, sigma: [B, 1, K]
    return (pi * mu).sum(dim=-1).squeeze(-1)  # [B]


# ============================================================================
# A) ABLATION STUDY
# ============================================================================

def ablation_study(model, data, device, n_eval=500, seq_len=257):
    """
    Measure impact of removing each memory slot using slot_off parameter.

    For POSTFIX, we use the model's built-in slot_off parameter.
    """
    model.eval()
    n_slots = model.n_memory_slots

    # Get evaluation batch
    idx = torch.randint(0, len(data), (n_eval,))
    batch = data[idx].to(device)

    # Handle different data formats
    if batch.dim() == 2 and batch.shape[1] >= seq_len:
        # Chunked data - flatten and re-chunk
        flat = batch.flatten()
        x = flat[:n_eval * (seq_len - 1)].view(n_eval, seq_len - 1)
        y = flat[seq_len - 1:(n_eval + 1) * seq_len - 1:seq_len][:n_eval]
    else:
        x = batch[:, :seq_len - 1]
        y = batch[:, seq_len - 1] if batch.shape[1] >= seq_len else batch[:, -1]

    # Baseline (all slots active)
    with torch.no_grad():
        pi, mu, sigma = model(x)
        base_nll = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=0).item()
        base_pred = mdn_mean_1step(pi, mu, sigma)
        base_bias = (base_pred - y).mean().item()

    # Ablate each slot
    ablation_results = []

    for slot_idx in range(n_slots):
        with torch.no_grad():
            pi, mu, sigma = model(x, slot_off=slot_idx)
            abl_nll = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=0).item()
            abl_pred = mdn_mean_1step(pi, mu, sigma)
            abl_bias = (abl_pred - y).mean().item()

        delta_nll = abl_nll - base_nll
        delta_bias = abs(abl_bias) - abs(base_bias)

        ablation_results.append({
            "slot": slot_idx,
            "delta_nll": delta_nll,
            "delta_bias": delta_bias,
            "abl_nll": abl_nll,
        })

    return {
        "base_nll": base_nll,
        "base_bias": base_bias,
        "ablations": ablation_results,
        "total_importance": sum(r['delta_nll'] for r in ablation_results),
    }


# ============================================================================
# B) SLOT SIMILARITY
# ============================================================================

def measure_slot_similarity(model):
    """
    Compute cosine similarity between memory slots (including slot_id if used).
    """
    mem = model.memory_bank.memory.detach()  # [M, D]

    if model.memory_bank.use_slot_id:
        slot_ids = torch.arange(model.n_memory_slots, device=mem.device)
        mem = mem + model.memory_bank.slot_id(slot_ids)

    # Normalize
    mem_norm = F.normalize(mem, dim=-1)

    # Cosine similarity matrix
    sim = mem_norm @ mem_norm.T  # [M, M]

    n_slots = model.n_memory_slots
    mask = ~torch.eye(n_slots, dtype=bool, device=sim.device)
    off_diag = sim[mask]

    return {
        "similarity_matrix": sim.cpu().numpy(),
        "mean_similarity": off_diag.mean().item(),
        "max_similarity": off_diag.max().item(),
        "min_similarity": off_diag.min().item(),
    }


# ============================================================================
# C) GRADIENT CORRELATION
# ============================================================================

def measure_gradient_correlation(model, data, device, n_batches=10, seq_len=257):
    """
    Measure gradient correlation between memory slots.

    High correlation = slots are learning the same thing (bad differentiation).
    Low correlation = slots are learning different aspects (good!).
    """
    model.train()
    batch_size = 64
    n_slots = model.n_memory_slots

    slot_grads = [[] for _ in range(n_slots)]

    for i in range(n_batches):
        idx = torch.randint(0, len(data), (batch_size,))
        batch = data[idx].to(device)

        # Handle data format
        flat = batch.flatten()
        x = flat[:batch_size * (seq_len - 1)].view(batch_size, seq_len - 1)
        y = flat[seq_len - 1:(batch_size + 1) * seq_len - 1:seq_len][:batch_size]

        # Forward
        pi, mu, sigma = model(x)
        loss = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=0)

        # Backward
        model.zero_grad()
        loss.backward()

        # Collect gradients
        mem_grad = model.memory_bank.memory.grad  # [M, D]
        for slot_idx in range(n_slots):
            slot_grads[slot_idx].append(mem_grad[slot_idx].detach().cpu().clone())

    # Stack and compute correlations
    slot_grads = [torch.stack(grads) for grads in slot_grads]
    grad_norms = [g.norm(dim=-1).mean().item() for g in slot_grads]

    flat_grads = [g.flatten().numpy() for g in slot_grads]

    corr_matrix = np.zeros((n_slots, n_slots))
    for i in range(n_slots):
        for j in range(n_slots):
            if np.std(flat_grads[i]) > 1e-10 and np.std(flat_grads[j]) > 1e-10:
                corr_matrix[i, j] = np.corrcoef(flat_grads[i], flat_grads[j])[0, 1]
            else:
                corr_matrix[i, j] = 0.0

    mask = ~np.eye(n_slots, dtype=bool)
    off_diag = corr_matrix[mask]

    model.eval()

    return {
        "grad_norms": grad_norms,
        "grad_correlation_matrix": corr_matrix,
        "mean_grad_correlation": np.mean(np.abs(off_diag)),
        "max_grad_correlation": np.max(np.abs(off_diag)),
        "grad_leader": int(np.argmax(grad_norms)),
    }


# ============================================================================
# D) READOUT WEIGHT DISTRIBUTION
# ============================================================================

def analyze_readout_weights(model):
    """
    Analyze the learned readout weights (softmax over slots).

    This shows which slots the model actually uses for prediction.
    """
    weights = model.memory_bank.get_readout_weights().detach().cpu().numpy()

    # Entropy (higher = more uniform)
    entropy = -np.sum(weights * np.log(weights + 1e-10))
    max_entropy = np.log(len(weights))

    return {
        "readout_weights": weights,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "uniformity": entropy / max_entropy,  # 1 = uniform, 0 = single slot
        "dominant_slot": int(np.argmax(weights)),
        "dominant_weight": float(np.max(weights)),
    }


# ============================================================================
# E) SLOT EFFECT NORM
# ============================================================================

def measure_slot_effect_norm(model, data, device, n_eval=100, seq_len=257):
    """
    Measure how much each slot affects the output prediction.

    Uses ablation to measure L2 distance in prediction space.
    """
    model.eval()
    n_slots = model.n_memory_slots

    idx = torch.randint(0, len(data), (n_eval,))
    batch = data[idx].to(device)

    flat = batch.flatten()
    x = flat[:n_eval * (seq_len - 1)].view(n_eval, seq_len - 1)

    # Baseline prediction
    with torch.no_grad():
        pi_base, mu_base, sigma_base = model(x)
        pred_base = mdn_mean_1step(pi_base, mu_base, sigma_base)

    slot_effects = []

    for slot_idx in range(n_slots):
        with torch.no_grad():
            pi, mu, sigma = model(x, slot_off=slot_idx)
            pred = mdn_mean_1step(pi, mu, sigma)

        effect = (pred_base - pred).pow(2).mean().sqrt().item()
        slot_effects.append(effect)

    total_effect = sum(slot_effects) + 1e-10
    slot_effect_pct = [e / total_effect for e in slot_effects]

    return {
        "slot_effects": slot_effects,
        "slot_effect_pct": slot_effect_pct,
        "dominant_slot": int(np.argmax(slot_effects)),
        "effect_entropy": -sum(p * np.log(p + 1e-10) for p in slot_effect_pct),
        "max_effect": max(slot_effects),
    }


# ============================================================================
# F) SLOT NORMS
# ============================================================================

def measure_slot_norms(model):
    """Measure L2 norm of each slot embedding."""
    mem = model.memory_bank.memory.detach()  # [M, D]

    if model.memory_bank.use_slot_id:
        slot_ids = torch.arange(model.n_memory_slots, device=mem.device)
        mem = mem + model.memory_bank.slot_id(slot_ids)

    norms = mem.norm(dim=-1).cpu().numpy()

    return {
        "slot_norms": norms,
        "mean_norm": float(np.mean(norms)),
        "max_norm": float(np.max(norms)),
        "norm_std": float(np.std(norms)),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="POSTFIX Memory Diagnostics")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="reports/E3")
    parser.add_argument("--seq-len", type=int, default=257)
    args = parser.parse_args()

    console.print("[bold magenta]═══ POSTFIX Memory Diagnostics (E3) ═══[/]\n")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load checkpoint
    console.print(f"[cyan]Loading {args.ckpt}[/]")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # Check architecture
    arch = ckpt.get("architecture", "PREFIX")
    if arch != "POSTFIX":
        console.print(f"[yellow]Warning: checkpoint architecture is '{arch}', not POSTFIX[/]")

    # Reconstruct config
    config_data = ckpt["config"]
    if isinstance(config_data, dict):
        config = MDNConfig(**config_data)
    else:
        config = config_data

    n_slots = ckpt.get("n_memory_slots", 8)
    use_slot_id = ckpt.get("use_slot_id", True)

    console.print(f"[green]  Memory slots: {n_slots}[/]")
    console.print(f"[green]  Slot-ID: {use_slot_id}[/]")
    console.print(f"[green]  Architecture: {arch}[/]")

    # Create model
    model = SpacingMDNPostfix(
        config,
        n_memory_slots=n_slots,
        use_slot_id=use_slot_id
    ).to(device)

    # Load weights
    state_dict = ckpt["model"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Load data
    data_dir = Path(args.data_dir)
    val_data = torch.load(data_dir / "val.pt", weights_only=False)
    console.print(f"[green]Val data: {val_data.shape}[/]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # A) ABLATION
    # ========================================================================
    console.print("\n[bold cyan]═══ A) ABLATION (slot importance) ═══[/]")

    ablation = ablation_study(model, val_data, device, seq_len=args.seq_len)
    console.print(f"Base NLL: {ablation['base_nll']:.4f}")
    console.print(f"Total importance (Σ Δ): {ablation['total_importance']:.4f}")

    console.print("\n[dim]Per-slot ablation:[/]")
    for res in ablation['ablations']:
        bar_len = int(max(0, res['delta_nll']) * 200)
        bar = '█' * min(bar_len, 30)
        status = "🔴" if res['delta_nll'] > 0.01 else "🟡" if res['delta_nll'] > 0.001 else "⚪"
        console.print(f"  Slot {res['slot']}: Δ={res['delta_nll']:+.4f} {status} {bar}")

    # ========================================================================
    # B) SIMILARITY
    # ========================================================================
    console.print("\n[bold cyan]═══ B) SLOT SIMILARITY ═══[/]")

    similarity = measure_slot_similarity(model)
    console.print(f"Mean similarity: {similarity['mean_similarity']:.3f}")
    console.print(f"Max similarity: {similarity['max_similarity']:.3f}")

    if similarity['mean_similarity'] > 0.8:
        console.print("[red]⚠ HIGH: slots nearly identical[/]")
    elif similarity['mean_similarity'] > 0.5:
        console.print("[yellow]⚠ Moderate similarity[/]")
    else:
        console.print("[green]✓ Slots well differentiated[/]")

    # ========================================================================
    # C) GRADIENT CORRELATION
    # ========================================================================
    console.print("\n[bold cyan]═══ C) GRADIENT CORRELATION ═══[/]")

    grad_corr = measure_gradient_correlation(model, val_data, device, seq_len=args.seq_len)
    console.print(f"Mean grad correlation: {grad_corr['mean_grad_correlation']:.3f}")
    console.print(f"Max grad correlation: {grad_corr['max_grad_correlation']:.3f}")
    console.print(f"Gradient leader: Slot {grad_corr['grad_leader']}")

    if grad_corr['mean_grad_correlation'] > 0.7:
        console.print("[yellow]⚠ Slots learning together (high correlation)[/]")
    else:
        console.print("[green]✓ Slots learning independently[/]")

    # ========================================================================
    # D) READOUT WEIGHTS
    # ========================================================================
    console.print("\n[bold cyan]═══ D) READOUT WEIGHTS ═══[/]")

    readout = analyze_readout_weights(model)
    console.print(f"Uniformity: {readout['uniformity']:.2f} (1=uniform, 0=single slot)")
    console.print(f"Dominant slot: {readout['dominant_slot']} (weight={readout['dominant_weight']:.3f})")

    console.print("\n[dim]Weight distribution:[/]")
    for i, w in enumerate(readout['readout_weights']):
        bar_len = int(w * 40)
        bar = '█' * bar_len
        console.print(f"  Slot {i}: {w:.4f} {bar}")

    # ========================================================================
    # E) SLOT EFFECT NORM
    # ========================================================================
    console.print("\n[bold cyan]═══ E) SLOT EFFECT NORM ═══[/]")

    slot_effect = measure_slot_effect_norm(model, val_data, device, seq_len=args.seq_len)
    console.print(f"Effect entropy: {slot_effect['effect_entropy']:.2f}")
    console.print(f"Dominant slot: {slot_effect['dominant_slot']} (effect={slot_effect['max_effect']:.4f})")

    console.print("\n[dim]Effect distribution (% of total):[/]")
    for i, pct in enumerate(slot_effect['slot_effect_pct']):
        bar_len = int(pct * 50)
        bar = '█' * bar_len
        console.print(f"  Slot {i}: {pct*100:.1f}% {bar}")

    # ========================================================================
    # F) SLOT NORMS
    # ========================================================================
    console.print("\n[bold cyan]═══ F) SLOT NORMS ═══[/]")

    norms = measure_slot_norms(model)
    console.print(f"Mean norm: {norms['mean_norm']:.3f}")
    console.print(f"Norm std: {norms['norm_std']:.3f}")

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    console.print("\n[bold cyan]═══ SUMMARY ═══[/]")

    summary_table = Table(title="POSTFIX Memory Diagnostics")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white")
    summary_table.add_column("Status", style="bold")

    # Key comparison with E1/E2 expectations
    max_ablation_delta = max(r['delta_nll'] for r in ablation['ablations'])

    summary_table.add_row(
        "Max Ablation Δ",
        f"{max_ablation_delta:.4f}",
        "✓ GOOD" if max_ablation_delta > 0.01 else "⚠ low"
    )
    summary_table.add_row(
        "Mean Grad Corr",
        f"{grad_corr['mean_grad_correlation']:.3f}",
        "✓ GOOD" if grad_corr['mean_grad_correlation'] < 0.7 else "⚠ high"
    )
    summary_table.add_row(
        "Mean Slot Sim",
        f"{similarity['mean_similarity']:.3f}",
        "✓" if similarity['mean_similarity'] < 0.5 else "⚠"
    )
    summary_table.add_row(
        "Readout Uniformity",
        f"{readout['uniformity']:.2f}",
        "✓" if readout['uniformity'] > 0.5 else "⚠"
    )
    summary_table.add_row(
        "Effect Entropy",
        f"{slot_effect['effect_entropy']:.2f}",
        "✓" if slot_effect['effect_entropy'] > 1.5 else "⚠"
    )

    console.print(summary_table)

    # ========================================================================
    # COMPARISON NOTE
    # ========================================================================
    console.print("\n[bold yellow]═══ E1/E2 vs E3 Expectations ═══[/]")
    console.print("""
E1/E2 (PREFIX) had:
  - Ablation Δ ≈ 0 (memory not causal)
  - Grad corr ≈ 0.9 (slots learning same thing)

E3 (POSTFIX) should have:
  - Ablation Δ > 0.01 (memory is essential)
  - Grad corr < 0.7 (slots learning different things)
""")

    if max_ablation_delta > 0.01 and grad_corr['mean_grad_correlation'] < 0.7:
        console.print("[bold green]🎉 POSTFIX WORKING! Memory is now causal![/]")
    elif max_ablation_delta > 0.01:
        console.print("[yellow]Ablation improved, but grad corr still high[/]")
    else:
        console.print("[red]⚠ Ablation still low - investigate[/]")

    # ========================================================================
    # PLOTS
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # A) Ablation
    ax = axes[0, 0]
    deltas = [r['delta_nll'] for r in ablation['ablations']]
    colors = ['red' if d > 0.01 else 'orange' if d > 0.001 else 'gray' for d in deltas]
    ax.bar(range(n_slots), deltas, color=colors, alpha=0.7)
    ax.axhline(y=0.01, color='r', linestyle='--', label='Threshold')
    ax.set_xlabel('Slot')
    ax.set_ylabel('Δ NLL')
    ax.set_title(f'A) Ablation (max={max_ablation_delta:.4f})')
    ax.legend()

    # B) Similarity heatmap
    ax = axes[0, 1]
    im = ax.imshow(similarity['similarity_matrix'], cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Slot')
    ax.set_ylabel('Slot')
    ax.set_title(f'B) Slot Similarity (mean={similarity["mean_similarity"]:.2f})')
    plt.colorbar(im, ax=ax)

    # C) Gradient correlation
    ax = axes[0, 2]
    im2 = ax.imshow(grad_corr['grad_correlation_matrix'], cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Slot')
    ax.set_ylabel('Slot')
    ax.set_title(f'C) Grad Correlation (mean={grad_corr["mean_grad_correlation"]:.2f})')
    plt.colorbar(im2, ax=ax)

    # D) Readout weights
    ax = axes[1, 0]
    ax.bar(range(n_slots), readout['readout_weights'], color='steelblue', alpha=0.7)
    ax.axhline(y=1/n_slots, color='r', linestyle='--', label='Uniform')
    ax.set_xlabel('Slot')
    ax.set_ylabel('Weight')
    ax.set_title(f'D) Readout Weights (uniformity={readout["uniformity"]:.2f})')
    ax.legend()

    # E) Slot effect
    ax = axes[1, 1]
    ax.bar(range(n_slots), slot_effect['slot_effect_pct'], color='coral', alpha=0.7)
    ax.axhline(y=1/n_slots, color='r', linestyle='--', label='Uniform')
    ax.set_xlabel('Slot')
    ax.set_ylabel('Effect %')
    ax.set_title(f'E) Slot Effect (entropy={slot_effect["effect_entropy"]:.2f})')
    ax.legend()

    # F) Slot norms
    ax = axes[1, 2]
    ax.bar(range(n_slots), norms['slot_norms'], color='green', alpha=0.7)
    ax.set_xlabel('Slot')
    ax.set_ylabel('L2 Norm')
    ax.set_title(f'F) Slot Norms (mean={norms["mean_norm"]:.2f})')

    plt.tight_layout()
    plt.savefig(output_dir / "postfix_diagnostics.png", dpi=150)
    plt.close()
    console.print(f"\n[green]Saved postfix_diagnostics.png[/]")

    # ========================================================================
    # JSONL LOG
    # ========================================================================
    jsonl_row = {
        "ckpt": str(args.ckpt),
        "architecture": "POSTFIX",
        "n_slots": n_slots,
        "base_nll": ablation['base_nll'],
        "max_ablation_delta": max_ablation_delta,
        "total_importance": ablation['total_importance'],
        "mean_similarity": similarity['mean_similarity'],
        "mean_grad_correlation": grad_corr['mean_grad_correlation'],
        "readout_uniformity": readout['uniformity'],
        "effect_entropy": slot_effect['effect_entropy'],
        "ablation_deltas": deltas,
        "readout_weights": readout['readout_weights'].tolist(),
        "slot_effects": slot_effect['slot_effects'],
    }

    jsonl_path = output_dir / "postfix_diagnostics.jsonl"
    write_jsonl(jsonl_path, jsonl_row)
    console.print(f"[green]Appended to {jsonl_path}[/]")

    console.print("\n[bold green]═══ Diagnostics Complete ═══[/]")


if __name__ == "__main__":
    main()
