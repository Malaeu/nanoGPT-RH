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

from train_mdn_postfix import SpacingMDNPostfix, mdn_loss_1step
from train_mdn import MDNConfig


# ============================================================================
# HELPERS
# ============================================================================

def sample_xy(data: torch.Tensor, n: int, T: int = 256, stride: int = 1, seed: int = None):
    """
    Unified sampler for (context, target) pairs.

    Args:
        data: 1D or 2D tensor. If 2D [N, chunk_len], flattens first.
        n: number of samples to return
        T: context length (x will have T tokens, y is the T+1-th)
        stride: step between consecutive samples
        seed: optional random seed for reproducibility

    Returns:
        x: [n, T] context windows
        y: [n] target values
    """
    # Flatten 2D data to 1D signal
    s = data.flatten()
    total_len = len(s)

    # Valid start indices: can sample x[i:i+T] and y=s[i+T]
    max_start = total_len - T - 1
    if max_start < n * stride:
        # Not enough data for strided sampling, use random
        if seed is not None:
            torch.manual_seed(seed)
        starts = torch.randint(0, max_start + 1, (n,))
    else:
        # Strided sampling for even coverage
        if seed is not None:
            torch.manual_seed(seed)
        offset = torch.randint(0, stride, (1,)).item()
        all_starts = torch.arange(offset, max_start + 1, stride)
        perm = torch.randperm(len(all_starts))[:n]
        starts = all_starts[perm]

    # Extract (x, y) pairs
    x = torch.stack([s[i:i+T] for i in starts])  # [n, T]
    y = torch.stack([s[i+T] for i in starts])     # [n]

    return x, y


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

def ablation_study(model, data, device, n_eval=500, T=256):
    """
    Measure impact of removing each memory slot using slot_off parameter.

    For POSTFIX, we use the model's built-in slot_off parameter.
    """
    model.eval()
    n_slots = model.n_memory_slots

    # Use unified sampler
    x, y = sample_xy(data, n_eval, T=T)
    x, y = x.to(device), y.to(device)

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

def measure_gradient_correlation(model, data, device, n_batches=10, T=256):
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
        # Use unified sampler
        x, y = sample_xy(data, batch_size, T=T)
        x, y = x.to(device), y.to(device)

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

def measure_slot_effect_norm(model, data, device, n_eval=100, T=256):
    """
    Measure how much each slot affects the output prediction.

    Uses ablation to measure L2 distance in prediction space.
    """
    model.eval()
    n_slots = model.n_memory_slots

    # Use unified sampler (y not needed for this test)
    x, _ = sample_xy(data, n_eval, T=T)
    x = x.to(device)

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
# G) ROLLOUT DRIFT (Err@h)
# ============================================================================

def measure_rollout_drift(model, data, device, horizons=[1, 10, 50, 200], n_eval=100, T=256):
    """
    Measure error accumulation during autoregressive rollout.

    Key insight: if model learned "real law", error grows slowly.
    If just local fit, error explodes quickly.

    Returns Err@h for each horizon h.
    """
    model.eval()

    # Flatten data once (critical for 2D data!)
    s = data.flatten()
    max_h = max(horizons)
    total_len = len(s)

    # Valid start positions
    max_start = total_len - T - max_h - 1
    starts = torch.randint(0, max_start + 1, (n_eval,))

    results = {h: {"mae": [], "bias": []} for h in horizons}

    with torch.no_grad():
        for i in starts:
            # Get ground truth sequence (enough for max horizon)
            gt_seq = s[i:i + T + max_h].to(device)

            # Start with true context
            context = gt_seq[:T].unsqueeze(0)  # [1, T]

            pred_seq = []
            for step in range(max(horizons)):
                pi, mu, sigma = model(context)
                # Use mean prediction
                pred = mdn_mean_1step(pi, mu, sigma)
                pred_seq.append(pred.item())

                # Shift context and add prediction
                # pred is [1] (batch), need to reshape to [1, 1] for cat
                context = torch.cat([context[:, 1:], pred.view(1, 1)], dim=1)

            # Compute errors at each horizon
            for h in horizons:
                pred_val = pred_seq[h - 1]
                true_val = gt_seq[T + h - 1].item()
                error = pred_val - true_val
                results[h]["mae"].append(abs(error))
                results[h]["bias"].append(error)

    # Aggregate
    output = {}
    for h in horizons:
        mae = np.mean(results[h]["mae"])
        bias = np.mean(results[h]["bias"])
        std = np.std(results[h]["mae"])
        output[f"err@{h}"] = {"mae": mae, "bias": bias, "std": std}

    # Error growth slope (linear fit of log(mae) vs log(h))
    maes = [output[f"err@{h}"]["mae"] for h in horizons]
    if min(maes) > 0:
        log_h = np.log(horizons)
        log_mae = np.log(maes)
        slope, intercept = np.polyfit(log_h, log_mae, 1)
        output["error_growth_slope"] = slope
    else:
        output["error_growth_slope"] = 0.0

    return output


# ============================================================================
# H) CROSS-BLOCK TEST (distribution shift)
# ============================================================================

def cross_block_test(model, data, device, n_blocks=4, n_eval_per_block=100, T=256):
    """
    Test NLL on different contiguous blocks of data.

    If model learned index/position rather than spacing statistics,
    NLL will vary significantly across blocks.
    """
    model.eval()

    # Flatten data once (critical for 2D data!)
    s = data.flatten()
    total_len = len(s)
    block_size = total_len // n_blocks

    block_results = []

    for block_idx in range(n_blocks):
        start = block_idx * block_size
        end = start + block_size - T - 1

        if end <= start:
            continue

        block_nlls = []
        for _ in range(n_eval_per_block):
            idx = np.random.randint(start, end)
            x = s[idx:idx + T].unsqueeze(0).to(device)
            y = s[idx + T].unsqueeze(0).to(device)

            with torch.no_grad():
                pi, mu, sigma = model(x)
                nll = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=0).item()
                block_nlls.append(nll)

        block_results.append({
            "block": block_idx,
            "start_pct": start / total_len * 100,
            "mean_nll": np.mean(block_nlls),
            "std_nll": np.std(block_nlls),
        })

    # Check for distribution shift
    nlls = [b["mean_nll"] for b in block_results]
    nll_range = max(nlls) - min(nlls)
    nll_cv = np.std(nlls) / (np.mean(nlls) + 1e-10)  # coefficient of variation

    return {
        "blocks": block_results,
        "nll_range": nll_range,
        "nll_cv": nll_cv,
        "has_distribution_shift": nll_cv > 0.1,
    }


# ============================================================================
# I) SLOT ATTENTION PROFILE
# ============================================================================

def measure_slot_attention_profile(model, data, device, n_eval=50, T=256, attn_layers="last"):
    """
    Analyze where each memory slot attends in the data sequence.

    For POSTFIX: memory slots are at positions T..T+M-1
    They can attend to data positions 0..T-1

    Args:
        attn_layers: "last" (only last layer), "mean3" (last 3), "meanAll" (all layers)

    Returns center-of-mass and effective receptive field per slot.
    """
    model.eval()
    M = model.n_memory_slots

    # Use unified sampler (y not needed for attention analysis)
    x, _ = sample_xy(data, n_eval, T=T)
    x = x.to(device)

    # Get attention weights
    with torch.no_grad():
        pi, mu, sigma, attentions = model(x, return_attention=True)

    # attentions is list of [B, n_head, L, L] per layer
    # L = T + M, memory positions are T:T+M

    # Aggregate attention across layers based on attn_layers mode
    if attn_layers == "last":
        attn_agg = attentions[-1]  # [B, n_head, T+M, T+M]
    elif attn_layers == "mean3":
        # Average last 3 layers (or all if fewer)
        n_layers = min(3, len(attentions))
        attn_agg = torch.stack(attentions[-n_layers:]).mean(dim=0)
    elif attn_layers == "meanAll":
        attn_agg = torch.stack(attentions).mean(dim=0)
    else:
        attn_agg = attentions[-1]

    # Average over heads
    attn_avg = attn_agg.mean(dim=1)  # [B, T+M, T+M]

    # Extract memory→data attention: rows T:T+M, cols 0:T
    mem_to_data = attn_avg[:, T:T+M, :T]  # [B, M, T]

    # Average over batch
    mem_to_data_avg = mem_to_data.mean(dim=0).cpu().numpy()  # [M, T]

    slot_profiles = []
    positions = np.arange(T)

    for slot_idx in range(M):
        attn_dist = mem_to_data_avg[slot_idx]  # [T]
        attn_dist = attn_dist / (attn_dist.sum() + 1e-10)  # normalize

        # Center of mass
        com = np.sum(positions * attn_dist)

        # Effective receptive field (positions with >5% of max attention)
        threshold = 0.05 * attn_dist.max()
        active_positions = np.where(attn_dist > threshold)[0]
        if len(active_positions) > 0:
            rf_start = active_positions[0]
            rf_end = active_positions[-1]
            rf_width = rf_end - rf_start + 1
        else:
            rf_start, rf_end, rf_width = 0, 0, 0

        # Entropy of attention
        entropy = -np.sum(attn_dist * np.log(attn_dist + 1e-10))

        slot_profiles.append({
            "slot": slot_idx,
            "center_of_mass": com,
            "rf_start": rf_start,
            "rf_end": rf_end,
            "rf_width": rf_width,
            "entropy": entropy,
            "attention_distribution": attn_dist,
        })

    # Check if slots specialize (different centers of mass)
    coms = [s["center_of_mass"] for s in slot_profiles]
    com_range = max(coms) - min(coms)
    com_std = np.std(coms)

    return {
        "slot_profiles": slot_profiles,
        "com_range": com_range,
        "com_std": com_std,
        "slots_specialize": com_std > T * 0.1,  # >10% of window spread
    }


# ============================================================================
# J) PERMUTATION SANITY (Slot-ID shuffle)
# ============================================================================

def permutation_sanity_test(model, data, device, n_perms=5, n_eval=200, T=256):
    """
    Test if model relies on slot-ID embeddings as a crutch.

    Shuffle slot IDs at inference → if NLL explodes, model uses ID
    instead of learned content.
    """
    model.eval()

    if not model.memory_bank.use_slot_id:
        return {"skipped": True, "reason": "no slot_id embeddings"}

    # Use unified sampler
    x, y = sample_xy(data, n_eval, T=T)
    x, y = x.to(device), y.to(device)

    # Baseline NLL
    with torch.no_grad():
        pi, mu, sigma = model(x)
        base_nll = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=0).item()

    # Save original slot_id weights
    original_weight = model.memory_bank.slot_id.weight.data.clone()

    perm_nlls = []
    for _ in range(n_perms):
        # Shuffle slot_id embeddings (wrap in no_grad for safety)
        with torch.no_grad():
            perm = torch.randperm(model.n_memory_slots)
            model.memory_bank.slot_id.weight.data = original_weight[perm]

            pi, mu, sigma = model(x)
            perm_nll = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=0).item()
        perm_nlls.append(perm_nll)

    # Restore (in no_grad)
    with torch.no_grad():
        model.memory_bank.slot_id.weight.data = original_weight

    mean_perm_nll = np.mean(perm_nlls)
    nll_increase = mean_perm_nll - base_nll
    relative_increase = nll_increase / (abs(base_nll) + 1e-10)

    return {
        "base_nll": base_nll,
        "mean_perm_nll": mean_perm_nll,
        "nll_increase": nll_increase,
        "relative_increase": relative_increase,
        "relies_on_slot_id": relative_increase > 0.1,  # >10% degradation
    }


# ============================================================================
# K) GRADIENT RANK / PCA
# ============================================================================

def measure_gradient_rank(model, data, device, n_batches=20, T=256):
    """
    Measure effective dimensionality of slot gradients via PCA.

    If slots are redundant, gradient matrix has low rank.
    If slots learn independent things, higher rank.

    Returns r90 = number of components explaining 90% variance.
    """
    model.train()
    batch_size = 32
    n_slots = model.n_memory_slots

    all_grads = []

    for _ in range(n_batches):
        # Use unified sampler
        x, y = sample_xy(data, batch_size, T=T)
        x, y = x.to(device), y.to(device)

        pi, mu, sigma = model(x)
        loss = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=0)

        model.zero_grad()
        loss.backward()

        mem_grad = model.memory_bank.memory.grad.detach().cpu().numpy()  # [M, D]
        all_grads.append(mem_grad)

    # Stack: [n_batches, M, D]
    all_grads = np.stack(all_grads)

    # Reshape to [n_batches * M, D] for PCA
    G = all_grads.reshape(-1, all_grads.shape[-1])

    # PCA
    G_centered = G - G.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(G_centered, full_matrices=False)
    except:
        model.eval()
        return {"error": "SVD failed", "r90": n_slots}

    # Explained variance
    var_explained = S ** 2
    var_cumsum = np.cumsum(var_explained)
    var_total = var_cumsum[-1]

    # r90: components for 90% variance
    r90 = np.searchsorted(var_cumsum, 0.9 * var_total) + 1
    r50 = np.searchsorted(var_cumsum, 0.5 * var_total) + 1

    # Effective rank (entropy-based)
    p = var_explained / (var_total + 1e-10)
    effective_rank = np.exp(-np.sum(p * np.log(p + 1e-10)))

    model.eval()

    return {
        "r90": int(r90),
        "r50": int(r50),
        "effective_rank": effective_rank,
        "n_slots": n_slots,
        "rank_ratio": r90 / n_slots,  # 1 = full rank, 0 = all same
        "top_5_variance_pct": var_cumsum[min(4, len(var_cumsum)-1)] / var_total * 100,
    }


# ============================================================================
# SEED AGGREGATION (--ckpt-glob mode)
# ============================================================================

def run_seed_aggregation(args):
    """
    Run diagnostics on multiple checkpoints and aggregate results.

    Usage: --ckpt-glob "out/mdn_postfix_E3_s*/best.pt"
    """
    import glob

    console.print("[bold magenta]═══ POSTFIX Seed Aggregation Mode ═══[/]\n")

    # Find checkpoints
    ckpt_paths = sorted(glob.glob(args.ckpt_glob))
    if not ckpt_paths:
        console.print(f"[red]No checkpoints found matching: {args.ckpt_glob}[/]")
        return

    console.print(f"[cyan]Found {len(ckpt_paths)} checkpoints:[/]")
    for p in ckpt_paths:
        console.print(f"  - {p}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    console.print(f"\n[cyan]Device: {device}[/]")

    # Load data
    val_path = Path(args.data_dir) / "val.pt"
    val_data = torch.load(val_path, map_location="cpu", weights_only=True)
    console.print(f"[cyan]Val data: {val_data.shape}[/]")

    T = args.seq_len - 1

    # Collect metrics from each checkpoint
    all_metrics = []

    for ckpt_path in ckpt_paths:
        console.print(f"\n[bold cyan]── Processing: {ckpt_path} ──[/]")

        # Extract seed from path (e.g., "s1337" -> "1337")
        seed_match = Path(ckpt_path).parent.name
        seed = seed_match.split("_s")[-1] if "_s" in seed_match else seed_match

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            # Rebuild model - config may be MDNConfig object or dict
            cfg = ckpt["config"]
            if hasattr(cfg, 'n_embd'):
                # MDNConfig object
                config = cfg
                n_memory_slots = getattr(cfg, 'n_memory_slots', 8)
                use_slot_id = getattr(cfg, 'use_slot_id', True)
            else:
                # Dict
                config = MDNConfig(
                    n_embd=cfg["n_embd"],
                    n_head=cfg["n_head"],
                    n_layer=cfg["n_layer"],
                    block_size=cfg["block_size"],
                    n_mdn_components=cfg["n_mdn_components"],
                    dropout=0.0,
                )
                n_memory_slots = cfg.get("n_memory_slots", 8)
                use_slot_id = cfg.get("use_slot_id", True)

            model = SpacingMDNPostfix(
                config,
                n_memory_slots=n_memory_slots,
                use_slot_id=use_slot_id,
            )
            model.load_state_dict(ckpt["model"])
            model = model.to(device)
            model.eval()

            # Run key diagnostics (subset for speed)
            ablation = ablation_study(model, val_data, device, T=T, n_eval=200)
            grad_corr = measure_gradient_correlation(model, val_data, device, T=T, n_batches=5)
            rollout = measure_rollout_drift(model, val_data, device, T=T, n_eval=50)
            perm_test = permutation_sanity_test(model, val_data, device, T=T, n_eval=100)
            grad_rank = measure_gradient_rank(model, val_data, device, T=T, n_batches=10)

            metrics = {
                "seed": seed,
                "ckpt": ckpt_path,
                "val_nll": ablation["base_nll"],
                "max_ablation_delta": max(r["delta_nll"] for r in ablation["ablations"]),
                "mean_grad_corr": grad_corr["mean_grad_correlation"],
                "error_growth_slope": rollout["error_growth_slope"],
                "perm_relative_increase": perm_test.get("relative_increase", None),
                "gradient_rank_ratio": grad_rank.get("rank_ratio", None),
            }
            all_metrics.append(metrics)
            console.print(f"  [green]✓[/] NLL={metrics['val_nll']:.4f}, ablation_Δ={metrics['max_ablation_delta']:.4f}")

        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/]")
            continue

    if not all_metrics:
        console.print("[red]No successful runs![/]")
        return

    # Aggregate table
    console.print("\n[bold magenta]═══ SEED COMPARISON TABLE ═══[/]\n")

    table = Table(title="Seed Comparison")
    table.add_column("Seed", style="cyan")
    table.add_column("Val NLL", style="green")
    table.add_column("Ablation Δ", style="yellow")
    table.add_column("Grad Corr", style="blue")
    table.add_column("Err Slope", style="magenta")
    table.add_column("Perm Inc %", style="red")
    table.add_column("Rank Ratio", style="white")

    for m in all_metrics:
        table.add_row(
            m["seed"],
            f"{m['val_nll']:.4f}",
            f"{m['max_ablation_delta']:.4f}",
            f"{m['mean_grad_corr']:.3f}",
            f"{m['error_growth_slope']:.3f}",
            f"{m['perm_relative_increase']*100:.1f}%" if m['perm_relative_increase'] else "N/A",
            f"{m['gradient_rank_ratio']:.2f}" if m['gradient_rank_ratio'] else "N/A",
        )

    # Add mean±std row
    def agg(key):
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

    table.add_row(
        "[bold]mean±std[/]",
        agg("val_nll"),
        agg("max_ablation_delta"),
        agg("mean_grad_corr"),
        agg("error_growth_slope"),
        agg("perm_relative_increase"),
        agg("gradient_rank_ratio"),
        style="bold",
    )

    console.print(table)

    # Save aggregated results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_path = output_dir / "seed_aggregation.json"
    with open(agg_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    console.print(f"\n[green]Saved aggregation to {agg_path}[/]")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="POSTFIX Memory Diagnostics")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Single checkpoint path")
    parser.add_argument("--ckpt-glob", type=str, default=None,
                        help="Glob pattern for multiple checkpoints (e.g., 'out/mdn_postfix_E3_s*/best.pt')")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="reports/E3")
    parser.add_argument("--seq-len", type=int, default=257)
    parser.add_argument("--attn-layers", type=str, default="last",
                        choices=["last", "mean3", "meanAll"],
                        help="How to aggregate attention layers for profile analysis")
    args = parser.parse_args()

    # Validate: need either --ckpt or --ckpt-glob
    if args.ckpt is None and args.ckpt_glob is None:
        parser.error("Either --ckpt or --ckpt-glob is required")

    # If glob pattern, run aggregation mode
    if args.ckpt_glob:
        run_seed_aggregation(args)
        return

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

    T = args.seq_len - 1  # context length
    ablation = ablation_study(model, val_data, device, T=T)
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

    grad_corr = measure_gradient_correlation(model, val_data, device, T=T)
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

    slot_effect = measure_slot_effect_norm(model, val_data, device, T=T)
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
    # G) ROLLOUT DRIFT
    # ========================================================================
    console.print("\n[bold cyan]═══ G) ROLLOUT DRIFT (Err@h) ═══[/]")

    rollout = measure_rollout_drift(model, val_data, device, T=T)
    console.print(f"Error growth slope: {rollout['error_growth_slope']:.3f}")
    console.print("\n[dim]Errors at horizon h:[/]")
    for key in sorted([k for k in rollout.keys() if k.startswith("err@")]):
        h = int(key.split("@")[1])
        err = rollout[key]
        console.print(f"  {key}: MAE={err['mae']:.4f}, bias={err['bias']:+.4f}")

    if rollout['error_growth_slope'] < 0.5:
        console.print("[green]✓ Low error growth - model generalizes well[/]")
    else:
        console.print("[yellow]⚠ High error growth - may be overfitting[/]")

    # ========================================================================
    # H) CROSS-BLOCK TEST
    # ========================================================================
    console.print("\n[bold cyan]═══ H) CROSS-BLOCK TEST ═══[/]")

    cross_block = cross_block_test(model, val_data, device, T=T)
    console.print(f"NLL range across blocks: {cross_block['nll_range']:.4f}")
    console.print(f"NLL coefficient of variation: {cross_block['nll_cv']:.3f}")

    console.print("\n[dim]Per-block NLL:[/]")
    for b in cross_block['blocks']:
        console.print(f"  Block {b['block']} ({b['start_pct']:.0f}%): NLL={b['mean_nll']:.4f} ± {b['std_nll']:.4f}")

    if cross_block['has_distribution_shift']:
        console.print("[yellow]⚠ Distribution shift detected across blocks[/]")
    else:
        console.print("[green]✓ Consistent across data blocks[/]")

    # ========================================================================
    # I) SLOT ATTENTION PROFILE
    # ========================================================================
    console.print("\n[bold cyan]═══ I) SLOT ATTENTION PROFILE ═══[/]")

    attn_profile = measure_slot_attention_profile(model, val_data, device, T=T, attn_layers=args.attn_layers)
    console.print(f"Attention layers mode: {args.attn_layers}")
    console.print(f"Center-of-mass range: {attn_profile['com_range']:.1f} positions")
    console.print(f"Center-of-mass std: {attn_profile['com_std']:.1f}")

    console.print("\n[dim]Slot receptive fields:[/]")
    for sp in attn_profile['slot_profiles']:
        console.print(f"  Slot {sp['slot']}: CoM={sp['center_of_mass']:.1f}, RF=[{sp['rf_start']}-{sp['rf_end']}] width={sp['rf_width']}")

    if attn_profile['slots_specialize']:
        console.print("[green]✓ Slots specialize to different positions[/]")
    else:
        console.print("[yellow]⚠ Slots have similar attention patterns[/]")

    # ========================================================================
    # J) PERMUTATION SANITY
    # ========================================================================
    console.print("\n[bold cyan]═══ J) PERMUTATION SANITY (Slot-ID) ═══[/]")

    perm_test = permutation_sanity_test(model, val_data, device, T=T)
    if perm_test.get("skipped"):
        console.print(f"[dim]Skipped: {perm_test['reason']}[/]")
    else:
        console.print(f"Base NLL: {perm_test['base_nll']:.4f}")
        console.print(f"Permuted NLL: {perm_test['mean_perm_nll']:.4f}")
        console.print(f"Relative increase: {perm_test['relative_increase']*100:.1f}%")

        if perm_test['relies_on_slot_id']:
            console.print("[yellow]⚠ Model relies heavily on slot-ID (>10% degradation)[/]")
        else:
            console.print("[green]✓ Model uses slot content, not just ID[/]")

    # ========================================================================
    # K) GRADIENT RANK / PCA
    # ========================================================================
    console.print("\n[bold cyan]═══ K) GRADIENT RANK (PCA) ═══[/]")

    grad_rank = measure_gradient_rank(model, val_data, device, T=T)
    if "error" in grad_rank:
        console.print(f"[red]{grad_rank['error']}[/]")
    else:
        console.print(f"Components for 90% variance (r90): {grad_rank['r90']} / {grad_rank['n_slots']}")
        console.print(f"Components for 50% variance (r50): {grad_rank['r50']}")
        console.print(f"Effective rank: {grad_rank['effective_rank']:.2f}")
        console.print(f"Top-5 components explain: {grad_rank['top_5_variance_pct']:.1f}%")

        if grad_rank['rank_ratio'] > 0.5:
            console.print("[green]✓ High effective rank - slots learn different things[/]")
        else:
            console.print("[yellow]⚠ Low rank - slots may be redundant[/]")

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
    summary_table.add_row(
        "Error Growth Slope",
        f"{rollout['error_growth_slope']:.3f}",
        "✓" if rollout['error_growth_slope'] < 0.5 else "⚠"
    )
    summary_table.add_row(
        "Cross-Block CV",
        f"{cross_block['nll_cv']:.3f}",
        "✓" if not cross_block['has_distribution_shift'] else "⚠"
    )
    summary_table.add_row(
        "Slots Specialize",
        f"CoM std={attn_profile['com_std']:.1f}",
        "✓" if attn_profile['slots_specialize'] else "⚠"
    )
    if not perm_test.get("skipped"):
        summary_table.add_row(
            "Slot-ID Reliance",
            f"{perm_test['relative_increase']*100:.1f}%",
            "✓" if not perm_test['relies_on_slot_id'] else "⚠"
        )
    if "error" not in grad_rank:
        summary_table.add_row(
            "Gradient Rank (r90)",
            f"{grad_rank['r90']}/{grad_rank['n_slots']}",
            "✓" if grad_rank['rank_ratio'] > 0.5 else "⚠"
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
        # A-F metrics
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
        # G) Rollout drift
        "error_growth_slope": rollout['error_growth_slope'],
        "err_at_horizons": {k: v for k, v in rollout.items() if k.startswith("err@")},
        "bias_drift": rollout.get('bias_drift', None),
        # H) Cross-block
        "cross_block_cv": cross_block['nll_cv'],
        "has_distribution_shift": cross_block['has_distribution_shift'],
        "block_results": cross_block['blocks'],
        # I) Attention profile
        "slot_attention_com_std": attn_profile['com_std'],
        "slots_specialize": attn_profile['slots_specialize'],
        "slot_profiles": attn_profile['slot_profiles'],
        # J) Permutation test
        "perm_test_skipped": perm_test.get("skipped", False),
        "perm_relative_increase": perm_test.get('relative_increase', None),
        "relies_on_slot_id": perm_test.get('relies_on_slot_id', None),
        # K) Gradient rank
        "gradient_rank_error": "error" in grad_rank,
        "gradient_r90": grad_rank.get('r90', None),
        "gradient_effective_rank": grad_rank.get('effective_rank', None),
        "gradient_rank_ratio": grad_rank.get('rank_ratio', None),
    }

    jsonl_path = output_dir / "postfix_diagnostics.jsonl"
    write_jsonl(jsonl_path, jsonl_row)
    console.print(f"[green]Appended to {jsonl_path}[/]")

    console.print("\n[bold green]═══ Diagnostics Complete ═══[/]")


if __name__ == "__main__":
    main()
