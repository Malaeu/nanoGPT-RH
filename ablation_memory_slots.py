#!/usr/bin/env python3
"""
Memory Slot Ablation: Find which slot kills rollout stability.

For each slot M0-M7, zero it out and measure Err@100.
This tells us if the problem is "memory in general" or "specific slots".
"""

import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

from model.memory_mdn import MemoryMDN, MemoryMDNConfig

console = Console()

CKPT_PATH = Path("out/mdn_memory_q3_runpod/best.pt")
DATA_DIR = Path("data/continuous_clean")
CONTEXT_LEN = 128
HORIZON = 100
N_TRAJ = 50

Q3_SLOT_NAMES = [
    "M0_SIGN", "M1_NORM", "M2_TORUS", "M3_SYMBOL",
    "M4_FLOOR", "M5_TOEPLITZ", "M6_PRIMECAP", "M7_GOAL"
]


def load_model(ckpt_path, device="cuda"):
    """Load MemoryMDN model."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    config = ckpt['config']
    model = MemoryMDN(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config


def rollout_error(model, data, device, context_len, horizon, n_traj,
                  ablate_slot=None, ablate_all=False, mode="mean"):
    """
    Compute rollout error with optional slot ablation.

    Args:
        ablate_slot: int or None - which slot to zero out
        ablate_all: bool - zero out ALL memory slots
        mode: "mean", "sample", or "median"
    """
    errors = []

    # Get original memory
    original_memory = model.memory_bank.memory.data.clone()

    # Apply ablation
    if ablate_all:
        model.memory_bank.memory.data.zero_()
    elif ablate_slot is not None:
        model.memory_bank.memory.data[ablate_slot].zero_()

    with torch.no_grad():
        for i in range(min(n_traj, len(data))):
            seq = data[i].to(device).unsqueeze(0)

            if seq.shape[1] < context_len + horizon:
                continue

            true_cum = seq[0, context_len:context_len+horizon].sum().item()

            pred_cum = 0.0
            context = seq[:, :context_len].clone()

            for step in range(horizon):
                result = model(context)
                pi = result['pi'][:, -1]
                mu = result['mu'][:, -1]
                sigma = result['sigma'][:, -1]

                if mode == "mean":
                    pred = torch.sum(pi * mu, dim=-1).item()
                elif mode == "sample":
                    # Sample from mixture
                    comp_idx = torch.multinomial(pi, 1).squeeze(-1)
                    mu_sel = mu[0, comp_idx]
                    sigma_sel = sigma[0, comp_idx]
                    pred = (mu_sel + sigma_sel * torch.randn(1, device=device)).item()
                    pred = max(0, min(pred, 10))  # Clamp
                elif mode == "median":
                    # Approximate median: weighted median of component means
                    # Sort by mu, find where cumsum(pi) crosses 0.5
                    sorted_idx = torch.argsort(mu[0])
                    sorted_pi = pi[0, sorted_idx]
                    sorted_mu = mu[0, sorted_idx]
                    cumsum = torch.cumsum(sorted_pi, dim=0)
                    median_idx = (cumsum >= 0.5).nonzero()[0].item()
                    pred = sorted_mu[median_idx].item()

                pred_cum += pred
                new_spacing = torch.tensor([[pred]], device=device)
                context = torch.cat([context[:, 1:], new_spacing], dim=1)

            errors.append(abs(pred_cum - true_cum))

    # Restore original memory
    model.memory_bank.memory.data.copy_(original_memory)

    return np.mean(errors) if errors else float('nan')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/]")

    # Load model and data
    console.print(f"[green]Loading model from {CKPT_PATH}[/]")
    model, config = load_model(CKPT_PATH, device)

    console.print(f"[green]Loading data from {DATA_DIR}[/]")
    val_data = torch.load(DATA_DIR / "val.pt", weights_only=True)
    console.print(f"[dim]Val data: {val_data.shape}[/]")

    # Results table
    table = Table(title=f"Memory Slot Ablation (Err@{HORIZON}, context={CONTEXT_LEN})")
    table.add_column("Condition", style="cyan")
    table.add_column("Mean Rollout", justify="right")
    table.add_column("Sample Rollout", justify="right")
    table.add_column("Median Rollout", justify="right")
    table.add_column("Notes", style="dim")

    results = {}

    # 1. Full model (no ablation)
    console.print("\n[bold]Testing FULL model (no ablation)...[/]")
    err_mean = rollout_error(model, val_data, device, CONTEXT_LEN, HORIZON, N_TRAJ, mode="mean")
    err_sample = rollout_error(model, val_data, device, CONTEXT_LEN, HORIZON, N_TRAJ, mode="sample")
    err_median = rollout_error(model, val_data, device, CONTEXT_LEN, HORIZON, N_TRAJ, mode="median")
    results['full'] = (err_mean, err_sample, err_median)
    table.add_row("FULL (no ablation)", f"{err_mean:.4f}", f"{err_sample:.4f}", f"{err_median:.4f}", "baseline")
    console.print(f"  mean={err_mean:.4f}, sample={err_sample:.4f}, median={err_median:.4f}")

    # 2. All memory zeroed
    console.print("\n[bold]Testing NO MEMORY (all slots zeroed)...[/]")
    err_mean = rollout_error(model, val_data, device, CONTEXT_LEN, HORIZON, N_TRAJ, ablate_all=True, mode="mean")
    err_sample = rollout_error(model, val_data, device, CONTEXT_LEN, HORIZON, N_TRAJ, ablate_all=True, mode="sample")
    err_median = rollout_error(model, val_data, device, CONTEXT_LEN, HORIZON, N_TRAJ, ablate_all=True, mode="median")
    results['no_memory'] = (err_mean, err_sample, err_median)

    # Compare to full
    delta_mean = err_mean - results['full'][0]
    note = "BETTER" if delta_mean < -0.1 else ("WORSE" if delta_mean > 0.1 else "~same")
    table.add_row("NO MEMORY (all zero)", f"{err_mean:.4f}", f"{err_sample:.4f}", f"{err_median:.4f}", note)
    console.print(f"  mean={err_mean:.4f}, sample={err_sample:.4f}, median={err_median:.4f} [{note}]")

    # 3. Ablate each slot individually
    console.print("\n[bold]Testing individual slot ablation...[/]")
    for slot_idx in range(8):
        slot_name = Q3_SLOT_NAMES[slot_idx]
        console.print(f"  Ablating {slot_name}...")

        err_mean = rollout_error(model, val_data, device, CONTEXT_LEN, HORIZON, N_TRAJ, ablate_slot=slot_idx, mode="mean")
        err_sample = rollout_error(model, val_data, device, CONTEXT_LEN, HORIZON, N_TRAJ, ablate_slot=slot_idx, mode="sample")
        err_median = rollout_error(model, val_data, device, CONTEXT_LEN, HORIZON, N_TRAJ, ablate_slot=slot_idx, mode="median")
        results[slot_name] = (err_mean, err_sample, err_median)

        delta_mean = err_mean - results['full'][0]
        if delta_mean < -0.3:
            note = "CULPRIT?"
        elif delta_mean < -0.1:
            note = "helps remove"
        elif delta_mean > 0.1:
            note = "needed"
        else:
            note = "neutral"

        table.add_row(f"No {slot_name}", f"{err_mean:.4f}", f"{err_sample:.4f}", f"{err_median:.4f}", note)

    # Print results
    console.print("\n")
    console.print(table)

    # Summary
    console.print("\n[bold yellow]SUMMARY[/]")

    full_mean = results['full'][0]
    no_mem_mean = results['no_memory'][0]

    if no_mem_mean < full_mean * 0.8:
        console.print("[green]MEMORY IS THE PROBLEM![/] Removing all memory improves rollout.")
    elif no_mem_mean > full_mean * 1.2:
        console.print("[red]Memory helps rollout.[/] Problem is elsewhere.")
    else:
        console.print("[yellow]Memory has mixed effect on rollout.[/]")

    # Find worst slots (removing them helps most)
    slot_deltas = []
    for slot_idx in range(8):
        slot_name = Q3_SLOT_NAMES[slot_idx]
        delta = results[slot_name][0] - full_mean
        slot_deltas.append((slot_name, delta))

    slot_deltas.sort(key=lambda x: x[1])

    console.print("\n[bold]Slots ranked by impact (removing them):[/]")
    for name, delta in slot_deltas:
        if delta < -0.1:
            console.print(f"  [green]{name}: {delta:+.4f}[/] (removing HELPS)")
        elif delta > 0.1:
            console.print(f"  [red]{name}: {delta:+.4f}[/] (removing HURTS)")
        else:
            console.print(f"  [dim]{name}: {delta:+.4f}[/] (neutral)")


if __name__ == "__main__":
    main()
