#!/usr/bin/env python3
"""Quick rigidity test for v9."""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from pathlib import Path

from statistics_long_range_v2 import (
    compute_delta3_correct,
    compute_number_variance_correct,
    generate_gue_spacings_correct,
    gue_delta3_theory,
)
from rich.progress import track

console = Console()


def generate_v9_trajectory(n_steps=2000, seed=42):
    """Generate trajectory with v9 model."""
    from train_snowball_v9 import SnowballV9GPT, SnowballV9Config

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    ckpt = torch.load("out/snowball_v9_best.pt", map_location=device, weights_only=False)
    config = SnowballV9Config(**ckpt["config"])
    model = SnowballV9GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    bin_centers = ckpt.get("bin_centers", np.load("data/bin_centers.npy"))
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32)

    val = torch.load("data/val.pt", weights_only=False)
    start_seq = val[0:1, :config.seq_len]

    memory_states = model.snowball.get_initial_state(1, device)

    generated_tokens = start_seq[0].tolist()
    generated_spacings = []

    window_size = min(128, config.seq_len - 1)

    for step in track(range(n_steps), description="Generating v9..."):
        if len(generated_tokens) >= window_size:
            window = torch.tensor([generated_tokens[-window_size:]], dtype=torch.long)
        else:
            window = torch.tensor([generated_tokens], dtype=torch.long)

        scale_val = bin_centers_t[window]

        with torch.no_grad():
            pred, loss, mem_attn, memory_states = model(
                window, targets=None, return_hidden=False, scale_val=scale_val,
                memory_states=memory_states, return_memory=True
            )

            mu = model.get_mu(pred)[:, -1].item()

            u = np.random.random()
            spacing = mu * np.sqrt(-4 * np.log(1 - u + 1e-10) / np.pi)
            spacing = np.clip(spacing, bin_centers[1], bin_centers[-2])

            token = np.abs(bin_centers - spacing).argmin()
            generated_tokens.append(int(token))
            generated_spacings.append(spacing)

    return np.array(generated_spacings)


def generate_v8_trajectory(n_steps=2000, seed=42):
    """Generate trajectory with v8 model for comparison."""
    from train_snowball import SnowballGPT, SnowballConfig

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    ckpt = torch.load("out/snowball_v8_best.pt", map_location=device, weights_only=False)
    config = SnowballConfig(**ckpt["config"])
    model = SnowballGPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    bin_centers = ckpt.get("bin_centers", np.load("data/bin_centers.npy"))
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32)

    val = torch.load("data/val.pt", weights_only=False)
    start_seq = val[0:1, :config.seq_len]

    memory_state = model.snowball.get_initial_state(1, device)

    generated_tokens = start_seq[0].tolist()
    generated_spacings = []

    window_size = min(128, config.seq_len - 1)

    for step in track(range(n_steps), description="Generating v8..."):
        if len(generated_tokens) >= window_size:
            window = torch.tensor([generated_tokens[-window_size:]], dtype=torch.long)
        else:
            window = torch.tensor([generated_tokens], dtype=torch.long)

        scale_val = bin_centers_t[window]

        with torch.no_grad():
            pred, loss, mem_attn, new_memory = model(
                window, targets=None, return_hidden=False, scale_val=scale_val,
                memory_state=memory_state, return_memory=True
            )

            mu = model.get_mu(pred)[:, -1].item()

            u = np.random.random()
            spacing = mu * np.sqrt(-4 * np.log(1 - u + 1e-10) / np.pi)
            spacing = np.clip(spacing, bin_centers[1], bin_centers[-2])

            token = np.abs(bin_centers - spacing).argmin()
            generated_tokens.append(int(token))
            generated_spacings.append(spacing)
            memory_state = new_memory

    return np.array(generated_spacings)


def main():
    console.print("[bold cyan]🌊 V9 vs V8 RIGIDITY COMPARISON[/]\n")

    n_steps = 2000

    # Generate trajectories
    v9_spacings = generate_v9_trajectory(n_steps)
    v9_spacings = v9_spacings / v9_spacings.mean()

    v8_spacings = generate_v8_trajectory(n_steps)
    v8_spacings = v8_spacings / v8_spacings.mean()

    # GUE reference
    console.print("Generating GUE...")
    gue_spacings = generate_gue_spacings_correct(n_steps + 200, seed=42)[:n_steps]

    # Compute Δ3
    L_values = np.array([5, 10, 20, 50, 100, 200, 500])
    L_values = L_values[L_values < n_steps // 3]

    console.print("Computing Δ3...")
    delta3_v9 = compute_delta3_correct(v9_spacings, L_values)
    delta3_v8 = compute_delta3_correct(v8_spacings, L_values)
    delta3_gue = compute_delta3_correct(gue_spacings, L_values)
    delta3_theory = gue_delta3_theory(L_values)

    # Display
    console.print("\n")
    table = Table(title="📊 Δ3(L) RIGIDITY: V9 vs V8")
    table.add_column("L", style="bold", justify="right")
    table.add_column("V9 (multi)", justify="right")
    table.add_column("V8 (single)", justify="right")
    table.add_column("GUE Sim", justify="right")
    table.add_column("GUE Theory", justify="right")

    for i, L in enumerate(L_values):
        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "N/A"

        # Color v9 green if better than v8
        v9_val = delta3_v9[i]
        v8_val = delta3_v8[i]
        gue_val = delta3_gue[i]

        v9_str = fmt(v9_val)
        v8_str = fmt(v8_val)

        # Better = closer to GUE
        if not np.isnan(v9_val) and not np.isnan(v8_val) and not np.isnan(gue_val):
            if abs(v9_val - gue_val) < abs(v8_val - gue_val):
                v9_str = f"[green]{v9_str}[/]"
            else:
                v8_str = f"[green]{v8_str}[/]"

        table.add_row(
            f"{L:.0f}",
            v9_str,
            v8_str,
            fmt(delta3_gue[i]),
            fmt(delta3_theory[i]),
        )

    console.print(table)

    # Correlations
    valid = ~np.isnan(delta3_v9) & ~np.isnan(delta3_v8) & ~np.isnan(delta3_gue)
    if valid.sum() >= 3:
        corr_v9_gue = np.corrcoef(delta3_v9[valid], delta3_gue[valid])[0, 1]
        corr_v8_gue = np.corrcoef(delta3_v8[valid], delta3_gue[valid])[0, 1]

        mae_v9 = np.mean(np.abs(delta3_v9[valid] - delta3_gue[valid]))
        mae_v8 = np.mean(np.abs(delta3_v8[valid] - delta3_gue[valid]))

        console.print("\n")
        diag = Table(title="💡 COMPARISON")
        diag.add_column("Model", style="bold")
        diag.add_column("Corr with GUE")
        diag.add_column("MAE vs GUE")
        diag.add_column("Verdict")

        v9_better = corr_v9_gue > corr_v8_gue and mae_v9 < mae_v8

        diag.add_row(
            "V9 (multi-scale)",
            f"[green]{corr_v9_gue:.3f}[/]" if v9_better else f"{corr_v9_gue:.3f}",
            f"[green]{mae_v9:.3f}[/]" if v9_better else f"{mae_v9:.3f}",
            "[green]BETTER[/]" if v9_better else ""
        )
        diag.add_row(
            "V8 (single-scale)",
            f"{corr_v8_gue:.3f}" if v9_better else f"[green]{corr_v8_gue:.3f}[/]",
            f"{mae_v8:.3f}" if v9_better else f"[green]{mae_v8:.3f}[/]",
            "" if v9_better else "[green]BETTER[/]"
        )

        console.print(diag)

        # Final verdict
        if v9_better:
            console.print("\n[bold green]🎉 V9 MULTI-SCALE IMPROVES RIGIDITY![/]")
        else:
            console.print("\n[bold yellow]⚠️ V9 did not improve over V8[/]")


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    main()
