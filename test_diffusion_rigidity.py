#!/usr/bin/env python3
"""Test Diffusion Zeta rigidity vs AR models."""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
from rich.progress import track

from statistics_long_range_v2 import (
    compute_delta3_correct,
    compute_number_variance_correct,
    generate_gue_spacings_correct,
    gue_delta3_theory,
)

console = Console()


def generate_diffusion_trajectory(n_samples=8, n_steps=100):
    """Generate trajectories using diffusion model."""
    from train_diffusion import ZetaDiffusionTransformer, DiffusionSchedule

    device = torch.device("cpu")

    ckpt = torch.load("out/diffusion_zeta_best.pt", map_location=device, weights_only=False)
    config = ckpt["config"]

    model = ZetaDiffusionTransformer(
        seq_len=config["seq_len"],
        dim=config["dim"],
        depth=config["depth"],
        n_heads=config["n_heads"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    schedule = DiffusionSchedule(timesteps=config["timesteps"], schedule_type="cosine")

    console.print(f"Generating {n_samples} diffusion samples ({n_steps} denoising steps)...")

    all_spacings = []

    for i in track(range(n_samples), description="Diffusion sampling..."):
        with torch.no_grad():
            # Start from noise
            x = torch.randn(1, config["seq_len"], 1)

            # Denoise (use fewer steps for speed)
            step_size = config["timesteps"] // n_steps
            for t_idx in range(config["timesteps"] - 1, -1, -step_size):
                t = torch.full((1,), t_idx, dtype=torch.long)
                x = schedule.p_sample(model, x, t)

            # Convert to spacings
            spacings = x.squeeze().numpy()
            spacings = np.clip(spacings, 0.01, 5.0)
            all_spacings.extend(spacings.tolist())

    return np.array(all_spacings)


def generate_ar_trajectory(model_path, model_type, n_steps=2000):
    """Generate trajectory with AR model."""
    if model_type == "v8":
        from train_snowball import SnowballGPT, SnowballConfig
        ConfigClass = SnowballConfig
        ModelClass = SnowballGPT
    elif model_type == "v9":
        from train_snowball_v9 import SnowballV9GPT, SnowballV9Config
        ConfigClass = SnowballV9Config
        ModelClass = SnowballV9GPT

    device = torch.device("cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ConfigClass(**ckpt["config"])
    model = ModelClass(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    bin_centers = ckpt.get("bin_centers", np.load("data/bin_centers.npy"))
    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32)

    val = torch.load("data/val.pt", weights_only=False)
    start_seq = val[0:1, :config.seq_len]

    if model_type == "v8":
        memory_state = model.snowball.get_initial_state(1, device)
    else:
        memory_state = model.snowball.get_initial_state(1, device)

    generated = []
    window_size = 128

    for step in track(range(n_steps), description=f"Generating {model_type}..."):
        if len(generated) >= window_size:
            tokens = generated[-window_size:]
        else:
            tokens = start_seq[0].tolist()[:window_size]

        window = torch.tensor([tokens], dtype=torch.long)
        scale_val = bin_centers_t[window]

        with torch.no_grad():
            if model_type == "v8":
                pred, loss, mem_attn, memory_state = model(
                    window, targets=None, return_hidden=False, scale_val=scale_val,
                    memory_state=memory_state, return_memory=True
                )
            else:
                pred, loss, mem_attn, memory_state = model(
                    window, targets=None, return_hidden=False, scale_val=scale_val,
                    memory_states=memory_state, return_memory=True
                )

            mu = model.get_mu(pred)[:, -1].item()
            u = np.random.random()
            spacing = mu * np.sqrt(-4 * np.log(1 - u + 1e-10) / np.pi)
            spacing = np.clip(spacing, bin_centers[1], bin_centers[-2])
            token = np.abs(bin_centers - spacing).argmin()
            generated.append(int(token))

    spacings = bin_centers[generated]
    return spacings


def main():
    console.print(Panel.fit(
        "[bold cyan]🧨 DIFFUSION vs AR RIGIDITY SHOWDOWN[/]\n"
        "Does full attention beat autoregressive?",
        title="THE MOMENT OF TRUTH"
    ))

    np.random.seed(42)
    torch.manual_seed(42)

    n_steps = 2000

    # Generate diffusion samples (multiple short sequences concatenated)
    n_diffusion_samples = n_steps // 255 + 1
    diffusion_spacings = generate_diffusion_trajectory(n_samples=n_diffusion_samples, n_steps=100)
    diffusion_spacings = diffusion_spacings[:n_steps]
    diffusion_spacings = diffusion_spacings / diffusion_spacings.mean()

    console.print(f"Diffusion: mean={diffusion_spacings.mean():.3f}, std={diffusion_spacings.std():.3f}")

    # Generate AR samples
    ar_spacings = generate_ar_trajectory("out/snowball_v8_best.pt", "v8", n_steps)
    ar_spacings = ar_spacings / ar_spacings.mean()

    console.print(f"AR v8: mean={ar_spacings.mean():.3f}, std={ar_spacings.std():.3f}")

    # GUE reference
    console.print("Generating GUE...")
    gue_spacings = generate_gue_spacings_correct(n_steps + 200, seed=42)[:n_steps]

    # Compute Δ3
    L_values = np.array([5, 10, 20, 50, 100, 200, 500])
    L_values = L_values[L_values < n_steps // 3]

    console.print("\nComputing Δ3...")
    delta3_diff = compute_delta3_correct(diffusion_spacings, L_values)
    delta3_ar = compute_delta3_correct(ar_spacings, L_values)
    delta3_gue = compute_delta3_correct(gue_spacings, L_values)
    delta3_theory = gue_delta3_theory(L_values)

    # Display
    console.print("\n")
    table = Table(title="📊 Δ3(L) RIGIDITY: DIFFUSION vs AR")
    table.add_column("L", style="bold", justify="right")
    table.add_column("Diffusion", justify="right")
    table.add_column("AR v8", justify="right")
    table.add_column("GUE Sim", justify="right")
    table.add_column("GUE Theory", justify="right")

    for i, L in enumerate(L_values):
        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "N/A"

        diff_val = delta3_diff[i]
        ar_val = delta3_ar[i]
        gue_val = delta3_gue[i]

        # Color better one green
        diff_str = fmt(diff_val)
        ar_str = fmt(ar_val)

        if not np.isnan(diff_val) and not np.isnan(ar_val) and not np.isnan(gue_val):
            if abs(diff_val - gue_val) < abs(ar_val - gue_val):
                diff_str = f"[green]{diff_str}[/]"
            else:
                ar_str = f"[green]{ar_str}[/]"

        table.add_row(
            f"{L:.0f}",
            diff_str,
            ar_str,
            fmt(delta3_gue[i]),
            fmt(delta3_theory[i]),
        )

    console.print(table)

    # Summary stats
    valid = ~np.isnan(delta3_diff) & ~np.isnan(delta3_ar) & ~np.isnan(delta3_gue)
    if valid.sum() >= 3:
        corr_diff = np.corrcoef(delta3_diff[valid], delta3_gue[valid])[0, 1]
        corr_ar = np.corrcoef(delta3_ar[valid], delta3_gue[valid])[0, 1]

        mae_diff = np.mean(np.abs(delta3_diff[valid] - delta3_gue[valid]))
        mae_ar = np.mean(np.abs(delta3_ar[valid] - delta3_gue[valid]))

        console.print("\n")
        diag = Table(title="💡 FINAL COMPARISON")
        diag.add_column("Model", style="bold")
        diag.add_column("Corr GUE")
        diag.add_column("MAE GUE")
        diag.add_column("Verdict")

        diff_wins = mae_diff < mae_ar

        diag.add_row(
            "🧨 Diffusion",
            f"[green]{corr_diff:.3f}[/]" if diff_wins else f"{corr_diff:.3f}",
            f"[green]{mae_diff:.3f}[/]" if diff_wins else f"{mae_diff:.3f}",
            "[bold green]WINNER![/]" if diff_wins else ""
        )
        diag.add_row(
            "AR v8",
            f"{corr_ar:.3f}" if diff_wins else f"[green]{corr_ar:.3f}[/]",
            f"{mae_ar:.3f}" if diff_wins else f"[green]{mae_ar:.3f}[/]",
            "" if diff_wins else "[bold green]WINNER![/]"
        )

        console.print(diag)

        # Final verdict
        console.print("\n")
        if diff_wins:
            verdict = (
                "[bold green]🎉 DIFFUSION BEATS AUTOREGRESSIVE![/]\n"
                "Full attention captures global rigidity better!\n"
                f"MAE improvement: {(1 - mae_diff/mae_ar)*100:.1f}%"
            )
        else:
            verdict = (
                "[bold yellow]⚠️ Diffusion did not beat AR[/]\n"
                "Need more training or architecture tuning."
            )

        console.print(Panel.fit(verdict, title="🎯 THE VERDICT", border_style="cyan"))

        # Rigidity growth analysis
        console.print("\n[bold]Rigidity Growth Analysis:[/]")
        console.print(f"  GUE Theory: Δ3 ~ {delta3_theory[-1]/delta3_theory[0]:.1f}x from L=5 to L={L_values[-1]}")
        console.print(f"  Diffusion:  Δ3 ~ {delta3_diff[-1]/delta3_diff[0]:.1f}x")
        console.print(f"  AR v8:      Δ3 ~ {delta3_ar[-1]/delta3_ar[0]:.1f}x")

        # For GUE, growth should be ~log(L_max/L_min) / π² ≈ 4.6 / π² ≈ 0.47
        # i.e., about 2.7x growth from L=5 to L=500


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    main()
