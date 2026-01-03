#!/usr/bin/env python3
"""
Evaluate SpacingMDN with probabilistic metrics.

Metrics:
- NLL (negative log-likelihood)
- CRPS (continuous ranked probability score)
- Calibration via PIT (probability integral transform)
- MAE/RMSE for point predictions
- Error accumulation for gamma reconstruction
"""

import argparse
import math
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt

from model.mdn import SpacingMDN, MDNConfig
from model.memory_mdn import MemoryMDN, MemoryMDNConfig

console = Console()

OUTPUT_DIR = Path("out/mdn_100M")
DEFAULT_DATA_DIR = Path("data/continuous_100M")
REPORT_DIR = Path("reports/mdn")


def load_model(ckpt_path: Path, device: str = "cuda"):
    """Load trained MDN model (SpacingMDN or MemoryMDN)."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    config = ckpt['config']

    # Detect model type from config
    if isinstance(config, MemoryMDNConfig):
        model = MemoryMDN(config).to(device)
        model_type = "MemoryMDN"
    else:
        model = SpacingMDN(config).to(device)
        model_type = "SpacingMDN"

    model.load_state_dict(ckpt['model'])
    model.eval()
    console.print(f"[green]Loaded {ckpt_path}[/]")
    console.print(f"[cyan]{model_type}[/]: {sum(p.numel() for p in model.parameters()):,} parameters")
    val_nll = ckpt.get('val_nll', None)
    nll_str = f"{val_nll:.4f}" if val_nll is not None else "N/A"
    console.print(f"[dim]Step {ckpt.get('step', '?')}, NLL={nll_str}[/]")
    return model, config


def gaussian_cdf(x, mu, sigma):
    """CDF of Gaussian distribution."""
    return 0.5 * (1 + torch.erf((x - mu) / (sigma * math.sqrt(2))))


def mixture_cdf(x, pi, mu, sigma):
    """CDF of Gaussian mixture."""
    # x: (B, T, 1), pi/mu/sigma: (B, T, K)
    x = x.unsqueeze(-1)  # (B, T, 1)
    component_cdfs = gaussian_cdf(x, mu, sigma)  # (B, T, K)
    return torch.sum(pi * component_cdfs, dim=-1)  # (B, T)


def compute_pit(model, data, device, n_samples=10000):
    """
    Compute PIT (Probability Integral Transform) values.

    For a well-calibrated model, PIT values should be uniform on [0, 1].
    """
    model.eval()
    pit_values = []

    with torch.no_grad():
        # Take subset
        idx = np.random.choice(len(data), min(n_samples // 256, len(data)), replace=False)
        subset = data[idx].to(device)

        result = model(subset)
        pi = result['pi'][:, :-1]      # (B, T-1, K)
        mu = result['mu'][:, :-1]
        sigma = result['sigma'][:, :-1]
        targets = subset[:, 1:]         # (B, T-1)

        # CDF at target values
        cdf_vals = mixture_cdf(targets, pi, mu, sigma)  # (B, T-1)
        pit_values = cdf_vals.flatten().cpu().numpy()

    return pit_values


def compute_crps(model, data, device, n_samples=5000, n_mc_samples=200):
    """
    Compute CRPS (Continuous Ranked Probability Score).

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent samples from predicted distribution.

    FIXED: Now uses ALIGNED targets from the same sequence.
    - Input x = seq[:, :-1] (positions 0 to T-2)
    - Target y = seq[:, 1:] (positions 1 to T-1)
    - Prediction at position t predicts y[:, t]

    Lower is better.
    """
    model.eval()
    crps_values = []

    with torch.no_grad():
        idx = np.random.choice(len(data), min(n_samples // 256, len(data)), replace=False)
        subset = data[idx].to(device)  # (B, T)

        # Model predicts s_{t+1} from context s_0..s_t
        result = model(subset)
        # pi[:, t] predicts target at position t+1
        pi = result['pi'][:, :-1]      # (B, T-1, K) - predictions for positions 1 to T-1
        mu = result['mu'][:, :-1]
        sigma = result['sigma'][:, :-1]
        targets = subset[:, 1:]         # (B, T-1) - aligned targets

        B, T_minus_1, K = pi.shape

        # Sample positions to evaluate (don't need all T-1 positions)
        n_positions = min(10, T_minus_1)
        pos_indices = np.random.choice(T_minus_1, n_positions, replace=False)

        for b in range(min(B, 50)):  # Limit batch samples for speed
            for t in pos_indices:
                # Sample from mixture at position t
                samples = []
                for _ in range(n_mc_samples):
                    k = torch.multinomial(pi[b, t], 1).item()
                    sample = mu[b, t, k] + sigma[b, t, k] * torch.randn(1, device=device)
                    samples.append(sample.item())
                samples = np.array(samples)

                # CORRECT: Target is the actual next spacing at aligned position
                y = targets[b, t].item()

                # CRPS estimation via sampling
                crps = np.mean(np.abs(samples - y)) - 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
                crps_values.append(crps)

    return np.array(crps_values)


def plot_pit_histogram(pit_values, save_path: Path):
    """Plot PIT histogram for calibration assessment."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(pit_values, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.axhline(1.0, color='red', linestyle='--', label='Uniform (perfect calibration)')
    ax.set_xlabel('PIT value')
    ax.set_ylabel('Density')
    ax.set_title('PIT Histogram')
    ax.legend()
    ax.set_xlim(0, 1)

    # Q-Q plot
    ax = axes[1]
    theoretical = np.linspace(0, 1, len(pit_values))
    empirical = np.sort(pit_values)
    ax.plot(theoretical, empirical, 'b.', alpha=0.3, markersize=1)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Empirical quantiles')
    ax.set_title('PIT Q-Q Plot')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.suptitle('MDN Calibration Assessment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    console.print(f"[green]Saved {save_path}[/]")
    plt.close()


def plot_error_accumulation(errors_mean, errors_sample, horizons, save_path: Path):
    """Plot error accumulation over prediction horizons."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(horizons, errors_mean, 'b-o', linewidth=2, label='Mean prediction')
    ax.plot(horizons, errors_sample, 'r-s', linewidth=2, label='Sample prediction')

    # Theoretical random walk: sqrt(n) * sigma
    sigma = 0.42  # GUE std
    rw_error = sigma * np.sqrt(horizons)
    ax.plot(horizons, rw_error, 'g--', linewidth=2, label=f'Random walk (σ={sigma})')

    ax.set_xlabel('Prediction Horizon (steps)')
    ax.set_ylabel('Cumulative Error |Δγ|')
    ax.set_title('Error Accumulation in γ Reconstruction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    console.print(f"[green]Saved {save_path}[/]")
    plt.close()


def test_error_accumulation(model, data, device, horizons=[10, 25, 50, 100, 200], context_len=128):
    """Test how errors accumulate over prediction horizons.

    Uses context_len=128 by default (model trained on seq_len=256).
    Short contexts (e.g. 10) cause poor performance as model never saw them.
    """
    model.eval()

    errors_mean = []
    errors_sample = []

    with torch.no_grad():
        for horizon in horizons:
            mean_errors = []
            sample_errors = []

            # Take multiple trajectories
            n_traj = min(100, len(data))
            for i in range(n_traj):
                seq = data[i].to(device).unsqueeze(0)  # (1, T)

                # Need context_len + horizon positions
                if seq.shape[1] < context_len + horizon:
                    continue

                # Ground truth cumulative sum from position context_len onwards
                true_cum = seq[0, context_len:context_len+horizon].sum().item()

                # Predict autoregressively using mean
                pred_cum_mean = 0.0
                context = seq[:, :context_len].clone()

                for step in range(horizon):
                    result = model(context)
                    pi = result['pi'][:, -1]
                    mu = result['mu'][:, -1]
                    pred = torch.sum(pi * mu, dim=-1).item()
                    pred_cum_mean += pred

                    # Update context (sliding window)
                    new_spacing = torch.tensor([[pred]], device=device)
                    context = torch.cat([context[:, 1:], new_spacing], dim=1)

                # Predict using sampling
                pred_cum_sample = 0.0
                context = seq[:, :context_len].clone()

                for step in range(horizon):
                    pred = model.sample(context).item()
                    pred_cum_sample += pred

                    new_spacing = torch.tensor([[pred]], device=device)
                    context = torch.cat([context[:, 1:], new_spacing], dim=1)

                mean_errors.append(abs(pred_cum_mean - true_cum))
                sample_errors.append(abs(pred_cum_sample - true_cum))

            errors_mean.append(np.mean(mean_errors))
            errors_sample.append(np.mean(sample_errors))

            console.print(f"[dim]Horizon {horizon}: mean_err={errors_mean[-1]:.4f}, sample_err={errors_sample[-1]:.4f}[/]")

    return np.array(errors_mean), np.array(errors_sample), np.array(horizons)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpacingMDN")
    parser.add_argument("--ckpt", type=Path, default=OUTPUT_DIR / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to data directory (default: data/continuous_100M)")
    parser.add_argument("--n-pit", type=int, default=50000)
    parser.add_argument("--n-crps", type=int, default=5000)
    args = parser.parse_args()

    data_dir = args.data_dir

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(Panel.fit(
        "[bold cyan]EVALUATING SpacingMDN[/]\n"
        "Probabilistic metrics: NLL, CRPS, Calibration",
        title="📊"
    ))

    # Load model and data
    model, config = load_model(args.ckpt, device)

    val_data = torch.load(data_dir / "val.pt", weights_only=False)
    val_data = torch.clamp(val_data, min=0.0, max=10.0)
    console.print(f"[green]Val data: {val_data.shape} from {data_dir}[/]")

    # 1. PIT for calibration
    console.print("\n[bold]1. Computing PIT values...[/]")
    pit_values = compute_pit(model, val_data, device, n_samples=args.n_pit)

    # KS test against uniform
    ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')
    console.print(f"   PIT mean: {pit_values.mean():.4f} (should be ~0.5)")
    console.print(f"   PIT std: {pit_values.std():.4f} (should be ~0.289)")
    console.print(f"   KS test: stat={ks_stat:.4f}, p={ks_pval:.4f}")

    plot_pit_histogram(pit_values, REPORT_DIR / "pit_histogram.png")

    # 2. CRPS
    console.print("\n[bold]2. Computing CRPS...[/]")
    crps_values = compute_crps(model, val_data, device, n_samples=args.n_crps)
    console.print(f"   CRPS mean: {crps_values.mean():.4f}")
    console.print(f"   CRPS std: {crps_values.std():.4f}")

    # 3. Error accumulation
    console.print("\n[bold]3. Testing error accumulation...[/]")
    errors_mean, errors_sample, horizons = test_error_accumulation(
        model, val_data, device,
        horizons=[10, 25, 50, 100, 200]
    )
    plot_error_accumulation(errors_mean, errors_sample, horizons,
                            REPORT_DIR / "error_accumulation.png")

    # Summary table
    console.print("\n")
    table = Table(title="[bold]MDN Evaluation Summary[/]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Interpretation", style="dim")

    table.add_row("PIT mean", f"{pit_values.mean():.4f}", "Should be ~0.5")
    table.add_row("PIT std", f"{pit_values.std():.4f}", "Should be ~0.289")
    table.add_row("KS p-value", f"{ks_pval:.4f}", ">0.05 = well calibrated")
    table.add_row("CRPS", f"{crps_values.mean():.4f}", "Lower is better")
    table.add_row("Error @ h=100", f"{errors_mean[3]:.4f}", "Cumulative |Δγ|")

    console.print(table)

    calibration = "GOOD" if ks_pval > 0.05 else "POOR"
    console.print(f"\n[bold]Calibration verdict: [{calibration}]{calibration}[/]")


if __name__ == "__main__":
    main()
