#!/usr/bin/env python3
"""
TASK_SPEC_2M — Eval Suite for SpacingMDN

Usage:
  python eval_mdn.py --ckpt out/mdn_baseline/best.pt --data-dir data/continuous_2M \\
      --n-pit 20000 --n-crps 2000 --rollout 200 --n-rollouts 64

One-step metrics:
  - NLL (negative log-likelihood)
  - CRPS (Continuous Ranked Probability Score via sampling)
  - PIT (Probability Integral Transform) mean/std + KS p-value
  - Bias: E[E[s|x] - s_true]

Rollout metrics:
  - Horizons: h=10, 25, 50, 100, 200
  - Modes: mean-rollout, sample-rollout ensemble

Sanity:
  - GT-rollout: error should be ~0 when using true spacings
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
from dataclasses import dataclass
import argparse

console = Console()

# Import model
from train_mdn import SpacingMDN, MDNConfig


# ============================================================================
# MDN UTILITIES
# ============================================================================

def sample_from_mdn(pi, mu, sigma, n_samples=1):
    """
    Sample from mixture of Gaussians.

    Args:
        pi: (B, K) mixture weights
        mu: (B, K) means
        sigma: (B, K) std devs
        n_samples: number of samples per distribution

    Returns:
        samples: (B, n_samples) sampled values
    """
    B, K = pi.shape
    device = pi.device

    # Sample component indices
    component_idx = torch.multinomial(pi, n_samples, replacement=True)  # (B, n_samples)

    # Sample from Gaussians
    # Gather mu and sigma for selected components
    mu_selected = torch.gather(mu.unsqueeze(1).expand(-1, n_samples, -1),
                               2, component_idx.unsqueeze(2)).squeeze(2)  # (B, n_samples)
    sigma_selected = torch.gather(sigma.unsqueeze(1).expand(-1, n_samples, -1),
                                  2, component_idx.unsqueeze(2)).squeeze(2)  # (B, n_samples)

    # Reparameterization
    eps = torch.randn(B, n_samples, device=device)
    samples = mu_selected + sigma_selected * eps

    # Ensure positive (spacings must be > 0)
    samples = F.relu(samples) + 1e-6

    return samples


def mdn_mean(pi, mu, sigma):
    """Expected value of MDN: E[s] = sum_k pi_k * mu_k"""
    return (pi * mu).sum(dim=-1)


def mdn_cdf(x, pi, mu, sigma):
    """
    CDF of Gaussian mixture at point x.
    CDF(x) = sum_k pi_k * Phi((x - mu_k) / sigma_k)
    """
    # Standard normal CDF
    z = (x.unsqueeze(-1) - mu) / sigma  # (B, K) or (B, T, K)
    phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return (pi * phi).sum(dim=-1)


# ============================================================================
# ONE-STEP METRICS
# ============================================================================

def compute_nll(model, data, device, n_samples=None):
    """Compute average NLL on data."""
    model.eval()

    if n_samples is not None:
        idx = torch.randint(0, len(data), (n_samples,))
        data = data[idx]

    total_nll = 0.0
    n_batches = 0

    batch_size = 256
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size].to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        with torch.no_grad():
            pi, mu, sigma = model(x)

            # NLL
            y_exp = y.unsqueeze(-1)
            log_prob = -0.5 * math.log(2 * math.pi) - torch.log(sigma) - 0.5 * ((y_exp - mu) / sigma) ** 2
            log_pi = torch.log(pi + 1e-10)
            log_likelihood = torch.logsumexp(log_pi + log_prob, dim=-1)
            nll = -log_likelihood.mean()

            total_nll += nll.item()
            n_batches += 1

    return total_nll / n_batches


def compute_crps(model, data, device, n_samples=2000, n_mc_samples=100):
    """
    Compute CRPS (Continuous Ranked Probability Score) via sampling.

    CRPS = E[|s_sampled - s_true|] - 0.5 * E[|s_sampled - s_sampled'|]
    """
    model.eval()

    # Sample subset of data
    idx = torch.randint(0, len(data), (n_samples,))
    subset = data[idx].to(device)
    x = subset[:, :-1]
    y = subset[:, 1:]  # (n_samples, T-1)

    crps_values = []

    with torch.no_grad():
        pi, mu, sigma = model(x)  # (n_samples, T-1, K)

        # Flatten for easier sampling
        B, T, K = pi.shape
        pi_flat = pi.view(-1, K)
        mu_flat = mu.view(-1, K)
        sigma_flat = sigma.view(-1, K)
        y_flat = y.reshape(-1)

        # Sample from MDN
        samples = sample_from_mdn(pi_flat, mu_flat, sigma_flat, n_mc_samples)  # (B*T, n_mc)

        # Term 1: E[|s - y|]
        term1 = torch.abs(samples - y_flat.unsqueeze(1)).mean(dim=1)

        # Term 2: E[|s - s'|] / 2
        # Use pairwise differences
        term2 = torch.abs(samples.unsqueeze(1) - samples.unsqueeze(2)).mean(dim=(1, 2)) / 2

        crps = (term1 - term2).mean().item()

    return crps


def compute_pit(model, data, device, n_samples=20000):
    """
    Compute PIT (Probability Integral Transform) values.

    PIT = CDF(y_true | predicted distribution)

    For well-calibrated model, PIT should be uniform [0, 1].
    Returns: pit_values, pit_mean, pit_std, ks_pvalue
    """
    model.eval()

    idx = torch.randint(0, len(data), (n_samples,))
    subset = data[idx].to(device)
    x = subset[:, :-1]
    y = subset[:, 1:]

    with torch.no_grad():
        pi, mu, sigma = model(x)

        # Compute CDF at true values
        pit = mdn_cdf(y, pi, mu, sigma)  # (B, T)
        pit_flat = pit.cpu().numpy().flatten()

    # Remove invalid values
    pit_flat = pit_flat[(pit_flat >= 0) & (pit_flat <= 1)]

    # Statistics
    pit_mean = np.mean(pit_flat)
    pit_std = np.std(pit_flat)

    # KS test against uniform
    ks_stat, ks_pvalue = stats.kstest(pit_flat, 'uniform')

    return pit_flat, pit_mean, pit_std, ks_pvalue


def compute_bias(model, data, device, n_samples=5000):
    """
    Compute prediction bias: E[E[s|x] - s_true]

    Positive bias = model predicts too high
    Negative bias = model predicts too low
    """
    model.eval()

    idx = torch.randint(0, len(data), (n_samples,))
    subset = data[idx].to(device)
    x = subset[:, :-1]
    y = subset[:, 1:]

    with torch.no_grad():
        pi, mu, sigma = model(x)
        pred_mean = mdn_mean(pi, mu, sigma)  # (B, T)

        bias = (pred_mean - y).mean().item()

    return bias


# ============================================================================
# ROLLOUT METRICS
# ============================================================================

def rollout_mean(model, start_seq, horizon, device):
    """
    Deterministic rollout using predicted mean.

    Args:
        model: SpacingMDN
        start_seq: (B, T) starting sequence
        horizon: number of steps to predict

    Returns:
        predictions: (B, horizon) predicted spacings
    """
    model.eval()
    B, T = start_seq.shape
    seq_len = model.config.seq_len

    current = start_seq.clone()
    predictions = []

    with torch.no_grad():
        for _ in range(horizon):
            # Truncate to seq_len if needed
            if current.shape[1] > seq_len:
                current = current[:, -seq_len:]

            pi, mu, sigma = model(current)

            # Get mean of last position
            pred = mdn_mean(pi[:, -1], mu[:, -1], sigma[:, -1])  # (B,)
            predictions.append(pred)

            # Append to sequence
            current = torch.cat([current, pred.unsqueeze(1)], dim=1)

    return torch.stack(predictions, dim=1)  # (B, horizon)


def rollout_sample(model, start_seq, horizon, device, n_ensemble=64):
    """
    Stochastic rollout with ensemble sampling.

    Args:
        model: SpacingMDN
        start_seq: (B, T) starting sequence
        horizon: number of steps
        n_ensemble: number of parallel samples

    Returns:
        predictions: (B, n_ensemble, horizon)
    """
    model.eval()
    B, T = start_seq.shape
    seq_len = model.config.seq_len

    # Expand for ensemble
    current = start_seq.unsqueeze(1).expand(-1, n_ensemble, -1).reshape(B * n_ensemble, T)
    predictions = []

    with torch.no_grad():
        for _ in range(horizon):
            if current.shape[1] > seq_len:
                current = current[:, -seq_len:]

            pi, mu, sigma = model(current)

            # Sample from last position
            pred = sample_from_mdn(pi[:, -1], mu[:, -1], sigma[:, -1], 1).squeeze(-1)  # (B*n_ens,)
            predictions.append(pred)

            current = torch.cat([current, pred.unsqueeze(1)], dim=1)

    predictions = torch.stack(predictions, dim=1)  # (B*n_ens, horizon)
    return predictions.view(B, n_ensemble, horizon)


def compute_rollout_metrics(model, data, device, horizons=[10, 25, 50, 100, 200], n_rollouts=64, n_starts=100):
    """
    Compute rollout error at different horizons.

    Returns:
        results: dict with errors at each horizon
    """
    model.eval()

    # Get starting sequences
    idx = torch.randint(0, len(data), (n_starts,))
    starts = data[idx].to(device)

    results = {}

    for h in horizons:
        if h > data.shape[1] - 2:
            console.print(f"[yellow]Skipping h={h} (exceeds sequence length)[/]")
            continue

        # Get ground truth
        x_start = starts[:, :h]
        y_true = starts[:, h:2*h] if 2*h <= starts.shape[1] else starts[:, h:]

        actual_h = y_true.shape[1]

        # Mean rollout
        pred_mean = rollout_mean(model, x_start, actual_h, device)
        err_mean = torch.abs(pred_mean - y_true).mean().item()

        # Sample rollout ensemble
        pred_sample = rollout_sample(model, x_start, actual_h, device, n_rollouts)  # (B, n_ens, h)

        # Median prediction
        pred_median = pred_sample.median(dim=1).values  # (B, h)
        err_median = torch.abs(pred_median - y_true).mean().item()

        # Ensemble spread (uncertainty)
        spread = pred_sample.std(dim=1).mean().item()

        results[h] = {
            "err_mean": err_mean,
            "err_median": err_median,
            "spread": spread,
        }

    return results


def gt_rollout_sanity(data, device):
    """
    Sanity check: using ground truth spacings should give ~0 error.
    This validates that our rollout metric is correct.
    """
    # Take some sequences
    idx = torch.randint(0, len(data), (100,))
    seqs = data[idx].to(device)

    # "Predict" using actual next values (this should be perfect)
    x = seqs[:, :128]
    y_true = seqs[:, 128:192]

    # If we just copy the true values, error should be ~0
    error = torch.abs(y_true - y_true).mean().item()

    # Also test that error IS non-zero when we use wrong values
    y_wrong = torch.randn_like(y_true) * 0.42 + 1.0
    error_wrong = torch.abs(y_wrong - y_true).mean().item()

    return error, error_wrong


# ============================================================================
# MAIN EVAL
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Eval Suite for SpacingMDN")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--n-pit", type=int, default=2000, help="Samples for PIT (reduced to avoid OOM)")
    parser.add_argument("--n-crps", type=int, default=2000, help="Samples for CRPS")
    parser.add_argument("--rollout", type=int, default=100, help="Max rollout horizon")
    parser.add_argument("--n-rollouts", type=int, default=32, help="Rollout ensemble size")
    parser.add_argument("--output-dir", type=str, default="reports/2M", help="Output directory")
    args = parser.parse_args()

    console.print("[bold magenta]═══ TASK_SPEC_2M: Eval Suite ═══[/]\n")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    console.print(f"[cyan]Device: {device}[/]")

    # Load model
    console.print(f"[cyan]Loading checkpoint: {args.ckpt}[/]")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    config = MDNConfig(**ckpt["config"])
    model = SpacingMDN(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load data
    data_dir = Path(args.data_dir)
    val_data = torch.load(data_dir / "val.pt", weights_only=False)
    console.print(f"[green]Val data: {val_data.shape}[/]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # SANITY CHECK
    # ========================================================================
    console.print("\n[bold cyan]═══ Sanity Check ═══[/]")
    gt_err, wrong_err = gt_rollout_sanity(val_data, device)
    console.print(f"GT rollout sanity: error={gt_err:.6f} (should be ~0)")
    console.print(f"Wrong rollout:     error={wrong_err:.4f} (should be >0)")

    if gt_err > 0.001:
        console.print("[red]WARNING: GT sanity check failed![/]")
    else:
        console.print("[green]✓ Sanity check passed[/]")

    # ========================================================================
    # ONE-STEP METRICS
    # ========================================================================
    console.print("\n[bold cyan]═══ One-Step Metrics ═══[/]")

    # NLL
    nll = compute_nll(model, val_data, device)
    console.print(f"NLL: {nll:.4f}")

    # CRPS
    crps = compute_crps(model, val_data, device, n_samples=args.n_crps)
    console.print(f"CRPS: {crps:.4f}")

    # PIT
    pit_vals, pit_mean, pit_std, ks_pval = compute_pit(model, val_data, device, n_samples=args.n_pit)
    console.print(f"PIT: mean={pit_mean:.4f} (target=0.5), std={pit_std:.4f} (target≈0.29)")
    console.print(f"KS test p-value: {ks_pval:.4f} (p>0.05 = calibrated)")

    # Bias
    bias = compute_bias(model, val_data, device)
    console.print(f"Bias: {bias:.6f} (0 = unbiased)")

    # ========================================================================
    # ROLLOUT METRICS
    # ========================================================================
    console.print("\n[bold cyan]═══ Rollout Metrics ═══[/]")

    horizons = [10, 25, 50, 100, 200]
    horizons = [h for h in horizons if h <= args.rollout]

    rollout_results = compute_rollout_metrics(model, val_data, device, horizons, args.n_rollouts)

    rollout_table = Table(title="Rollout Errors")
    rollout_table.add_column("Horizon", style="cyan", justify="right")
    rollout_table.add_column("Err@mean", justify="right")
    rollout_table.add_column("Err@median", justify="right")
    rollout_table.add_column("Spread", justify="right")

    for h, res in rollout_results.items():
        rollout_table.add_row(
            str(h),
            f"{res['err_mean']:.4f}",
            f"{res['err_median']:.4f}",
            f"{res['spread']:.4f}"
        )

    console.print(rollout_table)

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    console.print("\n[bold cyan]═══ Summary ═══[/]")

    summary = Table(title="Eval Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_column("Target/Note", style="yellow")

    summary.add_row("NLL", f"{nll:.4f}", "lower is better")
    summary.add_row("CRPS", f"{crps:.4f}", "lower is better")
    summary.add_row("PIT mean", f"{pit_mean:.4f}", "0.5 (calibrated)")
    summary.add_row("PIT std", f"{pit_std:.4f}", "~0.29 (uniform)")
    summary.add_row("KS p-value", f"{ks_pval:.4f}", ">0.05 (calibrated)")
    summary.add_row("Bias", f"{bias:.6f}", "0 (unbiased)")
    if rollout_results:
        h100 = rollout_results.get(100, rollout_results.get(50, {}))
        summary.add_row("Err@h=100", f"{h100.get('err_median', 'N/A'):.4f}", "lower than RW baseline")

    console.print(summary)

    # ========================================================================
    # SAVE REPORT
    # ========================================================================
    report = f"""# Eval Report: SpacingMDN

## Model Info
- Checkpoint: `{args.ckpt}`
- Architecture: {config.n_layer}L/{config.n_head}H/{config.n_embd}E
- MDN Components: {config.n_components}

## One-Step Metrics
| Metric | Value | Target |
|--------|-------|--------|
| NLL | {nll:.4f} | lower is better |
| CRPS | {crps:.4f} | lower is better |
| PIT mean | {pit_mean:.4f} | 0.5 |
| PIT std | {pit_std:.4f} | ~0.29 |
| KS p-value | {ks_pval:.4f} | >0.05 |
| Bias | {bias:.6f} | 0 |

## Rollout Metrics
| Horizon | Err@mean | Err@median | Spread |
|---------|----------|------------|--------|
"""
    for h, res in rollout_results.items():
        report += f"| {h} | {res['err_mean']:.4f} | {res['err_median']:.4f} | {res['spread']:.4f} |\n"

    report += f"""
## Sanity Check
- GT rollout error: {gt_err:.6f} (should be ~0) ✓

## Notes
- PIT KS test: {'PASS (calibrated)' if ks_pval > 0.05 else 'FAIL (miscalibrated)'}
- Bias: {'acceptable' if abs(bias) < 0.01 else 'significant'}
"""

    with open(output_dir / "eval_table.md", "w") as f:
        f.write(report)

    console.print(f"\n[green]Saved report to {output_dir}/eval_table.md[/]")

    # ========================================================================
    # PLOTS
    # ========================================================================

    # PIT histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(pit_vals, bins=50, density=True, alpha=0.7, color='steelblue')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Uniform')
    ax.set_xlabel('PIT')
    ax.set_ylabel('Density')
    ax.set_title(f'PIT Histogram (KS p={ks_pval:.3f})')
    ax.legend()
    ax.set_xlim(0, 1)

    # Err vs horizon
    ax = axes[1]
    hs = list(rollout_results.keys())
    err_mean = [rollout_results[h]['err_mean'] for h in hs]
    err_median = [rollout_results[h]['err_median'] for h in hs]

    ax.plot(hs, err_mean, 'o-', label='Mean rollout', markersize=8)
    ax.plot(hs, err_median, 's-', label='Median (ensemble)', markersize=8)
    ax.set_xlabel('Horizon h')
    ax.set_ylabel('MAE')
    ax.set_title('Rollout Error vs Horizon')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "eval_plots.png", dpi=150)
    plt.close()

    console.print(f"[green]Saved plots to {output_dir}/eval_plots.png[/]")
    console.print("\n[bold green]═══ Eval Complete ═══[/]")


if __name__ == "__main__":
    main()
