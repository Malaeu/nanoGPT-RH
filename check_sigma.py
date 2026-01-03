#!/usr/bin/env python3
"""
σ Diagnostic Script - Check for overconfidence (σ-collapse).

Critical check: Is negative NLL due to real learning or σ cheating?

Key metrics:
- σ_median, σ_p05, σ_p95 across all predictions
- If σ_p05 → 1e-4 (floor), model is overconfident
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from scipy import stats

from model.mdn import SpacingMDN

console = Console()

DATA_DIR = Path("data/continuous_100M_full")
if not DATA_DIR.exists():
    DATA_DIR = Path("data/continuous_100M")


def compute_sigma_stats(model, loader, device, n_batches=100):
    """Compute σ statistics across validation set."""
    model.eval()

    all_sigmas = []
    all_pis = []
    all_mus = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            x = batch[0].to(device)
            result = model(x)

            # Get MDN parameters: [batch, seq, K]
            pi = result['pi'].cpu().numpy()
            mu = result['mu'].cpu().numpy()
            sigma = result['sigma'].cpu().numpy()

            all_pis.append(pi)
            all_mus.append(mu)
            all_sigmas.append(sigma)
            all_targets.append(x[:, 1:].cpu().numpy())  # targets are shifted by 1

    all_sigmas = np.concatenate(all_sigmas, axis=0)  # [N, seq-1, K]
    all_pis = np.concatenate(all_pis, axis=0)
    all_mus = np.concatenate(all_mus, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return {
        'sigma': all_sigmas,
        'pi': all_pis,
        'mu': all_mus,
        'targets': all_targets,
    }


def analyze_sigma(data):
    """Analyze σ distribution."""
    sigma = data['sigma']
    pi = data['pi']

    # Flatten all sigmas
    sigma_flat = sigma.flatten()

    # Weighted sigma (by π) - more meaningful
    # weighted_sigma = Σ π_k * σ_k for each position
    weighted_sigma = (pi * sigma).sum(axis=-1).flatten()

    # Per-component analysis
    K = sigma.shape[-1]

    console.print(Panel.fit(
        "[bold cyan]σ Distribution Analysis[/]\n"
        "Checking for overconfidence (σ-collapse)",
        title="🔬"
    ))

    # Main stats table
    table = Table(title="σ Statistics (All Components)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Risk Level", style="yellow")

    # Compute percentiles
    p05 = np.percentile(sigma_flat, 5)
    p25 = np.percentile(sigma_flat, 25)
    p50 = np.percentile(sigma_flat, 50)
    p75 = np.percentile(sigma_flat, 75)
    p95 = np.percentile(sigma_flat, 95)

    # Risk assessment
    def risk_level(val, threshold_low=0.01, threshold_critical=0.001):
        if val < threshold_critical:
            return "[red bold]CRITICAL[/]"
        elif val < threshold_low:
            return "[yellow]WARNING[/]"
        else:
            return "[green]OK[/]"

    table.add_row("σ_min", f"{sigma_flat.min():.6f}", risk_level(sigma_flat.min()))
    table.add_row("σ_p05 (5th %tile)", f"{p05:.6f}", risk_level(p05))
    table.add_row("σ_p25", f"{p25:.6f}", risk_level(p25))
    table.add_row("σ_median", f"{p50:.6f}", risk_level(p50, 0.05, 0.01))
    table.add_row("σ_p75", f"{p75:.6f}", "[green]OK[/]")
    table.add_row("σ_p95", f"{p95:.6f}", "[green]OK[/]")
    table.add_row("σ_max", f"{sigma_flat.max():.6f}", "[green]OK[/]")
    table.add_row("σ_mean", f"{sigma_flat.mean():.6f}", "[dim]info[/]")

    console.print(table)

    # Weighted sigma stats
    console.print("\n[bold]π-Weighted σ (effective uncertainty):[/]")
    table2 = Table()
    table2.add_column("Metric", style="cyan")
    table2.add_column("Value", style="green")

    table2.add_row("weighted_σ_median", f"{np.median(weighted_sigma):.6f}")
    table2.add_row("weighted_σ_p05", f"{np.percentile(weighted_sigma, 5):.6f}")
    table2.add_row("weighted_σ_p95", f"{np.percentile(weighted_sigma, 95):.6f}")

    console.print(table2)

    # Per-component stats
    console.print(f"\n[bold]Per-Component σ (K={K} components):[/]")
    table3 = Table()
    table3.add_column("Component", style="cyan")
    table3.add_column("σ_median", style="green")
    table3.add_column("σ_p05", style="yellow")
    table3.add_column("π_mean", style="blue")

    for k in range(K):
        sigma_k = sigma[..., k].flatten()
        pi_k = pi[..., k].flatten()
        risk = risk_level(np.percentile(sigma_k, 5))
        table3.add_row(
            f"k={k}",
            f"{np.median(sigma_k):.4f}",
            f"{np.percentile(sigma_k, 5):.4f} {risk}",
            f"{pi_k.mean():.3f}"
        )

    console.print(table3)

    # Floor detection
    floor_threshold = 1e-3  # Close to sigma_min=1e-4
    at_floor = (sigma_flat < floor_threshold).mean() * 100

    console.print(f"\n[bold]Floor Detection:[/]")
    console.print(f"  σ values < {floor_threshold}: [{'red' if at_floor > 5 else 'green'}]{at_floor:.2f}%[/]")

    if at_floor > 5:
        console.print("[red bold]⚠️ OVERCONFIDENCE DETECTED![/]")
        console.print("[yellow]Model is pushing σ to floor. NLL is artificially low.[/]")
        console.print("[dim]Recommendations:[/]")
        console.print("  1. Increase sigma_min from 1e-4 to 1e-3 or 5e-3")
        console.print("  2. Add penalty: λ * E[max(0, σ_floor - σ)]")
        console.print("  3. Early stop based on calibration, not NLL")
    elif at_floor > 1:
        console.print("[yellow]⚠️ Mild σ-floor hitting. Monitor closely.[/]")
    else:
        console.print("[green]✓ No floor-hitting detected. σ distribution looks healthy.[/]")

    return {
        'sigma_median': p50,
        'sigma_p05': p05,
        'sigma_p95': p95,
        'weighted_sigma_median': np.median(weighted_sigma),
        'at_floor_pct': at_floor,
    }


def compute_calibration(data):
    """Compute PIT and coverage calibration metrics."""
    sigma = data['sigma']
    pi = data['pi']
    mu = data['mu']
    targets = data['targets']

    # Compute PIT for mixture model
    # CDF of mixture: F(x) = Σ π_k * Φ((x - μ_k) / σ_k)
    from scipy.stats import norm

    # Align shapes - targets may be shorter due to prediction shift
    seq_len = min(targets.shape[1], pi.shape[1])
    targets = targets[:, :seq_len]
    pi = pi[:, :seq_len]
    mu = mu[:, :seq_len]
    sigma = sigma[:, :seq_len]

    # Compute mixture CDF for each target
    pit_values = []

    for i in range(targets.shape[0]):
        for j in range(targets.shape[1]):
            t = targets[i, j]
            cdf = 0.0
            for k in range(pi.shape[-1]):
                cdf += pi[i, j, k] * norm.cdf(t, mu[i, j, k], sigma[i, j, k])
            pit_values.append(cdf)

    pit_values = np.array(pit_values)

    # PIT should be uniform [0, 1]
    ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')

    console.print(Panel.fit("[bold cyan]Calibration Analysis[/]", title="📊"))

    table = Table(title="PIT Statistics (should be Uniform[0,1])")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Ideal", style="dim")

    table.add_row("PIT mean", f"{pit_values.mean():.4f}", "0.5")
    table.add_row("PIT std", f"{pit_values.std():.4f}", "0.289")
    table.add_row("KS statistic", f"{ks_stat:.4f}", "<0.05")
    table.add_row("KS p-value", f"{ks_pval:.4f}", ">0.05")

    console.print(table)

    # Coverage analysis (using weighted mixture σ)
    # For each prediction, compute 50/90/95% intervals
    weighted_sigma = (pi * sigma).sum(axis=-1)
    weighted_mu = (pi * mu).sum(axis=-1)

    targets_flat = targets.flatten()
    mu_flat = weighted_mu.flatten()
    sigma_flat = weighted_sigma.flatten()

    # z-scores
    z = np.abs((targets_flat - mu_flat) / sigma_flat)

    cov_50 = (z < 0.674).mean() * 100  # 50% = ±0.674σ
    cov_90 = (z < 1.645).mean() * 100  # 90% = ±1.645σ
    cov_95 = (z < 1.96).mean() * 100   # 95% = ±1.96σ

    console.print("\n[bold]Coverage Analysis (weighted mean±kσ):[/]")
    table2 = Table()
    table2.add_column("Interval", style="cyan")
    table2.add_column("Nominal", style="dim")
    table2.add_column("Actual", style="green")
    table2.add_column("Status", style="yellow")

    def cov_status(actual, nominal):
        diff = actual - nominal
        if abs(diff) < 3:
            return "[green]✓ Well calibrated[/]"
        elif diff < -5:
            return "[red]⚠️ Overconfident[/]"
        else:
            return "[yellow]Under-confident[/]"

    table2.add_row("50%", "50%", f"{cov_50:.1f}%", cov_status(cov_50, 50))
    table2.add_row("90%", "90%", f"{cov_90:.1f}%", cov_status(cov_90, 90))
    table2.add_row("95%", "95%", f"{cov_95:.1f}%", cov_status(cov_95, 95))

    console.print(table2)

    # Verdict
    if ks_pval > 0.01 and abs(pit_values.mean() - 0.5) < 0.05:
        console.print("\n[green bold]✓ Calibration looks good![/]")
        console.print("[dim]Negative NLL is likely REAL progress, not σ-cheating.[/]")
    else:
        console.print("\n[yellow]⚠️ Calibration issues detected[/]")
        if pit_values.mean() > 0.55:
            console.print("[yellow]PIT > 0.5: Model predicts too high[/]")
        elif pit_values.mean() < 0.45:
            console.print("[yellow]PIT < 0.5: Model predicts too low[/]")

    return {
        'pit_mean': pit_values.mean(),
        'pit_std': pit_values.std(),
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'cov_50': cov_50,
        'cov_90': cov_90,
        'cov_95': cov_95,
    }


def main():
    parser = argparse.ArgumentParser(description="σ Diagnostic Check")
    parser.add_argument("--checkpoint", type=str, default="out/mdn_scaled/best.pt",
                        help="Path to checkpoint")
    parser.add_argument("--n-batches", type=int, default=50,
                        help="Number of batches to analyze")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    console.print(f"[cyan]Loading checkpoint: {args.checkpoint}[/]")
    ckpt = torch.load(args.checkpoint, weights_only=False)
    config = ckpt['config']

    model = SpacingMDN(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    console.print(f"[green]Model: {config.n_layer}L/{config.n_head}H/{config.n_embd}E[/]")
    console.print(f"[green]MDN components: {config.n_components}[/]")
    console.print(f"[green]sigma_min: {config.sigma_min}, sigma_max: {config.sigma_max}[/]")
    console.print(f"[dim]Step: {ckpt.get('step', 'N/A')}, Val NLL: {ckpt.get('val_nll', 'N/A'):.4f}[/]")

    # Load validation data
    val_data = torch.load(DATA_DIR / "val.pt", weights_only=False)
    val_data = torch.clamp(val_data, min=0.0, max=10.0)
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Analyze
    console.print(f"\n[cyan]Analyzing {args.n_batches} batches...[/]\n")
    data = compute_sigma_stats(model, val_loader, device, n_batches=args.n_batches)

    sigma_stats = analyze_sigma(data)
    calib_stats = compute_calibration(data)

    # Final verdict
    console.print("\n" + "="*60)
    console.print("[bold]FINAL VERDICT:[/]")

    if sigma_stats['at_floor_pct'] > 5:
        console.print("[red bold]❌ σ-COLLAPSE DETECTED[/]")
        console.print("[yellow]Negative NLL is mostly due to overconfidence, not learning.[/]")
        console.print("[dim]Action: Increase sigma_min or add σ penalty.[/]")
    elif calib_stats['ks_pval'] < 0.01:
        console.print("[yellow]⚠️ CALIBRATION WARNING[/]")
        console.print("[yellow]Model density not well calibrated.[/]")
    else:
        console.print("[green bold]✓ HEALTHY MODEL[/]")
        console.print("[green]Negative NLL appears to be genuine progress![/]")


if __name__ == "__main__":
    main()
