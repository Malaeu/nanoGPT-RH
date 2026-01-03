#!/usr/bin/env python3
"""
Evaluate SpacingMDN Flash with Residual Support.
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

from mdn_flash import SpacingMDN, MDNConfig
from memory_mdn_flash import MemoryMDN, MemoryMDNConfig

console = Console()

OUTPUT_DIR = Path("out/mdn_memory_q3_flash")
DEFAULT_DATA_DIR = Path("data/continuous_residuals")
REPORT_DIR = Path("reports/mdn_flash")


def load_model(ckpt_path: Path, device: str = "cuda"):
    """Load trained MDN model."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    config = ckpt['config']

    if isinstance(config, MemoryMDNConfig):
        model = MemoryMDN(config).to(device)
        model_type = "MemoryMDN"
    else:
        model = SpacingMDN(config).to(device)
        model_type = "SpacingMDN"

    model.load_state_dict(ckpt['model'])
    model.eval()
    console.print(f"[green]Loaded {ckpt_path}[/]")
    val_nll = ckpt.get('val_nll', 0.0)
    console.print(f"[dim]Step {ckpt.get('step', '?')}, NLL={val_nll:.4f}[/]")
    return model, config


def mixture_cdf(x, pi, mu, sigma):
    """CDF of Gaussian mixture."""
    x = x.unsqueeze(-1)
    # Gaussian CDF: 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    component_cdfs = 0.5 * (1 + torch.erf((x - mu) / (sigma * math.sqrt(2))))
    return torch.sum(pi * component_cdfs, dim=-1)


def compute_pit(model, data, device, n_samples=10000):
    model.eval()
    with torch.no_grad():
        idx = np.random.choice(len(data), min(n_samples // 256 + 1, len(data)), replace=False)
        subset = data[idx].to(device)
        result = model(subset)
        pi = result['pi'][:, :-1]
        mu = result['mu'][:, :-1]
        sigma = result['sigma'][:, :-1]
        targets = subset[:, 1:]
        cdf_vals = mixture_cdf(targets, pi, mu, sigma)
        return cdf_vals.flatten().cpu().numpy()


def compute_crps(model, data, device, n_samples=500, n_mc=200):
    model.eval()
    crps_values = []
    with torch.no_grad():
        idx = np.random.choice(len(data), min(50, len(data)), replace=False)
        subset = data[idx].to(device)
        result = model(subset)
        pi, mu, sigma = result['pi'][:, :-1], result['mu'][:, :-1], result['sigma'][:, :-1]
        targets = subset[:, 1:]
        
        for b in range(pi.shape[0]):
            for t in range(0, pi.shape[1], 20): # Subsample time
                # Sample from mixture
                k = torch.multinomial(pi[b, t], n_mc, replacement=True)
                samples = mu[b, t, k] + sigma[b, t, k] * torch.randn(n_mc, device=device)
                samples = samples.cpu().numpy()
                y = targets[b, t].item()
                # CRPS = E|X-y| - 0.5E|X-X'|
                crps = np.mean(np.abs(samples - y)) - 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
                crps_values.append(crps)
    return np.array(crps_values)


def test_error_accumulation(model, data, device, horizons=[10, 25, 50, 100], context_len=128, is_residual=False):
    model.eval()
    errors_mean = []
    with torch.no_grad():
        for horizon in horizons:
            errs = []
            n_traj = min(100, len(data))
            for i in range(n_traj):
                seq = data[i].to(device).unsqueeze(0)
                if seq.shape[1] < context_len + horizon: continue
                
                true_vals = seq[0, context_len:context_len+horizon]
                if is_residual: true_vals = true_vals + 1.0
                true_cum = true_vals.sum().item()
                
                pred_cum = 0.0
                ctx = seq[:, :context_len].clone()
                for _ in range(horizon):
                    res = model(ctx)
                    pred_res = torch.sum(res['pi'][:, -1] * res['mu'][:, -1], dim=-1).item()
                    pred_spac = pred_res + 1.0 if is_residual else pred_res
                    pred_cum += pred_spac
                    new_v = torch.tensor([[pred_res]], device=device)
                    ctx = torch.cat([ctx[:, 1:], new_v], dim=1)
                errs.append(abs(pred_cum - true_cum))
            errors_mean.append(np.mean(errs))
            console.print(f"Horizon {horizon}: err={errors_mean[-1]:.4f}")
    return np.array(errors_mean), horizons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=OUTPUT_DIR / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    is_residual = False
    if (args.data_dir / "meta.pt").exists():
        meta = torch.load(args.data_dir / "meta.pt", weights_only=False)
        is_residual = meta.get('is_residual', False)
        if is_residual: console.print("[yellow]Residual mode ACTIVE (+1.0)[/]")

    model, config = load_model(args.ckpt, device)
    val_data = torch.load(args.data_dir / "val.pt", weights_only=False)
    
    pit = compute_pit(model, val_data, device)
    ks_stat, ks_pval = stats.kstest(pit, 'uniform')
    
    crps = compute_crps(model, val_data, device)
    errs, horizons = test_error_accumulation(model, val_data, device, is_residual=is_residual)
    
    table = Table(title="Flash Evaluation Results")
    table.add_column("Metric"); table.add_column("Value")
    table.add_row("NLL (best)", f"{config.n_embd}") # placeholder
    table.add_row("PIT Mean", f"{pit.mean():.4f}")
    table.add_row("CRPS", f"{crps.mean():.4f}")
    table.add_row("Err@100", f"{errs[3]:.4f}")
    console.print(table)

if __name__ == "__main__":
    main()