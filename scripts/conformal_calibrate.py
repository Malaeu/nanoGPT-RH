#!/usr/bin/env python3
"""
Conformal Calibration for MDN Quantile Intervals

Split conformal prediction to guarantee coverage:
1. Split data into calib (50%) and test (50%)
2. Compute non-conformity scores on calib
3. Find adjustment q = (1-alpha)(1+1/n)-quantile
4. Verify coverage on test set

Reference: MASTER_SPEC.md §5.4, §10

Usage:
    python scripts/conformal_calibrate.py \
        --ckpt checkpoints/E4_s7_best.pt \
        --data-dir data/continuous_2M \
        --alpha 0.1 \
        --output results/calibrator.json
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class CalibrationResult:
    """Results from conformal calibration."""
    alpha: float
    adjustment_q: float
    calib_size: int
    test_size: int
    # Raw coverage (before adjustment)
    raw_coverage_80: float
    raw_coverage_90: float
    raw_width_mean: float
    # Calibrated coverage
    calib_coverage: float
    test_coverage: float
    test_width_mean: float
    # Status
    coverage_ok: bool
    width_reasonable: bool


def load_mdn_model(ckpt_path: Path, device: str = 'cpu'):
    """Load MDN model from checkpoint."""
    from train_mdn_postfix import SpacingMDNPostfix
    from train_mdn import MDNConfig

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']

    # Create config object if dict
    if isinstance(config, dict):
        config_obj = MDNConfig(**config)
    else:
        config_obj = config

    # Build model
    n_slots = ckpt.get('n_memory_slots', 8)
    slot_id_mode = ckpt.get('slot_id_mode', 'fixed')
    content_mode = ckpt.get('content_mode', 'normal')
    use_aux_loss = ckpt.get('use_aux_loss', False)

    model = SpacingMDNPostfix(
        config=config_obj,
        n_memory_slots=n_slots,
        slot_id_mode=slot_id_mode,
        content_mode=content_mode,
        use_aux_loss=use_aux_loss
    )
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    return model, config_obj


def compute_quantiles(model, x: torch.Tensor, device: str,
                      quantiles: list = [0.1, 0.5, 0.9]) -> dict:
    """
    Compute quantiles from MDN mixture.

    Returns dict with keys 'q0.1', 'q0.5', 'q0.9' etc.
    Each value is shape (batch,).
    """
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        # Model returns (pi, mu, sigma) tuple, shape [B, 1, K] for POSTFIX
        pi, mu, sigma = model(x)

        # Squeeze the seq dimension (POSTFIX returns [B, 1, K])
        pi = pi.squeeze(1)       # (batch, K)
        mu = mu.squeeze(1)       # (batch, K)
        sigma = sigma.squeeze(1) # (batch, K)

    # Compute quantiles via sampling from mixture
    n_samples = 1000
    batch_size = pi.shape[0]
    K = pi.shape[1]

    # Sample component indices
    component_samples = torch.multinomial(pi, n_samples, replacement=True)  # (batch, n_samples)

    # Sample from selected Gaussians
    samples = torch.zeros(batch_size, n_samples, device=device)
    for b in range(batch_size):
        for s in range(n_samples):
            k = component_samples[b, s]
            samples[b, s] = torch.normal(mu[b, k], sigma[b, k])

    # Compute quantiles
    result_q = {}
    for q in quantiles:
        q_val = torch.quantile(samples, q, dim=1)  # (batch,)
        result_q[f'q{q}'] = q_val.cpu().numpy()

    return result_q


def compute_nonconformity_scores(q_lo: np.ndarray, q_hi: np.ndarray,
                                  y_true: np.ndarray) -> np.ndarray:
    """
    Compute non-conformity scores.

    Score = max(q_lo - y, y - q_hi)
    Positive score means y is outside interval.
    """
    lower_violation = q_lo - y_true  # positive if y < q_lo
    upper_violation = y_true - q_hi  # positive if y > q_hi
    scores = np.maximum(lower_violation, upper_violation)
    return scores


def calibrate_intervals(model, data: torch.Tensor, device: str,
                        alpha: float = 0.1,
                        calib_fraction: float = 0.5) -> CalibrationResult:
    """
    Run split conformal calibration.

    Args:
        model: Trained MDN model
        data: Validation tensor (N, seq_len)
        device: 'cuda' or 'cpu'
        alpha: Target miscoverage rate (0.1 = 90% coverage)
        calib_fraction: Fraction of data for calibration

    Returns:
        CalibrationResult with adjustment factor and metrics
    """
    console.print(f"[cyan]Running conformal calibration (alpha={alpha})...[/]")

    N = data.shape[0]
    seq_len = data.shape[1]

    # Split into calib and test
    n_calib = int(N * calib_fraction)
    n_test = N - n_calib

    # Shuffle indices
    indices = np.random.permutation(N)
    calib_indices = indices[:n_calib]
    test_indices = indices[n_calib:]

    calib_data = data[calib_indices]
    test_data = data[test_indices]

    console.print(f"  Calib set: {n_calib} samples")
    console.print(f"  Test set: {n_test} samples")

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Compute raw quantiles on calib set
    # ═══════════════════════════════════════════════════════════════

    # Use positions 100:200 for calibration (avoid edges)
    batch_size = 64
    all_q_lo = []
    all_q_hi = []
    all_y_true = []

    console.print("[dim]Computing quantiles on calib set...[/]")

    for start in range(0, n_calib, batch_size):
        end = min(start + batch_size, n_calib)
        batch = calib_data[start:end]

        # Predict from context (positions 0:128) to target (position 128)
        context_len = min(128, seq_len - 1)
        x = batch[:, :context_len]
        y = batch[:, context_len].numpy()  # target

        q = compute_quantiles(model, x, device, quantiles=[0.1, 0.5, 0.9])
        all_q_lo.append(q['q0.1'])
        all_q_hi.append(q['q0.9'])
        all_y_true.append(y)

    q_lo_calib = np.concatenate(all_q_lo)
    q_hi_calib = np.concatenate(all_q_hi)
    y_true_calib = np.concatenate(all_y_true)

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Compute non-conformity scores
    # ═══════════════════════════════════════════════════════════════

    scores = compute_nonconformity_scores(q_lo_calib, q_hi_calib, y_true_calib)

    # Raw coverage (before adjustment)
    raw_coverage = np.mean((y_true_calib >= q_lo_calib) & (y_true_calib <= q_hi_calib))
    raw_width = np.mean(q_hi_calib - q_lo_calib)

    console.print(f"  Raw 80% coverage: {raw_coverage:.4f}")
    console.print(f"  Raw width: {raw_width:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Find adjustment quantile
    # ═══════════════════════════════════════════════════════════════

    # Conformal quantile: (1-alpha)(1 + 1/n)
    # For 90% coverage, alpha=0.1
    n = len(scores)
    target_quantile = (1 - alpha) * (1 + 1/n)
    target_quantile = min(target_quantile, 1.0)

    adjustment_q = float(np.quantile(scores, target_quantile))
    console.print(f"  Adjustment q: {adjustment_q:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Verify on test set
    # ═══════════════════════════════════════════════════════════════

    console.print("[dim]Verifying on test set...[/]")

    all_q_lo = []
    all_q_hi = []
    all_y_true = []

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        batch = test_data[start:end]

        context_len = min(128, seq_len - 1)
        x = batch[:, :context_len]
        y = batch[:, context_len].numpy()

        q = compute_quantiles(model, x, device, quantiles=[0.1, 0.5, 0.9])
        all_q_lo.append(q['q0.1'])
        all_q_hi.append(q['q0.9'])
        all_y_true.append(y)

    q_lo_test = np.concatenate(all_q_lo)
    q_hi_test = np.concatenate(all_q_hi)
    y_true_test = np.concatenate(all_y_true)

    # Apply adjustment
    q_lo_adjusted = q_lo_test - adjustment_q
    q_hi_adjusted = q_hi_test + adjustment_q

    # Calibrated coverage
    test_coverage = np.mean((y_true_test >= q_lo_adjusted) & (y_true_test <= q_hi_adjusted))
    test_width = np.mean(q_hi_adjusted - q_lo_adjusted)

    # Calib set coverage (with adjustment)
    q_lo_calib_adj = q_lo_calib - adjustment_q
    q_hi_calib_adj = q_hi_calib + adjustment_q
    calib_coverage = np.mean((y_true_calib >= q_lo_calib_adj) & (y_true_calib <= q_hi_calib_adj))

    console.print(f"  Calibrated coverage (calib): {calib_coverage:.4f}")
    console.print(f"  Calibrated coverage (test): {test_coverage:.4f}")
    console.print(f"  Calibrated width: {test_width:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Check thresholds
    # ═══════════════════════════════════════════════════════════════

    target_coverage = 1 - alpha
    coverage_ok = abs(test_coverage - target_coverage) < 0.03  # within 3%
    width_reasonable = test_width < 1.0  # not too wide

    # Also compute 80% coverage for reference
    q80_lo = q_lo_test - adjustment_q * 0.6  # rough scaling
    q80_hi = q_hi_test + adjustment_q * 0.6
    raw_coverage_80 = np.mean((y_true_test >= q_lo_test) & (y_true_test <= q_hi_test))

    return CalibrationResult(
        alpha=alpha,
        adjustment_q=adjustment_q,
        calib_size=n_calib,
        test_size=n_test,
        raw_coverage_80=raw_coverage_80,
        raw_coverage_90=raw_coverage,
        raw_width_mean=raw_width,
        calib_coverage=calib_coverage,
        test_coverage=test_coverage,
        test_width_mean=test_width,
        coverage_ok=coverage_ok,
        width_reasonable=width_reasonable
    )


def print_calibration_report(result: CalibrationResult):
    """Print formatted calibration report."""
    table = Table(title="Conformal Calibration Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Target", style="dim")
    table.add_column("Status", style="bold")

    target_cov = 1 - result.alpha

    table.add_row(
        "Alpha (miscoverage)",
        f"{result.alpha:.2f}",
        "-",
        ""
    )
    table.add_row(
        "Adjustment q",
        f"{result.adjustment_q:.4f}",
        "-",
        ""
    )
    table.add_row(
        "Raw coverage @90%",
        f"{result.raw_coverage_90:.4f}",
        "0.90",
        "[yellow]raw[/]"
    )
    table.add_row(
        "Calibrated coverage (test)",
        f"{result.test_coverage:.4f}",
        f"{target_cov:.2f} +/- 0.02",
        "[green]PASS[/]" if result.coverage_ok else "[red]FAIL[/]"
    )
    table.add_row(
        "Calibrated width",
        f"{result.test_width_mean:.4f}",
        "< 1.0",
        "[green]PASS[/]" if result.width_reasonable else "[red]FAIL[/]"
    )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Conformal Calibration for MDN Quantiles")
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Target miscoverage rate (0.1 = 90%% coverage)')
    parser.add_argument('--output', type=str, default='results/calibrator.json',
                       help='Output JSON path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for split')

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"[cyan]Device: {device}[/]")

    # Load model
    ckpt_path = Path(args.ckpt)
    console.print(f"[cyan]Loading model: {ckpt_path}[/]")
    model, config = load_mdn_model(ckpt_path, device)

    # Load data (use validation set)
    data_dir = Path(args.data_dir)
    val_path = data_dir / 'val.pt'
    console.print(f"[cyan]Loading data: {val_path}[/]")
    val_data = torch.load(val_path, weights_only=True)
    console.print(f"  Shape: {val_data.shape}")

    # Run calibration
    result = calibrate_intervals(
        model=model,
        data=val_data,
        device=device,
        alpha=args.alpha
    )

    # Print report
    print_calibration_report(result)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    result_dict = asdict(result)
    for k, v in result_dict.items():
        if hasattr(v, 'item'):  # numpy scalar
            result_dict[k] = v.item()
        elif isinstance(v, (np.floating, np.integer)):
            result_dict[k] = float(v) if isinstance(v, np.floating) else int(v)

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    console.print(f"\n[green]Saved calibration to: {output_path}[/]")

    # Summary
    if result.coverage_ok and result.width_reasonable:
        console.print("\n[green bold]CALIBRATION PASSED![/]")
        console.print(f"Use adjustment_q = {result.adjustment_q:.4f} to expand intervals")
    else:
        console.print("\n[red bold]CALIBRATION ISSUES:[/]")
        if not result.coverage_ok:
            console.print(f"  - Coverage {result.test_coverage:.4f} not within target range")
        if not result.width_reasonable:
            console.print(f"  - Width {result.test_width_mean:.4f} too large")


if __name__ == '__main__':
    main()
