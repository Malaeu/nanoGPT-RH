#!/usr/bin/env python3
"""
R0/R1 Ablation Study for Causal Zeta.

Compares:
- R0: Neural only (no filter)
- R1: Neural + Q3 rejector (calibrated on real-val)

Rejector validators (Q3-style, not arbitrary thresholds):
1. Sanity: no NaN/Inf, mean spacing ~ 1
2. R_t distribution gate: P5-P95 from real-val
3. Healing stability: not 5x worse than baseline

Usage:
    python -m causal_zeta.run_ablation --mode R0 --checkpoint out/best.pt
    python -m causal_zeta.run_ablation --mode R1 --checkpoint out/best.pt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import SpacingGPT
from causal_zeta.variables import LatentExtractor, RigidityCalculator

console = Console()


# =============================================================================
# SFF (Spectral Form Factor) - Long-range rigidity test
# =============================================================================

def compute_sff(spacings: np.ndarray, tau_values: np.ndarray = None) -> dict:
    """
    Compute Spectral Form Factor from spacings.

    K(τ) = (1/N) |Σ exp(i τ u_n)|²

    where u_n = cumsum(spacings) are unfolded coordinates.

    Returns:
        dict with tau_values, K_values, ramp_slope, ramp_rmse, plateau_level
    """
    # Build unfolded coordinates: u_0=0, u_{n+1}=u_n+s_n
    u = np.concatenate([[0], np.cumsum(spacings)])
    N = len(u)

    # Default tau range: [0.1, 2π] with log spacing
    if tau_values is None:
        tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    # Compute SFF: K(τ) = (1/N) |Σ exp(iτu_n)|²
    # Note: K(0) = N, so normalized K(τ)/K(0) gives range [0, 1]
    K_values = np.zeros(len(tau_values))
    for i, tau in enumerate(tau_values):
        phases = np.exp(1j * tau * u)
        K_values[i] = np.abs(np.sum(phases))**2 / N  # Single normalization

    # Extract metrics:
    # 1. Ramp region: τ in [0.5, 3] (typical for GUE)
    ramp_mask = (tau_values >= 0.5) & (tau_values <= 3.0)
    if np.sum(ramp_mask) > 2:
        tau_ramp = tau_values[ramp_mask]
        K_ramp = K_values[ramp_mask]
        # Linear fit: K = slope * τ + intercept
        slope, intercept = np.polyfit(tau_ramp, K_ramp, 1)
        K_fit = slope * tau_ramp + intercept
        rmse = np.sqrt(np.mean((K_ramp - K_fit)**2))
    else:
        slope, rmse = 0.0, 0.0

    # 2. Plateau region: τ > 4
    plateau_mask = tau_values > 4.0
    if np.sum(plateau_mask) > 0:
        plateau_level = np.mean(K_values[plateau_mask])
    else:
        plateau_level = K_values[-1]

    return {
        "tau_values": tau_values,
        "K_values": K_values,
        "ramp_slope": slope,
        "ramp_rmse": rmse,
        "plateau_level": plateau_level,
    }


def compute_sff_from_trajectories(results: list, real_spacings: np.ndarray = None) -> dict:
    """
    Compute SFF for real data and generated trajectories.

    Returns dict with keys: real, generated, and comparison metrics.
    """
    # Aggregate all spacings from accepted trajectories
    gen_spacings = []
    for r in results:
        if r.accepted:
            gen_spacings.extend(r.spacings)
    gen_spacings = np.array(gen_spacings)

    # Use same tau range for fair comparison
    tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    sff_gen = compute_sff(gen_spacings, tau_values)

    result = {
        "generated": sff_gen,
        "n_spacings": len(gen_spacings),
    }

    if real_spacings is not None:
        sff_real = compute_sff(real_spacings, tau_values)
        result["real"] = sff_real
        result["real_n_spacings"] = len(real_spacings)

        # Comparison metrics
        result["slope_ratio"] = sff_gen["ramp_slope"] / sff_real["ramp_slope"] if sff_real["ramp_slope"] != 0 else 0
        result["plateau_ratio"] = sff_gen["plateau_level"] / sff_real["plateau_level"] if sff_real["plateau_level"] != 0 else 0

    return result


@dataclass
class RejectorConfig:
    """Calibrated rejector thresholds from real-val."""
    R_t_p5: float = 0.0
    R_t_p95: float = 0.0
    R_t_mean: float = 1.0
    R_t_std: float = 0.0
    healing_baseline: float = 10.0
    healing_max_factor: float = 5.0  # reject if healing > 5x baseline


@dataclass
class TrajectoryResult:
    """Result of generating one trajectory."""
    tokens: np.ndarray  # [T,] bin indices
    spacings: np.ndarray  # [T,] actual spacing values
    R_t_values: np.ndarray  # [T,] rigidity values
    healing_time: float
    accepted: bool = True
    reject_reason: str = ""


def load_model_and_data(args):
    """Load model, data, and bin_centers."""
    console.print(f"[cyan]Loading checkpoint: {args.checkpoint}[/]")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = SpacingGPT(config)
    model.load_state_dict(ckpt["model"])

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()
    console.print(f"[green]Model loaded on {device}[/]")

    # Load data
    data_path = Path(args.data_dir)
    val_data = torch.load(data_path / "val.pt", weights_only=False)
    bin_centers = np.load(data_path / "bin_centers.npy")

    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
    )

    console.print(f"[cyan]Loaded {len(val_data)} validation sequences, {len(bin_centers)} bins[/]")

    return model, val_loader, bin_centers, device


def calibrate_rejector(val_loader, bin_centers, rigidity_window: int = 10, rejector_mode: str = "soft") -> RejectorConfig:
    """Calibrate rejector thresholds on real validation data."""
    console.print(Panel.fit("[bold magenta]Calibrating Rejector on Real-Val[/]"))

    rigidity_calc = RigidityCalculator(window=rigidity_window, bin_centers=bin_centers)

    all_R_t = []

    for (batch,) in val_loader:
        R_traj = rigidity_calc.compute(batch)  # [B, T]
        # Skip early positions (window warmup)
        R_valid = R_traj[:, rigidity_window:].cpu().numpy().flatten()
        all_R_t.extend(R_valid)

    all_R_t = np.array(all_R_t)

    # Percentile modes: soft (P5-P95), medium (P10-P90), hard (P20-P80)
    percentile_map = {
        "soft": (5, 95),
        "medium": (10, 90),
        "hard": (20, 80),
    }
    lo_pct, hi_pct = percentile_map.get(rejector_mode, (5, 95))

    config = RejectorConfig(
        R_t_p5=np.percentile(all_R_t, lo_pct),  # Actually uses lo_pct
        R_t_p95=np.percentile(all_R_t, hi_pct),  # Actually uses hi_pct
        R_t_mean=np.mean(all_R_t),
        R_t_std=np.std(all_R_t),
        healing_baseline=8.2,  # from Round 005
        healing_max_factor=5.0,
    )

    console.print(f"  Rejector mode: {rejector_mode} (P{lo_pct}-P{hi_pct})")
    console.print(f"  R_t P{lo_pct}:  {config.R_t_p5:.4f}")
    console.print(f"  R_t P{hi_pct}: {config.R_t_p95:.4f}")
    console.print(f"  R_t mean: {config.R_t_mean:.4f} ± {config.R_t_std:.4f}")
    console.print(f"  Healing baseline: {config.healing_baseline:.1f}")
    console.print(f"  Healing max: {config.healing_baseline * config.healing_max_factor:.1f}")

    return config


def generate_trajectory(
    model: SpacingGPT,
    context: torch.Tensor,
    traj_len: int,
    bin_centers: np.ndarray,
    rigidity_calc: RigidityCalculator,
    temperature: float = 1.0,
) -> TrajectoryResult:
    """Generate one trajectory and compute metrics."""
    device = context.device

    with torch.no_grad():
        generated = model.generate(context, traj_len, temperature)
        tokens = generated[0, context.shape[1]:].cpu().numpy()

    # Convert to spacings
    spacings = bin_centers[tokens]

    # Compute R_t
    tokens_tensor = torch.tensor(tokens, device=device).unsqueeze(0)
    R_traj = rigidity_calc.compute(tokens_tensor)[0].cpu().numpy()

    # Compute healing time using impulse response methodology
    # Measures how long until trajectory "forgets" its starting point
    # Based on deviation from running mean
    window = 10
    if len(spacings) > window * 2:
        # Running mean as "equilibrium"
        running_mean = np.convolve(spacings, np.ones(window)/window, mode='valid')
        # Deviation from running mean at each point
        deviations = np.abs(spacings[window-1:len(running_mean)+window-1] - running_mean)
        # Healing = first time deviation drops below 10% of initial
        initial_dev = deviations[0] if len(deviations) > 0 else 1.0
        threshold = max(0.1, initial_dev * 0.1)  # 10% of initial or 0.1
        healing_time = next((i for i, d in enumerate(deviations) if d < threshold), len(deviations))
    else:
        healing_time = len(spacings)

    return TrajectoryResult(
        tokens=tokens,
        spacings=spacings,
        R_t_values=R_traj,
        healing_time=healing_time,
    )


def apply_rejector(result: TrajectoryResult, config: RejectorConfig) -> TrajectoryResult:
    """Apply Q3 rejector to trajectory. Sets accepted=False if rejected."""
    # Validator 1: Sanity
    if np.any(np.isnan(result.spacings)) or np.any(np.isinf(result.spacings)):
        result.accepted = False
        result.reject_reason = "NaN/Inf in spacings"
        return result

    mean_s = np.mean(result.spacings)
    if abs(mean_s - 1.0) > 0.3:  # More than 30% off from 1.0
        result.accepted = False
        result.reject_reason = f"Mean spacing {mean_s:.3f} too far from 1.0"
        return result

    # Validator 2: R_t distribution gate
    R_valid = result.R_t_values[10:]  # Skip warmup
    if len(R_valid) > 0:
        R_p5 = np.percentile(R_valid, 5)
        R_p95 = np.percentile(R_valid, 95)

        # Check if trajectory R_t is within real-val range (with margin)
        margin = 0.5  # Allow some slack
        if R_p5 < config.R_t_p5 - margin or R_p95 > config.R_t_p95 + margin:
            result.accepted = False
            result.reject_reason = f"R_t out of range: [{R_p5:.2f}, {R_p95:.2f}] vs [{config.R_t_p5:.2f}, {config.R_t_p95:.2f}]"
            return result

    # Validator 3: Healing stability
    if result.healing_time > config.healing_baseline * config.healing_max_factor:
        result.accepted = False
        result.reject_reason = f"Healing {result.healing_time:.1f} > {config.healing_baseline * config.healing_max_factor:.1f}"
        return result

    result.accepted = True
    return result


def run_ablation(model, val_loader, bin_centers, args, device, rejector_config=None):
    """Run ablation study for one mode (R0 or R1)."""
    mode = args.mode
    console.print(Panel.fit(f"[bold blue]Running {mode} Ablation[/]"))

    rigidity_calc = RigidityCalculator(window=args.rigidity_window, bin_centers=bin_centers)

    # Get contexts from validation data
    contexts = []
    for (batch,) in val_loader:
        for i in range(batch.shape[0]):
            contexts.append(batch[i:i+1, :args.context_len].to(device))
            if len(contexts) >= args.n_traj:
                break
        if len(contexts) >= args.n_traj:
            break

    console.print(f"[cyan]Generating {args.n_traj} trajectories (len={args.traj_len})...[/]")

    results = []
    for i, ctx in enumerate(track(contexts, description=f"Generating {mode}")):
        torch.manual_seed(args.seed + i)
        result = generate_trajectory(
            model, ctx, args.traj_len, bin_centers, rigidity_calc, args.temperature
        )

        if mode == "R1" and rejector_config:
            result = apply_rejector(result, rejector_config)

        results.append(result)

    return results


def collect_real_spacings(val_loader, bin_centers: np.ndarray, max_spacings: int = 50000) -> np.ndarray:
    """Collect spacings from real validation data for SFF baseline."""
    all_spacings = []
    for (batch,) in val_loader:
        for i in range(batch.shape[0]):
            tokens = batch[i].numpy()
            spacings = bin_centers[tokens]
            all_spacings.extend(spacings)
            if len(all_spacings) >= max_spacings:
                break
        if len(all_spacings) >= max_spacings:
            break
    return np.array(all_spacings[:max_spacings])


def compute_metrics(results: list[TrajectoryResult], mode: str) -> dict:
    """Compute summary metrics from trajectory results."""
    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]

    # Aggregate R_t values
    all_R_t = []
    for r in accepted:
        all_R_t.extend(r.R_t_values[10:])  # Skip warmup
    all_R_t = np.array(all_R_t) if all_R_t else np.array([1.0])

    # Aggregate spacings
    all_spacings = []
    for r in accepted:
        all_spacings.extend(r.spacings)
    all_spacings = np.array(all_spacings) if all_spacings else np.array([1.0])

    # Healing times
    healing_times = [r.healing_time for r in accepted]

    # Rejection reasons
    reject_reasons = {}
    for r in rejected:
        reason = r.reject_reason.split(":")[0] if ":" in r.reject_reason else r.reject_reason
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    return {
        "mode": mode,
        "n_total": len(results),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "acceptance_rate": len(accepted) / len(results) if results else 0,
        "reject_reasons": reject_reasons,
        # R_t stats
        "R_t_mean": np.mean(all_R_t),
        "R_t_std": np.std(all_R_t),
        "R_t_p5": np.percentile(all_R_t, 5),
        "R_t_p50": np.percentile(all_R_t, 50),
        "R_t_p95": np.percentile(all_R_t, 95),
        # Spacing stats
        "S_mean": np.mean(all_spacings),
        "S_std": np.std(all_spacings),
        # Healing stats
        "healing_mean": np.mean(healing_times) if healing_times else 0,
        "healing_std": np.std(healing_times) if healing_times else 0,
        "healing_p50": np.percentile(healing_times, 50) if healing_times else 0,
        "healing_p95": np.percentile(healing_times, 95) if healing_times else 0,
    }


def print_sff_metrics(sff_result: dict, mode: str):
    """Pretty print SFF comparison metrics."""
    console.print(f"\n[bold magenta]SFF Results for {mode}:[/]")

    table = Table(title="Spectral Form Factor Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Real", style="green")
    table.add_column(mode, style="yellow")
    table.add_column("Ratio", style="blue")

    if "real" in sff_result:
        real = sff_result["real"]
        gen = sff_result["generated"]

        table.add_row(
            "Ramp Slope",
            f"{real['ramp_slope']:.6f}",
            f"{gen['ramp_slope']:.6f}",
            f"{sff_result['slope_ratio']:.3f}",
        )
        table.add_row(
            "Ramp RMSE",
            f"{real['ramp_rmse']:.6f}",
            f"{gen['ramp_rmse']:.6f}",
            "-",
        )
        table.add_row(
            "Plateau Level",
            f"{real['plateau_level']:.6f}",
            f"{gen['plateau_level']:.6f}",
            f"{sff_result['plateau_ratio']:.3f}",
        )
        table.add_row(
            "N spacings",
            f"{sff_result['real_n_spacings']}",
            f"{sff_result['n_spacings']}",
            "-",
        )
    else:
        gen = sff_result["generated"]
        table.add_row("Ramp Slope", "-", f"{gen['ramp_slope']:.6f}", "-")
        table.add_row("Ramp RMSE", "-", f"{gen['ramp_rmse']:.6f}", "-")
        table.add_row("Plateau Level", "-", f"{gen['plateau_level']:.6f}", "-")
        table.add_row("N spacings", "-", f"{sff_result['n_spacings']}", "-")

    console.print(table)


def print_metrics(metrics: dict):
    """Pretty print metrics."""
    console.print(f"\n[bold cyan]Results for {metrics['mode']}:[/]")

    table = Table(title=f"{metrics['mode']} Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Trajectories", f"{metrics['n_accepted']}/{metrics['n_total']}")
    table.add_row("Acceptance Rate", f"{metrics['acceptance_rate']:.1%}")
    table.add_row("", "")
    table.add_row("R_t mean ± std", f"{metrics['R_t_mean']:.4f} ± {metrics['R_t_std']:.4f}")
    table.add_row("R_t [P5, P50, P95]", f"[{metrics['R_t_p5']:.3f}, {metrics['R_t_p50']:.3f}, {metrics['R_t_p95']:.3f}]")
    table.add_row("", "")
    table.add_row("S mean ± std", f"{metrics['S_mean']:.4f} ± {metrics['S_std']:.4f}")
    table.add_row("", "")
    table.add_row("Healing mean ± std", f"{metrics['healing_mean']:.1f} ± {metrics['healing_std']:.1f}")
    table.add_row("Healing [P50, P95]", f"[{metrics['healing_p50']:.1f}, {metrics['healing_p95']:.1f}]")

    console.print(table)

    if metrics['reject_reasons']:
        console.print("\n[yellow]Rejection Reasons:[/]")
        for reason, count in sorted(metrics['reject_reasons'].items(), key=lambda x: -x[1]):
            console.print(f"  {reason}: {count}")


def generate_report(args, metrics: dict, rejector_config: RejectorConfig = None, sff_result: dict = None):
    """Generate markdown report."""
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Ablation Study: {args.mode}")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\n---\n")

    lines.append("## Configuration\n")
    lines.append(f"- Mode: **{args.mode}**")
    lines.append(f"- Checkpoint: `{args.checkpoint}`")
    lines.append(f"- N trajectories: {args.n_traj}")
    lines.append(f"- Trajectory length: {args.traj_len}")
    lines.append(f"- Context length: {args.context_len}")
    lines.append(f"- Rigidity window: {args.rigidity_window}")
    lines.append(f"- Seed: {args.seed}")

    if rejector_config and args.mode == "R1":
        lines.append("\n## Rejector Calibration (from real-val)\n")
        lines.append(f"- R_t P5: {rejector_config.R_t_p5:.4f}")
        lines.append(f"- R_t P95: {rejector_config.R_t_p95:.4f}")
        lines.append(f"- Healing baseline: {rejector_config.healing_baseline:.1f}")
        lines.append(f"- Healing max (5x): {rejector_config.healing_baseline * rejector_config.healing_max_factor:.1f}")

    lines.append("\n## Results\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Trajectories | {metrics['n_accepted']}/{metrics['n_total']} |")
    lines.append(f"| Acceptance Rate | {metrics['acceptance_rate']:.1%} |")
    lines.append(f"| R_t mean ± std | {metrics['R_t_mean']:.4f} ± {metrics['R_t_std']:.4f} |")
    lines.append(f"| R_t [P5, P50, P95] | [{metrics['R_t_p5']:.3f}, {metrics['R_t_p50']:.3f}, {metrics['R_t_p95']:.3f}] |")
    lines.append(f"| S mean ± std | {metrics['S_mean']:.4f} ± {metrics['S_std']:.4f} |")
    lines.append(f"| Healing mean ± std | {metrics['healing_mean']:.1f} ± {metrics['healing_std']:.1f} |")
    lines.append(f"| Healing [P50, P95] | [{metrics['healing_p50']:.1f}, {metrics['healing_p95']:.1f}] |")

    if metrics['reject_reasons']:
        lines.append("\n## Rejection Reasons\n")
        lines.append("| Reason | Count |")
        lines.append("|--------|-------|")
        for reason, count in sorted(metrics['reject_reasons'].items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")

    # SFF section
    if sff_result is not None:
        lines.append("\n## SFF (Spectral Form Factor)\n")
        lines.append("Long-range spectral rigidity test.")
        lines.append("")
        lines.append("| Metric | Real | Generated | Ratio |")
        lines.append("|--------|------|-----------|-------|")

        if "real" in sff_result:
            real = sff_result["real"]
            gen = sff_result["generated"]
            lines.append(f"| Ramp Slope | {real['ramp_slope']:.6f} | {gen['ramp_slope']:.6f} | {sff_result['slope_ratio']:.3f} |")
            lines.append(f"| Ramp RMSE | {real['ramp_rmse']:.6f} | {gen['ramp_rmse']:.6f} | - |")
            lines.append(f"| Plateau Level | {real['plateau_level']:.6f} | {gen['plateau_level']:.6f} | {sff_result['plateau_ratio']:.3f} |")
            lines.append(f"| N spacings | {sff_result['real_n_spacings']} | {sff_result['n_spacings']} | - |")
        else:
            gen = sff_result["generated"]
            lines.append(f"| Ramp Slope | - | {gen['ramp_slope']:.6f} | - |")
            lines.append(f"| Ramp RMSE | - | {gen['ramp_rmse']:.6f} | - |")
            lines.append(f"| Plateau Level | - | {gen['plateau_level']:.6f} | - |")
            lines.append(f"| N spacings | - | {sff_result['n_spacings']} | - |")

        # Interpretation
        if "real" in sff_result:
            lines.append("")
            slope_ratio = sff_result['slope_ratio']
            if 0.9 <= slope_ratio <= 1.1:
                lines.append("**Ramp**: ✓ PASS (within 10% of real)")
            elif 0.8 <= slope_ratio <= 1.2:
                lines.append(f"**Ramp**: ~ OK (ratio = {slope_ratio:.2f})")
            else:
                lines.append(f"**Ramp**: ✗ FAIL (ratio = {slope_ratio:.2f})")

    lines.append("\n---\n*End of report*")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    console.print(f"[green]Report saved to {report_path}[/]")


def main():
    parser = argparse.ArgumentParser(description="R0/R1 Ablation Study")
    parser.add_argument("--mode", type=str, required=True, choices=["R0", "R1"])
    parser.add_argument("--checkpoint", type=str, default="out/best.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--report", type=str, default=None)

    parser.add_argument("--n-traj", type=int, default=200)
    parser.add_argument("--traj-len", type=int, default=256)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--rigidity-window", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--sff", action="store_true", help="Compute Spectral Form Factor")
    parser.add_argument("--rejector-mode", type=str, default="soft", choices=["soft", "medium", "hard"],
                       help="Rejector strictness: soft (P5-P95), medium (P10-P90), hard (P20-P80)")

    args = parser.parse_args()

    if args.report is None:
        args.report = f"reports/ablation_{args.mode}.md"

    console.print(Panel.fit(
        f"[bold blue]R0/R1 ABLATION STUDY[/]\n"
        f"Mode: {args.mode}",
        title="Causal Zeta"
    ))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load
    model, val_loader, bin_centers, device = load_model_and_data(args)

    # Calibrate rejector on real-val (needed for R1, useful for reference in R0)
    rejector_config = calibrate_rejector(val_loader, bin_centers, args.rigidity_window, args.rejector_mode)

    # Run ablation
    results = run_ablation(
        model, val_loader, bin_centers, args, device,
        rejector_config=rejector_config if args.mode == "R1" else None
    )

    # Compute and print metrics
    metrics = compute_metrics(results, args.mode)
    print_metrics(metrics)

    # SFF analysis (if requested)
    sff_result = None
    if args.sff:
        console.print(Panel.fit("[bold magenta]Computing SFF (Spectral Form Factor)...[/]"))
        # Collect real spacings for baseline
        real_spacings = collect_real_spacings(val_loader, bin_centers, max_spacings=args.n_traj * args.traj_len)
        sff_result = compute_sff_from_trajectories(results, real_spacings)
        print_sff_metrics(sff_result, args.mode)

    # Generate report
    generate_report(args, metrics, rejector_config, sff_result)

    console.print(Panel.fit(
        f"[bold green]✓ {args.mode} Complete![/]\n"
        f"Acceptance: {metrics['acceptance_rate']:.1%}\n"
        + (f"SFF slope ratio: {sff_result['slope_ratio']:.3f}\n" if sff_result and 'slope_ratio' in sff_result else "")
        + f"Report: {args.report}",
        title="Results"
    ))


if __name__ == "__main__":
    main()
