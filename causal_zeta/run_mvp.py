#!/usr/bin/env python3
"""
Causal Zeta MVP: Core causal analysis pipeline (no LLM agents).

This script runs:
1. Extract: Collect causal states (Z_t, R_t, S_t, Y_t)
2. CI Tests: HSIC tests for graph validation
3. Intervention: do(S) with healing metrics
4. Validate: Sanity checks on spacing values
5. Report: Generate markdown report

Usage:
    python -m causal_zeta.run_mvp --checkpoint out/best.pt --report reports/round_001.md
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import SpacingGPT
from causal_zeta.variables import LatentExtractor, RigidityCalculator, collect_causal_states
from causal_zeta.graph import CausalGraph
from causal_zeta.ci_tests import run_ci_tests, graph_validation_tests, print_ci_results
from causal_zeta.interventions import SpacingIntervention
from causal_zeta.validators import Q3Validator

console = Console()


def load_model_and_data(args):
    """Load trained SpacingGPT, data, and bin_centers."""
    console.print(f"[cyan]Loading checkpoint: {args.checkpoint}[/]")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = SpacingGPT(config)
    model.load_state_dict(ckpt["model"])

    # Device
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
    meta = torch.load(data_path / "meta.pt", weights_only=False)

    # Load or compute bin_centers
    bin_centers_path = data_path / "bin_centers.npy"
    if bin_centers_path.exists():
        bin_centers = np.load(bin_centers_path)
        console.print(f"[cyan]Loaded bin_centers: {len(bin_centers)} values[/]")
    elif "bin_edges" in meta:
        edges = np.array(meta["bin_edges"])
        bin_centers = (edges[:-1] + edges[1:]) / 2
        np.save(bin_centers_path, bin_centers)
        console.print(f"[cyan]Computed bin_centers from edges: {len(bin_centers)} values[/]")
    else:
        raise ValueError("No bin_centers or bin_edges found!")

    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
    )

    console.print(f"[cyan]Loaded {len(val_data)} validation sequences[/]")

    return model, val_loader, meta, bin_centers, device


def run_extraction(model, val_loader, bin_centers, args, device):
    """Phase 1: Extract causal states."""
    console.print(Panel.fit("[bold magenta]Phase 1: Extraction[/]"))

    # Fit latent extractor
    latent_extractor = LatentExtractor(model, n_components=2)
    latent_extractor.fit(val_loader, n_samples=args.pca_fit_samples)

    # Rigidity calculator with bin_centers
    rigidity_calc = RigidityCalculator(window=args.rigidity_window, bin_centers=bin_centers)

    # Collect causal states
    console.print(f"[cyan]Collecting {args.n_windows} causal states (R_window={args.rigidity_window})...[/]")
    states = collect_causal_states(
        model, val_loader, latent_extractor, rigidity_calc,
        bin_centers, n_samples=args.n_windows,
        rigidity_window=args.rigidity_window
    )
    console.print(f"[green]Collected {len(states)} causal states[/]")

    # Compute statistics on spacing values
    S_vals = np.array([s.S_t for s in states])
    Y_vals = np.array([s.Y_t for s in states])
    R_vals = np.array([s.R_t for s in states])

    stats = {
        "S_mean": np.mean(S_vals),
        "S_std": np.std(S_vals),
        "S_min": np.min(S_vals),
        "S_max": np.max(S_vals),
        "Y_mean": np.mean(Y_vals),
        "R_mean": np.mean(R_vals),
        "R_std": np.std(R_vals),
    }

    table = Table(title="Spacing Statistics (on actual values)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Expected", style="yellow")
    table.add_row("S mean", f"{stats['S_mean']:.4f}", "≈ 1.0")
    table.add_row("S std", f"{stats['S_std']:.4f}", "≈ 0.42")
    table.add_row("S min", f"{stats['S_min']:.4f}", "> 0")
    table.add_row("S max", f"{stats['S_max']:.4f}", "< 4")
    table.add_row("R mean", f"{stats['R_mean']:.4f}", "≈ 1.0")
    console.print(table)

    return states, latent_extractor, rigidity_calc, stats


def compute_effect_size(states) -> dict:
    """
    Compute effect size (ΔR² and ΔMAE) for CI-1 test.

    Model 0: R_t ~ Ridge(S_{t-1}, Z_t)
    Model 1: R_t ~ Ridge(S_{t-1}, Z_t, S_deep)

    Returns:
        dict with delta_r2, delta_mae, r2_baseline, r2_with_lag
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_absolute_error

    # Extract features
    S_t1 = np.array([s.S_t for s in states])  # S_{t-1}
    Z_t = np.array([s.Z_t for s in states])   # [n, 2]
    S_deep = np.array([s.S_t_prev for s in states])  # S_deep (outside R_t window)
    R_t = np.array([s.R_t for s in states])   # target

    # Model 0 features: [S_{t-1}, Z_t[0], Z_t[1]]
    X0 = np.column_stack([S_t1, Z_t])

    # Model 1 features: [S_{t-1}, Z_t[0], Z_t[1], S_deep]
    X1 = np.column_stack([S_t1, Z_t, S_deep])

    # Fit Ridge regression
    ridge0 = Ridge(alpha=1.0).fit(X0, R_t)
    ridge1 = Ridge(alpha=1.0).fit(X1, R_t)

    # Predictions
    pred0 = ridge0.predict(X0)
    pred1 = ridge1.predict(X1)

    # Metrics
    r2_0 = r2_score(R_t, pred0)
    r2_1 = r2_score(R_t, pred1)
    mae_0 = mean_absolute_error(R_t, pred0)
    mae_1 = mean_absolute_error(R_t, pred1)

    delta_r2 = r2_1 - r2_0
    delta_mae = mae_0 - mae_1  # positive = adding S_deep improves

    return {
        "r2_baseline": r2_0,
        "r2_with_lag": r2_1,
        "delta_r2": delta_r2,
        "mae_baseline": mae_0,
        "mae_with_lag": mae_1,
        "delta_mae": delta_mae,
    }


def run_ci_tests_phase(states, args):
    """Phase 2: CI tests + effect size."""
    console.print(Panel.fit("[bold magenta]Phase 2: CI Tests[/]"))

    tests = graph_validation_tests()
    console.print(f"[cyan]Running {len(tests)} CI tests with {args.n_permutations} permutations...[/]")

    results = run_ci_tests(states, tests, n_permutations=args.n_permutations, threshold=0.05)
    print_ci_results(results)

    # Compute effect size for CI-1
    effect = compute_effect_size(states)
    console.print(f"\n[bold cyan]Effect Size (CI-1):[/]")
    console.print(f"  R² baseline (S,Z→R):  {effect['r2_baseline']:.4f}")
    console.print(f"  R² with lag (S,Z,Sd→R): {effect['r2_with_lag']:.4f}")
    console.print(f"  [bold]ΔR² = {effect['delta_r2']:.6f}[/]")
    console.print(f"  MAE baseline: {effect['mae_baseline']:.4f}")
    console.print(f"  MAE with lag: {effect['mae_with_lag']:.4f}")
    console.print(f"  [bold]ΔMAE = {effect['delta_mae']:.6f}[/]")

    # Effect size verdict
    if abs(effect['delta_r2']) < 1e-3 and abs(effect['delta_mae']) < 1e-3:
        console.print(f"  [green]Effect size: PASS-strong (ΔR² < 1e-3, ΔMAE < 1e-3)[/]")
        effect['verdict'] = 'PASS-strong'
    elif abs(effect['delta_r2']) < 5e-3:
        console.print(f"  [yellow]Effect size: PASS-ok (ΔR² < 5e-3)[/]")
        effect['verdict'] = 'PASS-ok'
    else:
        console.print(f"  [red]Effect size: SUS (ΔR² ≥ 1e-2, practically notable)[/]")
        effect['verdict'] = 'SUS'

    return results, effect


def run_intervention_phase(model, val_loader, bin_centers, args, device):
    """Phase 3: Intervention do(S)."""
    console.print(Panel.fit("[bold magenta]Phase 3: Intervention do(S)[/]"))

    # Get some validation contexts
    contexts = []
    for (batch,) in val_loader:
        contexts.append(batch[:1].to(device))
        if len(contexts) >= 5:
            break

    # Convert delta from spacing scale to bin scale
    # δ_spacing = 0.05 → δ_bin = 0.05 / bin_width
    bin_width = bin_centers[1] - bin_centers[0]  # Assuming uniform bins
    delta_bin = args.doS_delta / bin_width

    console.print(f"[cyan]Intervention: do(S += {args.doS_delta}) spacing = {delta_bin:.2f} bins[/]")

    intervention = SpacingIntervention(delta=delta_bin, position=10)

    results = []
    for i, ctx in enumerate(contexts):
        result = intervention.apply(model, ctx, n_generate=args.doS_steps, temperature=1.0, seed=args.seed + i)
        results.append(result)

    # Aggregate healing metrics
    healing_times = [r.healing_time for r in results]
    mean_healing = np.mean(healing_times)
    std_healing = np.std(healing_times)

    console.print(f"[green]Healing time: {mean_healing:.1f} ± {std_healing:.1f} steps[/]")

    return results, {"mean_healing": mean_healing, "std_healing": std_healing}


def run_sanity_checks(states, bin_centers):
    """Phase 4: Sanity validation."""
    console.print(Panel.fit("[bold magenta]Phase 4: Sanity Checks[/]"))

    S_vals = np.array([s.S_t for s in states])
    Y_vals = np.array([s.Y_t for s in states])

    checks = []

    # Check 1: Mean spacing ≈ 1.0
    mean_S = np.mean(S_vals)
    mean_ok = abs(mean_S - 1.0) < 0.1
    checks.append(("Mean spacing ≈ 1.0", mean_ok, f"{mean_S:.4f}"))

    # Check 2: Min spacing > 0 (repulsion)
    min_S = np.min(S_vals)
    min_ok = min_S > 0
    checks.append(("Min spacing > 0", min_ok, f"{min_S:.4f}"))

    # Check 3: No NaN/Inf
    nan_ok = not (np.any(np.isnan(S_vals)) or np.any(np.isinf(S_vals)))
    checks.append(("No NaN/Inf", nan_ok, "clean" if nan_ok else "CORRUPTED"))

    table = Table(title="Sanity Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Value", style="yellow")

    all_pass = True
    for name, passed, value in checks:
        status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        table.add_row(name, status, str(value))
        if not passed:
            all_pass = False

    console.print(table)

    return all_pass, checks


def generate_report(args, stats, ci_results, intervention_metrics, sanity_checks, sanity_pass, effect_size=None):
    """Generate markdown report."""
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Causal Zeta MVP Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\n---\n")

    # Config
    lines.append("## 1. Configuration\n")
    lines.append(f"- Checkpoint: `{args.checkpoint}`")
    lines.append(f"- Data: `{args.data_dir}`")
    lines.append(f"- Seed: {args.seed}")
    lines.append(f"- N windows: {args.n_windows}")
    lines.append(f"- PCA fit samples: {args.pca_fit_samples}")
    lines.append(f"- Rigidity window: {args.rigidity_window}")
    lines.append(f"- CI permutations: {args.n_permutations}")
    lines.append(f"- do(S) delta: {args.doS_delta}")
    lines.append(f"- do(S) steps: {args.doS_steps}")

    # Variable definitions
    lines.append("\n## 2. Variable Definitions\n")
    lines.append("- **Z_t** = PCA(n=2) of hidden_states[-1][:, -1, :] (last layer, last position)")
    lines.append(f"- **R_t** = Var(s[t-{args.rigidity_window}:t]) / 0.178 (on actual spacing values)")
    lines.append("- **S_t, Y_t** = spacing values via bin_centers lookup")

    # Sanity
    lines.append("\n## 3. Sanity Checks\n")
    lines.append(f"**Overall: {'PASS' if sanity_pass else 'FAIL'}**\n")
    lines.append("| Check | Status | Value |")
    lines.append("|-------|--------|-------|")
    for name, passed, value in sanity_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"| {name} | {status} | {value} |")

    lines.append(f"\n- S mean: {stats['S_mean']:.4f} (target: 1.0)")
    lines.append(f"- S std: {stats['S_std']:.4f} (target: ~0.42)")
    lines.append(f"- R mean: {stats['R_mean']:.4f} (target: ~1.0)")

    # CI Results
    lines.append("\n## 4. CI Test Results\n")
    lines.append("| Test | X | Y | Z | HSIC | p-value | Independent? |")
    lines.append("|------|---|---|---|------|---------|--------------|")
    for r in ci_results:
        z_str = ", ".join(r.conditioning_set) if r.conditioning_set else "∅"
        indep = "✓ Yes" if r.independent else "✗ No"
        lines.append(f"| {r.var_x} ⊥ {r.var_y} | {r.var_x} | {r.var_y} | {z_str} | {r.hsic_value:.6f} | {r.p_value:.4f} | {indep} |")

    lines.append("\n**Interpretation:**\n")
    for r in ci_results:
        # DAG v0.2: CI-1 tests R_t ⊥ S_deep | (S_{t-1}, Z_t)
        # S_deep = s[t - L_R - 1], first element OUTSIDE the R_t window
        if r.var_x == "R_t" and r.var_y in ("S_deep", "S_{t-2}"):
            if r.independent:
                lines.append(f"- **CI-1 PASS**: R_t ⊥ S_deep | (S_{{t-1}}, Z_t). Window L_R={args.rigidity_window} is sufficient.")
            else:
                lines.append(f"- **CI-1 FAIL**: R_t depends on S_deep beyond window. Consider expanding L_R.")
        # Legacy v0.1 test (for backwards compatibility)
        elif r.var_x == "R_t" and r.var_y == "S_{t-1}":
            if r.independent:
                lines.append("- **CI-A PASS**: Rigidity is global (via Z_t), not determined by local spacing.")
            else:
                lines.append("- **CI-A FAIL**: Rigidity depends on local spacing. Edge S_{t-1} → R_t needed (DAG v0.2).")
        # CI-2: Z_t informativeness
        elif r.var_x == "Y_t" and r.var_y == "Z_t":
            if r.independent:
                lines.append("- **CI-2 FAIL**: Z_t is NOT informative about Y_t. PCA latent mode is garbage!")
            else:
                lines.append("- **CI-2 PASS**: Z_t is informative about Y_t. Latent mode captures meaningful structure.")

    # Effect Size (if computed)
    if effect_size:
        lines.append("\n## 4.1 Effect Size (CI-1)\n")
        lines.append("Ridge regression: R_t ~ f(features)")
        lines.append("")
        lines.append("| Model | Features | R² | MAE |")
        lines.append("|-------|----------|-----|-----|")
        lines.append(f"| Baseline | S_{{t-1}}, Z_t | {effect_size['r2_baseline']:.4f} | {effect_size['mae_baseline']:.4f} |")
        lines.append(f"| With S_deep | S_{{t-1}}, Z_t, S_deep | {effect_size['r2_with_lag']:.4f} | {effect_size['mae_with_lag']:.4f} |")
        lines.append("")
        lines.append(f"- **ΔR² = {effect_size['delta_r2']:.6f}**")
        lines.append(f"- **ΔMAE = {effect_size['delta_mae']:.6f}**")
        lines.append(f"- **Verdict: {effect_size['verdict']}**")
        if effect_size['verdict'] == 'PASS-strong':
            lines.append("\nS_deep adds no predictive power → memory is truly short-range.")
        elif effect_size['verdict'] == 'PASS-ok':
            lines.append("\nS_deep adds minimal predictive power → memory is mostly short-range.")
        else:
            lines.append("\n⚠️ S_deep adds notable predictive power → p-value may be misleading!")

    # Intervention
    lines.append("\n## 5. Intervention do(S)\n")
    lines.append(f"- Delta: {args.doS_delta} (spacing scale)")
    lines.append(f"- Steps generated: {args.doS_steps}")
    lines.append(f"- **Mean healing time: {intervention_metrics['mean_healing']:.1f} ± {intervention_metrics['std_healing']:.1f} steps**")

    if intervention_metrics['mean_healing'] < 5:
        lines.append("\n**Interpretation**: Fast healing suggests strong spectral rigidity.")
    else:
        lines.append("\n**Interpretation**: Slow healing may indicate weak rigidity or model issues.")

    # Verdict
    lines.append("\n## 6. Verdict\n")
    lines.append(f"- Sanity: {'✓ PASS' if sanity_pass else '✗ FAIL'}")

    # DAG v0.2: CI-1 tests R⊥S_deep|S_{t-1},Z_t
    ci_1_pass = any(r.var_x == "R_t" and r.var_y in ("S_deep", "S_{t-2}") and r.independent for r in ci_results)
    # Legacy CI-A for backwards compatibility
    ci_a_pass = any(r.var_x == "R_t" and r.var_y == "S_{t-1}" and r.independent for r in ci_results)
    # CI-2: Z_t informativeness
    ci_2_pass = any(r.var_x == "Y_t" and r.var_y == "Z_t" and not r.independent for r in ci_results)

    # Check which test was run
    has_ci1 = any(r.var_x == "R_t" and r.var_y in ("S_deep", "S_{t-2}") for r in ci_results)
    has_cia = any(r.var_x == "R_t" and r.var_y == "S_{t-1}" for r in ci_results)

    if has_ci1:
        lines.append(f"- CI-1 (R⊥S₂|S₁,Z): {'✓ PASS' if ci_1_pass else '✗ FAIL'}")
    if has_cia:
        lines.append(f"- CI-A (R⊥S|Z): {'✓ PASS' if ci_a_pass else '✗ FAIL'}")
    lines.append(f"- CI-2 (Y~Z): {'✓ PASS' if ci_2_pass else '✗ FAIL'}")

    # Graph update decision based on CI tests
    graph_updates = []
    if has_cia and not ci_a_pass:
        graph_updates.append("ADD EDGE: S_{t-1} → R_t (done in DAG v0.2)")
    if has_ci1 and not ci_1_pass:
        graph_updates.append("EXPAND: R_t has longer memory, consider wider window")
    if not ci_2_pass:
        graph_updates.append("REVISE: Z_t definition (try different layer/position)")

    if not graph_updates and sanity_pass:
        lines.append("\n**Graph update decision: NO CHANGE** (all tests passed)")
    else:
        lines.append("\n**Graph update decision: UPDATE REQUIRED**")
        for update in graph_updates:
            lines.append(f"- ✏️ {update}")
        if not sanity_pass:
            lines.append("- ⚠️ Fix sanity issues first")

    lines.append("\n---\n")
    lines.append("*End of report*")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    console.print(f"[green]Report saved to {report_path}[/]")


def main():
    parser = argparse.ArgumentParser(description="Causal Zeta MVP")
    parser.add_argument("--checkpoint", type=str, default="out/best.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--report", type=str, default="reports/round_001.md")

    # Extraction
    parser.add_argument("--n-windows", type=int, default=2000)
    parser.add_argument("--pca-fit-samples", type=int, default=10000)
    parser.add_argument("--rigidity-window", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)

    # CI tests
    parser.add_argument("--n-permutations", "--ci-permutations", type=int, default=100,
                        dest="n_permutations", help="Number of permutations for CI tests")

    # Intervention
    parser.add_argument("--doS-delta", type=float, default=0.05, help="Delta in spacing scale")
    parser.add_argument("--doS-steps", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold blue]🔬 CAUSAL ZETA MVP 🔬[/]\n"
        "Core causal analysis (no LLM agents)",
        title="Causal Discovery"
    ))

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load
    model, val_loader, meta, bin_centers, device = load_model_and_data(args)

    # Phase 1: Extraction
    states, latent_extractor, rigidity_calc, stats = run_extraction(
        model, val_loader, bin_centers, args, device
    )

    # Phase 2: CI Tests + Effect Size
    ci_results, effect_size = run_ci_tests_phase(states, args)

    # Phase 3: Intervention
    intervention_results, intervention_metrics = run_intervention_phase(
        model, val_loader, bin_centers, args, device
    )

    # Phase 4: Sanity
    sanity_pass, sanity_checks = run_sanity_checks(states, bin_centers)

    # Generate report
    generate_report(args, stats, ci_results, intervention_metrics, sanity_checks, sanity_pass, effect_size)

    # Final summary
    console.print(Panel.fit(
        f"[bold green]✓ MVP Complete![/]\n\n"
        f"Sanity: {'PASS' if sanity_pass else 'FAIL'}\n"
        f"Healing time: {intervention_metrics['mean_healing']:.1f} steps\n"
        f"Report: {args.report}",
        title="Results"
    ))


if __name__ == "__main__":
    main()
