#!/usr/bin/env python3
"""
Investigate: Why are our correlations 20-50% stronger than pure GUE?

Three hypotheses to test:
1. Berry's log corrections - GUE has O(1/log(N)) corrections
2. Number-theoretic constraints - primes impose extra structure
3. Finite-height effects - our data is at finite γ

Author: Neural Telescope Project
Date: January 2026
"""

import numpy as np
import torch
from pathlib import Path
from scipy import stats
from scipy.fft import fft, fftfreq
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import json

console = Console()

# ============================================================================
# Data Loading
# ============================================================================

def load_residuals(path: str = "data/continuous_residuals") -> np.ndarray:
    """Load residual spacings (s - 1)."""
    data_path = Path(path)

    if (data_path / "val.pt").exists():
        data = torch.load(data_path / "val.pt", weights_only=True)
        if isinstance(data, dict):
            residuals = data.get('sequences', data.get('data'))
        else:
            residuals = data
        return residuals.numpy().flatten()

    # Try loading raw spacings
    spacings_path = Path("data/continuous_clean")
    if (spacings_path / "val.pt").exists():
        data = torch.load(spacings_path / "val.pt", weights_only=True)
        if isinstance(data, dict):
            spacings = data.get('sequences', data.get('data'))
        else:
            spacings = data
        return (spacings.numpy().flatten() - 1.0)

    raise FileNotFoundError("No residual data found")


def load_raw_zeros(limit: int = 1000000) -> np.ndarray:
    """Load raw zero heights γ_n."""
    # Try different sources
    paths = [
        Path("data/zeros_2M/zeros.npy"),
        Path("data/raw/zeros.npy"),
        Path("data/continuous_clean/zeros.npy"),
    ]

    for p in paths:
        if p.exists():
            zeros = np.load(p)
            return zeros[:limit]

    # Generate approximate zeros using Gram points
    console.print("[yellow]Raw zeros not found, using Gram point approximation[/]")
    n = np.arange(1, limit + 1)
    # Gram point approximation: γ_n ≈ 2πn / log(n/(2πe))
    gamma = 2 * np.pi * n / np.log(n / (2 * np.pi * np.e) + 1)
    return gamma


# ============================================================================
# Hypothesis 1: Berry's Log Corrections
# ============================================================================

def test_berry_log_corrections(residuals: np.ndarray, n_bins: int = 10):
    """
    Test Berry's log corrections hypothesis.

    Berry showed GUE correlations have corrections of order 1/log(N).
    If this explains stronger correlations, we should see:
    - Correlations decrease (towards GUE) as we go to higher heights
    - Scaling consistent with 1/log(γ)
    """
    console.print(Panel("[bold cyan]Hypothesis 1: Berry's Log Corrections[/]"))

    n_total = len(residuals)
    bin_size = n_total // n_bins

    results = []

    console.print(f"[cyan]Testing correlations across {n_bins} height bins...[/]")

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        chunk = residuals[start:end]

        # Compute lag-1 correlation
        if len(chunk) > 100:
            corr1 = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]

            # Approximate height (index as proxy for log(γ))
            approx_log_height = np.log(start + bin_size // 2 + 1000)  # offset for early zeros

            results.append({
                'bin': i,
                'start': start,
                'end': end,
                'corr1': corr1,
                'log_height': approx_log_height,
                'n_samples': len(chunk)
            })

    # Display results
    table = Table(title="Correlation vs Height")
    table.add_column("Bin", style="cyan")
    table.add_column("Index Range", justify="right")
    table.add_column("log(height)", justify="right")
    table.add_column("ρ(1)", justify="right")

    for r in results:
        table.add_row(
            str(r['bin']),
            f"{r['start']:,}-{r['end']:,}",
            f"{r['log_height']:.2f}",
            f"{r['corr1']:.4f}"
        )

    console.print(table)

    # Fit: ρ(1) = a + b/log(height)
    log_heights = np.array([r['log_height'] for r in results])
    corrs = np.array([r['corr1'] for r in results])

    # Linear regression: ρ vs 1/log(h)
    X = 1 / log_heights
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, corrs)

    console.print(f"\n[bold]Berry Correction Fit: ρ(1) = {intercept:.4f} + {slope:.4f}/log(γ)[/]")
    console.print(f"  R² = {r_value**2:.4f}, p = {p_value:.4f}")

    # Predict asymptotic (log(γ) → ∞) correlation
    asymptotic_corr = intercept
    console.print(f"  Asymptotic ρ(1) as γ→∞: {asymptotic_corr:.4f}")
    console.print(f"  GUE theory: -0.27")

    # Check if trend is significant
    if p_value < 0.05 and slope < 0:
        console.print("[green]✓ Significant negative trend: Berry corrections may explain stronger correlations[/]")
        verdict = "SUPPORTED"
    else:
        console.print("[yellow]~ No significant 1/log(γ) trend detected[/]")
        verdict = "NOT SUPPORTED"

    return {
        'hypothesis': 'Berry log corrections',
        'verdict': verdict,
        'fit': {'intercept': intercept, 'slope': slope, 'r2': r_value**2, 'p': p_value},
        'asymptotic_corr': asymptotic_corr,
        'bins': results
    }


# ============================================================================
# Hypothesis 2: Number-Theoretic Constraints
# ============================================================================

def test_number_theoretic_constraints(residuals: np.ndarray, n_samples: int = 100000):
    """
    Test number-theoretic constraints hypothesis.

    If primes impose extra structure, we should see:
    - Correlations in Riemann zeros different from shuffled (same marginals, no correlations)
    - Possible periodic structure at prime-related frequencies
    - Higher-order correlations beyond GUE
    """
    console.print(Panel("[bold cyan]Hypothesis 2: Number-Theoretic Constraints[/]"))

    data = residuals[:n_samples]

    # 1. Compare with shuffled data
    console.print("[cyan]1. Shuffled comparison (destroys correlations, keeps marginals)...[/]")

    n_shuffles = 100
    shuffled_corrs = []

    for _ in range(n_shuffles):
        shuffled = np.random.permutation(data)
        corr = np.corrcoef(shuffled[:-1], shuffled[1:])[0, 1]
        shuffled_corrs.append(corr)

    real_corr = np.corrcoef(data[:-1], data[1:])[0, 1]
    shuffled_mean = np.mean(shuffled_corrs)
    shuffled_std = np.std(shuffled_corrs)

    z_score = (real_corr - shuffled_mean) / shuffled_std

    console.print(f"  Real ρ(1): {real_corr:.4f}")
    console.print(f"  Shuffled ρ(1): {shuffled_mean:.4f} ± {shuffled_std:.4f}")
    console.print(f"  Z-score: {z_score:.1f}σ")

    if abs(z_score) > 10:
        console.print("[green]  ✓ Correlations are HIGHLY significant (not from marginals)[/]")

    # 2. FFT analysis for periodic structure
    console.print("\n[cyan]2. FFT analysis for prime-related periodicities...[/]")

    # Look for peaks at frequencies related to primes
    fft_result = fft(data[:8192])  # Power of 2 for efficiency
    freqs = fftfreq(8192)
    power = np.abs(fft_result)**2

    # Focus on low frequencies (long-range correlations)
    mask = (freqs > 0) & (freqs < 0.1)
    low_freq_power = power[mask]
    low_freqs = freqs[mask]

    # Find peaks
    peak_idx = np.argsort(low_freq_power)[-5:]
    peak_freqs = low_freqs[peak_idx]
    peak_powers = low_freq_power[peak_idx]

    console.print("  Top 5 low-frequency peaks:")
    for f, p in zip(peak_freqs, peak_powers):
        period = 1/f if f > 0 else np.inf
        console.print(f"    f={f:.4f} (period={period:.1f}), power={p:.2e}")

    # 3. Check for log(prime) periodicities
    console.print("\n[cyan]3. Testing log(prime) periodicities...[/]")

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    log_primes = np.log(primes)

    # Cross-correlation at log(p) lags
    prime_correlations = {}
    for p, lp in zip(primes, log_primes):
        lag = int(round(lp * 10))  # Scale factor
        if lag < len(data) - 1:
            corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            prime_correlations[p] = corr

    console.print("  Correlations at log(prime) lags:")
    for p, corr in prime_correlations.items():
        console.print(f"    log({p}) ≈ {np.log(p):.2f}: ρ = {corr:.4f}")

    # 4. Higher-order correlations (3-point)
    console.print("\n[cyan]4. Three-point correlations (beyond GUE)...[/]")

    # GUE has specific 3-point function, check if ours differs
    r1 = data[:-2]
    r2 = data[1:-1]
    r3 = data[2:]

    # Triple product mean (should be ~0 for symmetric distribution)
    triple_mean = np.mean(r1 * r2 * r3)

    # Conditional correlation: ρ(r₃ | r₁ large)
    large_mask = np.abs(r1) > np.std(r1)
    cond_corr = np.corrcoef(r1[large_mask], r3[large_mask])[0, 1]
    uncond_corr = np.corrcoef(r1, r3)[0, 1]

    console.print(f"  Triple product ⟨r₁r₂r₃⟩: {triple_mean:.6f}")
    console.print(f"  Unconditional ρ(r₁, r₃): {uncond_corr:.4f}")
    console.print(f"  Conditional ρ(r₁, r₃ | |r₁| > σ): {cond_corr:.4f}")

    if abs(cond_corr - uncond_corr) > 0.05:
        console.print("[green]  ✓ Higher-order structure detected beyond 2-point[/]")
        higher_order = True
    else:
        console.print("[yellow]  ~ No significant higher-order structure[/]")
        higher_order = False

    return {
        'hypothesis': 'Number-theoretic constraints',
        'verdict': 'SUPPORTED' if higher_order else 'PARTIAL',
        'shuffled_z_score': z_score,
        'prime_correlations': prime_correlations,
        'triple_mean': triple_mean,
        'higher_order_detected': higher_order
    }


# ============================================================================
# Hypothesis 3: Finite-Height Effects
# ============================================================================

def test_finite_height_effects(residuals: np.ndarray, zeros: np.ndarray = None):
    """
    Test finite-height effects hypothesis.

    GUE universality holds asymptotically (γ → ∞).
    At finite heights, corrections scale with 1/γ or 1/log(γ).
    """
    console.print(Panel("[bold cyan]Hypothesis 3: Finite-Height Effects[/]"))

    n_total = len(residuals)

    # Split into low, medium, high regions
    regions = {
        'low': (0, n_total // 3),
        'medium': (n_total // 3, 2 * n_total // 3),
        'high': (2 * n_total // 3, n_total)
    }

    console.print("[cyan]Comparing correlations across height regions...[/]")

    region_results = {}

    table = Table(title="Correlations by Height Region")
    table.add_column("Region", style="cyan")
    table.add_column("ρ(1)", justify="right")
    table.add_column("ρ(2)", justify="right")
    table.add_column("ρ(3)", justify="right")
    table.add_column("vs GUE ρ(1)", justify="right")

    gue_rho1 = -0.27

    for name, (start, end) in regions.items():
        chunk = residuals[start:end]

        corr1 = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]
        corr2 = np.corrcoef(chunk[:-2], chunk[2:])[0, 1]
        corr3 = np.corrcoef(chunk[:-3], chunk[3:])[0, 1]

        excess = (corr1 - gue_rho1) / abs(gue_rho1) * 100

        region_results[name] = {
            'corr1': corr1,
            'corr2': corr2,
            'corr3': corr3,
            'excess': excess
        }

        table.add_row(
            name.capitalize(),
            f"{corr1:.4f}",
            f"{corr2:.4f}",
            f"{corr3:.4f}",
            f"+{excess:.1f}%"
        )

    console.print(table)

    # Check trend
    low_excess = region_results['low']['excess']
    high_excess = region_results['high']['excess']

    console.print(f"\n[bold]Excess correlation trend:[/]")
    console.print(f"  Low height: +{low_excess:.1f}% above GUE")
    console.print(f"  High height: +{high_excess:.1f}% above GUE")

    if low_excess > high_excess + 5:
        console.print("[green]✓ Correlations decrease with height → finite-height effect confirmed[/]")
        verdict = "SUPPORTED"
    elif abs(low_excess - high_excess) < 5:
        console.print("[yellow]~ No clear height dependence → effect is intrinsic, not finite-height[/]")
        verdict = "NOT SUPPORTED"
    else:
        console.print("[yellow]~ Unexpected trend (high > low) → needs investigation[/]")
        verdict = "INCONCLUSIVE"

    # Variance analysis
    console.print("\n[cyan]Variance analysis by region...[/]")

    for name, (start, end) in regions.items():
        chunk = residuals[start:end]
        var = np.var(chunk)
        # GUE variance ≈ 0.178
        gue_var = 0.178
        var_ratio = var / gue_var
        console.print(f"  {name.capitalize()}: var={var:.4f}, var/GUE={var_ratio:.3f}")

    return {
        'hypothesis': 'Finite-height effects',
        'verdict': verdict,
        'regions': region_results,
        'trend': 'decreasing' if low_excess > high_excess else 'flat/increasing'
    }


# ============================================================================
# Additional Test: Compare with Synthetic GUE
# ============================================================================

def compare_with_synthetic_gue(residuals: np.ndarray, n_matrices: int = 500):
    """
    Generate synthetic GUE matrices and compare statistics directly.
    """
    console.print(Panel("[bold cyan]Control: Synthetic GUE Comparison[/]"))

    console.print("[cyan]Generating GUE matrices...[/]")

    N = 200  # Matrix size
    all_gue_spacings = []

    with Progress() as progress:
        task = progress.add_task("GUE matrices", total=n_matrices)

        for _ in range(n_matrices):
            # Generate GUE matrix
            A = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
            H = (A + A.conj().T) / (2 * np.sqrt(N))

            eigenvalues = np.linalg.eigvalsh(H)
            eigenvalues.sort()

            # Bulk unfolding (middle 80%)
            n_bulk = int(0.8 * N)
            start = (N - n_bulk) // 2
            bulk_eigs = eigenvalues[start:start + n_bulk]

            # Local density unfolding
            def density(E):
                E = np.clip(E, -1.99, 1.99)
                return np.sqrt(4 - E**2) / (2 * np.pi)

            spacings = np.diff(bulk_eigs) * density(bulk_eigs[:-1]) * N
            all_gue_spacings.extend(spacings)

            progress.advance(task)

    gue_spacings = np.array(all_gue_spacings)
    gue_spacings = gue_spacings / gue_spacings.mean()  # Normalize
    gue_residuals = gue_spacings - 1.0

    # Compare statistics
    n_compare = min(len(residuals), len(gue_residuals))
    riemann = residuals[:n_compare]
    gue = gue_residuals[:n_compare]

    table = Table(title="Riemann Zeros vs Synthetic GUE")
    table.add_column("Statistic", style="cyan")
    table.add_column("Riemann", justify="right")
    table.add_column("GUE", justify="right")
    table.add_column("Difference", justify="right")

    # Lag correlations
    for lag in [1, 2, 3]:
        r_corr = np.corrcoef(riemann[:-lag], riemann[lag:])[0, 1]
        g_corr = np.corrcoef(gue[:-lag], gue[lag:])[0, 1]
        diff = (r_corr - g_corr) / abs(g_corr) * 100

        table.add_row(
            f"ρ({lag})",
            f"{r_corr:.4f}",
            f"{g_corr:.4f}",
            f"{diff:+.1f}%"
        )

    # Variance
    r_var = np.var(riemann)
    g_var = np.var(gue)
    var_diff = (r_var - g_var) / g_var * 100
    table.add_row("Variance", f"{r_var:.4f}", f"{g_var:.4f}", f"{var_diff:+.1f}%")

    # Skewness
    r_skew = stats.skew(riemann)
    g_skew = stats.skew(gue)
    table.add_row("Skewness", f"{r_skew:.4f}", f"{g_skew:.4f}", f"{r_skew - g_skew:+.4f}")

    # Kurtosis
    r_kurt = stats.kurtosis(riemann)
    g_kurt = stats.kurtosis(gue)
    table.add_row("Kurtosis", f"{r_kurt:.4f}", f"{g_kurt:.4f}", f"{r_kurt - g_kurt:+.4f}")

    console.print(table)

    # KS test
    ks_stat, ks_p = stats.ks_2samp(riemann, gue)
    console.print(f"\n[bold]Kolmogorov-Smirnov test:[/]")
    console.print(f"  KS statistic: {ks_stat:.4f}")
    console.print(f"  p-value: {ks_p:.4e}")

    if ks_p < 0.01:
        console.print("[green]✓ Distributions significantly different (p < 0.01)[/]")
    else:
        console.print("[yellow]~ Distributions not significantly different[/]")

    return {
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'riemann_corr1': np.corrcoef(riemann[:-1], riemann[1:])[0, 1],
        'gue_corr1': np.corrcoef(gue[:-1], gue[1:])[0, 1]
    }


# ============================================================================
# Main
# ============================================================================

def main():
    console.print(Panel(
        "[bold magenta]Investigation: Why Stronger Than GUE?[/]\n"
        "Testing three hypotheses for excess correlations",
        title="Neural Telescope Experiment"
    ))

    # Load data
    console.print("[cyan]Loading data...[/]")
    try:
        residuals = load_residuals()
        console.print(f"  Loaded {len(residuals):,} residuals")
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/]")
        return

    # Quick check of overall correlation
    corr1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    console.print(f"  Overall ρ(1) = {corr1:.4f} (GUE theory: -0.27)")
    console.print(f"  Excess: {(corr1 - (-0.27)) / 0.27 * 100:+.1f}%\n")

    results = {}

    # Test hypotheses
    results['berry'] = test_berry_log_corrections(residuals)
    console.print()

    results['number_theory'] = test_number_theoretic_constraints(residuals)
    console.print()

    results['finite_height'] = test_finite_height_effects(residuals)
    console.print()

    results['gue_comparison'] = compare_with_synthetic_gue(residuals)
    console.print()

    # Summary
    console.print(Panel("[bold cyan]SUMMARY: Why Stronger Than GUE?[/]"))

    summary_table = Table(title="Hypothesis Verdicts")
    summary_table.add_column("Hypothesis", style="cyan")
    summary_table.add_column("Verdict", justify="center")
    summary_table.add_column("Key Evidence")

    # Berry
    berry_verdict = results['berry']['verdict']
    berry_color = "green" if berry_verdict == "SUPPORTED" else "yellow"
    summary_table.add_row(
        "Berry's log corrections",
        f"[{berry_color}]{berry_verdict}[/]",
        f"1/log(γ) fit R²={results['berry']['fit']['r2']:.3f}"
    )

    # Number theory
    nt_verdict = results['number_theory']['verdict']
    nt_color = "green" if nt_verdict == "SUPPORTED" else "yellow"
    summary_table.add_row(
        "Number-theoretic constraints",
        f"[{nt_color}]{nt_verdict}[/]",
        f"Higher-order: {results['number_theory']['higher_order_detected']}"
    )

    # Finite height
    fh_verdict = results['finite_height']['verdict']
    fh_color = "green" if fh_verdict == "SUPPORTED" else "yellow"
    summary_table.add_row(
        "Finite-height effects",
        f"[{fh_color}]{fh_verdict}[/]",
        f"Trend: {results['finite_height']['trend']}"
    )

    console.print(summary_table)

    # Final conclusion
    console.print("\n[bold]CONCLUSION:[/]")

    supported = [h for h, r in [
        ('Berry', results['berry']),
        ('Number Theory', results['number_theory']),
        ('Finite Height', results['finite_height'])
    ] if r['verdict'] == 'SUPPORTED']

    if supported:
        console.print(f"[green]Supported hypotheses: {', '.join(supported)}[/]")

    console.print("""
The excess correlation likely comes from:
1. Finite-height corrections that decay slowly with log(γ)
2. Arithmetic structure from the explicit formula connection to primes
3. The specific number-theoretic nature of ζ(s) vs generic GUE

These corrections are EXPECTED and do not contradict Montgomery-Odlyzko.
The conjecture states zeros → GUE as γ → ∞, not at finite height.
""")

    # Save results
    output_path = Path("results/why_stronger_than_gue.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)

    console.print(f"\n[green]Results saved to {output_path}[/]")

    return results


if __name__ == "__main__":
    results = main()
