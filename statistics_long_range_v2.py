#!/usr/bin/env python3
"""
LONG-RANGE SPECTRAL STATISTICS v2 — FIXED

Исправления:
1. Правильный Δ3 по Dyson-Mehta: fit линии к N(E), не к positions
2. Правильный GUE: симуляция + semicircle unfolding
3. Правильные данные: spacings → cumsum → N(x)
"""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from pathlib import Path

console = Console()


# ============================================================================
# CORRECT Δ3 COMPUTATION (Dyson-Mehta)
# ============================================================================

def compute_delta3_correct(spacings, L_values, n_samples=50):
    """
    Correct Dyson-Mehta Δ3(L) computation.

    Δ3(L) = min_{a,b} (1/L) ∫_x^{x+L} [N(E) - aE - b]² dE

    where N(E) is the staircase counting function.

    For discrete levels:
    Δ3(L) = min_{a,b} (1/n) Σ [i - a*E_i - b]²

    where we sum over levels E_i in the window [x, x+L].
    """
    # Convert spacings to unfolded positions (cumulative sum, mean=1)
    positions = np.cumsum(spacings)
    N = len(positions)

    delta3 = []

    for L in L_values:
        if L >= positions[-1] - positions[0]:
            delta3.append(np.nan)
            continue

        d3_samples = []

        # Sample windows
        max_start = positions[-1] - L
        start_positions = np.linspace(positions[0], max_start, n_samples)

        for x0 in start_positions:
            # Find levels in [x0, x0+L]
            mask = (positions >= x0) & (positions < x0 + L)
            levels_in_window = positions[mask]
            n_levels = len(levels_in_window)

            if n_levels < 3:
                continue

            # Level indices within window (1, 2, 3, ...)
            n_i = np.arange(1, n_levels + 1)

            # Levels relative to window start
            E_i = levels_in_window - x0

            # Least squares fit: n = a*E + b
            # Normal equations
            sum_E = np.sum(E_i)
            sum_E2 = np.sum(E_i ** 2)
            sum_n = np.sum(n_i)
            sum_nE = np.sum(n_i * E_i)

            denom = n_levels * sum_E2 - sum_E ** 2
            if abs(denom) < 1e-10:
                continue

            a = (n_levels * sum_nE - sum_E * sum_n) / denom
            b = (sum_n * sum_E2 - sum_E * sum_nE) / denom

            # Compute Δ3 = (1/L) * Σ(n_i - a*E_i - b)²
            residuals = n_i - a * E_i - b
            d3 = np.sum(residuals ** 2) / L

            d3_samples.append(d3)

        if d3_samples:
            delta3.append(np.mean(d3_samples))
        else:
            delta3.append(np.nan)

    return np.array(delta3)


def compute_number_variance_correct(spacings, L_values, n_samples=100):
    """
    Correct number variance Σ²(L).

    Σ²(L) = <n²> - <n>² where n is number of levels in interval L.
    """
    positions = np.cumsum(spacings)
    N = len(positions)

    sigma2 = []

    for L in L_values:
        if L >= positions[-1] - positions[0]:
            sigma2.append(np.nan)
            continue

        max_start = positions[-1] - L
        start_positions = np.linspace(positions[0], max_start, n_samples)

        counts = []
        for x0 in start_positions:
            mask = (positions >= x0) & (positions < x0 + L)
            counts.append(mask.sum())

        if counts:
            counts = np.array(counts)
            sigma2.append(np.var(counts))
        else:
            sigma2.append(np.nan)

    return np.array(sigma2)


# ============================================================================
# CORRECT GUE SIMULATION
# ============================================================================

def generate_gue_spacings_correct(N, seed=42):
    """
    Generate GUE eigenvalue spacings with proper unfolding.

    1. Generate N×N random Hermitian matrix from GUE
    2. Diagonalize to get eigenvalues
    3. Unfold using semicircle law
    4. Compute spacings
    """
    np.random.seed(seed)

    # GUE: H = (A + A†)/√(2N) where A has i.i.d. complex Gaussians
    A = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
    H = (A + A.conj().T) / np.sqrt(2 * N)

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues = np.sort(eigenvalues)

    # Semicircle law unfolding
    # For GUE with our normalization, eigenvalues lie in [-2, 2]
    # Density: ρ(x) = (1/2π) √(4 - x²)
    # Cumulative: N(x) = (N/2) + (N/π)[x√(4-x²)/4 + arcsin(x/2)]

    def semicircle_cdf(x):
        """CDF of semicircle law on [-2, 2]."""
        x = np.clip(x, -2 + 1e-10, 2 - 1e-10)
        return 0.5 + (1/np.pi) * (x * np.sqrt(4 - x**2) / 4 + np.arcsin(x / 2))

    # Unfold: each eigenvalue maps to its expected rank
    unfolded = N * semicircle_cdf(eigenvalues)

    # Spacings
    spacings = np.diff(unfolded)

    # Normalize to mean 1
    spacings = spacings / np.mean(spacings)

    return spacings


# ============================================================================
# GUE THEORETICAL VALUES
# ============================================================================

def gue_delta3_theory(L):
    """Theoretical Δ3(L) for GUE (Dyson-Mehta)."""
    # For large L: Δ3(L) ≈ (1/π²) ln(2πL) + γ - 5/4 - π²/8 ≈ (1/π²) ln(L) + 0.057
    # More accurate for all L from Mehta's book
    return (1 / np.pi**2) * (np.log(2 * np.pi * L) + np.euler_gamma - 5/4)


def gue_sigma2_theory(L):
    """Theoretical Σ²(L) for GUE."""
    # For large L: Σ²(L) ≈ (2/π²) ln(2πL) + const
    return (2 / np.pi**2) * np.log(2 * np.pi * L) + 0.442


def poisson_delta3_theory(L):
    """Δ3(L) for Poisson (uncorrelated levels)."""
    return L / 15


def poisson_sigma2_theory(L):
    """Σ²(L) for Poisson."""
    return L


# ============================================================================
# SNOWBALL GENERATION (same as before)
# ============================================================================

def generate_snowball_trajectory(model, config, bin_centers, device, n_steps=1000, seed=42):
    """Generate long trajectory using snowball model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)

    val = torch.load("data/val.pt", weights_only=False)
    start_seq = val[0:1, :config.seq_len].to(device)

    memory_state = model.snowball.get_initial_state(1, device)

    generated_tokens = start_seq[0].tolist()
    generated_spacings = []

    window_size = min(128, config.seq_len - 1)

    for step in track(range(n_steps), description="Generating..."):
        if len(generated_tokens) >= window_size:
            window = torch.tensor([generated_tokens[-window_size:]], dtype=torch.long, device=device)
        else:
            window = torch.tensor([generated_tokens], dtype=torch.long, device=device)

        scale_val = bin_centers_t[window]

        with torch.no_grad():
            pred, loss, mem_attn, new_memory = model(
                window, targets=None, return_hidden=False, scale_val=scale_val,
                memory_state=memory_state, return_memory=True
            )

            mu = model.get_mu(pred)[:, -1].item()

            # Sample from Wigner distribution
            u = np.random.random()
            spacing = mu * np.sqrt(-4 * np.log(1 - u + 1e-10) / np.pi)
            spacing = np.clip(spacing, bin_centers[1], bin_centers[-2])

            token = np.abs(bin_centers - spacing).argmin()

            generated_tokens.append(int(token))
            generated_spacings.append(spacing)

            memory_state = new_memory

    return np.array(generated_spacings)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis_v2(n_steps=2000):
    """Run fixed long-range statistics analysis."""

    console.print(Panel.fit(
        "[bold cyan]🎻 LONG-RANGE SPECTRAL STATISTICS v2[/]\n"
        "FIXED: Δ3 computation + GUE unfolding",
        title="CALIBRATED SYMPHONY CHECK"
    ))

    # Load snowball model
    console.print("\n[cyan]Loading Snowball v8...[/]")
    from train_snowball import SnowballGPT, SnowballConfig

    ckpt = torch.load("out/snowball_v8_best.pt", map_location="cpu", weights_only=False)
    config = SnowballConfig(**ckpt["config"])
    model = SnowballGPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    bin_centers = ckpt.get("bin_centers", np.load("data/bin_centers.npy"))
    device = torch.device("cpu")

    # Generate trajectory
    console.print(f"\n[cyan]Generating {n_steps} step trajectory...[/]")
    model_spacings = generate_snowball_trajectory(
        model, config, bin_centers, device, n_steps=n_steps
    )
    model_spacings = model_spacings / model_spacings.mean()  # Normalize
    console.print(f"  Mean: {model_spacings.mean():.3f}, Std: {model_spacings.std():.3f}")

    # Real zeta spacings
    console.print("\n[cyan]Loading real zeta spacings...[/]")
    val = torch.load("data/val.pt", weights_only=False)
    real_tokens = val[:n_steps // 256 + 2].flatten().numpy()[:n_steps]
    real_spacings = bin_centers[real_tokens]
    real_spacings = real_spacings / real_spacings.mean()
    console.print(f"  Mean: {real_spacings.mean():.3f}, Std: {real_spacings.std():.3f}")

    # GUE reference (FIXED)
    console.print("\n[cyan]Generating GUE ensemble (FIXED)...[/]")
    gue_spacings = generate_gue_spacings_correct(n_steps + 200, seed=42)[:n_steps]
    console.print(f"  Mean: {gue_spacings.mean():.3f}, Std: {gue_spacings.std():.3f}")

    # Shuffled
    console.print("\n[cyan]Creating shuffled baseline...[/]")
    shuffled_spacings = real_spacings.copy()
    np.random.shuffle(shuffled_spacings)

    # Compute statistics with FIXED functions
    L_values = np.array([5, 10, 20, 50, 100, 200, 500])
    L_values = L_values[L_values < n_steps // 3]

    console.print("\n[cyan]Computing Δ3(L) with FIXED algorithm...[/]")
    delta3_model = compute_delta3_correct(model_spacings, L_values)
    delta3_real = compute_delta3_correct(real_spacings, L_values)
    delta3_gue = compute_delta3_correct(gue_spacings, L_values)
    delta3_shuffled = compute_delta3_correct(shuffled_spacings, L_values)
    delta3_theory = gue_delta3_theory(L_values)
    delta3_poisson = poisson_delta3_theory(L_values)

    console.print("\n[cyan]Computing Σ²(L)...[/]")
    sigma2_model = compute_number_variance_correct(model_spacings, L_values)
    sigma2_real = compute_number_variance_correct(real_spacings, L_values)
    sigma2_gue = compute_number_variance_correct(gue_spacings, L_values)
    sigma2_theory = gue_sigma2_theory(L_values)

    # Display results
    console.print("\n")
    table = Table(title="📊 Δ3(L) RIGIDITY (FIXED)")
    table.add_column("L", style="bold", justify="right")
    table.add_column("Snowball", justify="right")
    table.add_column("Real Zeta", justify="right")
    table.add_column("GUE Sim", justify="right")
    table.add_column("GUE Theory", justify="right")
    table.add_column("Shuffled", justify="right")
    table.add_column("Poisson", justify="right")

    for i, L in enumerate(L_values):
        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "N/A"

        table.add_row(
            f"{L:.0f}",
            fmt(delta3_model[i]),
            fmt(delta3_real[i]),
            fmt(delta3_gue[i]),
            fmt(delta3_theory[i]),
            fmt(delta3_shuffled[i]),
            fmt(delta3_poisson[i]),
        )

    console.print(table)

    # Number variance table
    console.print("\n")
    table2 = Table(title="📊 Σ²(L) NUMBER VARIANCE")
    table2.add_column("L", style="bold", justify="right")
    table2.add_column("Snowball", justify="right")
    table2.add_column("Real Zeta", justify="right")
    table2.add_column("GUE Sim", justify="right")
    table2.add_column("GUE Theory", justify="right")

    for i, L in enumerate(L_values):
        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "N/A"

        table2.add_row(
            f"{L:.0f}",
            fmt(sigma2_model[i]),
            fmt(sigma2_real[i]),
            fmt(sigma2_gue[i]),
            fmt(sigma2_theory[i]),
        )

    console.print(table2)

    # Diagnosis
    valid_mask = ~np.isnan(delta3_model) & ~np.isnan(delta3_gue) & ~np.isnan(delta3_real)

    if valid_mask.sum() >= 3:
        # Correlations
        corr_gue = np.corrcoef(delta3_model[valid_mask], delta3_gue[valid_mask])[0, 1]
        corr_real = np.corrcoef(delta3_model[valid_mask], delta3_real[valid_mask])[0, 1]
        corr_theory = np.corrcoef(delta3_model[valid_mask], delta3_theory[valid_mask])[0, 1]

        # MAE
        mae_gue = np.mean(np.abs(delta3_model[valid_mask] - delta3_gue[valid_mask]))
        mae_real = np.mean(np.abs(delta3_model[valid_mask] - delta3_real[valid_mask]))
        mae_poisson = np.mean(np.abs(delta3_model[valid_mask] - delta3_poisson[valid_mask]))

        console.print("\n")
        diag_table = Table(title="💡 RIGIDITY DIAGNOSIS (FIXED)")
        diag_table.add_column("Comparison", style="bold")
        diag_table.add_column("Correlation")
        diag_table.add_column("MAE")
        diag_table.add_column("Verdict")

        def verdict(corr, mae, ref_name):
            if corr > 0.95 and mae < 0.05:
                return f"[green]MATCHES {ref_name}[/]"
            elif corr > 0.85:
                return f"[yellow]CLOSE to {ref_name}[/]"
            elif corr > 0.5:
                return f"[cyan]PARTIAL {ref_name}[/]"
            else:
                return f"[red]DIFFERS from {ref_name}[/]"

        diag_table.add_row("vs GUE Sim", f"{corr_gue:.3f}", f"{mae_gue:.3f}", verdict(corr_gue, mae_gue, "GUE"))
        diag_table.add_row("vs GUE Theory", f"{corr_theory:.3f}", f"-", verdict(corr_theory, 0, "Theory"))
        diag_table.add_row("vs Real Zeta", f"{corr_real:.3f}", f"{mae_real:.3f}", verdict(corr_real, mae_real, "Zeta"))
        diag_table.add_row("vs Poisson", "-", f"{mae_poisson:.3f}", "[green]DIFFERS[/]" if mae_poisson > 1 else "[red]TOO CLOSE[/]")

        console.print(diag_table)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Δ3(L)
    ax = axes[0, 0]
    valid = ~np.isnan(delta3_model)
    ax.loglog(L_values[valid], delta3_model[valid], 'b-o', label='Snowball v8', linewidth=2, markersize=8)
    valid = ~np.isnan(delta3_real)
    ax.loglog(L_values[valid], delta3_real[valid], 'g-s', label='Real Zeta', linewidth=2, markersize=8)
    valid = ~np.isnan(delta3_gue)
    ax.loglog(L_values[valid], delta3_gue[valid], 'r-^', label='GUE Sim', linewidth=2, markersize=8)
    ax.loglog(L_values, delta3_theory, 'k--', label='GUE Theory', linewidth=2)
    ax.loglog(L_values, delta3_poisson, 'gray', linestyle=':', label='Poisson', linewidth=2)
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('Δ3(L)', fontsize=12)
    ax.set_title('Spectral Rigidity Δ3(L) — FIXED', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Σ²(L)
    ax = axes[0, 1]
    valid = ~np.isnan(sigma2_model)
    ax.loglog(L_values[valid], sigma2_model[valid], 'b-o', label='Snowball v8', linewidth=2, markersize=8)
    valid = ~np.isnan(sigma2_real)
    ax.loglog(L_values[valid], sigma2_real[valid], 'g-s', label='Real Zeta', linewidth=2, markersize=8)
    valid = ~np.isnan(sigma2_gue)
    ax.loglog(L_values[valid], sigma2_gue[valid], 'r-^', label='GUE Sim', linewidth=2, markersize=8)
    ax.loglog(L_values, sigma2_theory, 'k--', label='GUE Theory', linewidth=2)
    ax.loglog(L_values, L_values, 'gray', linestyle=':', label='Poisson', linewidth=2)
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('Σ²(L)', fontsize=12)
    ax.set_title('Number Variance Σ²(L)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spacing histogram
    ax = axes[1, 0]
    bins = np.linspace(0, 4, 50)
    ax.hist(model_spacings, bins=bins, density=True, alpha=0.5, label='Snowball v8', color='blue')
    ax.hist(real_spacings, bins=bins, density=True, alpha=0.5, label='Real Zeta', color='green')
    ax.hist(gue_spacings, bins=bins, density=True, alpha=0.5, label='GUE Sim', color='red')

    s = np.linspace(0.01, 4, 100)
    wigner = (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4)
    ax.plot(s, wigner, 'k--', linewidth=2, label='Wigner Surmise')

    ax.set_xlabel('Spacing s', fontsize=12)
    ax.set_ylabel('P(s)', fontsize=12)
    ax.set_title('Spacing Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Comparison bar chart
    ax = axes[1, 1]
    if valid_mask.sum() >= 3:
        metrics = ['Corr GUE', 'Corr Zeta', 'Corr Theory']
        values = [corr_gue, corr_real, corr_theory]
        colors = ['red', 'green', 'black']
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Correlation')
        ax.set_title('Model Correlations with References')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('reports/long_range_statistics_v2.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/long_range_statistics_v2.png[/]")

    # Save
    np.savez(
        "reports/long_range_stats_v2.npz",
        L_values=L_values,
        delta3_model=delta3_model,
        delta3_real=delta3_real,
        delta3_gue=delta3_gue,
        delta3_theory=delta3_theory,
        sigma2_model=sigma2_model,
    )
    console.print(f"[green]Results saved: reports/long_range_stats_v2.npz[/]")

    # Final verdict
    console.print("\n")

    if valid_mask.sum() >= 3:
        if corr_gue > 0.9 and corr_theory > 0.9:
            if corr_real > 0.85:
                diagnosis = (
                    "[bold green]🎻 FULL SUCCESS![/]\n"
                    "Model matches GUE AND real zeta rigidity.\n"
                    "Long-range spectral structure captured!"
                )
            else:
                diagnosis = (
                    "[bold yellow]⚠️ UNIVERSAL RMT CAPTURED[/]\n"
                    f"Matches GUE (corr={corr_gue:.3f}) but differs from zeta (corr={corr_real:.3f}).\n"
                    "Model learned generic RMT, not zeta-specific structure."
                )
        elif corr_gue > 0.7:
            diagnosis = (
                "[bold yellow]⚠️ PARTIAL RMT STRUCTURE[/]\n"
                f"Some GUE-like behavior (corr={corr_gue:.3f}).\n"
                "Long-range correlations present but imperfect."
            )
        else:
            diagnosis = (
                "[bold red]🔴 RIGIDITY NOT CAPTURED[/]\n"
                f"Poor match with GUE (corr={corr_gue:.3f}).\n"
                "Model doesn't show proper spectral rigidity."
            )

        console.print(Panel.fit(diagnosis, title="🎯 FINAL VERDICT", border_style="cyan"))

    return {
        "delta3_model": delta3_model,
        "delta3_gue": delta3_gue,
        "corr_gue": corr_gue if valid_mask.sum() >= 3 else None,
        "corr_real": corr_real if valid_mask.sum() >= 3 else None,
    }


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    run_analysis_v2(n_steps=2000)
