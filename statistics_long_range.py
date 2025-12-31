#!/usr/bin/env python3
"""
LONG-RANGE SPECTRAL STATISTICS

Глобальные метрики для проверки "поймала ли модель оркестр":
1. Δ3(L) — spectral rigidity (Dyson-Mehta statistic)
2. SFF K(τ) — spectral form factor (ramp/plateau structure)
3. Number variance Σ²(L)

Сравниваем:
- Snowball v8 generated trajectories
- Real zeta zeros (unfolded)
- GUE ensemble (Monte Carlo)
- Shuffled test (destroyed correlations)
"""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

console = Console()


# ============================================================================
# SPECTRAL STATISTICS
# ============================================================================

def compute_delta3(spacings, L_values):
    """
    Compute Dyson-Mehta Δ3(L) statistic.

    Δ3(L) measures how well the integrated level density can be
    approximated by a straight line over interval L.

    For GUE: Δ3(L) ≈ (1/π²) ln(L) + const for large L
    For Poisson: Δ3(L) = L/15

    Args:
        spacings: array of unfolded spacings (mean=1)
        L_values: array of L values to compute Δ3 at

    Returns:
        delta3: array of Δ3(L) values
    """
    # Convert spacings to positions (cumulative sum)
    positions = np.cumsum(spacings)
    N = len(positions)

    delta3 = []

    for L in L_values:
        # Number of windows we can fit
        n_windows = max(1, int(N - L))
        d3_samples = []

        for start in range(0, n_windows, max(1, n_windows // 50)):  # Sample ~50 windows
            # Find levels in window [start, start+L]
            mask = (positions >= positions[start]) & (positions < positions[start] + L)
            levels_in_window = positions[mask] - positions[start]
            n_levels = len(levels_in_window)

            if n_levels < 3:
                continue

            # Fit best straight line: N(x) = ax + b
            # Minimize ∫[N(x) - ax - b]² dx
            x = levels_in_window
            n = np.arange(1, n_levels + 1)

            # Least squares fit
            A = np.column_stack([x, np.ones_like(x)])
            coeffs, residuals, rank, s = np.linalg.lstsq(A, n, rcond=None)

            # Compute Δ3 = (1/L) ∫[N(x) - ax - b]² dx ≈ (1/L) Σ[n_i - a*x_i - b]²
            predicted = A @ coeffs
            d3 = np.mean((n - predicted) ** 2)
            d3_samples.append(d3)

        if d3_samples:
            delta3.append(np.mean(d3_samples))
        else:
            delta3.append(np.nan)

    return np.array(delta3)


def compute_number_variance(spacings, L_values):
    """
    Compute number variance Σ²(L).

    Σ²(L) = <(n - <n>)²> where n is number of levels in interval L

    For GUE: Σ²(L) ≈ (2/π²) ln(L) + const for large L
    For Poisson: Σ²(L) = L
    """
    positions = np.cumsum(spacings)
    N = len(positions)

    sigma2 = []

    for L in L_values:
        n_windows = max(1, int(N - L))
        counts = []

        for start in range(0, n_windows, max(1, n_windows // 100)):
            mask = (positions >= positions[start]) & (positions < positions[start] + L)
            counts.append(mask.sum())

        if counts:
            counts = np.array(counts)
            sigma2.append(np.var(counts))
        else:
            sigma2.append(np.nan)

    return np.array(sigma2)


def compute_sff(spacings, tau_values):
    """
    Compute spectral form factor K(τ).

    K(τ) = |∑_n exp(2πi τ x_n)|² / N

    For GUE:
    - Linear ramp: K(τ) ≈ τ for τ < 1
    - Plateau: K(τ) ≈ 1 for τ ≥ 1

    τ is in units of mean level spacing.
    """
    positions = np.cumsum(spacings)
    positions = positions - positions.mean()  # Center
    N = len(positions)

    sff = []

    for tau in tau_values:
        # Fourier sum
        phase = 2 * np.pi * tau * positions
        z = np.sum(np.exp(1j * phase))
        K = np.abs(z) ** 2 / N
        sff.append(K)

    return np.array(sff)


# ============================================================================
# GUE REFERENCE
# ============================================================================

def gue_delta3_theory(L):
    """Theoretical Δ3(L) for GUE."""
    # Δ3(L) ≈ (1/π²) ln(2πL) - 0.007 for large L
    return (1 / np.pi**2) * np.log(2 * np.pi * L) - 0.007


def gue_sigma2_theory(L):
    """Theoretical Σ²(L) for GUE."""
    # Σ²(L) ≈ (2/π²) ln(2πL) + 0.422 for large L
    return (2 / np.pi**2) * np.log(2 * np.pi * L) + 0.422


def poisson_delta3_theory(L):
    """Theoretical Δ3(L) for Poisson (no correlations)."""
    return L / 15


def generate_gue_spacings(N, seed=42):
    """Generate GUE eigenvalue spacings via Monte Carlo."""
    np.random.seed(seed)

    # Random Hermitian matrix
    H = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    H = (H + H.conj().T) / 2

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues.sort()

    # Unfold using semicircle law
    # For GUE, density is semicircle: ρ(x) = (1/π) sqrt(2N - x²)
    # Unfolding: x → ∫ρ(t)dt
    # Approximate unfolding for simplicity:
    from scipy.interpolate import interp1d

    # Empirical CDF as unfolding
    n = np.arange(1, N + 1)
    unfolded = interp1d(eigenvalues, n, fill_value="extrapolate")(eigenvalues)

    # Spacings
    spacings = np.diff(unfolded)
    spacings = spacings / spacings.mean()  # Normalize to mean 1

    return spacings


# ============================================================================
# GENERATION WITH SNOWBALL
# ============================================================================

def generate_snowball_trajectory(model, config, bin_centers, device, n_steps=1000, seed=42):
    """Generate long trajectory using snowball model with recurrent memory."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32).to(device)

    # Start with a sequence from validation data
    val = torch.load("data/val.pt", weights_only=False)
    start_seq = val[0:1, :config.seq_len].to(device)

    # Initialize memory
    memory_state = model.snowball.get_initial_state(1, device)

    generated_tokens = start_seq[0].tolist()
    generated_spacings = []

    # Window size for processing
    window_size = min(128, config.seq_len - 1)

    for step in track(range(n_steps), description="Generating trajectory..."):
        # Get last window
        if len(generated_tokens) >= window_size:
            window = torch.tensor([generated_tokens[-window_size:]], dtype=torch.long, device=device)
        else:
            window = torch.tensor([generated_tokens], dtype=torch.long, device=device)

        # Forward pass
        scale_val = bin_centers_t[window]

        with torch.no_grad():
            pred, loss, mem_attn, new_memory = model(
                window, targets=None, return_hidden=False, scale_val=scale_val,
                memory_state=memory_state, return_memory=True
            )

            # Get predicted mu for last position
            mu = model.get_mu(pred)[:, -1].item()

            # Sample from Wigner distribution centered at mu
            # P(s) ∝ s * exp(-π s² / (4 mu²))
            # Sample using rejection or inverse CDF

            # Simple approximation: sample spacing around mu
            # Wigner mode is at s = sqrt(2/π) * mu ≈ 0.8 * mu
            # Use Wigner-like sampling

            u = np.random.random()
            # Inverse CDF of Wigner: s = mu * sqrt(-4 ln(1-u) / π)
            spacing = mu * np.sqrt(-4 * np.log(1 - u + 1e-10) / np.pi)
            spacing = np.clip(spacing, bin_centers[1], bin_centers[-2])

            # Find nearest bin
            token = np.abs(bin_centers - spacing).argmin()

            generated_tokens.append(int(token))
            generated_spacings.append(spacing)

            # Update memory
            memory_state = new_memory

    return np.array(generated_spacings)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis(n_steps=2000):
    """Run full long-range statistics analysis."""

    console.print(Panel.fit(
        "[bold cyan]🎻 LONG-RANGE SPECTRAL STATISTICS[/]\n"
        "Δ3 Rigidity + SFF Form Factor + Σ² Number Variance",
        title="SYMPHONY CHECK"
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
    console.print(f"\n[cyan]Generating {n_steps} step trajectory with snowball...[/]")
    model_spacings = generate_snowball_trajectory(
        model, config, bin_centers, device, n_steps=n_steps
    )
    console.print(f"  Mean spacing: {model_spacings.mean():.3f}")
    console.print(f"  Std spacing: {model_spacings.std():.3f}")

    # Load real zeta spacings
    console.print("\n[cyan]Loading real zeta spacings...[/]")
    val = torch.load("data/val.pt", weights_only=False)
    # Flatten and convert to spacings
    real_tokens = val[:n_steps // 256 + 1].flatten().numpy()[:n_steps]
    real_spacings = bin_centers[real_tokens]
    real_spacings = real_spacings / real_spacings.mean()  # Normalize
    console.print(f"  Mean spacing: {real_spacings.mean():.3f}")

    # Generate GUE reference
    console.print("\n[cyan]Generating GUE ensemble...[/]")
    gue_spacings = generate_gue_spacings(n_steps + 100, seed=42)[:n_steps]
    console.print(f"  Mean spacing: {gue_spacings.mean():.3f}")

    # Shuffled test (destroys correlations)
    console.print("\n[cyan]Creating shuffled baseline...[/]")
    shuffled_spacings = real_spacings.copy()
    np.random.shuffle(shuffled_spacings)

    # Compute statistics
    L_values = np.array([5, 10, 20, 50, 100, 200])
    L_values = L_values[L_values < n_steps // 2]

    tau_values = np.linspace(0.01, 3.0, 100)

    console.print("\n[cyan]Computing Δ3(L) rigidity...[/]")
    delta3_model = compute_delta3(model_spacings, L_values)
    delta3_real = compute_delta3(real_spacings, L_values)
    delta3_gue = compute_delta3(gue_spacings, L_values)
    delta3_shuffled = compute_delta3(shuffled_spacings, L_values)
    delta3_theory = gue_delta3_theory(L_values)
    delta3_poisson = poisson_delta3_theory(L_values)

    console.print("\n[cyan]Computing Σ²(L) number variance...[/]")
    sigma2_model = compute_number_variance(model_spacings, L_values)
    sigma2_real = compute_number_variance(real_spacings, L_values)
    sigma2_gue = compute_number_variance(gue_spacings, L_values)
    sigma2_theory = gue_sigma2_theory(L_values)

    console.print("\n[cyan]Computing SFF K(τ)...[/]")
    sff_model = compute_sff(model_spacings, tau_values)
    sff_real = compute_sff(real_spacings, tau_values)
    sff_gue = compute_sff(gue_spacings, tau_values)
    sff_shuffled = compute_sff(shuffled_spacings, tau_values)

    # Display results
    console.print("\n")
    table = Table(title="📊 Δ3(L) RIGIDITY COMPARISON")
    table.add_column("L", style="bold", justify="right")
    table.add_column("Snowball v8", justify="right")
    table.add_column("Real Zeta", justify="right")
    table.add_column("GUE Sim", justify="right")
    table.add_column("GUE Theory", justify="right")
    table.add_column("Shuffled", justify="right")

    for i, L in enumerate(L_values):
        table.add_row(
            f"{L:.0f}",
            f"{delta3_model[i]:.3f}" if not np.isnan(delta3_model[i]) else "N/A",
            f"{delta3_real[i]:.3f}" if not np.isnan(delta3_real[i]) else "N/A",
            f"{delta3_gue[i]:.3f}" if not np.isnan(delta3_gue[i]) else "N/A",
            f"{delta3_theory[i]:.3f}",
            f"{delta3_shuffled[i]:.3f}" if not np.isnan(delta3_shuffled[i]) else "N/A",
        )

    console.print(table)

    # Diagnosis
    console.print("\n")

    # Compare model to real and GUE
    valid_mask = ~np.isnan(delta3_model) & ~np.isnan(delta3_real)
    if valid_mask.sum() > 2:
        corr_real = np.corrcoef(delta3_model[valid_mask], delta3_real[valid_mask])[0, 1]
        corr_gue = np.corrcoef(delta3_model[valid_mask], delta3_gue[valid_mask])[0, 1]

        # Mean absolute error
        mae_real = np.mean(np.abs(delta3_model[valid_mask] - delta3_real[valid_mask]))
        mae_gue = np.mean(np.abs(delta3_model[valid_mask] - delta3_gue[valid_mask]))
        mae_poisson = np.mean(np.abs(delta3_model[valid_mask] - delta3_poisson[valid_mask]))

        diag_table = Table(title="💡 RIGIDITY DIAGNOSIS")
        diag_table.add_column("Comparison", style="bold")
        diag_table.add_column("Correlation")
        diag_table.add_column("MAE")
        diag_table.add_column("Verdict")

        def verdict(corr, mae):
            if corr > 0.9 and mae < 0.1:
                return "[green]EXCELLENT[/]"
            elif corr > 0.7:
                return "[yellow]GOOD[/]"
            else:
                return "[red]POOR[/]"

        diag_table.add_row("vs Real Zeta", f"{corr_real:.3f}", f"{mae_real:.3f}", verdict(corr_real, mae_real))
        diag_table.add_row("vs GUE", f"{corr_gue:.3f}", f"{mae_gue:.3f}", verdict(corr_gue, mae_gue))
        diag_table.add_row("vs Poisson", "-", f"{mae_poisson:.3f}", "[red]Should differ[/]" if mae_poisson > 0.5 else "[green]OK[/]")

        console.print(diag_table)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Δ3(L)
    ax = axes[0, 0]
    ax.loglog(L_values, delta3_model, 'b-o', label='Snowball v8', linewidth=2, markersize=8)
    ax.loglog(L_values, delta3_real, 'g-s', label='Real Zeta', linewidth=2, markersize=8)
    ax.loglog(L_values, delta3_gue, 'r-^', label='GUE Sim', linewidth=2, markersize=8)
    ax.loglog(L_values, delta3_theory, 'k--', label='GUE Theory', linewidth=2)
    ax.loglog(L_values, delta3_poisson, 'gray', linestyle=':', label='Poisson', linewidth=2)
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('Δ3(L)', fontsize=12)
    ax.set_title('Spectral Rigidity Δ3(L)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Σ²(L)
    ax = axes[0, 1]
    ax.loglog(L_values, sigma2_model, 'b-o', label='Snowball v8', linewidth=2, markersize=8)
    ax.loglog(L_values, sigma2_real, 'g-s', label='Real Zeta', linewidth=2, markersize=8)
    ax.loglog(L_values, sigma2_gue, 'r-^', label='GUE Sim', linewidth=2, markersize=8)
    ax.loglog(L_values, sigma2_theory, 'k--', label='GUE Theory', linewidth=2)
    ax.loglog(L_values, L_values, 'gray', linestyle=':', label='Poisson', linewidth=2)
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('Σ²(L)', fontsize=12)
    ax.set_title('Number Variance Σ²(L)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SFF
    ax = axes[1, 0]
    ax.plot(tau_values, sff_model, 'b-', label='Snowball v8', linewidth=2, alpha=0.8)
    ax.plot(tau_values, sff_real, 'g-', label='Real Zeta', linewidth=2, alpha=0.8)
    ax.plot(tau_values, sff_gue, 'r-', label='GUE Sim', linewidth=2, alpha=0.8)
    ax.plot(tau_values, sff_shuffled, 'gray', linestyle=':', label='Shuffled', linewidth=2)
    # GUE theory: linear ramp then plateau
    tau_ramp = tau_values[tau_values <= 1]
    tau_plateau = tau_values[tau_values > 1]
    ax.plot(tau_ramp, tau_ramp, 'k--', linewidth=2, alpha=0.5)
    ax.axhline(y=1, xmin=1/3, xmax=1, color='k', linestyle='--', linewidth=2, alpha=0.5, label='GUE Theory')
    ax.set_xlabel('τ', fontsize=12)
    ax.set_ylabel('K(τ)', fontsize=12)
    ax.set_title('Spectral Form Factor', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)

    # Spacing histogram comparison
    ax = axes[1, 1]
    bins = np.linspace(0, 4, 50)
    ax.hist(model_spacings, bins=bins, density=True, alpha=0.5, label='Snowball v8', color='blue')
    ax.hist(real_spacings, bins=bins, density=True, alpha=0.5, label='Real Zeta', color='green')

    # Wigner surmise
    s = np.linspace(0, 4, 100)
    wigner = (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4)
    ax.plot(s, wigner, 'k--', linewidth=2, label='Wigner Surmise')

    ax.set_xlabel('Spacing s', fontsize=12)
    ax.set_ylabel('P(s)', fontsize=12)
    ax.set_title('Spacing Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/long_range_statistics.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/long_range_statistics.png[/]")

    # Save results
    np.savez(
        "reports/long_range_stats.npz",
        L_values=L_values,
        tau_values=tau_values,
        delta3_model=delta3_model,
        delta3_real=delta3_real,
        delta3_gue=delta3_gue,
        sigma2_model=sigma2_model,
        sff_model=sff_model,
        sff_real=sff_real,
    )
    console.print(f"[green]Results saved: reports/long_range_stats.npz[/]")

    # Final diagnosis
    console.print("\n")
    if valid_mask.sum() > 2 and corr_real > 0.8:
        diagnosis = (
            "[bold green]🎻 MODEL CAPTURED LONG-RANGE STRUCTURE![/]\n"
            "Δ3(L) rigidity matches real zeta zeros.\n"
            "The overcompensation on lag-1 is a local artifact.\n"
            "Long-range physics is preserved!"
        )
    elif valid_mask.sum() > 2 and corr_gue > 0.8:
        diagnosis = (
            "[bold yellow]⚠️ MODEL SHOWS GUE-LIKE BEHAVIOR[/]\n"
            "Matches GUE but not exactly real zeta.\n"
            "Generic RMT captured, specific zeta features missing."
        )
    else:
        diagnosis = (
            "[bold red]🔴 LONG-RANGE STRUCTURE NOT CAPTURED[/]\n"
            "Model doesn't match real zeta or GUE rigidity.\n"
            "Need better architecture or training."
        )

    console.print(Panel.fit(diagnosis, title="🎯 FINAL VERDICT", border_style="cyan"))

    return {
        "delta3_model": delta3_model,
        "delta3_real": delta3_real,
        "corr_real": corr_real if valid_mask.sum() > 2 else None,
    }


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    run_analysis(n_steps=2000)
