#!/usr/bin/env python3
"""
MATH RIGOR AUDIT: Checking Pi & Normalization

Critical Question: Did we forget 2π in the exponent?

SFF Formula variants:
- exp(i·τ·u)      - What we used (no 2π)
- exp(2πi·τ·u)    - Textbook Fourier standard

If 13.99 was caused by missing 2π, this audit will reveal it.
"""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def run_audit():
    console.print(Panel.fit("[bold red]🚨 MATH RIGOR AUDIT: Checking Pi & Normalization 🚨[/]"))

    # 1. LOAD DATA
    console.print("\n[yellow]Loading data...[/]")
    val_data = torch.load('data/val.pt', weights_only=False)
    bin_centers = np.load('data/bin_centers.npy')

    # Extract ALL spacings (not just 50!)
    raw_spacings = []
    for i in range(len(val_data)):  # ALL trajectories
        s = bin_centers[val_data[i].numpy()]
        raw_spacings.extend(s)
    raw_spacings = np.array(raw_spacings)

    # 2. MEAN SPACING CHECK (CRITICAL!)
    mean_val = np.mean(raw_spacings)
    console.print(f"[cyan]Raw Mean Spacing: {mean_val:.6f}[/]")

    # FORCE NORMALIZATION
    # В теории RMT среднее расстояние должно быть строго 1.0
    spacings_norm = raw_spacings / mean_val
    console.print(f"[green]Normalized Mean Spacing -> {np.mean(spacings_norm):.6f}[/]")

    # Unfold (Cumulative sum)
    u = np.concatenate([[0], np.cumsum(spacings_norm)])
    N = len(u)
    console.print(f"[cyan]Total unfolded points: {N:,}[/]")

    # 3. GENERATE POISSON (CONTROL GROUP)
    np.random.seed(42)
    spacings_poisson = np.random.exponential(scale=1.0, size=len(raw_spacings))
    u_poisson = np.concatenate([[0], np.cumsum(spacings_poisson)])

    # 4. SFF FORMULA CHECK
    # SAME tau range as verify_normalization_artifact.py!
    tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    def calc_sff_curve(u_arr, formula_type):
        """Compute full SFF curve."""
        vals = []
        for tau in tau_values:
            if formula_type == 'no_pi':
                arg = 1j * tau * u_arr
            elif formula_type == 'with_2pi':
                arg = 2j * np.pi * tau * u_arr

            # K(tau) = |sum(exp)|² / N
            val = np.abs(np.sum(np.exp(arg)))**2 / len(u_arr)
            vals.append(val)
        return np.array(vals)

    def get_plateau(K_vals, tau_vals):
        """Get plateau (same as verify_normalization: tau > 4)."""
        mask = tau_vals > 4.0
        if mask.sum() > 0:
            return np.mean(K_vals[mask])
        else:
            return np.mean(K_vals[-10:])

    def calc_sff_val(u_arr, formula_type):
        K = calc_sff_curve(u_arr, formula_type)
        return get_plateau(K, tau_values)

    console.print("\n[yellow]Computing SFF with both formulas...[/]")

    # --- RUN TESTS ---

    # Case 1: Real Data
    real_no_pi = calc_sff_val(u, 'no_pi')
    real_2pi = calc_sff_val(u, 'with_2pi')

    # Case 2: Poisson
    pois_no_pi = calc_sff_val(u_poisson, 'no_pi')
    pois_2pi = calc_sff_val(u_poisson, 'with_2pi')

    # 5. REPORT
    console.print("\n")
    table = Table(title="🏆 AUDIT RESULTS: SFF PLATEAU HEIGHT")
    table.add_column("Data Source", style="bold")
    table.add_column("exp(i·τ·u)\n(Our Method)", justify="right", style="cyan")
    table.add_column("exp(2πi·τ·u)\n(Textbook)", justify="right", style="magenta")
    table.add_column("Ratio\n(no_pi/2pi)", justify="right")
    table.add_column("Conclusion")

    ratio_real = real_no_pi / real_2pi if real_2pi > 0 else float('inf')
    ratio_pois = pois_no_pi / pois_2pi if pois_2pi > 0 else float('inf')

    table.add_row(
        "Real Zeros",
        f"{real_no_pi:.4f}",
        f"{real_2pi:.4f}",
        f"{ratio_real:.2f}x",
        "[bold green]Physics[/]"
    )
    table.add_row(
        "Poisson (Noise)",
        f"{pois_no_pi:.4f}",
        f"{pois_2pi:.4f}",
        f"{ratio_pois:.2f}x",
        "[grey]Baseline[/]"
    )

    console.print(table)

    # KEY RATIOS
    console.print("\n[bold]KEY COMPARISONS:[/]")
    console.print(f"  Real/Poisson (no_pi):  {real_no_pi/pois_no_pi:.2f}x")
    console.print(f"  Real/Poisson (with_2pi): {real_2pi/pois_2pi:.2f}x")

    # INTERPRETATION
    console.print("\n[bold]═══ VERDICT ═══[/]\n")

    if abs(real_2pi - 1.0) < 0.3 and abs(pois_2pi - 1.0) < 0.3:
        console.print(Panel.fit(
            "[yellow]⚠️ With 2π, both collapse to ~1.0![/]\n\n"
            "This means:\n"
            "- The physics is in the RAMP region, not the plateau\n"
            "- Our high values (13.99) were from wrong frequency scaling\n"
            "- Need to analyze ramp slope instead of plateau",
            title="RAMP PHYSICS",
            border_style="yellow"
        ))
    elif real_2pi > pois_2pi * 1.2:  # 20% difference is significant
        console.print(Panel.fit(
            f"[green]✅ Real Zeros still beats Poisson![/]\n\n"
            f"With 2π formula:\n"
            f"  Real:    {real_2pi:.4f}\n"
            f"  Poisson: {pois_2pi:.4f}\n"
            f"  Ratio:   {real_2pi/pois_2pi:.2f}x\n\n"
            f"[green]Structure is ROBUST regardless of π convention![/]",
            title="✅ STRUCTURE CONFIRMED",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[red]❌ With 2π, difference vanishes![/]\n\n"
            "13.99 was likely an artifact of frequency scaling.\n"
            "The Riemann structure may be weaker than we thought.",
            title="⚠️ POSSIBLE ARTIFACT",
            border_style="red"
        ))


if __name__ == "__main__":
    run_audit()
