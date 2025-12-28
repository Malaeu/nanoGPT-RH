#!/usr/bin/env python3
"""
JITTER ROBUSTNESS TEST: Is the SFF spike real or a binning artifact?

Hypothesis:
- If 2π spike disappears with jitter → Binning artifact (quantization echo)
- If 2π spike survives jitter → Real physics (zeros sit on quasi-lattice)

Method:
1. Load binned data (bin_centers[token_ids])
2. Add uniform jitter within bin width: s' = s + U(-w/2, w/2)
3. Compute SFF for original and jittered
4. Compare spike heights
"""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt

console = Console()


def compute_sff(spacings, tau_values):
    """Compute Spectral Form Factor"""
    spacings = spacings / np.mean(spacings)  # Normalize mean to 1
    u = np.concatenate([[0], np.cumsum(spacings)])
    N = len(u)

    sff = []
    for tau in tau_values:
        phases = tau * u
        complex_sum = np.sum(np.exp(1j * phases))
        k = (np.abs(complex_sum) ** 2) / N
        sff.append(k)

    return np.array(sff)


def run_jitter_test():
    console.print(Panel.fit(
        "[bold cyan]🎲 JITTER ROBUSTNESS TEST[/]\n"
        "Testing if SFF spikes are binning artifacts...",
        title="CRITICAL SANITY CHECK"
    ))

    # Load data
    val_data = torch.load('data/val.pt', weights_only=False)
    bin_centers = np.load('data/bin_centers.npy')

    # Compute bin width
    bin_width = bin_centers[1] - bin_centers[0]
    console.print(f"[dim]Bin width: {bin_width:.6f}[/]")

    # Convert to continuous spacings
    raw_spacings = []
    for i in range(len(val_data)):
        s = bin_centers[val_data[i].numpy()]
        raw_spacings.extend(s)
    raw_spacings = np.array(raw_spacings)

    console.print(f"[dim]Total spacings: {len(raw_spacings):,}[/]")

    # Tau range (avoiding exact 2π, focus on region around it)
    tau_values = np.linspace(0.5, 10.0, 2000)

    # Test multiple jitter levels
    jitter_levels = [0, 0.25, 0.5, 0.75, 1.0]  # Fraction of bin_width
    results = {}

    console.print("\n[bold]Computing SFF for different jitter levels...[/]")

    for jitter_frac in jitter_levels:
        jitter_amount = jitter_frac * bin_width

        if jitter_frac == 0:
            spacings = raw_spacings.copy()
            label = "No jitter (original)"
        else:
            # Add uniform jitter within bin
            jitter = np.random.uniform(-jitter_amount/2, jitter_amount/2, len(raw_spacings))
            spacings = raw_spacings + jitter
            # Ensure no negative spacings
            spacings = np.maximum(spacings, 0.001)
            label = f"Jitter ±{jitter_amount/2:.4f}"

        sff = compute_sff(spacings, tau_values)

        # Find peak near 2π
        pi2_region = (tau_values > 5.5) & (tau_values < 7.0)
        peak_in_region = np.max(sff[pi2_region])
        peak_tau = tau_values[pi2_region][np.argmax(sff[pi2_region])]

        # Background (plateau region away from resonances)
        bg_region = (tau_values > 3.0) & (tau_values < 5.0)
        background = np.mean(sff[bg_region])

        results[jitter_frac] = {
            'sff': sff,
            'peak': peak_in_region,
            'peak_tau': peak_tau,
            'background': background,
            'snr': peak_in_region / background,  # Signal-to-noise ratio
            'label': label
        }

        console.print(f"  {label}: peak={peak_in_region:.2f} at τ={peak_tau:.3f}, bg={background:.3f}, SNR={peak_in_region/background:.1f}x")

    # Summary table
    console.print("\n")
    table = Table(title="🎲 JITTER TEST RESULTS")
    table.add_column("Jitter Level", style="bold")
    table.add_column("Peak Height", justify="right")
    table.add_column("Peak τ", justify="right")
    table.add_column("Background", justify="right")
    table.add_column("SNR", justify="right")
    table.add_column("Verdict", style="bold")

    original_peak = results[0]['peak']

    for jitter_frac, r in results.items():
        # Verdict
        if jitter_frac == 0:
            verdict = "[dim]Baseline[/]"
        else:
            retention = r['peak'] / original_peak
            if retention > 0.8:
                verdict = "[green]SURVIVES ✓[/]"
            elif retention > 0.5:
                verdict = "[yellow]DEGRADED[/]"
            else:
                verdict = "[red]ARTIFACT ✗[/]"

        table.add_row(
            r['label'],
            f"{r['peak']:.2f}",
            f"{r['peak_tau']:.3f}",
            f"{r['background']:.3f}",
            f"{r['snr']:.1f}x",
            verdict
        )

    console.print(table)

    # Final verdict
    full_jitter = results[1.0]
    retention = full_jitter['peak'] / original_peak

    console.print("\n")
    if retention > 0.7:
        console.print(Panel.fit(
            f"[bold green]✅ SPIKE IS REAL PHYSICS![/]\n\n"
            f"Peak retention at full jitter: {retention:.1%}\n"
            f"The 2π spike survives bin-width noise.\n\n"
            f"[green]Zeros genuinely sit on a quasi-lattice.[/]",
            title="🏆 VERDICT: NOT AN ARTIFACT",
            border_style="green"
        ))
    elif retention > 0.3:
        console.print(Panel.fit(
            f"[bold yellow]⚠️ MIXED SIGNAL[/]\n\n"
            f"Peak retention at full jitter: {retention:.1%}\n"
            f"Partial artifact, partial physics.\n\n"
            f"[yellow]Need more investigation.[/]",
            title="⚠️ VERDICT: INCONCLUSIVE",
            border_style="yellow"
        ))
    else:
        console.print(Panel.fit(
            f"[bold red]❌ SPIKE IS BINNING ARTIFACT![/]\n\n"
            f"Peak retention at full jitter: {retention:.1%}\n"
            f"The spike disappears with sub-bin noise.\n\n"
            f"[red]DO NOT trust Memory Bank prime frequencies![/]",
            title="🚫 VERDICT: ARTIFACT",
            border_style="red"
        ))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: SFF comparison
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for (jitter_frac, r), color in zip(results.items(), colors):
        ax.plot(tau_values, r['sff'], label=r['label'], color=color,
                linewidth=2 if jitter_frac in [0, 1.0] else 0.8,
                alpha=1.0 if jitter_frac in [0, 1.0] else 0.5)

    ax.axvline(x=2*np.pi, color='red', linestyle='--', alpha=0.5, label='2π')
    ax.set_xlabel('τ')
    ax.set_ylabel('SFF K(τ)')
    ax.set_title('SFF vs Jitter Level')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)

    # Right: Peak retention curve
    ax = axes[1]
    jitter_fracs = list(results.keys())
    retentions = [results[j]['peak'] / original_peak for j in jitter_fracs]
    ax.plot(jitter_fracs, retentions, 'bo-', markersize=10, linewidth=2)
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='70% threshold')
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='30% threshold')
    ax.fill_between([0, 1], [0.7, 0.7], [1, 1], alpha=0.2, color='green', label='Real Physics')
    ax.fill_between([0, 1], [0, 0], [0.3, 0.3], alpha=0.2, color='red', label='Artifact')
    ax.set_xlabel('Jitter (fraction of bin width)')
    ax.set_ylabel('Peak Retention')
    ax.set_title('2π Peak Survival vs Jitter')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/jitter_test.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]📊 Plot saved: reports/jitter_test.png[/]")


if __name__ == "__main__":
    run_jitter_test()
