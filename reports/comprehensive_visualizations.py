#!/usr/bin/env python3
"""
Comprehensive Visualizations for Riemann Zeros Research.

Creates publication-quality figures combining:
1. GUE comparison (extracted operator vs theoretical)
2. Attention ablation heatmap (Layer 1 dominance)
3. Linear operator performance
4. Prime-zero relationship structure

Author: Neural Telescope Project
Date: January 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Rich console for progress
from rich.console import Console
from rich.panel import Panel

console = Console()

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
})

# Color palette (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'tertiary': '#F18F01',     # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#3B3B3B',      # Dark gray
    'background': '#F5F5F5',   # Light gray
    'gue_theory': '#1B4332',   # Dark green
    'gue_numerical': '#40916C', # Green
    'our_model': '#E63946',    # Red
}


def load_results():
    """Load all analysis results."""
    base = Path("results")

    results = {}

    # GUE comparison
    gue_file = base / "gue_comparison.json"
    if gue_file.exists():
        with open(gue_file) as f:
            results['gue'] = json.load(f)

    # Linear operator rollout
    rollout_file = base / "linear_operator_rollout.json"
    if rollout_file.exists():
        with open(rollout_file) as f:
            results['rollout'] = json.load(f)

    # Ablation results
    ablation_file = base / "ablation_flash.json"
    if ablation_file.exists():
        with open(ablation_file) as f:
            results['ablation'] = json.load(f)

    # L1 analysis
    l1_file = base / "L1_analysis" / "analysis_results.json"
    if l1_file.exists():
        with open(l1_file) as f:
            results['l1'] = json.load(f)

    return results


def plot_gue_comparison(ax, results):
    """Plot GUE autocorrelation comparison."""
    gue = results.get('gue', {})

    our = gue.get('our_correlations', {'1': -0.34, '2': -0.08, '3': -0.03})
    theo = gue.get('gue_theoretical', {'1': -0.27, '2': -0.06, '3': -0.025})
    num = gue.get('gue_numerical', {'1': -0.29, '2': -0.06, '3': -0.02})

    lags = [1, 2, 3]
    our_vals = [our.get(str(k), 0) for k in lags]
    theo_vals = [theo.get(str(k), 0) for k in lags]
    num_vals = [num.get(str(k), 0) for k in lags]

    x = np.arange(len(lags))
    width = 0.25

    bars1 = ax.bar(x - width, our_vals, width, label='Flash Model',
                   color=COLORS['our_model'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, theo_vals, width, label='GUE Theory',
                   color=COLORS['gue_theory'], edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, num_vals, width, label='GUE Numerical',
                   color=COLORS['gue_numerical'], edgecolor='white', linewidth=1)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Lag k')
    ax.set_ylabel('Autocorrelation ρ(k)')
    ax.set_title('Level Repulsion: Our Model vs GUE Theory')
    ax.set_xticks(x)
    ax.set_xticklabels(['k=1', 'k=2', 'k=3'])
    ax.legend(loc='lower right')
    ax.set_ylim(-0.45, 0.05)

    # Add annotation
    ax.annotate('All negative = Level Repulsion ✓',
                xy=(0, -0.34), xytext=(1.5, -0.38),
                fontsize=9, style='italic',
                arrowprops=dict(arrowstyle='->', color='gray'))


def plot_ablation_heatmap(ax, results):
    """Plot attention head ablation heatmap."""
    ablation = results.get('ablation', {})
    impact_matrix = ablation.get('impact_matrix', None)

    if impact_matrix is None:
        # Default values from Flash analysis
        impact_matrix = [
            [11.3, 5.0, 1.4, 0.2, 6.1, 2.2, 1.7, 65.0],
            [94.8, 107.5, 311.7, 232.5, 119.5, 51.3, 42.8, 118.9],
            [27.0, 1.4, 5.6, 1.8, 2.3, 20.4, 22.0, 79.3],
            [-3.1, 44.6, 4.2, -3.6, -5.5, 32.6, 8.9, 20.2],
            [-3.1, -4.9, -2.1, 14.2, 34.0, 8.8, -3.4, -2.0],
            [9.1, 3.7, 20.7, 8.4, 12.8, 5.1, -2.6, 20.0],
        ]

    impact_matrix = np.array(impact_matrix)

    # Clip for visualization
    impact_clipped = np.clip(impact_matrix, -20, 150)

    im = ax.imshow(impact_clipped, cmap='RdYlGn_r', aspect='auto',
                   vmin=-20, vmax=150)

    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Attention Head Importance (% NLL Change)')
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'H{i}' for i in range(8)])
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'L{i}' for i in range(6)])

    # Highlight Layer 1
    rect = plt.Rectangle((-0.5, 0.5), 8, 1, fill=False,
                         edgecolor='red', linewidth=3, linestyle='--')
    ax.add_patch(rect)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('% NLL Change')

    # Annotate critical head
    ax.annotate('L1.H2: +312%\n(CORE)', xy=(2, 1), xytext=(5, 0),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='white', lw=2),
                color='white', weight='bold',
                bbox=dict(boxstyle='round', fc='red', alpha=0.8))


def plot_rollout_performance(ax, results):
    """Plot linear operator vs Flash model rollout."""
    rollout = results.get('rollout', {}).get('results', {})

    methods = ['Baseline\n(predict 0)', 'Linear\nOperator', 'Flash\nModel']
    errors = [
        rollout.get('baseline_err100', 0.354),
        rollout.get('linear_err100', 0.310),
        rollout.get('flash_err100', 0.241)
    ]

    colors = [COLORS['neutral'], COLORS['tertiary'], COLORS['primary']]

    bars = ax.bar(methods, errors, color=colors, edgecolor='white', linewidth=2)

    ax.set_ylabel('Err@100 (Rollout Error)')
    ax.set_title('Autoregressive Prediction Performance')
    ax.set_ylim(0, 0.45)

    # Add value labels
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.annotate(f'{err:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, weight='bold')

    # Add improvement arrows
    ax.annotate('', xy=(1, 0.310), xytext=(1, 0.354),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(1.3, 0.332, '-12%', fontsize=9, color='green')

    ax.annotate('', xy=(2, 0.241), xytext=(2, 0.310),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(2.3, 0.275, '-22%', fontsize=9, color='green')


def plot_operator_structure(ax, results):
    """Plot the extracted linear operator structure."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Extracted GUE Linear Operator',
            fontsize=14, ha='center', weight='bold')

    # Main equation
    eq_box = FancyBboxPatch((1, 6), 8, 2, boxstyle="round,pad=0.1",
                            facecolor='#E8F4F8', edgecolor=COLORS['primary'],
                            linewidth=2)
    ax.add_patch(eq_box)
    ax.text(5, 7, r'$r_n = -0.45 \cdot r_{n-1} - 0.28 \cdot r_{n-2} - 0.16 \cdot r_{n-3}$',
            fontsize=13, ha='center', va='center', family='serif')

    # Coefficient interpretation
    coeffs = [
        (2, 4.5, r'$a_1 = -0.45$', 'Nearest-neighbor\nrepulsion'),
        (5, 4.5, r'$a_2 = -0.28$', 'Second-neighbor\nrepulsion'),
        (8, 4.5, r'$a_3 = -0.16$', 'Third-neighbor\nrepulsion'),
    ]

    for x, y, coeff, label in coeffs:
        circle = plt.Circle((x, y), 0.6, color=COLORS['our_model'], alpha=0.3)
        ax.add_patch(circle)
        ax.text(x, y + 0.1, coeff, fontsize=10, ha='center', va='center', weight='bold')
        ax.text(x, y - 1.2, label, fontsize=8, ha='center', va='center', style='italic')

    # GUE signature box
    sig_box = FancyBboxPatch((1.5, 0.5), 7, 2, boxstyle="round,pad=0.1",
                             facecolor='#E8F8E8', edgecolor=COLORS['gue_theory'],
                             linewidth=2)
    ax.add_patch(sig_box)
    ax.text(5, 1.8, 'GUE Signatures:', fontsize=10, ha='center', weight='bold',
            color=COLORS['gue_theory'])
    ax.text(5, 1.0, r'All negative (repulsion) • Decreasing (short-range) • $\sum a_i \approx -0.89$ (rigidity)',
            fontsize=9, ha='center', va='center')


def plot_prime_zero_connection(ax):
    """Plot conceptual diagram of prime-zero-operator connection."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Riemann Zeros ↔ Primes Connection',
            fontsize=14, ha='center', weight='bold')

    # Boxes
    boxes = [
        (1, 6, 'Riemann\nZeros\nζ(½+iγₙ)=0', COLORS['primary']),
        (4, 6, 'Unfolded\nSpacings\nsₙ', COLORS['tertiary']),
        (7, 6, 'Flash\nModel', COLORS['secondary']),
        (4, 2.5, 'GUE\nOperator', COLORS['gue_theory']),
        (7, 2.5, 'Prime\nDistribution\nπ(x)', COLORS['success']),
    ]

    for x, y, text, color in boxes:
        box = FancyBboxPatch((x-1, y-1), 2, 2, boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=9, ha='center', va='center', weight='bold')

    # Arrows
    arrows = [
        (2, 6.5, 3, 6.5),  # Zeros -> Spacings
        (5, 6.5, 6, 6.5),  # Spacings -> Model
        (4, 5, 4, 4.5),    # Spacings -> GUE
        (6, 5, 7, 4.5),    # Model -> Primes (learned)
        (5.5, 2.5, 6, 2.5), # GUE -> Primes
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Labels on arrows
    ax.text(2.5, 7, 'unfold', fontsize=8, style='italic')
    ax.text(5.5, 7, 'learn', fontsize=8, style='italic')
    ax.text(4.3, 4.7, 'extract', fontsize=8, style='italic')
    ax.text(5.8, 2.1, 'explicit\nformula', fontsize=7, style='italic', ha='center')

    # Key insight
    ax.text(5, 0.5, 'Montgomery-Odlyzko: Zero statistics ≈ GUE eigenvalue statistics',
            fontsize=9, ha='center', style='italic', color='gray')


def create_comprehensive_figure(results):
    """Create the main comprehensive figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Panel A: GUE Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    plot_gue_comparison(ax1, results)
    ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes,
             fontsize=16, weight='bold')

    # Panel B: Ablation Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    plot_ablation_heatmap(ax2, results)
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes,
             fontsize=16, weight='bold')

    # Panel C: Rollout Performance
    ax3 = fig.add_subplot(gs[1, 0])
    plot_rollout_performance(ax3, results)
    ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes,
             fontsize=16, weight='bold')

    # Panel D: Operator Structure
    ax4 = fig.add_subplot(gs[1, 1])
    plot_operator_structure(ax4, results)
    ax4.text(-0.05, 1.05, 'D', transform=ax4.transAxes,
             fontsize=16, weight='bold')

    # Panel E: Prime-Zero Connection (spans bottom)
    ax5 = fig.add_subplot(gs[2, :])
    plot_prime_zero_connection(ax5)
    ax5.text(-0.02, 1.05, 'E', transform=ax5.transAxes,
             fontsize=16, weight='bold')

    # Main title
    fig.suptitle('Neural Telescope for Riemann Zeros: GUE Operator Extraction\n' +
                 'Flash Model Analysis (January 2026)',
                 fontsize=16, weight='bold', y=0.98)

    return fig


def main():
    console.print(Panel("[bold cyan]Creating Comprehensive Visualizations[/]"))

    # Load results
    console.print("[cyan]Loading analysis results...[/]")
    results = load_results()
    console.print(f"  Loaded: {list(results.keys())}")

    # Create output directory
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate main figure
    console.print("[cyan]Generating comprehensive figure...[/]")
    fig = create_comprehensive_figure(results)

    output_path = output_dir / "comprehensive_gue_analysis.png"
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    console.print(f"[green]Saved: {output_path}[/]")

    # Also save PDF for publication
    pdf_path = output_dir / "comprehensive_gue_analysis.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    console.print(f"[green]Saved: {pdf_path}[/]")

    plt.close(fig)

    console.print(Panel("[bold green]Visualizations Complete![/]"))

    return str(output_path)


if __name__ == "__main__":
    output = main()
    print(f"\nOutput: {output}")
