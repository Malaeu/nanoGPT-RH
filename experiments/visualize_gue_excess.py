#!/usr/bin/env python3
"""
Visualize the GUE excess correlation findings.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console

console = Console()

# Load results
results_path = Path("results/why_stronger_than_gue.json")
with open(results_path) as f:
    results = json.load(f)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Berry corrections fit
ax1 = axes[0, 0]
bins = results['berry']['bins']
log_heights = [b['log_height'] for b in bins]
corrs = [b['corr1'] for b in bins]

ax1.scatter(log_heights, corrs, s=100, c='red', edgecolor='white', zorder=3)

# Fit line
fit = results['berry']['fit']
x_fit = np.linspace(min(log_heights), max(log_heights), 100)
y_fit = fit['intercept'] + fit['slope'] / x_fit
ax1.plot(x_fit, y_fit, 'b--', linewidth=2, label=f"ρ = {fit['intercept']:.4f} + {fit['slope']:.4f}/log(γ)")

# GUE reference
ax1.axhline(-0.27, color='green', linestyle=':', linewidth=2, label='GUE Theory (-0.27)')
ax1.axhline(-0.29, color='green', linestyle='-.', linewidth=1, alpha=0.7, label='GUE Numerical (-0.29)')

ax1.set_xlabel('log(γ) [Height]', fontsize=12)
ax1.set_ylabel('ρ(1) Autocorrelation', fontsize=12)
ax1.set_title('A. Berry Log Corrections Test', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel B: Riemann vs GUE correlations
ax2 = axes[0, 1]

lags = [1, 2, 3]
riemann_corrs = [-0.339, -0.078, -0.030]  # From experiment
gue_corrs = [-0.292, -0.061, -0.021]  # Numerical GUE

x = np.arange(len(lags))
width = 0.35

bars1 = ax2.bar(x - width/2, riemann_corrs, width, label='Riemann Zeros', color='#E63946', edgecolor='white')
bars2 = ax2.bar(x + width/2, gue_corrs, width, label='GUE (numerical)', color='#40916C', edgecolor='white')

ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_xlabel('Lag k', fontsize=12)
ax2.set_ylabel('Autocorrelation ρ(k)', fontsize=12)
ax2.set_title('B. Riemann vs GUE Correlations', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['k=1', 'k=2', 'k=3'])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add percentage difference
for i, (r, g) in enumerate(zip(riemann_corrs, gue_corrs)):
    diff = (r - g) / abs(g) * 100
    ax2.annotate(f'{diff:.0f}%', xy=(i, min(r, g) - 0.015), ha='center', fontsize=10, color='red')

# Panel C: Height regions (no trend!)
ax3 = axes[1, 0]

regions = ['Low\n(γ < 10⁵)', 'Medium\n(10⁵ < γ < 10⁶)', 'High\n(γ > 10⁶)']
region_corrs = [-0.3391, -0.3389, -0.3390]  # From experiment
errors = [0.0005, 0.0005, 0.0005]  # Approximate errors

ax3.bar(regions, region_corrs, color=['#1B4332', '#40916C', '#74C69D'], edgecolor='white', linewidth=2)
ax3.errorbar(regions, region_corrs, yerr=errors, fmt='none', color='black', capsize=5)

ax3.axhline(-0.27, color='red', linestyle='--', linewidth=2, label='GUE Theory')
ax3.set_ylabel('ρ(1) Autocorrelation', fontsize=12)
ax3.set_title('C. No Height Dependence (Intrinsic Effect!)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.set_ylim(-0.36, -0.25)
ax3.grid(True, alpha=0.3, axis='y')

# Annotation
ax3.annotate('FLAT!\nExcess is\nINTRINSIC', xy=(1, -0.339), xytext=(1.5, -0.32),
            fontsize=11, fontweight='bold', color='#E63946',
            arrowprops=dict(arrowstyle='->', color='#E63946'))

# Panel D: Summary diagram
ax4 = axes[1, 1]
ax4.axis('off')

# Create text summary
summary_text = """
CONCLUSION: Why Riemann > GUE?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Berry's log corrections: SUPPORTED
   • Small 1/log(γ) trend detected
   • R² = 0.52, p = 0.018
   • But asymptotic still -0.34 ≠ -0.27

✗ Finite-height effects: REJECTED
   • Correlations CONSTANT across heights
   • Effect is INTRINSIC, not artifact

~ Number theory: PARTIAL
   • 102σ from shuffled (real correlations)
   • No clear log(prime) periodicities

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY INSIGHT:
Riemann zeros have 16-40% STRONGER correlations
than generic GUE eigenvalues. This is an
INTRINSIC property of ζ(s), not finite-height.

Possible source: Arithmetic corrections from
the explicit formula connecting zeros ↔ primes.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#333333', linewidth=2))

ax4.set_title('D. Experimental Summary', fontsize=13, fontweight='bold')

# Main title
fig.suptitle('Investigation: Why Riemann Zeros Have Stronger Correlations Than GUE?',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_path = Path("reports/figures/why_stronger_than_gue.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
console.print(f"[green]Saved: {output_path}[/]")

plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
console.print(f"[green]Saved: {output_path.with_suffix('.pdf')}[/]")

plt.close()
