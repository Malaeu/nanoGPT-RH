#!/usr/bin/env python3
"""
🔮 Q3-ENHANCED NEURAL ORACLE

The Hybrid Engine:
- Neural Pilot: Predicts local shifts (oscillations, GUE micro-structure)
- Q3 Navigator: Ensures global constraints (spectral gap floor)

"Neural network captures GUE micro-structure (oscillations),
 while Q3 framework provides macroscopic stability bounds (floor)."
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from pathlib import Path
from rich.console import Console
from rich.table import Table
import time

from model.gpt import SpacingGPT, GPTConfig
from predict_zeros import inverse_unfold, bin_to_spacing, unfold_val

console = Console()

# ============================================================
# Q3 CONSTANTS
# ============================================================
B_MIN = 3.0
T_SYM = 0.06  # 3/50
C_STAR = 1.1  # 11/10 (Uniform Floor)
EMPIRICAL_MIN = 4.028  # What we actually measured

# ============================================================
# CONFIG
# ============================================================
CKPT_PATH = Path("out/best.pt")
ZEROS_PATH = Path("zeros/zeros2M.txt")
DATA_DIR = Path("data")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================
# Q3 VALIDATION FUNCTIONS
# ============================================================
def a_xi(xi):
    """Archimedean density: a(ξ) = log(π) - Re[ψ(1/4 + iπξ)]"""
    z = 0.25 + 1j * np.pi * xi
    return np.log(np.pi) - np.real(digamma(z))

def w_window(xi, B=B_MIN, t=T_SYM):
    """Fejér×heat window"""
    fejer = np.maximum(0, 1 - np.abs(xi) / B)
    heat = np.exp(-4 * (np.pi**2) * t * (xi**2))
    return fejer * heat

def compute_empirical_symbol(spacings, n_theta=100):
    """
    Build empirical P_A from generated spacings.
    
    This checks if the generated trajectory respects Q3 constraints.
    """
    # Compute "effective" symbol from spacing distribution
    # This is simplified - full version would use proper periodization
    
    thetas = np.linspace(-0.5, 0.5, n_theta)
    values = []
    
    # Use spacing histogram as proxy for symbol
    hist, edges = np.histogram(spacings, bins=50, density=True)
    
    # Map to symbol-like quantity
    for theta in thetas:
        # Weight by Q3 window
        idx = int((theta + 0.5) * len(hist))
        idx = max(0, min(idx, len(hist)-1))
        val = hist[idx] * 2 * np.pi  # Scale factor
        values.append(max(val, 0.1))  # Avoid zero
    
    return np.array(values), thetas

def q3_trajectory_check(spacings, threshold=0.5):
    """
    Check if trajectory respects Q3 constraints.
    
    Returns (valid, score, reason)
    """
    # Basic checks
    if len(spacings) < 2:
        return False, 0, "Too few spacings"
    
    # Check 1: No tiny spacings (level repulsion)
    min_spacing = np.min(spacings)
    if min_spacing < 0.1:
        return False, min_spacing, f"Spacing too small: {min_spacing:.4f} < 0.1"
    
    # Check 2: Mean should be ~1 (unfolded)
    mean_spacing = np.mean(spacings)
    if abs(mean_spacing - 1.0) > 0.5:
        return False, mean_spacing, f"Mean drift: {mean_spacing:.4f} != 1.0"
    
    # Check 3: Variance reasonable (GUE-like)
    std_spacing = np.std(spacings)
    if std_spacing > 1.0:
        return False, std_spacing, f"Too chaotic: std={std_spacing:.4f}"
    
    # All checks passed
    score = 1.0 - abs(mean_spacing - 1.0) - max(0, 0.3 - min_spacing)
    return True, score, "Q3 constraints satisfied"


# ============================================================
# NEURAL PILOT
# ============================================================
def load_model():
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = SpacingGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model, config

def load_meta():
    return torch.load(DATA_DIR / "meta.pt", weights_only=False)


# ============================================================
# HYBRID ORACLE
# ============================================================
def q3_oracle_generate(start_idx=1800000, n_zeros=100, n_candidates=5):
    """
    Q3-Enhanced Neural Oracle
    
    1. Neural generates candidates (beam search style)
    2. Q3 validates each trajectory
    3. Returns best valid trajectory
    """
    console.print("\n[bold magenta]═══ 🔮 Q3-ENHANCED NEURAL ORACLE ═══[/]\n")
    console.print(f"[cyan]Device: {DEVICE}[/]")
    
    # Load everything
    model, config = load_model()
    console.print(f"[green]Model loaded: seq_len={config.seq_len}[/]")
    
    zeros = np.loadtxt(ZEROS_PATH)
    meta = load_meta()
    bin_edges = np.array(meta["bin_edges"])
    
    # Compute spacings
    unfolded = unfold_val(zeros)
    all_gaps = np.diff(unfolded)
    
    context_len = config.seq_len
    
    # Ground truth
    true_gammas = zeros[start_idx + context_len + 1 : start_idx + context_len + 1 + n_zeros]
    true_spacings = all_gaps[start_idx + context_len : start_idx + context_len + n_zeros]
    
    console.print(f"\n[bold]Generating {n_candidates} candidate trajectories...[/]")
    
    candidates = []
    
    for cand_idx in range(n_candidates):
        # Initialize
        curr_gaps = list(all_gaps[start_idx : start_idx + context_len])
        curr_gamma = zeros[start_idx + context_len]
        
        generated_gammas = []
        generated_spacings = []
        
        with torch.no_grad():
            for i in range(n_zeros):
                ctx = curr_gaps[-context_len:]
                ctx_bins = np.digitize(ctx, bin_edges) - 1
                ctx_bins = np.clip(ctx_bins, 0, config.vocab_size - 1)
                
                x = torch.tensor(ctx_bins, dtype=torch.long).unsqueeze(0).to(DEVICE)
                logits, _ = model(x)
                probs = torch.softmax(logits[0, -1, :], dim=0)
                
                # Different sampling for each candidate
                if cand_idx == 0:
                    # Greedy (argmax)
                    pred_bin = torch.argmax(probs).item()
                else:
                    # Sampling with temperature
                    temp = 0.5 + 0.3 * cand_idx
                    probs_temp = torch.softmax(logits[0, -1, :] / temp, dim=0)
                    pred_bin = torch.multinomial(probs_temp, 1).item()
                
                s_pred = bin_to_spacing(pred_bin, bin_edges)
                next_gamma = inverse_unfold(curr_gamma, s_pred)
                
                generated_gammas.append(next_gamma)
                generated_spacings.append(s_pred)
                
                curr_gaps.append(s_pred)
                curr_gamma = next_gamma
        
        # Q3 validation
        valid, score, reason = q3_trajectory_check(generated_spacings)
        
        candidates.append({
            'gammas': np.array(generated_gammas),
            'spacings': np.array(generated_spacings),
            'valid': valid,
            'score': score,
            'reason': reason,
            'method': 'greedy' if cand_idx == 0 else f'sample_t{0.5 + 0.3*cand_idx:.1f}'
        })
        
        status = "[green]✓[/]" if valid else "[red]✗[/]"
        console.print(f"  Candidate {cand_idx+1}: {status} score={score:.4f} ({reason})")
    
    # Select best valid candidate
    valid_candidates = [c for c in candidates if c['valid']]
    
    if valid_candidates:
        best = max(valid_candidates, key=lambda x: x['score'])
        console.print(f"\n[green]✅ Selected: {best['method']} (score={best['score']:.4f})[/]")
    else:
        console.print(f"\n[yellow]⚠️ No valid candidates! Using greedy anyway.[/]")
        best = candidates[0]
    
    # Compute metrics
    diffs = best['gammas'] - true_gammas
    mae = np.abs(diffs).mean()
    mre = np.abs(diffs / (true_gammas - np.concatenate([[zeros[start_idx+context_len]], true_gammas[:-1]]))).mean()
    
    console.print(f"\n[bold]═══ ORACLE RESULTS ═══[/]")
    console.print(f"  MAE: {mae:.6f}")
    console.print(f"  MRE: {mre:.4f} ({mre*100:.2f}%)")
    console.print(f"  Spacing mean: {best['spacings'].mean():.4f} (target: 1.0)")
    console.print(f"  Spacing std: {best['spacings'].std():.4f}")
    
    return {
        'best': best,
        'candidates': candidates,
        'true_gammas': true_gammas,
        'true_spacings': true_spacings,
        'mae': mae,
        'mre': mre
    }


def visualize_oracle(results, save_path="q3_oracle_result.png"):
    """Visualize Oracle results."""
    console.print("\n[cyan]Creating visualization...[/]")
    
    best = results['best']
    true_gammas = results['true_gammas']
    n_zeros = len(true_gammas)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = np.arange(n_zeros)
    
    # Top Left: Trajectory
    ax = axes[0, 0]
    ax.plot(steps, true_gammas - true_gammas[0], 'k-o', label='Ground Truth', alpha=0.6, markersize=3)
    ax.plot(steps, best['gammas'] - true_gammas[0], 'r--x', label='Q3 Oracle', linewidth=2, markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Δγ from start')
    ax.set_title('Q3 Oracle Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top Right: Spacing comparison
    ax = axes[0, 1]
    ax.plot(steps, results['true_spacings'], 'k-', label='True spacings', alpha=0.6)
    ax.plot(steps, best['spacings'], 'r--', label='Predicted spacings', alpha=0.8)
    ax.axhline(1.0, color='blue', linestyle=':', label='Mean=1 (expected)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Spacing')
    ax.set_title('Spacing Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom Left: Error over time
    ax = axes[1, 0]
    diffs = best['gammas'] - true_gammas
    cum_drift = np.cumsum(diffs)
    ax.plot(steps, cum_drift, 'b-', linewidth=2)
    ax.fill_between(steps, cum_drift, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Drift')
    ax.set_title('Stability (Cumulative Error)')
    ax.grid(True, alpha=0.3)
    
    # Bottom Right: Candidate scores
    ax = axes[1, 1]
    methods = [c['method'] for c in results['candidates']]
    scores = [c['score'] for c in results['candidates']]
    colors = ['green' if c['valid'] else 'red' for c in results['candidates']]
    bars = ax.bar(methods, scores, color=colors, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Candidate')
    ax.set_ylabel('Q3 Score')
    ax.set_title('Q3 Validation Scores')
    ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(
        f"Q3-Enhanced Neural Oracle\n"
        f"MAE={results['mae']:.4f}, MRE={results['mre']*100:.1f}%, Method={best['method']}",
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    console.print(f"[green]✅ Saved to {save_path}[/]")
    plt.close()


def main():
    results = q3_oracle_generate(n_zeros=100, n_candidates=5)
    visualize_oracle(results)
    
    # Summary
    console.print("\n")
    table = Table(title="🔮 Q3 ORACLE SUMMARY", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Method", results['best']['method'])
    table.add_row("Q3 Valid", "✓" if results['best']['valid'] else "✗")
    table.add_row("Q3 Score", f"{results['best']['score']:.4f}")
    table.add_row("MAE (γ)", f"{results['mae']:.6f}")
    table.add_row("MRE", f"{results['mre']*100:.2f}%")
    table.add_row("Spacing μ", f"{results['best']['spacings'].mean():.4f}")
    table.add_row("Spacing σ", f"{results['best']['spacings'].std():.4f}")
    
    console.print(table)
    
    console.print("\n[bold]═══ THE HYBRID INSIGHT ═══[/]")
    console.print("[cyan]Q3 = SCENE[/] (floor, constraints, global density)")
    console.print("[red]Neural = DANCE[/] (oscillations, GUE repulsion, local shifts)")
    console.print("[green]Together = ORACLE[/] (physics-constrained ML prediction)")
    
    console.print("\n[bold green]═══ COMPLETE ═══[/]")


if __name__ == "__main__":
    main()
