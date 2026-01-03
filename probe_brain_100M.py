#!/usr/bin/env python3
"""
PROBE BRAIN 100M: Fourier Analysis of Memory Bank
==================================================

Analyze learned memory vectors from MemoryBankGPT trained on 100M zeros.
Looking for prime number rhythms (Selberg trace formula patterns).

Expected patterns: m·ln(p) frequencies where p is prime
"""

import torch
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
from pathlib import Path
import sys

console = Console()

OUTPUT_DIR = Path("out/memory_bank_100M")
REPORT_DIR = Path("reports/100M")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Import config
sys.path.insert(0, '.')
from train_memory_bank_100M import MemoryBankConfig


def load_model():
    """Load trained MemoryBankGPT from 100M."""
    try:
        ckpt = torch.load(OUTPUT_DIR / 'best.pt', weights_only=False)
        console.print(f"[green]Loaded: {OUTPUT_DIR}/best.pt[/]")
        return ckpt
    except FileNotFoundError:
        try:
            ckpt = torch.load(OUTPUT_DIR / 'final.pt', weights_only=False)
            console.print(f"[yellow]Loaded: {OUTPUT_DIR}/final.pt[/]")
            return ckpt
        except FileNotFoundError:
            console.print("[red]No model found! Run train_memory_bank_100M.py first.[/]")
            return None


def analyze_fft(vec):
    """Analyze vector via FFT. Returns RAW frequencies."""
    dim = len(vec)
    fft_vals = np.abs(fft(vec))
    freqs = fftfreq(dim, d=1.0)

    pos_mask = freqs > 0
    fft_pos = fft_vals[pos_mask]
    freq_pos = freqs[pos_mask]

    dom_idx = np.argmax(fft_pos)
    dom_freq_raw = freq_pos[dom_idx]
    dom_power = fft_pos[dom_idx]

    # Find all significant peaks
    peaks, properties = find_peaks(fft_pos, height=dom_power * 0.3, distance=3)
    peak_freqs = freq_pos[peaks]
    peak_powers = fft_pos[peaks]

    return {
        'dom_freq_raw': dom_freq_raw,
        'dom_power': dom_power,
        'fft': fft_pos,
        'freqs': freq_pos,
        'total_energy': np.sum(fft_pos ** 2),
        'peak_freqs': peak_freqs,
        'peak_powers': peak_powers,
    }


def null_hypothesis_test(memory_vectors, n_random=1000):
    """Compare trained memory to random vectors."""
    n_slots, dim = memory_vectors.shape

    trained_powers = []
    for i in range(n_slots):
        result = analyze_fft(memory_vectors[i])
        trained_powers.append(result['dom_power'])

    avg_trained_power = np.mean(trained_powers)

    random_powers = []
    for _ in range(n_random):
        random_vec = np.random.randn(dim) * memory_vectors.std()
        result = analyze_fft(random_vec)
        random_powers.append(result['dom_power'])

    random_powers = np.array(random_powers)
    p_value = np.mean(random_powers >= avg_trained_power)

    return {
        'p_value': p_value,
        'trained_avg': avg_trained_power,
        'random_mean': np.mean(random_powers),
        'random_std': np.std(random_powers),
        'z_score': (avg_trained_power - np.mean(random_powers)) / np.std(random_powers)
    }


def match_prime_harmonics(freq, dim):
    """
    Try to match frequency to m·ln(p) pattern.

    freq is in cycles per index (0 to 0.5)
    period = 1/freq (in indices)

    We look for matches to m·ln(p) where p is prime.
    """
    if freq <= 0:
        return None, None, None

    period = 1.0 / freq

    # Convert to "physical" frequency by scaling with dimension
    # This gives us the number of oscillations over the full embedding
    phys_freq = freq * dim

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    best_match = None
    best_error = float('inf')
    best_formula = None

    for p in primes:
        ln_p = np.log(p)
        for m in range(1, 20):
            target = m * ln_p
            error = abs(phys_freq - target) / target if target > 0 else float('inf')
            if error < best_error:
                best_error = error
                best_match = target
                best_formula = f"{m}·ln({p})"

    # Also check for π and 2π
    for target, formula in [(np.pi, "π"), (2*np.pi, "2π"), (np.e, "e")]:
        error = abs(phys_freq - target) / target
        if error < best_error:
            best_error = error
            best_match = target
            best_formula = formula

    return best_formula, best_match, best_error


def probe():
    console.print(Panel.fit(
        "[bold cyan]PROBING 100M MEMORY BANK[/]\n"
        "Analyzing learned frequencies for prime rhythms",
        title="🧠"
    ))

    ckpt = load_model()
    if ckpt is None:
        return

    state_dict = ckpt['model']
    config = ckpt.get('config')

    memory_key = None
    for key in state_dict.keys():
        if 'memory_bank.memory' in key and 'proj' not in key:
            memory_key = key
            break

    if memory_key is None:
        console.print("[red]Memory bank not found![/]")
        return

    memory_vectors = state_dict[memory_key].cpu().numpy()
    n_slots, dim = memory_vectors.shape
    console.print(f"[dim]Memory shape: {n_slots} slots × {dim} dimensions[/]")

    # Null hypothesis test
    console.print("\n[bold]Running Null Hypothesis Test...[/]")
    null_result = null_hypothesis_test(memory_vectors)

    console.print(f"  Trained avg power: {null_result['trained_avg']:.4f}")
    console.print(f"  Random mean power: {null_result['random_mean']:.4f} ± {null_result['random_std']:.4f}")
    console.print(f"  Z-score: {null_result['z_score']:.2f}")
    console.print(f"  P-value: {null_result['p_value']:.4f}")

    # Analyze each slot
    console.print("\n")
    table = Table(title="🔬 100M MEMORY SLOT ANALYSIS")
    table.add_column("Slot", style="bold")
    table.add_column("Raw Freq", justify="right", style="cyan")
    table.add_column("Phys Freq", justify="right", style="cyan")
    table.add_column("Power", justify="right")
    table.add_column("Best Match", style="green")
    table.add_column("Error", justify="right")

    results = []
    for i in range(n_slots):
        r = analyze_fft(memory_vectors[i])
        results.append(r)

        phys_freq = r['dom_freq_raw'] * dim
        formula, match, error = match_prime_harmonics(r['dom_freq_raw'], dim)

        table.add_row(
            f"Slot {i}",
            f"{r['dom_freq_raw']:.4f}",
            f"{phys_freq:.3f}",
            f"{r['dom_power']:.4f}",
            formula if formula else "N/A",
            f"{error:.3f}" if error else "N/A"
        )

    console.print(table)

    # Verdict
    console.print("\n")
    if null_result['p_value'] < 0.05:
        verdict_color = "green"
        verdict_text = "SIGNIFICANT"
        verdict_msg = f"Memory structure is statistically significant (p={null_result['p_value']:.4f} < 0.05)"
    elif null_result['p_value'] < 0.1:
        verdict_color = "yellow"
        verdict_text = "MARGINAL"
        verdict_msg = f"Marginal significance (p={null_result['p_value']:.4f})"
    else:
        verdict_color = "red"
        verdict_text = "NOT SIGNIFICANT"
        verdict_msg = f"Could be random noise (p={null_result['p_value']:.4f})"

    console.print(Panel.fit(
        f"[bold {verdict_color}]{verdict_text}[/]\n\n"
        f"{verdict_msg}\n\n"
        f"[dim]100M zeros = 50x more data than original 2M experiment[/]",
        title="📊 STATISTICAL VERDICT",
        border_style=verdict_color
    ))

    # Plot
    n_cols = 4
    n_rows = (n_slots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows))
    axes = axes.flatten() if n_slots > 4 else [axes] if n_slots == 1 else axes

    for i, r in enumerate(results):
        ax = axes[i]
        ax.plot(r['freqs'], r['fft'], 'b-', linewidth=0.8)

        dom_idx = np.argmax(r['fft'])
        ax.scatter([r['freqs'][dom_idx]], [r['fft'][dom_idx]],
                   c='red', s=50, zorder=5, label=f'Peak: {r["dom_freq_raw"]:.4f}')

        ax.set_xlabel('Frequency')
        ax.set_ylabel('FFT Magnitude')
        ax.set_title(f'Memory Slot {i}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_slots, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'100M Memory Bank FFT Analysis (p-value: {null_result["p_value"]:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'brain_probe_100M.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: {REPORT_DIR}/brain_probe_100M.png[/]")

    # Raw stats
    console.print("\n[bold]Raw Memory Vector Statistics:[/]")
    for i, vec in enumerate(memory_vectors):
        console.print(f"  Slot {i}: mean={vec.mean():.4f}, std={vec.std():.4f}, norm={np.linalg.norm(vec):.4f}")


if __name__ == "__main__":
    probe()
