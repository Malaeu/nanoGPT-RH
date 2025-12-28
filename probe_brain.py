#!/usr/bin/env python3
"""
PROBE BRAIN: Fourier Analysis of Memory Bank (FIXED VERSION)

We trained MemoryBankGPT with 4 memory slots.
Now we analyze: Did the model discover any structure?

IMPORTANT: We report RAW frequencies and compare against NULL HYPOTHESIS
(random vectors) to determine statistical significance.
"""

import torch
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
import sys

console = Console()

# Need to import MemoryBankConfig for pickle to work
sys.path.insert(0, '.')
from train_memory_bank import MemoryBankConfig


def load_model():
    """Load trained MemoryBankGPT"""
    try:
        ckpt = torch.load('out/memory_bank_best.pt', weights_only=False)
        console.print("[green]Loaded: out/memory_bank_best.pt[/]")
        return ckpt
    except FileNotFoundError:
        try:
            ckpt = torch.load('out/memory_bank_final.pt', weights_only=False)
            console.print("[yellow]Loaded: out/memory_bank_final.pt (not best)[/]")
            return ckpt
        except FileNotFoundError:
            console.print("[red]No trained model found! Run train_memory_bank.py first.[/]")
            return None


def analyze_fft(vec):
    """
    Analyze a single vector via FFT.
    Returns RAW frequencies (cycles per index).
    """
    dim = len(vec)
    fft_vals = np.abs(fft(vec))
    freqs = fftfreq(dim, d=1.0)

    # Only positive frequencies
    pos_mask = freqs > 0
    fft_pos = fft_vals[pos_mask]
    freq_pos = freqs[pos_mask]

    # Find dominant frequency (index with max power)
    dom_idx = np.argmax(fft_pos)
    dom_freq_raw = freq_pos[dom_idx]  # In cycles per index (0 to 0.5)
    dom_power = fft_pos[dom_idx]

    return {
        'dom_freq_raw': dom_freq_raw,
        'dom_power': dom_power,
        'fft': fft_pos,
        'freqs': freq_pos,
        'total_energy': np.sum(fft_pos ** 2)
    }


def null_hypothesis_test(memory_vectors, n_random=1000):
    """
    Generate random vectors and compare their FFT structure
    to trained memory vectors.

    Returns p-value: probability of seeing such structure by chance.
    """
    n_slots, dim = memory_vectors.shape

    # Get trained memory stats
    trained_powers = []
    for i in range(n_slots):
        result = analyze_fft(memory_vectors[i])
        trained_powers.append(result['dom_power'])

    avg_trained_power = np.mean(trained_powers)

    # Generate random vectors and measure
    random_powers = []
    for _ in range(n_random):
        random_vec = np.random.randn(dim) * memory_vectors.std()
        result = analyze_fft(random_vec)
        random_powers.append(result['dom_power'])

    random_powers = np.array(random_powers)

    # P-value: how often does random beat trained?
    p_value = np.mean(random_powers >= avg_trained_power)

    return {
        'p_value': p_value,
        'trained_avg': avg_trained_power,
        'random_mean': np.mean(random_powers),
        'random_std': np.std(random_powers),
        'z_score': (avg_trained_power - np.mean(random_powers)) / np.std(random_powers)
    }


def probe():
    console.print(Panel.fit(
        "[bold cyan]🧠 PROBING THE BRAIN (FIXED)[/]\n"
        "Analyzing memory vectors with NULL HYPOTHESIS test",
        title="HONEST ANALYSIS"
    ))

    # Load model
    ckpt = load_model()
    if ckpt is None:
        return

    # Extract memory vectors
    state_dict = ckpt['model']

    memory_key = None
    for key in state_dict.keys():
        if 'memory_bank.memory' in key and 'proj' not in key:
            memory_key = key
            break

    if memory_key is None:
        console.print("[red]Memory bank not found in model![/]")
        return

    memory_vectors = state_dict[memory_key].cpu().numpy()
    n_slots, dim = memory_vectors.shape
    console.print(f"[dim]Memory shape: {n_slots} slots × {dim} dimensions[/]")

    # Null hypothesis test
    console.print("\n[bold]Running Null Hypothesis Test (1000 random vectors)...[/]")
    null_result = null_hypothesis_test(memory_vectors)

    console.print(f"  Trained avg power: {null_result['trained_avg']:.4f}")
    console.print(f"  Random mean power: {null_result['random_mean']:.4f} ± {null_result['random_std']:.4f}")
    console.print(f"  Z-score: {null_result['z_score']:.2f}")
    console.print(f"  P-value: {null_result['p_value']:.4f}")

    # Analyze each slot
    console.print("\n")
    table = Table(title="🔬 MEMORY SLOT ANALYSIS (RAW FREQUENCIES)")
    table.add_column("Slot", style="bold")
    table.add_column("Raw Freq", justify="right", style="cyan")
    table.add_column("Period (indices)", justify="right")
    table.add_column("Power", justify="right")
    table.add_column("Energy", justify="right")

    results = []
    for i in range(n_slots):
        r = analyze_fft(memory_vectors[i])
        results.append(r)

        # Period in indices
        period = 1.0 / r['dom_freq_raw'] if r['dom_freq_raw'] > 0 else float('inf')

        table.add_row(
            f"Slot {i}",
            f"{r['dom_freq_raw']:.4f}",
            f"{period:.1f}",
            f"{r['dom_power']:.4f}",
            f"{r['total_energy']:.1f}"
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
        verdict_msg = f"Could be random noise (p={null_result['p_value']:.4f} >= 0.05)"

    console.print(Panel.fit(
        f"[bold {verdict_color}]{verdict_text}[/]\n\n"
        f"{verdict_msg}\n\n"
        f"[dim]Note: RAW frequencies reported.\n"
        f"No arbitrary scaling applied.\n"
        f"Period = 1/freq (in index units)[/]",
        title=f"📊 STATISTICAL VERDICT",
        border_style=verdict_color
    ))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, r in enumerate(results):
        ax = axes[i // 2, i % 2]
        ax.plot(r['freqs'], r['fft'], 'b-', linewidth=0.8)

        # Mark dominant frequency
        dom_idx = np.argmax(r['fft'])
        ax.scatter([r['freqs'][dom_idx]], [r['fft'][dom_idx]],
                   c='red', s=50, zorder=5, label=f'Peak: {r["dom_freq_raw"]:.4f}')

        ax.set_xlabel('Frequency (cycles/index)')
        ax.set_ylabel('FFT Magnitude')
        ax.set_title(f'Memory Slot {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Memory Bank FFT Analysis (p-value: {null_result["p_value"]:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig('reports/brain_probe.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]Plot saved: reports/brain_probe.png[/]")

    # Raw stats
    console.print("\n[bold]Raw Memory Vector Statistics:[/]")
    for i, vec in enumerate(memory_vectors):
        console.print(f"  Slot {i}: mean={vec.mean():.4f}, std={vec.std():.4f}")


if __name__ == "__main__":
    probe()
