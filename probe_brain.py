#!/usr/bin/env python3
"""
PROBE BRAIN: Fourier Analysis of Memory Bank

We trained MemoryBankGPT with 4 memory slots.
Now we analyze: Did the model discover prime rhythms?

Key frequencies to search:
- 2π ≈ 6.283 (lattice resonance)
- 6.644 (the Monster - beat frequency)
- m·ln(p) (prime orbits)

If memory vectors contain these frequencies → AI discovered arithmetic!
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


# Prime dictionary for matching
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
PRIME_FREQS = {}
for p in PRIMES:
    for m in range(1, 10):
        PRIME_FREQS[f"{m}ln({p})"] = m * np.log(p)

# Add 2πk
for k in range(1, 10):
    PRIME_FREQS[f"{k}×2π"] = k * 2 * np.pi

# The Monster
PRIME_FREQS["MONSTER"] = 6.644


def load_model():
    """Load trained MemoryBankGPT"""
    try:
        ckpt = torch.load('out/memory_bank_best.pt', weights_only=False)
        console.print("[green]✅ Loaded: out/memory_bank_best.pt[/]")
        return ckpt
    except FileNotFoundError:
        try:
            ckpt = torch.load('out/memory_bank_final.pt', weights_only=False)
            console.print("[yellow]⚠️ Loaded: out/memory_bank_final.pt (not best)[/]")
            return ckpt
        except FileNotFoundError:
            console.print("[red]❌ No trained model found! Run train_memory_bank.py first.[/]")
            return None


def analyze_memory_fourier(memory_vectors):
    """
    Analyze memory vectors via Fourier transform.
    Look for dominant frequencies.
    """
    n_slots, dim = memory_vectors.shape
    console.print(f"\n[bold]Memory Shape: {n_slots} slots × {dim} dimensions[/]")

    results = []

    for slot_idx in range(n_slots):
        vec = memory_vectors[slot_idx]

        # Fourier transform
        fft_vals = np.abs(fft(vec))
        freqs = fftfreq(dim, d=1.0)  # Normalized frequencies

        # Only positive frequencies
        pos_mask = freqs > 0
        fft_pos = fft_vals[pos_mask]
        freq_pos = freqs[pos_mask]

        # Scale frequencies to match our tau scale (heuristic: multiply by 2π)
        # This is because memory dimension maps to "spectral bandwidth"
        scaled_freqs = freq_pos * 2 * np.pi * dim / 10  # Heuristic scaling

        # Find peaks
        peaks, props = find_peaks(fft_pos, height=np.mean(fft_pos) * 2)

        # Store results
        results.append({
            'slot': slot_idx,
            'fft': fft_pos,
            'freqs': scaled_freqs,
            'peaks': peaks,
            'peak_freqs': scaled_freqs[peaks] if len(peaks) > 0 else [],
            'peak_heights': fft_pos[peaks] if len(peaks) > 0 else [],
            'dominant_freq': scaled_freqs[np.argmax(fft_pos)],
            'energy': np.sum(fft_pos ** 2)
        })

    return results


def match_to_primes(freq):
    """Find best match in prime dictionary"""
    best_label = "???"
    min_err = float('inf')

    for label, f_theo in PRIME_FREQS.items():
        err = abs(freq - f_theo)
        if err < min_err:
            min_err = err
            best_label = label

    return best_label, min_err


def probe():
    console.print(Panel.fit(
        "[bold red]🧠⛏️ PROBING THE BRAIN[/]\n"
        "Searching for prime rhythms in memory...",
        title="AI ARCHAEOLOGY"
    ))

    # Load model
    ckpt = load_model()
    if ckpt is None:
        return

    # Extract memory vectors
    state_dict = ckpt['model']

    # Find memory parameter
    memory_key = None
    for key in state_dict.keys():
        if 'memory_bank.memory' in key and 'proj' not in key:
            memory_key = key
            break

    if memory_key is None:
        console.print("[red]❌ Memory bank not found in model![/]")
        return

    memory_vectors = state_dict[memory_key].cpu().numpy()
    console.print(f"[cyan]Found memory: {memory_key}[/]")

    # Analyze
    results = analyze_memory_fourier(memory_vectors)

    # Display results
    console.print("\n")
    table = Table(title="🔬 MEMORY SLOT ANALYSIS")
    table.add_column("Slot", style="bold")
    table.add_column("Dominant Freq", justify="right")
    table.add_column("Best Match", style="green")
    table.add_column("Error", justify="right")
    table.add_column("Energy", justify="right")
    table.add_column("Verdict", style="bold")

    found_primes = []
    found_monster = False

    for r in results:
        dom_freq = r['dominant_freq']
        best_match, err = match_to_primes(dom_freq)

        verdict = "[grey]Noise[/]"
        if err < 0.5:
            if "ln" in best_match:
                verdict = "[green]PRIME![/]"
                found_primes.append(best_match)
            elif "2π" in best_match:
                verdict = "[cyan]Lattice[/]"
            elif best_match == "MONSTER":
                verdict = "[bold red]MONSTER![/]"
                found_monster = True

        table.add_row(
            f"Slot {r['slot']}",
            f"{dom_freq:.3f}",
            best_match,
            f"{err:.3f}",
            f"{r['energy']:.1f}",
            verdict
        )

    console.print(table)

    # Final verdict
    console.print("\n")
    if found_primes or found_monster:
        console.print(Panel.fit(
            f"[bold green]🎉 AI DISCOVERED ARITHMETIC![/]\n\n"
            f"Found primes: {', '.join(found_primes) if found_primes else 'None'}\n"
            f"Found Monster (6.644): {'YES!' if found_monster else 'No'}\n\n"
            f"[green]The neural network learned prime rhythms from data![/]",
            title="🏆 DISCOVERY",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[yellow]No clear prime signatures found.[/]\n\n"
            "Possible reasons:\n"
            "- Model needs more training\n"
            "- Memory slots learned different features\n"
            "- Frequency scaling needs adjustment",
            title="⚠️ INCONCLUSIVE",
            border_style="yellow"
        ))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, r in enumerate(results):
        ax = axes[i // 2, i % 2]
        ax.plot(r['freqs'], r['fft'], 'b-', linewidth=0.5)

        # Mark peaks
        if len(r['peaks']) > 0:
            ax.scatter(r['peak_freqs'], r['peak_heights'], c='red', s=30, zorder=5)

        # Mark known frequencies
        for label, f in [("2π", 2*np.pi), ("Monster", 6.644)]:
            ax.axvline(x=f, color='green', linestyle='--', alpha=0.5, label=label)

        ax.set_xlabel('Scaled Frequency')
        ax.set_ylabel('FFT Magnitude')
        ax.set_title(f'Memory Slot {i} (dominant: {r["dominant_freq"]:.2f})')
        ax.set_xlim(0, 15)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/brain_probe.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]📊 Plot saved: reports/brain_probe.png[/]")

    # Also analyze raw memory vectors directly
    console.print("\n[bold]Raw Memory Vector Statistics:[/]")
    for i, vec in enumerate(memory_vectors):
        console.print(f"  Slot {i}: mean={vec.mean():.4f}, std={vec.std():.4f}, "
                     f"min={vec.min():.4f}, max={vec.max():.4f}")


if __name__ == "__main__":
    probe()
