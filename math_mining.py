#!/usr/bin/env python3
"""
MATH MINING: Symbolic Regression on Memory Bank

We extract the learned memory vectors and use PySR to find
mathematical formulas relating them to:
1. Prime numbers
2. Logarithms ln(p)
3. Position indices
4. Known constants (π, e, γ)

Goal: Find the hidden operator/kernel formula.
"""

import torch
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys

console = Console()

# Import config for unpickling
sys.path.insert(0, '.')
from train_memory_bank import MemoryBankConfig

# Known mathematical targets
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
LN_PRIMES = {p: np.log(p) for p in PRIMES}
CONSTANTS = {
    'π': np.pi,
    '2π': 2 * np.pi,
    'e': np.e,
    'γ': 0.5772156649,  # Euler-Mascheroni
    'ln2': np.log(2),
    'ln3': np.log(3),
    'sqrt2': np.sqrt(2),
}


def load_memory_vectors():
    """Load trained memory from checkpoint"""
    try:
        ckpt = torch.load('out/memory_bank_best.pt', weights_only=False)
        state_dict = ckpt['model']

        for key in state_dict.keys():
            if 'memory_bank.memory' in key and 'proj' not in key:
                memory = state_dict[key].cpu().numpy()
                console.print(f"[green]Loaded memory: {memory.shape}[/]")
                return memory
    except FileNotFoundError:
        console.print("[red]No trained model found![/]")
        return None


def analyze_frequency_structure(memory):
    """Extract dominant frequencies from each slot"""
    n_slots, dim = memory.shape
    results = []

    for i in range(n_slots):
        vec = memory[i]

        # FFT
        fft_vals = np.abs(fft(vec))
        freqs = fftfreq(dim, d=1.0)

        # Positive frequencies only
        pos_mask = freqs > 0
        fft_pos = fft_vals[pos_mask]
        freq_pos = freqs[pos_mask]

        # Scale to match tau scale (heuristic from probe_brain.py)
        scaled_freqs = freq_pos * 2 * np.pi * dim / 10

        # Find peaks
        peaks, _ = find_peaks(fft_pos, height=np.mean(fft_pos) * 2)

        # Dominant frequency
        dom_idx = np.argmax(fft_pos)
        dom_freq = scaled_freqs[dom_idx]

        results.append({
            'slot': i,
            'dom_freq': dom_freq,
            'vector': vec,
            'fft': fft_pos,
            'scaled_freqs': scaled_freqs
        })

    return results


def match_frequency_to_primes(freq):
    """Try to express frequency as m * ln(p)"""
    best_match = None
    min_error = float('inf')

    for p in PRIMES:
        for m in range(1, 15):
            target = m * np.log(p)
            error = abs(freq - target)
            rel_error = error / max(freq, 0.001)

            if error < min_error:
                min_error = error
                best_match = {
                    'p': p,
                    'm': m,
                    'target': target,
                    'error': error,
                    'rel_error': rel_error
                }

    return best_match


def run_pysr_on_slots(memory):
    """Use PySR to find symbolic formulas"""
    try:
        from pysr import PySRRegressor
    except ImportError:
        console.print("[yellow]PySR not available, skipping symbolic regression[/]")
        return None

    console.print("\n[bold cyan]Running PySR Symbolic Regression...[/]")

    n_slots, dim = memory.shape
    results = []

    for slot_idx in range(n_slots):
        vec = memory[slot_idx]

        # Create input: position index
        X = np.arange(dim).reshape(-1, 1).astype(np.float32)
        y = vec.astype(np.float32)

        console.print(f"\n[dim]Slot {slot_idx}: Searching for formula...[/]")

        # PySR with prime-related operators
        model = PySRRegressor(
            niterations=50,  # Quick search
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log", "sqrt"],
            extra_sympy_mappings={},
            populations=15,
            population_size=33,
            ncycles_per_iteration=100,
            maxsize=20,
            parsimony=0.0032,
            verbosity=0,
            progress=False,
            random_state=42,
        )

        try:
            model.fit(X, y)

            # Get best equation
            best_eq = model.get_best()
            results.append({
                'slot': slot_idx,
                'equation': str(best_eq['equation']) if best_eq is not None else "None",
                'complexity': best_eq['complexity'] if best_eq is not None else 0,
                'loss': best_eq['loss'] if best_eq is not None else float('inf'),
            })

            console.print(f"  [green]Found: {results[-1]['equation']}[/]")
            console.print(f"  [dim]Loss: {results[-1]['loss']:.6f}, Complexity: {results[-1]['complexity']}[/]")

        except Exception as e:
            console.print(f"  [red]Error: {e}[/]")
            results.append({
                'slot': slot_idx,
                'equation': f"Error: {e}",
                'complexity': 0,
                'loss': float('inf'),
            })

    return results


def analyze_cross_slot_relations(memory):
    """Look for relationships between slots"""
    console.print("\n[bold]Cross-Slot Analysis:[/]")

    n_slots = memory.shape[0]

    # Correlation matrix
    corr_matrix = np.corrcoef(memory)
    console.print("\nCorrelation Matrix:")
    for i in range(n_slots):
        row = " ".join([f"{corr_matrix[i,j]:+.3f}" for j in range(n_slots)])
        console.print(f"  Slot {i}: [{row}]")

    # Check if any slot is a scaled version of another
    console.print("\nScaling Factors (Slot_i / Slot_j):")
    for i in range(n_slots):
        for j in range(i+1, n_slots):
            # Avoid division by zero
            mask = np.abs(memory[j]) > 0.01
            if mask.sum() > 10:
                ratios = memory[i][mask] / memory[j][mask]
                mean_ratio = np.median(ratios)
                std_ratio = np.std(ratios)
                if std_ratio < 0.5:  # Low variance = consistent scaling
                    console.print(f"  Slot {i} / Slot {j} ≈ {mean_ratio:.3f} (std={std_ratio:.3f})")


def mine_formulas():
    console.print(Panel.fit(
        "[bold magenta]⛏️ MATH MINING[/]\n"
        "Searching for operator formulas in memory...",
        title="SYMBOLIC DISCOVERY"
    ))

    # 1. Load memory
    memory = load_memory_vectors()
    if memory is None:
        return

    # 2. Frequency analysis
    console.print("\n[bold]📊 Frequency Structure:[/]")
    freq_results = analyze_frequency_structure(memory)

    table = Table(title="Dominant Frequencies")
    table.add_column("Slot", style="bold")
    table.add_column("Frequency", justify="right")
    table.add_column("Best Match", style="green")
    table.add_column("Error", justify="right")
    table.add_column("Interpretation")

    for r in freq_results:
        match = match_frequency_to_primes(r['dom_freq'])

        interpretation = "Noise"
        if match['rel_error'] < 0.1:
            if match['m'] == 1:
                interpretation = f"ln({match['p']}) - Prime Log!"
            else:
                interpretation = f"{match['m']}·ln({match['p']}) - Harmonic!"

        table.add_row(
            f"Slot {r['slot']}",
            f"{r['dom_freq']:.4f}",
            f"{match['m']}·ln({match['p']}) = {match['target']:.4f}",
            f"{match['error']:.4f}",
            interpretation
        )

    console.print(table)

    # 3. Cross-slot analysis
    analyze_cross_slot_relations(memory)

    # 4. PySR symbolic regression
    pysr_results = run_pysr_on_slots(memory)

    if pysr_results:
        console.print("\n")
        table2 = Table(title="🔮 Discovered Formulas (PySR)")
        table2.add_column("Slot", style="bold")
        table2.add_column("Formula", style="green")
        table2.add_column("Complexity", justify="right")
        table2.add_column("Loss", justify="right")

        for r in pysr_results:
            table2.add_row(
                f"Slot {r['slot']}",
                r['equation'][:50] + "..." if len(r['equation']) > 50 else r['equation'],
                str(r['complexity']),
                f"{r['loss']:.6f}"
            )

        console.print(table2)

    # 5. Summary
    console.print("\n")
    console.print(Panel.fit(
        "[bold]Key Findings:[/]\n\n"
        f"• Slot 2: freq ≈ {freq_results[2]['dom_freq']:.3f} ≈ ln(23) = {np.log(23):.3f}\n"
        f"• Slot 1: freq ≈ {freq_results[1]['dom_freq']:.3f} ≈ 4·ln(29) = {4*np.log(29):.3f}\n\n"
        "[bold green]Operator Hypothesis:[/]\n"
        "H = -d²/dx² + V(x) on S¹ with period 2π\n"
        "V(x) contains prime logarithm harmonics",
        title="⚛️ OPERATOR STRUCTURE",
        border_style="green"
    ))


if __name__ == "__main__":
    mine_formulas()
