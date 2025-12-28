#!/usr/bin/env python3
"""
MATH MINING: Symbolic Regression on Memory Bank (FIXED VERSION)

We extract the learned memory vectors and use PySR to find
mathematical formulas. Reports RAW data with statistical tests.

NO arbitrary scaling. NO hardcoded conclusions.
"""

import torch
import numpy as np
from scipy.fft import fft, fftfreq
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys

console = Console()

# Import config for unpickling
sys.path.insert(0, '.')
from train_memory_bank import MemoryBankConfig


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


def analyze_fft_raw(vec):
    """Analyze vector via FFT. Returns RAW frequencies (cycles/index)."""
    dim = len(vec)
    fft_vals = np.abs(fft(vec))
    freqs = fftfreq(dim, d=1.0)

    pos_mask = freqs > 0
    fft_pos = fft_vals[pos_mask]
    freq_pos = freqs[pos_mask]

    dom_idx = np.argmax(fft_pos)
    return {
        'dom_freq_raw': freq_pos[dom_idx],
        'dom_power': fft_pos[dom_idx],
        'fft': fft_pos,
        'freqs': freq_pos
    }


def null_hypothesis_test(memory, n_random=1000):
    """Test if memory structure is statistically significant."""
    n_slots, dim = memory.shape

    # Trained memory powers
    trained_powers = [analyze_fft_raw(memory[i])['dom_power'] for i in range(n_slots)]
    avg_trained = np.mean(trained_powers)

    # Random baseline
    random_powers = []
    for _ in range(n_random):
        random_vec = np.random.randn(dim) * memory.std()
        random_powers.append(analyze_fft_raw(random_vec)['dom_power'])

    random_powers = np.array(random_powers)
    p_value = np.mean(random_powers >= avg_trained)
    z_score = (avg_trained - np.mean(random_powers)) / np.std(random_powers)

    return {
        'p_value': p_value,
        'z_score': z_score,
        'trained_avg': avg_trained,
        'random_mean': np.mean(random_powers),
        'random_std': np.std(random_powers)
    }


def run_pysr_on_slots(memory):
    """Use PySR to find symbolic formulas"""
    try:
        from pysr import PySRRegressor
    except ImportError:
        console.print("[yellow]PySR not available[/]")
        return None

    console.print("\n[bold]Running PySR Symbolic Regression...[/]")

    n_slots, dim = memory.shape
    results = []

    for slot_idx in range(n_slots):
        vec = memory[slot_idx]
        X = np.arange(dim).reshape(-1, 1).astype(np.float32)
        y = vec.astype(np.float32)

        console.print(f"[dim]Slot {slot_idx}: Searching...[/]")

        model = PySRRegressor(
            niterations=50,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log", "sqrt"],
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
            best_eq = model.get_best()
            results.append({
                'slot': slot_idx,
                'equation': str(best_eq['equation']) if best_eq else "None",
                'complexity': best_eq['complexity'] if best_eq else 0,
                'loss': best_eq['loss'] if best_eq else float('inf'),
            })
            console.print(f"  Found: {results[-1]['equation']}")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/]")
            results.append({'slot': slot_idx, 'equation': str(e), 'complexity': 0, 'loss': float('inf')})

    return results


def analyze_cross_slot_relations(memory):
    """Look for relationships between slots"""
    console.print("\n[bold]Cross-Slot Correlation:[/]")
    n_slots = memory.shape[0]
    corr = np.corrcoef(memory)

    for i in range(n_slots):
        row = " ".join([f"{corr[i,j]:+.2f}" for j in range(n_slots)])
        console.print(f"  Slot {i}: [{row}]")


def mine_formulas():
    console.print(Panel.fit(
        "[bold cyan]⛏️ MATH MINING (FIXED)[/]\n"
        "RAW frequencies, no arbitrary scaling",
        title="HONEST ANALYSIS"
    ))

    # Load memory
    memory = load_memory_vectors()
    if memory is None:
        return

    n_slots, dim = memory.shape

    # Null hypothesis test
    console.print("\n[bold]Null Hypothesis Test (1000 random vectors):[/]")
    null = null_hypothesis_test(memory)
    console.print(f"  Trained avg power: {null['trained_avg']:.4f}")
    console.print(f"  Random: {null['random_mean']:.4f} ± {null['random_std']:.4f}")
    console.print(f"  Z-score: {null['z_score']:.2f}")
    console.print(f"  P-value: {null['p_value']:.4f}")

    # FFT analysis with RAW frequencies
    console.print("\n")
    table = Table(title="📊 FFT Analysis (RAW FREQUENCIES)")
    table.add_column("Slot", style="bold")
    table.add_column("Raw Freq", justify="right", style="cyan")
    table.add_column("Period", justify="right")
    table.add_column("Power", justify="right")

    for i in range(n_slots):
        r = analyze_fft_raw(memory[i])
        period = 1.0 / r['dom_freq_raw'] if r['dom_freq_raw'] > 0 else float('inf')
        table.add_row(
            f"Slot {i}",
            f"{r['dom_freq_raw']:.4f}",
            f"{period:.1f} indices",
            f"{r['dom_power']:.4f}"
        )

    console.print(table)

    # Cross-slot analysis
    analyze_cross_slot_relations(memory)

    # PySR
    pysr_results = run_pysr_on_slots(memory)

    if pysr_results:
        console.print("\n")
        table2 = Table(title="🔮 PySR Formulas")
        table2.add_column("Slot", style="bold")
        table2.add_column("Formula", style="green")
        table2.add_column("Loss", justify="right")

        for r in pysr_results:
            eq_str = r['equation'][:45] + "..." if len(r['equation']) > 45 else r['equation']
            table2.add_row(f"Slot {r['slot']}", eq_str, f"{r['loss']:.6f}")

        console.print(table2)

    # Verdict
    console.print("\n")
    if null['p_value'] < 0.05:
        verdict = "[green]SIGNIFICANT[/] - Memory structure is non-random"
    elif null['p_value'] < 0.1:
        verdict = "[yellow]MARGINAL[/] - Weak evidence"
    else:
        verdict = "[red]NOT SIGNIFICANT[/] - Could be random noise"

    console.print(Panel.fit(
        f"[bold]Statistical Verdict:[/] {verdict}\n"
        f"P-value: {null['p_value']:.4f}\n\n"
        f"[dim]Note: RAW frequencies reported (cycles/index).\n"
        f"No arbitrary scaling or hardcoded conclusions.[/]",
        title="📊 RESULT",
        border_style="cyan"
    ))


if __name__ == "__main__":
    mine_formulas()
