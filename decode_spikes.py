import numpy as np
import itertools
from rich.console import Console
from rich.table import Table

console = Console()

def decode_spikes():
    console.print("[bold cyan]🔐 DECODING SPIKES: Searching for Primes[/]")

    # 1. Load Spikes
    try:
        data = np.load("reports/spikes_data.npz")
        tau = data["tau"]
        sff = data["sff"]
        peaks_idx = data["peaks"]

        # Filter strong peaks only
        strong_peaks_mask = sff[peaks_idx] > 10.0
        peak_taus = tau[peaks_idx][strong_peaks_mask]
        peak_heights = sff[peaks_idx][strong_peaks_mask]

        console.print(f"Loaded {len(peak_taus)} strong peaks (SFF > 10.0)")
    except FileNotFoundError:
        console.print("[red]❌ Report file not found. Run analyze_spikes.py first.[/]")
        return

    # 2. Generate Prime Dictionary (The "Periodic Orbits")
    # T_p = m * ln(p)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    harmonics = range(1, 10)

    prime_freqs = {}
    for p in primes:
        for m in harmonics:
            freq = m * np.log(p)
            label = f"{m}ln({p})"
            prime_freqs[label] = freq

    # Add combinations? (ln p + ln q = ln pq)
    # This is covered by ln(composite) if we only took primes, but let's keep simple.

    # Also add standard 2pi harmonics (The "Naive Lattice")
    for k in range(1, 10):
        prime_freqs[f"{k}×2π"] = k * 2 * np.pi

    # 3. Matcher
    table = Table(title="🕵️ SPIKE DECODER RESULTS")
    table.add_column("Observed τ", style="bold yellow", justify="right")
    table.add_column("Height", justify="right")
    table.add_column("Best Match", style="green")
    table.add_column("Error", justify="right")
    table.add_column("Physics", style="italic")

    # Scaling Factor Mystery:
    # Unfolded coordinate u corresponds to Mean Density.
    # Standard theory says physical time T relates to unfolded tau via Heisenberg time.
    # But let's look for DIRECT matches first.

    for t_obs, h in zip(peak_taus, peak_heights):
        best_label = "???"
        min_err = float("inf")

        for label, t_theo in prime_freqs.items():
            err = abs(t_obs - t_theo)
            if err < min_err:
                min_err = err
                best_label = label

        # Check relative error
        rel_err = min_err / t_obs

        match_str = f"{best_label}"
        err_str = f"{min_err:.4f}"

        physics_comment = "Noise?"
        if min_err < 0.05:
            if "π" in best_label:
                physics_comment = "Mean Spacing (Lattice)"
            else:
                physics_comment = "Prime Orbit!"

        if h > 50: # Highlight the monsters
            table.add_row(f"{t_obs:.3f}", f"[bold magenta]{h:.1f}[/]", match_str, err_str, physics_comment)
        elif min_err < 0.05:
            table.add_row(f"{t_obs:.3f}", f"{h:.1f}", match_str, err_str, physics_comment)

    console.print(table)

    # 4. Global Scaling Check
    # Maybe we missed a factor of 2pi or something?
    # Try to find alpha such that Tau = alpha * ln(p)
    console.print("\n[bold]🔍 Global Scaling Check:[/]")
    # Take the biggest non-2pi peak
    candidates = [t for t, label in zip(peak_taus, [best_label]*len(peak_taus)) if abs(t - 2*np.pi) > 0.5]
    if candidates:
        target = candidates[0] # First strong anomaly
        console.print(f"Checking anomaly at τ={target:.3f}...")
        for p in primes:
            ratio = target / np.log(p)
            console.print(f"  Could it be scaled ln({p})? Factor = {ratio:.4f}")

if __name__ == "__main__":
    decode_spikes()
