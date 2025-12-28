import torch
import numpy as np
from scipy.signal import find_peaks
from rich.console import Console
from rich.table import Table

console = Console()

def analyze_spikes():
    console.print("[bold yellow]🔍 SCANNING FOR RESONANCE SPIKES (The Comb)[/]")

    # 1. Load ALL Data
    val_data = torch.load('data/val.pt', weights_only=False)
    bin_centers = np.load('data/bin_centers.npy')

    raw_spacings = []
    for i in range(len(val_data)):
        s = bin_centers[val_data[i].numpy()]
        raw_spacings.extend(s)
    raw_spacings = np.array(raw_spacings)

    # 2. Unfold (Standard)
    # Mean spacing -> 1.0
    spacings = raw_spacings / np.mean(raw_spacings)
    u = np.concatenate([[0], np.cumsum(spacings)])
    N = len(u)

    console.print(f"Data points: {N}")
    console.print(f"Mean spacing: {np.mean(spacings):.6f} (Normalized)")

    # 3. Compute SFF over wide range
    # Tau от 0.5 до 65 (это захватит первые ~10 пиков по 2pi)
    # Шаг должен быть мелким, чтобы не пропустить пик!
    console.print("Computing High-Res SFF...")

    tau_max = 70.0
    resolution = 20000 # Высокое разрешение
    tau_values = np.linspace(0.5, tau_max, resolution)

    # Vectorized SFF
    # K(tau) = |sum(exp(i*tau*u))|^2 / N
    # Чтобы не убить память, делаем чанками по tau

    sff_vals = []
    chunk_size = 1000

    # Используем GPU если есть, для скорости (тут тяжелая тригонометрия)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    u_tensor = torch.tensor(u, device=device, dtype=torch.float32)

    for i in range(0, len(tau_values), chunk_size):
        taus = torch.tensor(tau_values[i:i+chunk_size], device=device, dtype=torch.float32)
        # Broadcasting: [Tau, 1] * [1, N] = [Tau, N]
        phases = taus.unsqueeze(1) * u_tensor.unsqueeze(0)
        # Sum over N
        complex_sum = torch.exp(1j * phases).sum(dim=1)
        k = (complex_sum.abs() ** 2) / N
        sff_vals.append(k.cpu().numpy())

    sff = np.concatenate(sff_vals)

    # 4. Find Peaks
    # Ищем пики, которые выше фона (фон ~1.0)
    peaks, properties = find_peaks(sff, height=5.0, distance=50) # height=5 отсечет шум

    # 5. Report
    table = Table(title=f"🏔️ DETECTED SPIKES (The Skeleton)")
    table.add_column("Rank")
    table.add_column("Tau (Position)", justify="right")
    table.add_column("Multiple of 2π", justify="right")
    table.add_column("Height (SFF)", justify="right")
    table.add_column("Decay Check", justify="right")

    detected_spikes = []

    for idx, p in enumerate(peaks):
        tau_p = tau_values[p]
        height = sff[p]
        ratio = tau_p / (2 * np.pi)

        # Check integer alignment
        is_integer = abs(ratio - round(ratio)) < 0.05
        mult_str = f"{ratio:.2f} × 2π"
        if is_integer:
            mult_str = f"[bold green]{mult_str}[/]"

        detected_spikes.append(height)

        # Decay info (сравниваем с предыдущим пиком, если есть)
        decay = "-"
        if idx > 0:
            prev_h = detected_spikes[idx-1]
            diff = (height - prev_h) / prev_h
            if diff < -0.05: decay = f"[red]{diff:.1%}[/]" # Падает
            elif diff > 0.05: decay = f"[green]+{diff:.1%}[/]" # Растет
            else: decay = "[grey]Stable[/]"

        table.add_row(str(idx+1), f"{tau_p:.3f}", mult_str, f"{height:.1f}", decay)

    console.print(table)

    # Save plot data for user
    np.savez("reports/spikes_data.npz", tau=tau_values, sff=sff, peaks=peaks)
    console.print("[dim]Data saved to reports/spikes_data.npz[/]")

if __name__ == "__main__":
    analyze_spikes()
