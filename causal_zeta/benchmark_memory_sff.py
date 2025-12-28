#!/usr/bin/env python3
"""
SFF Battle: Amnesiac vs Elephant Memory

Red Corner (Amnesiac): Standard generation, no memory, seq_len=256 max
Blue Corner (Elephant): Warmup + memory-augmented generation

The question: Does infinite context (via RMT memory) improve long-range
spectral structure (SFF plateau)?

```
           SFF K(τ)
              ^
              |        _____ Blue (Elephant) — higher plateau?
              |       /
              |      /_____ Red (Amnesiac) — lower plateau
              |     /
              |____/
              +---------------> τ

  If Blue > Red → Memory captures long-range order!
```
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import SpacingGPT, GPTConfig
from causal_zeta.rmt_causal import CausalInfiniteOracle, OracleConfig

console = Console()


def compute_sff(spacings: np.ndarray, tau_values: np.ndarray = None) -> dict:
    """Compute Spectral Form Factor."""
    u = np.concatenate([[0], np.cumsum(spacings)])
    N = len(u)

    if tau_values is None:
        tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    K_values = np.zeros(len(tau_values))
    for i, tau in enumerate(tau_values):
        phases = np.exp(1j * tau * u)
        K_values[i] = np.abs(np.sum(phases))**2 / N

    # Ramp: τ in [0.5, 3]
    ramp_mask = (tau_values >= 0.5) & (tau_values <= 3.0)
    if np.sum(ramp_mask) > 2:
        tau_ramp = tau_values[ramp_mask]
        K_ramp = K_values[ramp_mask]
        slope, _ = np.polyfit(tau_ramp, K_ramp, 1)
    else:
        slope = 0.0

    # Plateau: τ > 4
    plateau_mask = tau_values > 4.0
    plateau = np.mean(K_values[plateau_mask]) if np.sum(plateau_mask) > 0 else K_values[-1]

    return {
        "tau": tau_values,
        "K": K_values,
        "ramp_slope": slope,
        "plateau": plateau,
        "N": N,
    }


class ElephantGenerator:
    """
    Memory-augmented generator.

    Strategy: Generate in windows, update memory between windows.
    Memory biases the initial hidden state of each new window.
    """

    def __init__(self, model: SpacingGPT, bin_centers: np.ndarray, device):
        self.model = model
        self.bin_centers = bin_centers
        self.device = device
        self.config = model.config

        # Memory state
        self.memory_state = None
        self.memory_alpha = 0.5

    def warmup(self, tokens: torch.Tensor, n_windows: int = 50):
        """
        Warmup memory by processing real data.

        Args:
            tokens: [N,] or [B, T] real tokens
            n_windows: number of windows to process
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        # Initialize memory
        self.memory_state = torch.zeros(1, 1, self.config.n_embd, device=self.device)

        seq_len = self.config.seq_len
        total_tokens = tokens.shape[1]

        windows_processed = 0
        for start in range(0, total_tokens - seq_len, seq_len):
            if windows_processed >= n_windows:
                break

            window = tokens[:, start:start+seq_len].to(self.device)

            # Get hidden states
            hidden_states = self.model.get_hidden_states(window)
            last_hidden = hidden_states[-1]  # [1, T, C]

            # Update memory with EMA
            summary = last_hidden[:, -1:, :]
            self.memory_state = self.memory_alpha * self.memory_state + (1 - self.memory_alpha) * summary

            windows_processed += 1

        console.print(f"[cyan]Elephant warmup: {windows_processed} windows, memory norm = {torch.norm(self.memory_state).item():.4f}[/]")

    def generate_with_memory(self, context: torch.Tensor, n_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate tokens using memory-augmented approach.

        Strategy: Generate in windows, inject memory influence at each window boundary.
        """
        seq_len = self.config.seq_len
        generated = context.clone()

        # Generate in windows
        n_windows = (n_tokens + seq_len - 1) // seq_len

        for w in range(n_windows):
            # How many tokens to generate in this window
            remaining = n_tokens - (generated.shape[1] - context.shape[1])
            if remaining <= 0:
                break
            tokens_this_window = min(seq_len, remaining)

            # Get current context (last seq_len tokens)
            curr_context = generated[:, -seq_len:] if generated.shape[1] >= seq_len else generated

            # Generate tokens for this window
            for _ in range(tokens_this_window):
                # Crop to max seq_len
                idx_cond = generated[:, -seq_len:] if generated.shape[1] > seq_len else generated

                # Get logits
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature

                # MEMORY INJECTION: bias logits based on memory
                # Simple approach: add memory influence to logits
                if self.memory_state is not None:
                    # Project memory to vocab space
                    mem_flat = self.memory_state.squeeze()  # [C]
                    mem_bias = torch.matmul(mem_flat, self.model.lm_head.weight.T)  # [vocab]
                    # Soft influence (scaled down)
                    logits = logits + 0.1 * mem_bias.unsqueeze(0)

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, idx_next], dim=1)

            # Update memory at window boundary
            if self.memory_state is not None:
                window_tokens = generated[:, -tokens_this_window:]
                hidden_states = self.model.get_hidden_states(window_tokens)
                summary = hidden_states[-1][:, -1:, :]
                self.memory_state = self.memory_alpha * self.memory_state + (1 - self.memory_alpha) * summary

        return generated[:, context.shape[1]:]  # Return only generated part


def run_battle(model, val_data, bin_centers, args, device):
    """
    Epic battle: Amnesiac vs Elephant!
    """
    console.print(Panel.fit(
        "[bold red]🥊 SFF BATTLE 🥊[/]\n"
        "[red]Red Corner: Amnesiac (no memory)[/]\n"
        "[blue]Blue Corner: Elephant (∞ memory)[/]",
        title="FIGHT!"
    ))

    # Common parameters
    n_traj = args.n_traj
    traj_len = args.traj_len
    context_len = args.context_len

    # Prepare contexts from validation data
    contexts = []
    for i in range(min(n_traj, len(val_data))):
        ctx = val_data[i:i+1, :context_len].to(device)
        contexts.append(ctx)

    tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    # =========================================================================
    # RED CORNER: Amnesiac (standard generation)
    # =========================================================================
    console.print("\n[bold red]🔴 RED CORNER: Amnesiac[/]")

    red_spacings = []
    for ctx in track(contexts, description="Red generating"):
        with torch.no_grad():
            generated = model.generate(ctx, traj_len, temperature=1.0)
            tokens = generated[0, context_len:].cpu().numpy()
            spacings = bin_centers[tokens]
            red_spacings.extend(spacings)

    red_spacings = np.array(red_spacings)
    sff_red = compute_sff(red_spacings, tau_values)

    console.print(f"  N spacings: {len(red_spacings)}")
    console.print(f"  Ramp slope: {sff_red['ramp_slope']:.6f}")
    console.print(f"  [bold]Plateau: {sff_red['plateau']:.4f}[/]")

    # =========================================================================
    # BLUE CORNER: Elephant (memory-augmented)
    # =========================================================================
    console.print("\n[bold blue]🔵 BLUE CORNER: Elephant[/]")

    # Create elephant generator
    elephant = ElephantGenerator(model, bin_centers, device)

    # Warmup on validation data
    warmup_data = val_data[:args.warmup_windows].reshape(-1)
    elephant.warmup(warmup_data, n_windows=args.warmup_windows)

    blue_spacings = []
    for ctx in track(contexts, description="Blue generating"):
        with torch.no_grad():
            generated = elephant.generate_with_memory(ctx, traj_len, temperature=1.0)
            tokens = generated[0].cpu().numpy()
            spacings = bin_centers[tokens]
            blue_spacings.extend(spacings)

    blue_spacings = np.array(blue_spacings)
    sff_blue = compute_sff(blue_spacings, tau_values)

    console.print(f"  N spacings: {len(blue_spacings)}")
    console.print(f"  Ramp slope: {sff_blue['ramp_slope']:.6f}")
    console.print(f"  [bold]Plateau: {sff_blue['plateau']:.4f}[/]")

    # =========================================================================
    # REAL DATA BASELINE
    # =========================================================================
    console.print("\n[bold green]🟢 REFERENCE: Real Data[/]")

    real_spacings = []
    for i in range(min(n_traj * 2, len(val_data))):
        tokens = val_data[i].numpy()
        spacings = bin_centers[tokens]
        real_spacings.extend(spacings)

    real_spacings = np.array(real_spacings[:len(red_spacings)])
    sff_real = compute_sff(real_spacings, tau_values)

    console.print(f"  N spacings: {len(real_spacings)}")
    console.print(f"  Ramp slope: {sff_real['ramp_slope']:.6f}")
    console.print(f"  [bold]Plateau: {sff_real['plateau']:.4f}[/]")

    # =========================================================================
    # RESULTS
    # =========================================================================
    console.print("\n")

    table = Table(title="🏆 SFF BATTLE RESULTS 🏆")
    table.add_column("Fighter", style="bold")
    table.add_column("Ramp Slope", justify="right")
    table.add_column("Plateau", justify="right", style="bold")
    table.add_column("vs Real", justify="right")

    table.add_row(
        "[green]Real Data[/]",
        f"{sff_real['ramp_slope']:.6f}",
        f"{sff_real['plateau']:.4f}",
        "1.00",
    )
    table.add_row(
        "[red]🔴 Amnesiac[/]",
        f"{sff_red['ramp_slope']:.6f}",
        f"{sff_red['plateau']:.4f}",
        f"{sff_red['plateau']/sff_real['plateau']:.2f}",
    )
    table.add_row(
        "[blue]🔵 Elephant[/]",
        f"{sff_blue['ramp_slope']:.6f}",
        f"{sff_blue['plateau']:.4f}",
        f"{sff_blue['plateau']/sff_real['plateau']:.2f}",
    )

    console.print(table)

    # Winner
    console.print("\n")
    if sff_blue['plateau'] > sff_red['plateau'] * 1.1:
        console.print(Panel.fit(
            "[bold blue]🏆 ELEPHANT WINS! 🏆[/]\n\n"
            f"Plateau improvement: {(sff_blue['plateau']/sff_red['plateau'] - 1)*100:.1f}%\n\n"
            "Memory captures long-range order!",
            title="VICTORY"
        ))
    elif sff_red['plateau'] > sff_blue['plateau'] * 1.1:
        console.print(Panel.fit(
            "[bold red]🏆 AMNESIAC WINS! 🏆[/]\n\n"
            "Memory didn't help. Need different approach.",
            title="RESULT"
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]🤝 DRAW 🤝[/]\n\n"
            "No significant difference. Memory effect is subtle.",
            title="RESULT"
        ))

    # ASCII SFF comparison
    console.print("\n[bold]SFF Curves (ASCII):[/]")
    console.print("```")
    max_K = max(max(sff_real['K']), max(sff_red['K']), max(sff_blue['K']))
    for i in range(0, len(tau_values), 10):
        tau = tau_values[i]
        k_real = int(sff_real['K'][i] / max_K * 30)
        k_red = int(sff_red['K'][i] / max_K * 30)
        k_blue = int(sff_blue['K'][i] / max_K * 30)

        line = f"τ={tau:.2f} |"
        line += "G" * k_real + " " * (30 - k_real) + "|"
        line += "R" * k_red + " " * (30 - k_red) + "|"
        line += "B" * k_blue
        console.print(line)
    console.print("        [Green=Real, Red=Amnesiac, Blue=Elephant]")
    console.print("```")

    return {
        "real": sff_real,
        "red": sff_red,
        "blue": sff_blue,
    }


def main():
    parser = argparse.ArgumentParser(description="SFF Battle: Amnesiac vs Elephant")
    parser.add_argument("--checkpoint", type=str, default="out/best.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-traj", type=int, default=50)
    parser.add_argument("--traj-len", type=int, default=512)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--warmup-windows", type=int, default=100)
    args = parser.parse_args()

    # Load model
    console.print(f"[cyan]Loading: {args.checkpoint}[/]")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = SpacingGPT(config)
    model.load_state_dict(ckpt["model"])

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()
    console.print(f"[green]Model on {device}[/]")

    # Load data
    data_path = Path(args.data_dir)
    val_data = torch.load(data_path / "val.pt", weights_only=False)
    bin_centers = np.load(data_path / "bin_centers.npy")

    # FIGHT!
    results = run_battle(model, val_data, bin_centers, args, device)

    console.print("\n[bold green]Battle complete![/]")


if __name__ == "__main__":
    main()
