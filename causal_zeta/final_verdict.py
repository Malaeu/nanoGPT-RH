#!/usr/bin/env python3
"""
FINAL VERDICT: The Redemption
Исправленный аудит. Используем WINDOW-BASED логику (которая дала 78%),
но добавляем корректный Placebo контроль.

Мы ищем "Sweet Spot" памяти.
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

# Импортируем базу (чтобы не дублировать код SFF)
from causal_zeta.benchmark_memory_sff import compute_sff, ElephantGenerator
from model.gpt import SpacingGPT

console = Console()

class PlaceboGenerator(ElephantGenerator):
    """
    Тот же код, что у Слона (Elephant), но память — это фикция.
    Мы обновляем её, но заливаем туда ШУМ той же статистики (mean, std),
    что и у реального скрытого состояния.
    """
    def warmup(self, tokens, n_windows=50):
        # Fake warmup: просто крутим счетчик, но память держим случайной
        self.memory_state = torch.randn(1, 1, self.config.n_embd, device=self.device)
        # Масштабируем шум под типичную норму активаций (~4.0)
        self.memory_state = self.memory_state * 4.0
        console.print(f"[yellow]Placebo warmup: {n_windows} windows (Noise injection)[/]")

    def generate_with_memory(self, context, n_tokens, temperature=1.0):
        # Полная копия логики Elephant, но при обновлении памяти — подмена на шум
        seq_len = self.config.seq_len
        generated = context.clone()
        n_windows = (n_tokens + seq_len - 1) // seq_len

        for w in range(n_windows):
            remaining = n_tokens - (generated.shape[1] - context.shape[1])
            if remaining <= 0: break
            tokens_this_window = min(seq_len, remaining)

            # --- GENERATION STEP (Same as Elephant) ---
            for _ in range(tokens_this_window):
                idx_cond = generated[:, -seq_len:] if generated.shape[1] > seq_len else generated
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature

                # INJECTION (Same mechanism, but memory_state is noise)
                if self.memory_state is not None:
                    mem_flat = self.memory_state.squeeze()
                    mem_bias = torch.matmul(mem_flat, self.model.lm_head.weight.T)
                    logits = logits + 0.1 * mem_bias.unsqueeze(0)

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, idx_next], dim=1)

            # --- UPDATE STEP (THE SABOTAGE) ---
            # Вместо умного обновления из hidden states, мы обновляем шум
            if self.memory_state is not None:
                # Генерируем новый шум
                noise = torch.randn_like(self.memory_state)
                # Сохраняем "энергию" (норму), чтобы плацебо было честным по амплитуде
                current_norm = torch.norm(self.memory_state)
                self.memory_state = (noise / torch.norm(noise)) * current_norm

        return generated[:, context.shape[1]:]


def run_final_audit(model, val_data, bin_centers, args, device):
    console.print(Panel.fit(
        "[bold magenta]🔮 FINAL VERDICT: WINDOW-BASED AUDIT 🔮[/]\n"
        "Comparing: Amnesiac vs Elephant (Smart) vs Placebo (Noise)\n"
        "Logic: Window-updates (like the original 78% run)",
        title="PROTOCOL v2.0"
    ))

    # Common params
    n_traj = args.n_traj
    traj_len = args.traj_len
    ctx_len = args.context_len

    # Contexts
    contexts = [val_data[i:i+1, :ctx_len].to(device) for i in range(n_traj)]
    tau_values = np.logspace(-1, np.log10(2*np.pi), 100)

    # --- 1. AMNESIAC (Baseline) ---
    console.print("\n[bold red]🔴 RED CORNER: Amnesiac (Baseline)[/]")
    red_spacings = []
    for ctx in track(contexts, description="Amnesiac generating"):
        with torch.no_grad():
            gen = model.generate(ctx, traj_len)
            s = bin_centers[gen[0, ctx_len:].cpu().numpy()]
            red_spacings.extend(s)

    sff_red = compute_sff(np.array(red_spacings), tau_values)
    console.print(f"Plateau: {sff_red['plateau']:.4f}")

    # --- 2. ELEPHANT (Smart Memory) ---
    console.print("\n[bold blue]🔵 BLUE CORNER: Elephant (Smart Window Memory)[/]")
    elephant = ElephantGenerator(model, bin_centers, device)

    # Warmup (Real Data)
    warmup_tokens = val_data[:args.warmup_windows].reshape(-1)
    elephant.warmup(warmup_tokens, n_windows=args.warmup_windows)

    blue_spacings = []
    for ctx in track(contexts, description="Elephant generating"):
        with torch.no_grad():
            gen = elephant.generate_with_memory(ctx, traj_len)
            s = bin_centers[gen[0].cpu().numpy()]
            blue_spacings.extend(s)

    sff_blue = compute_sff(np.array(blue_spacings), tau_values)
    console.print(f"Plateau: {sff_blue['plateau']:.4f}")

    # --- 3. PLACEBO (Noise Memory) ---
    console.print("\n[bold yellow]🟡 YELLOW CORNER: Placebo (Noise Injection)[/]")
    placebo = PlaceboGenerator(model, bin_centers, device)
    placebo.warmup(warmup_tokens, n_windows=args.warmup_windows) # Fake warmup

    yellow_spacings = []
    for ctx in track(contexts, description="Placebo generating"):
        with torch.no_grad():
            gen = placebo.generate_with_memory(ctx, traj_len)
            s = bin_centers[gen[0].cpu().numpy()]
            yellow_spacings.extend(s)

    sff_yellow = compute_sff(np.array(yellow_spacings), tau_values)
    console.print(f"Plateau: {sff_yellow['plateau']:.4f}")

    # --- SUMMARY TABLE ---
    table = Table(title="🏆 FINAL VERDICT RESULTS 🏆")
    table.add_column("Agent", style="bold")
    table.add_column("Plateau", justify="right")
    table.add_column("vs Baseline", justify="right")
    table.add_column("Conclusion", justify="right")

    # Metrics
    base = sff_red['plateau']

    # Logic
    def verdict(val, base):
        diff = (val - base) / base
        if diff > 0.10: return "[green]SIGNIFICANT GAIN[/]"
        if diff < -0.10: return "[red]DEGRADATION[/]"
        return "[grey]NO EFFECT[/]"

    table.add_row("🔴 Amnesiac", f"{base:.4f}", "1.00x", "Baseline")
    table.add_row("🔵 Elephant", f"{sff_blue['plateau']:.4f}", f"{sff_blue['plateau']/base:.2f}x", verdict(sff_blue['plateau'], base))
    table.add_row("🟡 Placebo", f"{sff_yellow['plateau']:.4f}", f"{sff_yellow['plateau']/base:.2f}x", verdict(sff_yellow['plateau'], base))

    console.print(table)

    # Final Check
    if sff_blue['plateau'] > sff_yellow['plateau'] * 1.1:
        console.print("[bold green]✅ CONFIRMED: Information > Noise. The 78% gain was real physics![/]")
    else:
        console.print("[bold red]❌ BUSTED: Smart memory is no better than noise. Architecture needs training.[/]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="out/best.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-traj", type=int, default=50) # Increased for stability
    parser.add_argument("--traj-len", type=int, default=512)
    parser.add_argument("--warmup-windows", type=int, default=50) # 12.8k tokens
    parser.add_argument("--context-len", type=int, default=64)
    args = parser.parse_args()

    # Load Model (Standard loading)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SpacingGPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])

    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    model.to(device).eval()

    val_data = torch.load(f"{args.data_dir}/val.pt", weights_only=False)
    bin_centers = np.load(f"{args.data_dir}/bin_centers.npy")

    run_final_audit(model, val_data, bin_centers, args, device)

if __name__ == "__main__":
    main()
