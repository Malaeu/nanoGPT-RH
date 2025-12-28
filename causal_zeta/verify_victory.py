#!/usr/bin/env python3
"""
Operation Reality Check: Попытка УНИЧТОЖИТЬ результат +78% SFF.

Если результат выживет — он настоящий.
Если нет — мы нашли баг.

3 стресс-теста:
1. PLACEBO CONTROL - случайный шум vs умная память
2. LEAKAGE CHECK - проверка на копипаст warmup данных
3. STABILITY - 100 траекторий для статистической значимости

Usage:
    python -m causal_zeta.verify_victory --checkpoint out/best.pt
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import SpacingGPT

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

    plateau_mask = tau_values > 4.0
    plateau = np.mean(K_values[plateau_mask]) if np.sum(plateau_mask) > 0 else K_values[-1]

    ramp_mask = (tau_values >= 0.5) & (tau_values <= 3.0)
    if np.sum(ramp_mask) > 2:
        tau_ramp = tau_values[ramp_mask]
        K_ramp = K_values[ramp_mask]
        slope, _ = np.polyfit(tau_ramp, K_ramp, 1)
    else:
        slope = 0.0

    return {"tau": tau_values, "K": K_values, "plateau": plateau, "ramp_slope": slope, "N": N}


class ElephantGenerator:
    """Умный генератор с памятью (оригинал)."""

    def __init__(self, model: SpacingGPT, bin_centers: np.ndarray, device):
        self.model = model
        self.bin_centers = bin_centers
        self.device = device
        self.config = model.config
        self.memory_state = None
        self.memory_alpha = 0.5
        self.warmup_tokens = None  # Сохраняем для проверки утечки

    def warmup(self, tokens: torch.Tensor, n_windows: int = 50):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        self.memory_state = torch.zeros(1, 1, self.config.n_embd, device=self.device)
        self.warmup_tokens = tokens.cpu().numpy().flatten()  # СОХРАНЯЕМ!

        seq_len = self.config.seq_len
        total_tokens = tokens.shape[1]

        windows_processed = 0
        for start in range(0, total_tokens - seq_len, seq_len):
            if windows_processed >= n_windows:
                break
            window = tokens[:, start:start+seq_len].to(self.device)
            hidden_states = self.model.get_hidden_states(window)
            summary = hidden_states[-1][:, -1:, :]
            self.memory_state = self.memory_alpha * self.memory_state + (1 - self.memory_alpha) * summary
            windows_processed += 1

    def generate_with_memory(self, context: torch.Tensor, n_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        seq_len = self.config.seq_len
        generated = context.clone()

        for _ in range(n_tokens):
            idx_cond = generated[:, -seq_len:] if generated.shape[1] > seq_len else generated
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # MEMORY INJECTION
            if self.memory_state is not None:
                mem_flat = self.memory_state.squeeze()
                mem_bias = torch.matmul(mem_flat, self.model.lm_head.weight.T)
                logits = logits + 0.1 * mem_bias.unsqueeze(0)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, idx_next], dim=1)

            # Update memory periodically
            if generated.shape[1] % seq_len == 0:
                window = generated[:, -seq_len:]
                hidden_states = self.model.get_hidden_states(window)
                summary = hidden_states[-1][:, -1:, :]
                self.memory_state = self.memory_alpha * self.memory_state + (1 - self.memory_alpha) * summary

        return generated[:, context.shape[1]:]


class PlaceboGenerator:
    """
    ПЛАЦЕБО: Вместо умной памяти — СЛУЧАЙНЫЙ ШУМ той же амплитуды.
    Если результат такой же — значит дело не в информации, а в шуме.
    """

    def __init__(self, model: SpacingGPT, bin_centers: np.ndarray, device):
        self.model = model
        self.bin_centers = bin_centers
        self.device = device
        self.config = model.config
        self.noise_scale = None  # Будет вычислена из реального Elephant

    def calibrate_noise(self, elephant_memory_norm: float):
        """Калибровать шум по норме реальной памяти."""
        self.noise_scale = elephant_memory_norm

    def generate_with_noise(self, context: torch.Tensor, n_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        seq_len = self.config.seq_len
        generated = context.clone()
        n_embd = self.config.n_embd

        for _ in range(n_tokens):
            idx_cond = generated[:, -seq_len:] if generated.shape[1] > seq_len else generated
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # RANDOM NOISE вместо умной памяти!
            random_mem = torch.randn(n_embd, device=self.device)
            if self.noise_scale is not None:
                random_mem = random_mem * self.noise_scale / torch.norm(random_mem)
            noise_bias = torch.matmul(random_mem, self.model.lm_head.weight.T)
            logits = logits + 0.1 * noise_bias.unsqueeze(0)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, idx_next], dim=1)

        return generated[:, context.shape[1]:]


class AmnesiacGenerator:
    """Базовый генератор без памяти (контроль)."""

    def __init__(self, model: SpacingGPT, bin_centers: np.ndarray, device):
        self.model = model
        self.bin_centers = bin_centers
        self.device = device
        self.config = model.config

    def generate(self, context: torch.Tensor, n_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        seq_len = self.config.seq_len
        generated = context.clone()

        for _ in range(n_tokens):
            idx_cond = generated[:, -seq_len:] if generated.shape[1] > seq_len else generated
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, idx_next], dim=1)

        return generated[:, context.shape[1]:]


def test_placebo_control(model, val_data, bin_centers, args, device) -> dict:
    """
    ТЕСТ 1: PLACEBO CONTROL
    Elephant vs Random Noise — кто победит?
    """
    console.print(Panel.fit("[bold red]ТЕСТ 1: PLACEBO CONTROL[/]\nУмная память vs Случайный шум", title="🧪"))

    n_traj = args.n_traj
    traj_len = args.traj_len
    context_len = args.context_len

    contexts = []
    for i in range(min(n_traj, len(val_data))):
        ctx = val_data[i:i+1, :context_len].to(device)
        contexts.append(ctx)

    # Elephant (умный)
    console.print("\n[blue]🐘 Running Elephant (Smart Memory)...[/]")
    elephant = ElephantGenerator(model, bin_centers, device)
    warmup_data = val_data[:args.warmup_windows].reshape(-1)
    elephant.warmup(warmup_data, n_windows=args.warmup_windows)

    elephant_memory_norm = torch.norm(elephant.memory_state).item()
    console.print(f"  Memory norm: {elephant_memory_norm:.4f}")

    elephant_spacings = []
    for ctx in track(contexts, description="Elephant generating"):
        with torch.no_grad():
            generated = elephant.generate_with_memory(ctx, traj_len, temperature=1.0)
            tokens = generated[0].cpu().numpy()
            spacings = bin_centers[tokens]
            elephant_spacings.extend(spacings)

    sff_elephant = compute_sff(np.array(elephant_spacings))

    # Placebo (шум)
    console.print("\n[yellow]🎲 Running Placebo (Random Noise)...[/]")
    placebo = PlaceboGenerator(model, bin_centers, device)
    placebo.calibrate_noise(elephant_memory_norm)

    placebo_spacings = []
    for ctx in track(contexts, description="Placebo generating"):
        with torch.no_grad():
            generated = placebo.generate_with_noise(ctx, traj_len, temperature=1.0)
            tokens = generated[0].cpu().numpy()
            spacings = bin_centers[tokens]
            placebo_spacings.extend(spacings)

    sff_placebo = compute_sff(np.array(placebo_spacings))

    # Amnesiac (контроль)
    console.print("\n[red]🔴 Running Amnesiac (No intervention)...[/]")
    amnesiac = AmnesiacGenerator(model, bin_centers, device)

    amnesiac_spacings = []
    for ctx in track(contexts, description="Amnesiac generating"):
        with torch.no_grad():
            generated = amnesiac.generate(ctx, traj_len, temperature=1.0)
            tokens = generated[0].cpu().numpy()
            spacings = bin_centers[tokens]
            amnesiac_spacings.extend(spacings)

    sff_amnesiac = compute_sff(np.array(amnesiac_spacings))

    # Результаты
    table = Table(title="PLACEBO CONTROL RESULTS")
    table.add_column("Generator", style="bold")
    table.add_column("SFF Plateau", justify="right")
    table.add_column("vs Amnesiac", justify="right")

    table.add_row("🔴 Amnesiac (baseline)", f"{sff_amnesiac['plateau']:.4f}", "1.00x")
    table.add_row("🎲 Placebo (noise)", f"{sff_placebo['plateau']:.4f}",
                  f"{sff_placebo['plateau']/sff_amnesiac['plateau']:.2f}x")
    table.add_row("🐘 Elephant (smart)", f"{sff_elephant['plateau']:.4f}",
                  f"{sff_elephant['plateau']/sff_amnesiac['plateau']:.2f}x")

    console.print(table)

    # Вердикт
    elephant_vs_placebo = sff_elephant['plateau'] / sff_placebo['plateau'] if sff_placebo['plateau'] > 0 else 0
    if elephant_vs_placebo > 1.2:
        verdict = "✅ PASS: Elephant > Placebo (информация работает!)"
        status = "PASS"
    elif elephant_vs_placebo > 1.05:
        verdict = "⚠️ MARGINAL: Elephant немного лучше Placebo"
        status = "MARGINAL"
    else:
        verdict = "❌ FAIL: Elephant ≈ Placebo (это просто шум!)"
        status = "FAIL"

    console.print(f"\n[bold]{verdict}[/]")
    console.print(f"Elephant/Placebo ratio: {elephant_vs_placebo:.2f}x")

    return {
        "status": status,
        "elephant_plateau": sff_elephant['plateau'],
        "placebo_plateau": sff_placebo['plateau'],
        "amnesiac_plateau": sff_amnesiac['plateau'],
        "elephant_vs_placebo": elephant_vs_placebo,
        "elephant_warmup_tokens": elephant.warmup_tokens,
    }


def test_leakage_check(elephant_warmup_tokens: np.ndarray, generated_tokens: np.ndarray) -> dict:
    """
    ТЕСТ 2: LEAKAGE CHECK
    Сравнить сгенерированные токены с warmup данными.
    """
    console.print(Panel.fit("[bold red]ТЕСТ 2: LEAKAGE CHECK[/]\nПроверка на копипаст warmup данных", title="🔍"))

    warmup_set = set(elephant_warmup_tokens.tolist())
    generated_set = set(generated_tokens.tolist())

    # Точные совпадения последовательностей (sliding window)
    warmup_str = ''.join(map(str, elephant_warmup_tokens[:10000]))  # Ограничиваем
    gen_str = ''.join(map(str, generated_tokens[:10000]))

    # Ищем повторяющиеся подстроки длиной >= 10
    overlap_count = 0
    window_size = 10
    for i in range(len(gen_str) - window_size):
        substr = gen_str[i:i+window_size]
        if substr in warmup_str:
            overlap_count += 1

    overlap_pct = overlap_count / max(1, len(gen_str) - window_size) * 100

    # Распределение токенов
    warmup_hist = np.bincount(elephant_warmup_tokens, minlength=256)
    gen_hist = np.bincount(generated_tokens, minlength=256)

    # Корреляция распределений (это нормально что высокая — same domain)
    distribution_corr = np.corrcoef(warmup_hist, gen_hist)[0, 1]

    # Результаты
    table = Table(title="LEAKAGE CHECK RESULTS")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status", justify="center")

    table.add_row(
        "Sequence overlap (10-grams)",
        f"{overlap_pct:.2f}%",
        "< 5%",
        "✅" if overlap_pct < 5 else "❌"
    )
    table.add_row(
        "Distribution correlation",
        f"{distribution_corr:.4f}",
        "< 0.95 (expected high)",
        "✅" if distribution_corr < 0.99 else "⚠️"
    )

    console.print(table)

    if overlap_pct < 5:
        verdict = "✅ PASS: Нет утечки данных"
        status = "PASS"
    elif overlap_pct < 20:
        verdict = "⚠️ WARNING: Небольшое перекрытие, проверить"
        status = "WARNING"
    else:
        verdict = "❌ FAIL: Модель копипастит warmup!"
        status = "FAIL"

    console.print(f"\n[bold]{verdict}[/]")

    return {
        "status": status,
        "overlap_pct": overlap_pct,
        "distribution_corr": distribution_corr,
    }


def test_stability(model, val_data, bin_centers, args, device) -> dict:
    """
    ТЕСТ 3: STABILITY
    100 траекторий для статистической значимости.
    """
    console.print(Panel.fit("[bold red]ТЕСТ 3: STABILITY[/]\n100 траекторий для статистики", title="📊"))

    n_traj = 100  # Фиксированно 100
    traj_len = args.traj_len
    context_len = args.context_len

    contexts = []
    for i in range(min(n_traj, len(val_data))):
        ctx = val_data[i:i+1, :context_len].to(device)
        contexts.append(ctx)

    # Elephant
    elephant = ElephantGenerator(model, bin_centers, device)
    warmup_data = val_data[:args.warmup_windows].reshape(-1)
    elephant.warmup(warmup_data, n_windows=args.warmup_windows)

    # Собираем plateau для каждой траектории
    elephant_plateaus = []
    amnesiac_plateaus = []

    amnesiac = AmnesiacGenerator(model, bin_centers, device)

    console.print("\n[blue]Running 100 trajectories...[/]")
    for ctx in track(contexts, description="Generating"):
        with torch.no_grad():
            # Elephant
            gen_e = elephant.generate_with_memory(ctx, traj_len, temperature=1.0)
            tokens_e = gen_e[0].cpu().numpy()
            spacings_e = bin_centers[tokens_e]
            sff_e = compute_sff(spacings_e)
            elephant_plateaus.append(sff_e['plateau'])

            # Amnesiac
            gen_a = amnesiac.generate(ctx, traj_len, temperature=1.0)
            tokens_a = gen_a[0].cpu().numpy()
            spacings_a = bin_centers[tokens_a]
            sff_a = compute_sff(spacings_a)
            amnesiac_plateaus.append(sff_a['plateau'])

    elephant_plateaus = np.array(elephant_plateaus)
    amnesiac_plateaus = np.array(amnesiac_plateaus)

    # Статистика
    e_mean, e_std = np.mean(elephant_plateaus), np.std(elephant_plateaus)
    a_mean, a_std = np.mean(amnesiac_plateaus), np.std(amnesiac_plateaus)

    # t-test
    t_stat, p_value = stats.ttest_ind(elephant_plateaus, amnesiac_plateaus)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((e_std**2 + a_std**2) / 2)
    cohens_d = (e_mean - a_mean) / pooled_std if pooled_std > 0 else 0

    # Результаты
    table = Table(title="STABILITY RESULTS (N=100)")
    table.add_column("Generator", style="bold")
    table.add_column("Mean Plateau", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("95% CI", justify="right")

    e_ci = 1.96 * e_std / np.sqrt(len(elephant_plateaus))
    a_ci = 1.96 * a_std / np.sqrt(len(amnesiac_plateaus))

    table.add_row("🔴 Amnesiac", f"{a_mean:.4f}", f"{a_std:.4f}", f"±{a_ci:.4f}")
    table.add_row("🐘 Elephant", f"{e_mean:.4f}", f"{e_std:.4f}", f"±{e_ci:.4f}")

    console.print(table)

    console.print(f"\n[bold]Statistical Tests:[/]")
    console.print(f"  t-statistic: {t_stat:.4f}")
    console.print(f"  p-value: {p_value:.6f}")
    console.print(f"  Cohen's d: {cohens_d:.4f}")
    console.print(f"  Improvement: {(e_mean/a_mean - 1)*100:.1f}%")

    if p_value < 0.01 and cohens_d > 0.5:
        verdict = "✅ PASS: Статистически значимо (p<0.01, d>0.5)"
        status = "PASS"
    elif p_value < 0.05:
        verdict = "⚠️ MARGINAL: Значимо, но слабый эффект"
        status = "MARGINAL"
    else:
        verdict = "❌ FAIL: Не значимо статистически"
        status = "FAIL"

    console.print(f"\n[bold]{verdict}[/]")

    return {
        "status": status,
        "elephant_mean": e_mean,
        "elephant_std": e_std,
        "amnesiac_mean": a_mean,
        "amnesiac_std": a_std,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "improvement_pct": (e_mean/a_mean - 1)*100,
    }


def main():
    parser = argparse.ArgumentParser(description="Operation Reality Check")
    parser.add_argument("--checkpoint", type=str, default="out/best.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-traj", type=int, default=50, help="Trajectories for placebo test")
    parser.add_argument("--traj-len", type=int, default=512)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--warmup-windows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    console.print(Panel.fit(
        "[bold red]🔬 OPERATION REALITY CHECK 🔬[/]\n\n"
        "Попытка УНИЧТОЖИТЬ результат +78% SFF.\n"
        "Если выживет — он настоящий.",
        title="ЖЕСТКИЙ АУДИТ"
    ))

    # Load model
    console.print(f"\n[cyan]Loading: {args.checkpoint}[/]")
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

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================

    results = {}

    # Test 1: Placebo Control
    console.print("\n" + "="*60)
    placebo_result = test_placebo_control(model, val_data, bin_centers, args, device)
    results["placebo"] = placebo_result

    # Collect generated tokens for leakage check
    elephant = ElephantGenerator(model, bin_centers, device)
    warmup_data = val_data[:args.warmup_windows].reshape(-1)
    elephant.warmup(warmup_data, n_windows=args.warmup_windows)

    generated_tokens = []
    for i in range(min(10, len(val_data))):
        ctx = val_data[i:i+1, :args.context_len].to(device)
        with torch.no_grad():
            gen = elephant.generate_with_memory(ctx, args.traj_len, temperature=1.0)
            generated_tokens.extend(gen[0].cpu().numpy().tolist())

    # Test 2: Leakage Check
    console.print("\n" + "="*60)
    leakage_result = test_leakage_check(
        placebo_result["elephant_warmup_tokens"],
        np.array(generated_tokens)
    )
    results["leakage"] = leakage_result

    # Test 3: Stability
    console.print("\n" + "="*60)
    stability_result = test_stability(model, val_data, bin_centers, args, device)
    results["stability"] = stability_result

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    console.print("\n" + "="*60)
    console.print(Panel.fit("[bold]🏆 FINAL REALITY CHECK VERDICT 🏆[/]", title="РЕЗУЛЬТАТЫ"))

    table = Table(title="Summary")
    table.add_column("Test", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Key Metric", justify="right")

    table.add_row(
        "1. Placebo Control",
        "✅" if placebo_result["status"] == "PASS" else "❌",
        f"Elephant/Placebo: {placebo_result['elephant_vs_placebo']:.2f}x"
    )
    table.add_row(
        "2. Leakage Check",
        "✅" if leakage_result["status"] == "PASS" else "❌",
        f"Overlap: {leakage_result['overlap_pct']:.2f}%"
    )
    table.add_row(
        "3. Stability (N=100)",
        "✅" if stability_result["status"] == "PASS" else "❌",
        f"p={stability_result['p_value']:.4f}, d={stability_result['cohens_d']:.2f}"
    )

    console.print(table)

    # Overall verdict
    passed = sum(1 for r in [placebo_result, leakage_result, stability_result] if r["status"] == "PASS")

    console.print("\n")
    if passed == 3:
        console.print(Panel.fit(
            "[bold green]🎉 ПОБЕДА ПОДТВЕРЖДЕНА! 🎉[/]\n\n"
            "Все 3 теста пройдены.\n"
            "Результат +78% SFF — НАСТОЯЩИЙ!\n\n"
            "Memory действительно захватывает long-range structure.",
            title="✅ VERIFIED"
        ))
    elif passed >= 2:
        console.print(Panel.fit(
            "[bold yellow]⚠️ ЧАСТИЧНАЯ ПОБЕДА ⚠️[/]\n\n"
            f"Пройдено {passed}/3 тестов.\n"
            "Результат вероятно настоящий, но нужна осторожность.",
            title="⚠️ NEEDS REVIEW"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]❌ РЕЗУЛЬТАТ УНИЧТОЖЕН ❌[/]\n\n"
            f"Пройдено только {passed}/3 тестов.\n"
            "Победа была иллюзией. Ищем баг.",
            title="❌ DEBUNKED"
        ))

    return results


if __name__ == "__main__":
    main()
