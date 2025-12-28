#!/usr/bin/env python3
"""
RMT Causal Oracle: Infinite Context + Causal Interventions

The "Snowball" Memory:
```
Окно 1:      [MEM₀, x₁, x₂, ..., x₂₅₆] → hidden → MEM₁
Окно 2:      [MEM₁, x₂₅₇, ..., x₅₁₂]   → hidden → MEM₂
Окно N:      [MEMₙ₋₁, ...]             → hidden → MEMₙ
                 ↑
          "Сжатая Вселенная"
          (всё прошлое в одном векторе)
```

Каждый MEM тащит информацию о ВСЁМ предыдущем контексте,
даже если это миллион токенов назад.

Usage:
    python -m causal_zeta.rmt_causal --checkpoint out/best.pt
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import SpacingGPT

console = Console()


@dataclass
class OracleConfig:
    """RMT Oracle configuration."""
    memory_alpha: float = 0.5        # EMA weight for memory update
    gue_variance: float = 0.178      # σ² for GUE with mean=1
    rigidity_window: int = 10        # Window for R_t calculation
    pca_components: int = 2          # Latent dimension for Z_t


class CausalInfiniteOracle:
    """
    RMT-style Oracle with infinite context and causal interventions.

    Key idea: memory_state accumulates information across windows,
    allowing the model to "remember" arbitrarily long history.
    """

    def __init__(self, model: SpacingGPT, bin_centers: np.ndarray, config: OracleConfig = None):
        self.model = model
        self.model_config = model.config
        self.bin_centers = bin_centers
        self.config = config or OracleConfig()

        self.device = next(model.parameters()).device

        # Memory state: "compressed universe" vector
        self.memory_state = None

        # PCA for latent mode Z_t (fitted lazily)
        self.pca = None
        self._pca_fitted = False

        # History for analysis
        self.memory_history = []
        self.prediction_history = []

    def init_memory(self, method: str = "zero"):
        """
        Initialize memory state.

        Args:
            method: "zero", "noise", or "learned"
        """
        n_embd = self.model_config.n_embd

        if method == "zero":
            self.memory_state = torch.zeros(1, 1, n_embd, device=self.device)
        elif method == "noise":
            self.memory_state = torch.randn(1, 1, n_embd, device=self.device) * 0.02
        elif method == "learned":
            # Use mean of embedding layer as initialization
            with torch.no_grad():
                self.memory_state = self.model.transformer.wte.weight.mean(dim=0, keepdim=True).unsqueeze(0)
        else:
            raise ValueError(f"Unknown init method: {method}")

        self.memory_history = []
        self.prediction_history = []
        console.print(f"[cyan]Memory initialized ({method}): norm = {torch.norm(self.memory_state).item():.4f}[/]")

    def update_memory(self, context_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Snowball Memory Update.

        Process a window and update memory with new information.

        Args:
            context_tokens: [B, T] token indices

        Returns:
            logits: [B, T, vocab_size]
            hidden_last: [B, 1, n_embd] - last hidden state
        """
        if self.memory_state is None:
            self.init_memory()

        # Get hidden states from model
        hidden_states = self.model.get_hidden_states(context_tokens)
        last_layer_hidden = hidden_states[-1]  # [B, T, C]

        # Get logits for prediction
        with torch.no_grad():
            logits, _ = self.model(context_tokens)

        # Extract summary: last token's hidden state
        summary = last_layer_hidden[:, -1:, :]  # [B, 1, C]

        # Snowball update: EMA blend of old and new
        alpha = self.config.memory_alpha
        self.memory_state = alpha * self.memory_state + (1 - alpha) * summary

        # Track history
        self.memory_history.append(torch.norm(self.memory_state).item())

        return logits, last_layer_hidden

    def fit_pca(self, hidden_states_list: list):
        """
        Fit PCA on collected hidden states for Z_t projection.

        Args:
            hidden_states_list: list of [C,] numpy arrays
        """
        if len(hidden_states_list) < 10:
            console.print("[yellow]Not enough samples for PCA, need 10+[/]")
            return

        X = np.stack(hidden_states_list)
        self.pca = PCA(n_components=self.config.pca_components)
        self.pca.fit(X)
        self._pca_fitted = True

        console.print(f"[green]PCA fitted: {self.config.pca_components} components, "
                      f"explained variance = {sum(self.pca.explained_variance_ratio_):.2%}[/]")

    def compute_causal_vars(self, spacings: np.ndarray, hidden_state: np.ndarray) -> dict:
        """
        Compute causal graph variables.

        Args:
            spacings: recent spacing values
            hidden_state: [C,] hidden state vector

        Returns:
            dict with Z_t, R_t, S_t
        """
        # Z_t: Latent mode (PCA projection)
        if self._pca_fitted:
            z_t = self.pca.transform([hidden_state])[0]
        else:
            z_t = np.zeros(self.config.pca_components)

        # R_t: Rigidity (local variance / GUE variance)
        window = self.config.rigidity_window
        if len(spacings) >= window:
            r_t = np.var(spacings[-window:]) / self.config.gue_variance
        else:
            r_t = 1.0

        # S_t: Last spacing
        s_t = spacings[-1] if len(spacings) > 0 else 1.0

        return {
            "Z_t": z_t,
            "R_t": r_t,
            "S_t": s_t,
            "memory_norm": torch.norm(self.memory_state).item() if self.memory_state is not None else 0,
        }

    def predict_with_intervention(
        self,
        context_tokens: torch.Tensor,
        context_spacings: np.ndarray,
        do_S: float = None,
    ) -> dict:
        """
        Make prediction with optional causal intervention do(S).

        Args:
            context_tokens: [1, T] context token indices
            context_spacings: corresponding spacing values
            do_S: if not None, replace last spacing with this value

        Returns:
            dict with prediction, causal vars, intervention info
        """
        # INTERVENTION: do(S)
        if do_S is not None:
            # Replace last token with intervened value
            intervened_bin = self._spacing_to_bin(do_S)
            context_tokens = context_tokens.clone()
            context_tokens[0, -1] = intervened_bin

            context_spacings = context_spacings.copy()
            context_spacings[-1] = do_S

        # Run model with memory
        logits, hidden = self.update_memory(context_tokens)

        # Get prediction
        last_logits = logits[0, -1, :]  # [vocab_size]
        probs = F.softmax(last_logits, dim=-1)
        pred_bin = torch.argmax(probs).item()
        pred_spacing = self.bin_centers[pred_bin]

        # Compute causal variables
        hidden_last = hidden[0, -1, :].cpu().numpy()
        causal_vars = self.compute_causal_vars(context_spacings, hidden_last)

        return {
            "prediction": pred_spacing,
            "pred_bin": pred_bin,
            "pred_probs": probs.cpu().numpy(),
            "intervention": do_S,
            **causal_vars,
        }

    def _spacing_to_bin(self, spacing: float) -> int:
        """Convert spacing value to bin index."""
        diffs = np.abs(self.bin_centers - spacing)
        return int(np.argmin(diffs))

    def measure_causal_effect(
        self,
        context_tokens: torch.Tensor,
        context_spacings: np.ndarray,
        delta: float = 0.2,
    ) -> dict:
        """
        Measure causal effect of intervention do(S + delta).

        Args:
            context_tokens: [1, T] context
            context_spacings: spacing values
            delta: intervention size (added to last spacing)

        Returns:
            dict with baseline, intervened, and effect size
        """
        # Save memory state
        mem_backup = self.memory_state.clone() if self.memory_state is not None else None

        # Baseline (no intervention)
        baseline = self.predict_with_intervention(context_tokens, context_spacings, do_S=None)

        # Restore memory
        self.memory_state = mem_backup.clone() if mem_backup is not None else None

        # Intervened (do(S + delta))
        original_s = context_spacings[-1]
        do_s = original_s + delta
        intervened = self.predict_with_intervention(context_tokens, context_spacings, do_S=do_s)

        # Effect size
        effect = {
            "delta_prediction": intervened["prediction"] - baseline["prediction"],
            "delta_R_t": intervened["R_t"] - baseline["R_t"],
            "baseline": baseline,
            "intervened": intervened,
            "delta_S": delta,
        }

        return effect


def run_infinite_context_test(model, val_loader, bin_centers, args, device):
    """
    Test infinite context by processing long sequences with RMT memory.
    """
    console.print(Panel.fit("[bold blue]INFINITE CONTEXT TEST[/]", title="RMT Oracle"))

    oracle = CausalInfiniteOracle(model, bin_centers)
    oracle.init_memory(method="zero")

    # Collect hidden states for PCA
    hidden_samples = []

    # Process validation data in windows
    n_windows = 0
    all_spacings = []

    for (batch,) in track(val_loader, description="Processing windows"):
        for i in range(batch.shape[0]):
            tokens = batch[i:i+1, :].to(device)
            spacings = bin_centers[batch[i].numpy()]
            all_spacings.extend(spacings)

            # Update memory
            logits, hidden = oracle.update_memory(tokens)

            # Collect for PCA
            hidden_samples.append(hidden[0, -1, :].cpu().numpy())

            n_windows += 1
            if n_windows >= args.n_windows:
                break

        if n_windows >= args.n_windows:
            break

    console.print(f"[cyan]Processed {n_windows} windows[/]")
    console.print(f"[cyan]Total spacings seen: {len(all_spacings)}[/]")
    console.print(f"[cyan]Final memory norm: {torch.norm(oracle.memory_state).item():.4f}[/]")

    # Fit PCA
    oracle.fit_pca(hidden_samples)

    # Test causal intervention
    console.print("\n[bold magenta]Testing Causal Intervention do(S + 0.3):[/]")

    # Use last window for test
    test_tokens = batch[0:1, :].to(device)
    test_spacings = bin_centers[batch[0].numpy()]

    effect = oracle.measure_causal_effect(test_tokens, test_spacings, delta=0.3)

    table = Table(title="Causal Effect")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", style="green")
    table.add_column("Intervened", style="yellow")
    table.add_column("Effect", style="magenta")

    table.add_row(
        "Prediction",
        f"{effect['baseline']['prediction']:.4f}",
        f"{effect['intervened']['prediction']:.4f}",
        f"{effect['delta_prediction']:+.4f}",
    )
    table.add_row(
        "R_t",
        f"{effect['baseline']['R_t']:.4f}",
        f"{effect['intervened']['R_t']:.4f}",
        f"{effect['delta_R_t']:+.4f}",
    )
    table.add_row(
        "Z_t[0]",
        f"{effect['baseline']['Z_t'][0]:.4f}",
        f"{effect['intervened']['Z_t'][0]:.4f}",
        f"{effect['intervened']['Z_t'][0] - effect['baseline']['Z_t'][0]:+.4f}",
    )

    console.print(table)

    # Memory stability plot (text-based)
    console.print("\n[bold]Memory Norm Over Time:[/]")
    max_norm = max(oracle.memory_history)
    for i in range(0, len(oracle.memory_history), max(1, len(oracle.memory_history)//20)):
        bar_len = int(oracle.memory_history[i] / max_norm * 40)
        console.print(f"  {i:4d} |{'█' * bar_len}")

    return oracle


def main():
    parser = argparse.ArgumentParser(description="RMT Causal Oracle")
    parser.add_argument("--checkpoint", type=str, default="out/best.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-windows", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Load model
    console.print(f"[cyan]Loading checkpoint: {args.checkpoint}[/]")
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

    # Load data
    data_path = Path(args.data_dir)
    val_data = torch.load(data_path / "val.pt", weights_only=False)
    bin_centers = np.load(data_path / "bin_centers.npy")

    from torch.utils.data import DataLoader, TensorDataset
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Run test
    oracle = run_infinite_context_test(model, val_loader, bin_centers, args, device)

    console.print(Panel.fit("[bold green]✓ RMT Oracle Test Complete![/]"))


if __name__ == "__main__":
    main()
