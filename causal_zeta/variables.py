"""
Causal Variables for Zeta Spacing Analysis.

Z_t: Latent mode (PCA of hidden states) - the "driver"
R_t: Rigidity proxy (local variance) - the "constraint"
S_t: Observed spacing (from data/model)
Y_t: Target spacing (next token prediction)
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Optional
import sys
sys.path.insert(0, '..')
from model.gpt import SpacingGPT


@dataclass
class CausalState:
    """State of causal variables at time t."""
    t: int
    S_t: float          # Current spacing S_{t-1} (observed)
    Z_t: np.ndarray     # Latent mode [2,] from PCA
    R_t: float          # Rigidity proxy
    Y_t: Optional[float] = None  # Target (next spacing)
    S_t_prev: Optional[float] = None  # S_{t-2} for CI tests (DAG v0.2)
    H_t: Optional[np.ndarray] = None  # Full hidden state (for debugging)


class LatentExtractor:
    """
    Extract Z_t (latent mode) from SpacingGPT hidden states.

    Z_t = PCA(n=2) of last layer hidden states.
    - Z_t[0]: Phase component (oscillation position)
    - Z_t[1]: Amplitude/energy component
    """

    GUE_VARIANCE = 0.178  # Expected variance of single GUE spacing (mean=1)

    def __init__(self, model: SpacingGPT, n_components: int = 2):
        self.model = model
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        self.device = next(model.parameters()).device

    def fit(self, data_loader, n_samples: int = 10000):
        """
        Fit PCA on hidden states from validation data.

        Args:
            data_loader: DataLoader with spacing sequences
            n_samples: Number of samples to use for fitting
        """
        self.model.eval()
        hidden_collection = []

        with torch.no_grad():
            for (batch,) in data_loader:
                if len(hidden_collection) * batch.shape[1] >= n_samples:
                    break

                batch = batch.to(self.device)
                hidden_states = self.model.get_hidden_states(batch)
                # Last layer, last token position
                h_last = hidden_states[-1]  # [B, T, D]
                # Flatten to [B*T, D]
                h_flat = h_last.reshape(-1, h_last.shape[-1]).cpu().numpy()
                hidden_collection.append(h_flat)

        all_hidden = np.concatenate(hidden_collection, axis=0)[:n_samples]
        self.pca.fit(all_hidden)
        self.fitted = True

        # Print explained variance
        explained = self.pca.explained_variance_ratio_
        print(f"[LatentExtractor] Fitted PCA on {len(all_hidden)} samples")
        print(f"  Z_t[0] explains {explained[0]*100:.1f}% variance")
        print(f"  Z_t[1] explains {explained[1]*100:.1f}% variance")

        return self

    def extract(self, idx: torch.Tensor, position: int = -1) -> np.ndarray:
        """
        Extract Z_t from input sequence.

        Args:
            idx: [B, T] tensor of token indices
            position: Which position to extract (-1 = last)

        Returns:
            Z_t: [B, 2] numpy array of latent coordinates
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before extract()")

        self.model.eval()
        with torch.no_grad():
            idx = idx.to(self.device)
            hidden_states = self.model.get_hidden_states(idx)
            h_last = hidden_states[-1]  # [B, T, D]
            h_t = h_last[:, position, :]  # [B, D]

        Z_t = self.pca.transform(h_t.cpu().numpy())
        return Z_t

    def extract_trajectory(self, idx: torch.Tensor) -> np.ndarray:
        """
        Extract Z_t for all positions in sequence.

        Returns:
            Z_trajectory: [B, T, 2] array
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before extract()")

        self.model.eval()
        with torch.no_grad():
            idx = idx.to(self.device)
            hidden_states = self.model.get_hidden_states(idx)
            h_last = hidden_states[-1]  # [B, T, D]
            B, T, D = h_last.shape
            h_flat = h_last.reshape(-1, D).cpu().numpy()

        Z_flat = self.pca.transform(h_flat)
        Z_trajectory = Z_flat.reshape(B, T, self.n_components)
        return Z_trajectory


class RigidityCalculator:
    """
    Compute R_t (rigidity proxy) from spacing sequences.

    R_t = Var(s[t-L:t]) / GUE_variance

    Where:
    - s = ACTUAL spacing values (not bin indices!)
    - GUE_variance ~ 0.178 for mean-1 normalized spacings

    CRITICAL: Must convert bin indices → spacing values via bin_centers!

    Interpretation:
    - R_t ~ 1.0: Normal GUE behavior
    - R_t < 1.0: Super-rigid (crystal-like)
    - R_t > 1.0: Sub-rigid (Poisson-like breakdown)
    """

    GUE_VARIANCE = 0.178

    def __init__(self, window: int = 10, bin_centers: np.ndarray = None):
        """
        Args:
            window: Number of past spacings to use for variance
            bin_centers: Array of bin center values [vocab_size,]
                         If None, assumes input is already spacing values
        """
        self.window = window
        self.bin_centers = bin_centers
        if bin_centers is not None:
            self.bin_centers_tensor = None  # Will be set on first use

    def _to_spacing(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert bin indices to spacing values."""
        if self.bin_centers is None:
            return tokens.float()

        # Lazy init tensor on correct device
        if self.bin_centers_tensor is None or self.bin_centers_tensor.device != tokens.device:
            self.bin_centers_tensor = torch.tensor(
                self.bin_centers, dtype=torch.float32, device=tokens.device
            )

        # Index into bin_centers
        return self.bin_centers_tensor[tokens.long()]

    def compute(self, tokens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Compute R_t for token sequence.

        Args:
            tokens: [B, T] tensor of bin indices (0..vocab_size-1)
            normalize: If True, divide by GUE_VARIANCE

        Returns:
            R_t: [B, T] tensor (padded with 1.0 for early positions)
        """
        B, T = tokens.shape
        R = torch.ones(B, T, device=tokens.device)

        if T < self.window:
            return R

        # Convert to spacing values
        spacings = self._to_spacing(tokens)

        # Compute rolling variance on ACTUAL spacing values
        for t in range(self.window, T):
            window_data = spacings[:, t-self.window:t]
            var = torch.var(window_data, dim=1, unbiased=True)
            R[:, t] = var

        if normalize:
            R = R / self.GUE_VARIANCE

        return R

    def compute_single(self, tokens: torch.Tensor) -> float:
        """
        Compute R_t for last position only.

        Args:
            tokens: [T,] tensor of bin indices

        Returns:
            R_t: scalar rigidity value
        """
        if len(tokens) < self.window:
            return 1.0

        # Convert to spacing values
        spacings = self._to_spacing(tokens[-self.window:])
        var = torch.var(spacings, unbiased=True).item()
        return var / self.GUE_VARIANCE


def collect_causal_states(
    model: SpacingGPT,
    data_loader,
    latent_extractor: LatentExtractor,
    rigidity_calc: RigidityCalculator,
    bin_centers: np.ndarray,
    n_samples: int = 1000,
    rigidity_window: int = 10,
) -> list[CausalState]:
    """
    Collect causal states (Z_t, R_t, S_t, Y_t) from data.

    Args:
        model: Trained SpacingGPT
        data_loader: Validation data
        latent_extractor: Fitted LatentExtractor
        rigidity_calc: RigidityCalculator with bin_centers set
        bin_centers: Array mapping bin index → spacing value
        n_samples: Max states to collect
        rigidity_window: Window size for R_t calculation.
                        S_t_prev (S_deep) is set to batch[t - window - 1]
                        to ensure it's OUTSIDE the R_t window.

    Returns:
        List of CausalState objects
    """
    states = []
    device = next(model.parameters()).device

    def bin_to_spacing(b: int) -> float:
        """Convert bin index to actual spacing value."""
        return float(bin_centers[b])

    model.eval()
    sample_idx = 0

    with torch.no_grad():
        for (batch,) in data_loader:
            batch = batch.to(device)
            B, T = batch.shape

            # Get Z_t trajectory
            Z_traj = latent_extractor.extract_trajectory(batch)  # [B, T, 2]

            # Get R_t trajectory (rigidity_calc handles bin→spacing conversion)
            R_traj = rigidity_calc.compute(batch)  # [B, T]

            # Collect states for each position
            # Start at rigidity_window + 2 to have room for S_deep outside R_t window
            start_t = rigidity_window + 2
            for b in range(B):
                for t in range(start_t, T-1):
                    # S_deep = batch[t - rigidity_window - 1] is OUTSIDE R_t window
                    # R_t uses batch[t-rigidity_window : t], so t-rigidity_window-1 is safe
                    deep_idx = t - rigidity_window - 1
                    state = CausalState(
                        t=sample_idx,
                        S_t=bin_to_spacing(batch[b, t].item()),
                        Z_t=Z_traj[b, t],
                        R_t=R_traj[b, t].item(),
                        Y_t=bin_to_spacing(batch[b, t+1].item()),
                        S_t_prev=bin_to_spacing(batch[b, deep_idx].item()),  # S_deep (outside R_t window)
                    )
                    states.append(state)
                    sample_idx += 1

                    if sample_idx >= n_samples:
                        return states

    return states


if __name__ == "__main__":
    # Quick test
    print("Testing CausalState creation...")
    state = CausalState(
        t=0,
        S_t=1.05,
        Z_t=np.array([0.1, -0.2]),
        R_t=0.95,
        Y_t=0.87,
    )
    print(f"  S_t={state.S_t:.3f}, Z_t={state.Z_t}, R_t={state.R_t:.3f}, Y_t={state.Y_t:.3f}")
    print("OK!")
