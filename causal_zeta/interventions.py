"""
Interventions (do-operations) for Causal Zeta.

We can't intervene on the "real" Riemann zeta function.
But we CAN intervene on our generative model (SpacingGPT).

Key interventions:
1. do(S_t := S_t + delta) - Perturb spacing, measure healing
2. do(Z_t := Z_t + delta_vec) - Perturb latent phase
3. do(R_t := constant) - Fix rigidity, observe output
4. do(H_t := perturbed) - Add noise to hidden state
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
import sys
sys.path.insert(0, '..')
from model.gpt import SpacingGPT


@dataclass
class InterventionResult:
    """Result of a do-operation."""
    intervention_name: str
    target_variable: str
    delta: float | np.ndarray
    baseline_trajectory: np.ndarray  # [T,] original
    intervened_trajectory: np.ndarray  # [T,] after intervention
    healing_curve: np.ndarray  # |baseline - intervened| over time
    healing_time: int  # Steps until |diff| < threshold
    q3_violations: int  # Number of Q3 constraint violations


class Intervention:
    """Base class for causal interventions."""

    def __init__(self, name: str, target: str):
        self.name = name
        self.target = target

    def apply(self, model: SpacingGPT, context: torch.Tensor) -> torch.Tensor:
        """Apply intervention and return modified trajectory."""
        raise NotImplementedError


class SpacingIntervention(Intervention):
    """
    Intervention: do(S_t := S_t + delta).

    Perturb a spacing value and observe trajectory healing.
    This tests the "rigidity" hypothesis: how quickly does the
    system return to normal after a perturbation?
    """

    def __init__(self, delta: float = 0.05, position: int = 10):
        """
        Args:
            delta: Amount to add to spacing (in bin units)
            position: Which position to perturb
        """
        super().__init__("do(S_t)", "S_{t-1}")
        self.delta = delta
        self.position = position

    def apply(
        self,
        model: SpacingGPT,
        context: torch.Tensor,
        n_generate: int = 50,
        temperature: float = 1.0,
        seed: int = 42,
        n_trials: int = 10,
    ) -> InterventionResult:
        """
        Apply spacing intervention with impulse response measurement.

        Uses multiple trials with different seeds to measure the TRUE effect
        of intervention (not confounded by same-seed sampling).

        Args:
            model: SpacingGPT model
            context: [1, T] input context (bin indices)
            n_generate: Steps to generate after intervention
            temperature: Sampling temperature
            seed: Base random seed
            n_trials: Number of trials for averaging

        Returns:
            InterventionResult with impulse response curves
        """
        device = context.device
        vocab_size = model.config.vocab_size

        model.eval()

        baseline_trajs = []
        interv_trajs = []

        for trial in range(n_trials):
            trial_seed = seed + trial * 1000

            # Baseline trajectory (unperturbed)
            torch.manual_seed(trial_seed)
            with torch.no_grad():
                baseline_context = context.clone()
                baseline_traj = model.generate(
                    baseline_context, n_generate, temperature
                )[0, context.shape[1]:].cpu().numpy()
                baseline_trajs.append(baseline_traj)

            # Intervened trajectory (perturbed at position)
            torch.manual_seed(trial_seed + 500)  # DIFFERENT seed!
            with torch.no_grad():
                interv_context = context.clone()
                old_val = interv_context[0, self.position].item()
                # delta is in spacing scale, convert to bins
                new_val = int(np.clip(old_val + self.delta, 0, vocab_size - 1))
                interv_context[0, self.position] = new_val

                interv_traj = model.generate(
                    interv_context, n_generate, temperature
                )[0, context.shape[1]:].cpu().numpy()
                interv_trajs.append(interv_traj)

        # Convert to arrays
        baseline_trajs = np.array(baseline_trajs)  # [n_trials, n_generate]
        interv_trajs = np.array(interv_trajs)

        # Impulse Response: deviation from baseline mean at each timestep
        baseline_mean = np.mean(baseline_trajs, axis=0)
        interv_mean = np.mean(interv_trajs, axis=0)

        # Healing curve: |E[interv] - E[baseline]| over time
        healing = np.abs(interv_mean - baseline_mean)

        # Healing time: first step where impulse response < 0.5 bins
        # (50% of typical bin width)
        healing_time = n_generate
        initial_impulse = healing[0] if len(healing) > 0 else 1.0
        for t in range(len(healing)):
            if healing[t] < max(0.5, initial_impulse * 0.1):  # 10% of initial
                healing_time = t
                break

        return InterventionResult(
            intervention_name=self.name,
            target_variable=self.target,
            delta=self.delta,
            baseline_trajectory=baseline_mean,  # Mean trajectory
            intervened_trajectory=interv_mean,
            healing_curve=healing,
            healing_time=healing_time,
            q3_violations=0,
        )


class LatentIntervention(Intervention):
    """
    Intervention: do(Z_t := Z_t + delta_vec).

    Perturb the latent mode (phase vector) via hidden state manipulation.
    Uses "control vector" style intervention.
    """

    def __init__(self, delta_vec: np.ndarray = None, layer: int = -1):
        """
        Args:
            delta_vec: [2,] perturbation in PCA space
            layer: Which layer to perturb (-1 = last)
        """
        super().__init__("do(Z_t)", "Z_t")
        self.delta_vec = delta_vec if delta_vec is not None else np.array([0.1, 0.1])
        self.layer = layer
        self.pca = None  # Must be set from LatentExtractor

    def set_pca(self, pca):
        """Set PCA transform from LatentExtractor."""
        self.pca = pca

    def apply(
        self,
        model: SpacingGPT,
        context: torch.Tensor,
        n_generate: int = 50,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> InterventionResult:
        """
        Apply latent intervention via hidden state hook.
        """
        if self.pca is None:
            raise RuntimeError("Must call set_pca() before apply()")

        device = context.device

        # Convert delta_vec to hidden space
        # delta_hidden = PCA.inverse_transform(delta_vec) - PCA.inverse_transform(0)
        zero_vec = np.zeros_like(self.delta_vec)
        delta_hidden = self.pca.inverse_transform([self.delta_vec])[0] - \
                       self.pca.inverse_transform([zero_vec])[0]
        delta_hidden = torch.tensor(delta_hidden, device=device, dtype=torch.float32)

        # Baseline
        torch.manual_seed(seed)
        model.eval()
        with torch.no_grad():
            baseline_traj = model.generate(
                context.clone(), n_generate, temperature
            )[0, context.shape[1]:].cpu().numpy()

        # Intervened: add hook to perturb hidden state
        def perturb_hook(module, input, output):
            # output: [B, T, D]
            output = output.clone()
            output[:, -1, :] = output[:, -1, :] + delta_hidden
            return output

        # Register hook on target layer
        target_layer = model.transformer.h[self.layer]
        handle = target_layer.register_forward_hook(perturb_hook)

        torch.manual_seed(seed)
        with torch.no_grad():
            interv_traj = model.generate(
                context.clone(), n_generate, temperature
            )[0, context.shape[1]:].cpu().numpy()

        handle.remove()

        # Healing curve
        healing = np.abs(baseline_traj - interv_traj)
        healing_time = len(healing)
        for t in range(len(healing)):
            if healing[t] < 1:
                healing_time = t
                break

        return InterventionResult(
            intervention_name=self.name,
            target_variable=self.target,
            delta=self.delta_vec,
            baseline_trajectory=baseline_traj,
            intervened_trajectory=interv_traj,
            healing_curve=healing,
            healing_time=healing_time,
            q3_violations=0,
        )


class RigidityIntervention(Intervention):
    """
    Intervention: do(R_t := constant).

    Force rigidity to a fixed value by constraining output distribution.
    This is a "soft" intervention that clips/modifies logits.
    """

    def __init__(self, target_rigidity: float = 1.0, window: int = 10):
        """
        Args:
            target_rigidity: Desired R_t value (1.0 = GUE normal)
            window: Window for computing rigidity
        """
        super().__init__("do(R_t)", "R_t")
        self.target_rigidity = target_rigidity
        self.window = window

    def apply(
        self,
        model: SpacingGPT,
        context: torch.Tensor,
        n_generate: int = 50,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> InterventionResult:
        """
        Apply rigidity clamping during generation.
        """
        device = context.device
        vocab_size = model.config.vocab_size
        GUE_VAR = 0.178

        # Baseline
        torch.manual_seed(seed)
        model.eval()
        with torch.no_grad():
            baseline_traj = model.generate(
                context.clone(), n_generate, temperature
            )[0, context.shape[1]:].cpu().numpy()

        # Intervened: generate with rigidity constraint
        torch.manual_seed(seed)
        generated = context.clone()
        interv_traj = []

        target_var = self.target_rigidity * GUE_VAR

        with torch.no_grad():
            for _ in range(n_generate):
                # Get logits
                idx_cond = generated if generated.size(1) <= model.config.seq_len else \
                           generated[:, -model.config.seq_len:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature

                # Current variance in recent tokens
                if generated.shape[1] >= self.window:
                    recent = generated[0, -self.window:].float()
                    current_var = torch.var(recent).item()
                else:
                    current_var = target_var

                # Modify logits to steer toward target rigidity
                # If current_var > target: prefer tokens closer to mean
                # If current_var < target: prefer tokens farther from mean
                mean_val = recent.mean().item() if generated.shape[1] >= self.window else vocab_size / 2

                if current_var > target_var:
                    # Penalize extreme tokens
                    token_vals = torch.arange(vocab_size, device=device).float()
                    penalty = 0.1 * (token_vals - mean_val).abs()
                    logits = logits - penalty
                elif current_var < target_var * 0.8:
                    # Boost extreme tokens slightly
                    token_vals = torch.arange(vocab_size, device=device).float()
                    boost = 0.05 * (token_vals - mean_val).abs()
                    logits = logits + boost

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                interv_traj.append(next_token.item())

        interv_traj = np.array(interv_traj)

        healing = np.abs(baseline_traj - interv_traj)
        healing_time = len(healing)
        for t in range(len(healing)):
            if healing[t] < 1:
                healing_time = t
                break

        return InterventionResult(
            intervention_name=self.name,
            target_variable=self.target,
            delta=self.target_rigidity,
            baseline_trajectory=baseline_traj,
            intervened_trajectory=interv_traj,
            healing_curve=healing,
            healing_time=healing_time,
            q3_violations=0,
        )


class HiddenNoiseIntervention(Intervention):
    """
    Intervention: do(H_t := H_t + noise).

    Add Gaussian noise to hidden state and observe cascade.
    Tests if hidden state is the "root cause".
    """

    def __init__(self, noise_std: float = 0.01, layer: int = -1):
        super().__init__("do(H_t+noise)", "H_t")
        self.noise_std = noise_std
        self.layer = layer

    def apply(
        self,
        model: SpacingGPT,
        context: torch.Tensor,
        n_generate: int = 50,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> InterventionResult:
        """Apply noise to hidden state."""
        device = context.device

        # Baseline
        torch.manual_seed(seed)
        model.eval()
        with torch.no_grad():
            baseline_traj = model.generate(
                context.clone(), n_generate, temperature
            )[0, context.shape[1]:].cpu().numpy()

        # Noise hook
        def noise_hook(module, input, output):
            noise = torch.randn_like(output) * self.noise_std
            return output + noise

        target_layer = model.transformer.h[self.layer]
        handle = target_layer.register_forward_hook(noise_hook)

        torch.manual_seed(seed)
        with torch.no_grad():
            interv_traj = model.generate(
                context.clone(), n_generate, temperature
            )[0, context.shape[1]:].cpu().numpy()

        handle.remove()

        healing = np.abs(baseline_traj - interv_traj)
        healing_time = len(healing)
        for t in range(len(healing)):
            if healing[t] < 1:
                healing_time = t
                break

        return InterventionResult(
            intervention_name=self.name,
            target_variable=self.target,
            delta=self.noise_std,
            baseline_trajectory=baseline_traj,
            intervened_trajectory=interv_traj,
            healing_curve=healing,
            healing_time=healing_time,
            q3_violations=0,
        )


class InterventionSuite:
    """Run a suite of interventions and aggregate results."""

    def __init__(self, interventions: list[Intervention] = None):
        if interventions is None:
            interventions = [
                SpacingIntervention(delta=0.05, position=10),
                SpacingIntervention(delta=-0.05, position=10),
                RigidityIntervention(target_rigidity=0.5),
                RigidityIntervention(target_rigidity=1.5),
                HiddenNoiseIntervention(noise_std=0.01),
                HiddenNoiseIntervention(noise_std=0.05),
            ]
        self.interventions = interventions

    def run_all(
        self,
        model: SpacingGPT,
        contexts: list[torch.Tensor],
        **kwargs,
    ) -> list[InterventionResult]:
        """
        Run all interventions on multiple contexts.

        Args:
            model: SpacingGPT model
            contexts: List of context tensors
            **kwargs: Passed to intervention.apply()

        Returns:
            List of all results
        """
        all_results = []

        for ctx in contexts:
            for interv in self.interventions:
                result = interv.apply(model, ctx, **kwargs)
                all_results.append(result)

        return all_results

    def summarize(self, results: list[InterventionResult]) -> dict:
        """Aggregate results into summary statistics."""
        summary = {}

        for interv in self.interventions:
            name = interv.name
            relevant = [r for r in results if r.intervention_name == name]

            if relevant:
                healing_times = [r.healing_time for r in relevant]
                summary[name] = {
                    "mean_healing_time": np.mean(healing_times),
                    "std_healing_time": np.std(healing_times),
                    "healed_count": sum(1 for t in healing_times if t < 50),
                    "total_count": len(relevant),
                }

        return summary


def print_intervention_results(results: list[InterventionResult]):
    """Pretty print intervention results."""
    print("\n" + "=" * 60)
    print("INTERVENTION RESULTS")
    print("=" * 60)

    for r in results:
        print(f"\n{r.intervention_name} (target: {r.target_variable})")
        print(f"  Delta: {r.delta}")
        print(f"  Healing time: {r.healing_time} steps")
        print(f"  Q3 violations: {r.q3_violations}")
        print(f"  Max deviation: {r.healing_curve.max():.2f}")


if __name__ == "__main__":
    print("Intervention module loaded.")
    print("Available interventions:")
    print("  - SpacingIntervention: do(S_t)")
    print("  - LatentIntervention: do(Z_t)")
    print("  - RigidityIntervention: do(R_t)")
    print("  - HiddenNoiseIntervention: do(H_t+noise)")
