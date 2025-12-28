"""
Q3 Constraint Validators for Causal Zeta.

Q3 constraints are used as VALIDATORS, not teachers.
We don't train the model to satisfy Q3 - we check if
generated trajectories are "physically plausible".

Key constraints from Q3:
C1: Rigidity - variance bounded (spectral form factor)
C2: Repulsion - no zero spacings (level repulsion)
C3: Floor/Cap - bounds from ρ(t) formulation
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from scipy import stats


@dataclass
class ValidationResult:
    """Result of Q3 validation."""
    constraint_name: str
    passed: bool
    value: float
    threshold: float
    details: str


class Q3Validator:
    """
    Validator for Q3-derived constraints on spacing trajectories.

    These are NOT trained into the model. They are external validators
    that tell us if a generated trajectory is "physically plausible"
    according to RMT/Q3 theory.

    CRITICAL: Always convert bin indices → spacing values via bin_centers!
    """

    # Physical constants from Q3/GUE
    GUE_MEAN_SPACING = 1.0  # After unfolding
    GUE_VARIANCE = 0.178    # Theoretical variance for normalized spacings
    MIN_SPACING = 0.01      # Repulsion floor (no exact zeros)
    MAX_SPACING = 4.0       # Practical cap (very large gaps rare)

    # Q3-derived constants (from PROSHKA)
    FLOOR_C_STAR = 11 / 10  # c* = 11/10 uniform floor
    CAP_RHO_1 = 1 / 25      # ρ(1) < 1/25 prime cap

    def __init__(self, bin_centers: np.ndarray = None):
        """
        Args:
            bin_centers: Array mapping bin index → spacing value [vocab_size,]
                         If None, assumes input is already spacing values
        """
        self.bin_centers = bin_centers

    def bin_to_spacing(self, bins: np.ndarray) -> np.ndarray:
        """Convert bin indices to spacing values using bin_centers."""
        if self.bin_centers is None:
            return bins.astype(float)
        return self.bin_centers[bins.astype(int)]

    def validate_trajectory(
        self,
        trajectory: np.ndarray,
        is_bins: bool = True,
    ) -> list[ValidationResult]:
        """
        Run all Q3 validators on a trajectory.

        Args:
            trajectory: [T,] array of spacings or bins
            is_bins: If True, convert from bin indices

        Returns:
            List of ValidationResult for each constraint
        """
        if is_bins:
            spacings = self.bin_to_spacing(trajectory)
        else:
            spacings = trajectory

        results = []
        results.append(self.check_rigidity(spacings))
        results.append(self.check_repulsion(spacings))
        results.append(self.check_mean_constraint(spacings))
        results.append(self.check_spacing_bounds(spacings))
        results.append(self.check_gue_distribution(spacings))

        return results

    def check_rigidity(self, spacings: np.ndarray, window: int = 20) -> ValidationResult:
        """
        C1: Rigidity constraint.

        In GUE, the number variance is suppressed (log growth vs linear).
        We check: Var(spacings) / GUE_VARIANCE < threshold.
        """
        if len(spacings) < window:
            return ValidationResult(
                "C1_Rigidity", True, 0.0, 2.0,
                "Trajectory too short for rigidity check"
            )

        # Compute local variances over windows
        variances = []
        for i in range(0, len(spacings) - window, window // 2):
            window_data = spacings[i:i+window]
            variances.append(np.var(window_data))

        mean_var = np.mean(variances)
        normalized = mean_var / self.GUE_VARIANCE

        # Threshold: variance should be within 2x of GUE expected
        threshold = 2.0
        passed = normalized < threshold

        return ValidationResult(
            constraint_name="C1_Rigidity",
            passed=passed,
            value=normalized,
            threshold=threshold,
            details=f"Normalized variance {normalized:.3f} (threshold {threshold})"
        )

    def check_repulsion(self, spacings: np.ndarray) -> ValidationResult:
        """
        C2: Repulsion constraint.

        Level repulsion: P(s→0) ~ s^β where β=2 for GUE.
        Check: no spacing below MIN_SPACING threshold.
        """
        min_spacing = np.min(spacings)
        n_tiny = np.sum(spacings < self.MIN_SPACING)
        fraction_tiny = n_tiny / len(spacings)

        # Allow at most 1% tiny spacings
        threshold = 0.01
        passed = fraction_tiny < threshold

        return ValidationResult(
            constraint_name="C2_Repulsion",
            passed=passed,
            value=fraction_tiny,
            threshold=threshold,
            details=f"Min spacing {min_spacing:.4f}, {n_tiny} below {self.MIN_SPACING}"
        )

    def check_mean_constraint(self, spacings: np.ndarray) -> ValidationResult:
        """
        Mean spacing should be close to 1.0 (after unfolding).
        """
        mean = np.mean(spacings)
        deviation = abs(mean - self.GUE_MEAN_SPACING)

        # Allow 10% deviation
        threshold = 0.1
        passed = deviation < threshold

        return ValidationResult(
            constraint_name="C_Mean",
            passed=passed,
            value=deviation,
            threshold=threshold,
            details=f"Mean spacing {mean:.4f}, deviation {deviation:.4f}"
        )

    def check_spacing_bounds(self, spacings: np.ndarray) -> ValidationResult:
        """
        Spacing bounds: no spacing should exceed MAX_SPACING.
        """
        max_spacing = np.max(spacings)
        n_large = np.sum(spacings > self.MAX_SPACING)
        fraction_large = n_large / len(spacings)

        threshold = 0.01
        passed = fraction_large < threshold

        return ValidationResult(
            constraint_name="C_Bounds",
            passed=passed,
            value=fraction_large,
            threshold=threshold,
            details=f"Max spacing {max_spacing:.4f}, {n_large} above {self.MAX_SPACING}"
        )

    def check_gue_distribution(self, spacings: np.ndarray) -> ValidationResult:
        """
        Check if spacing distribution matches Wigner surmise.

        Uses Kolmogorov-Smirnov test against GUE theoretical.
        """
        def wigner_surmise_cdf(s):
            """CDF of Wigner surmise: P(s) = (π*s/2) * exp(-π*s²/4)"""
            return 1 - np.exp(-np.pi * s**2 / 4)

        # K-S test
        ks_stat, p_value = stats.kstest(spacings, wigner_surmise_cdf)

        # Pass if p-value > 0.01 (not strongly rejecting GUE)
        threshold = 0.01
        passed = p_value > threshold

        return ValidationResult(
            constraint_name="C_GUE_Distribution",
            passed=passed,
            value=p_value,
            threshold=threshold,
            details=f"K-S statistic {ks_stat:.4f}, p-value {p_value:.4f}"
        )

    def count_violations(self, results: list[ValidationResult]) -> int:
        """Count number of failed constraints."""
        return sum(1 for r in results if not r.passed)

    def validate_batch(
        self,
        trajectories: np.ndarray,
        is_bins: bool = True,
    ) -> dict:
        """
        Validate a batch of trajectories.

        Args:
            trajectories: [B, T] array
            is_bins: If True, convert from bins

        Returns:
            Summary statistics
        """
        all_results = []
        for i in range(len(trajectories)):
            results = self.validate_trajectory(trajectories[i], is_bins)
            all_results.append(results)

        # Aggregate
        constraint_names = [r.constraint_name for r in all_results[0]]
        summary = {}

        for name in constraint_names:
            passed_count = sum(
                1 for results in all_results
                for r in results if r.constraint_name == name and r.passed
            )
            summary[name] = {
                "passed": passed_count,
                "total": len(all_results),
                "rate": passed_count / len(all_results),
            }

        # Overall
        fully_passed = sum(
            1 for results in all_results
            if all(r.passed for r in results)
        )
        summary["overall"] = {
            "fully_valid": fully_passed,
            "total": len(all_results),
            "rate": fully_passed / len(all_results),
        }

        return summary


class SFFValidator:
    """
    Spectral Form Factor validator.

    The SFF K(τ) shows characteristic GUE behavior:
    - Linear ramp for τ < 1
    - Plateau at 1 for τ ≥ 1

    This is a more sophisticated rigidity test.
    """

    def compute_sff(self, spacings: np.ndarray, tau_max: float = 2.0, n_points: int = 100):
        """
        Compute spectral form factor from spacings.

        K(τ) = |Σ_n exp(2πi * u_n * τ)|² / N

        where u_n = Σ_{k<n} s_k (unfolded positions).
        """
        # Reconstruct unfolded positions
        u = np.cumsum(spacings)
        N = len(u)

        tau_vals = np.linspace(0.01, tau_max, n_points)
        K_vals = []

        for tau in tau_vals:
            # Fourier sum
            phases = 2 * np.pi * u * tau
            fourier_sum = np.sum(np.exp(1j * phases))
            K = np.abs(fourier_sum)**2 / N
            K_vals.append(K)

        return tau_vals, np.array(K_vals)

    def validate_sff(self, spacings: np.ndarray) -> ValidationResult:
        """
        Validate SFF behavior.

        Check that SFF shows ramp -> plateau transition.
        """
        tau, K = self.compute_sff(spacings)

        # Check ramp region (τ < 0.5)
        ramp_mask = tau < 0.5
        ramp_K = K[ramp_mask]
        ramp_tau = tau[ramp_mask]

        # Linear fit on ramp
        if len(ramp_tau) > 2:
            slope, intercept = np.polyfit(ramp_tau, ramp_K, 1)
        else:
            slope = 0

        # Check plateau region (τ > 1.0)
        plateau_mask = tau > 1.0
        plateau_K = K[plateau_mask]
        plateau_mean = np.mean(plateau_K) if len(plateau_K) > 0 else 0
        plateau_std = np.std(plateau_K) if len(plateau_K) > 0 else 1

        # Pass if:
        # 1. Ramp has positive slope
        # 2. Plateau is relatively flat (std/mean < 0.5)
        ramp_ok = slope > 0
        plateau_ok = plateau_std / (plateau_mean + 1e-6) < 0.5

        passed = ramp_ok and plateau_ok

        return ValidationResult(
            constraint_name="C_SFF",
            passed=passed,
            value=slope,
            threshold=0.0,
            details=f"Ramp slope {slope:.3f}, plateau std/mean {plateau_std/(plateau_mean+1e-6):.3f}"
        )


def print_validation_results(results: list[ValidationResult]):
    """Pretty print validation results."""
    print("\n" + "=" * 60)
    print("Q3 CONSTRAINT VALIDATION")
    print("=" * 60)

    for r in results:
        status = "[PASS]" if r.passed else "[FAIL]"
        color = "\033[92m" if r.passed else "\033[91m"
        reset = "\033[0m"

        print(f"\n{color}{status}{reset} {r.constraint_name}")
        print(f"    Value: {r.value:.4f}")
        print(f"    Threshold: {r.threshold}")
        print(f"    Details: {r.details}")

    # Summary
    n_pass = sum(1 for r in results if r.passed)
    print(f"\n{'=' * 60}")
    print(f"Summary: {n_pass}/{len(results)} constraints passed")
    if n_pass == len(results):
        print("\033[92mTrajectory is Q3-valid!\033[0m")
    else:
        print("\033[91mTrajectory has Q3 violations\033[0m")
    print("=" * 60)


if __name__ == "__main__":
    # Test with synthetic GUE-like data
    np.random.seed(42)

    # Generate Wigner surmise samples
    n = 1000
    u = np.random.random(n)
    # Inverse CDF: s = sqrt(-4/π * ln(1-u))
    spacings = np.sqrt(-4 / np.pi * np.log(1 - u))

    print("Testing Q3Validator with synthetic GUE data...")
    validator = Q3Validator()
    results = validator.validate_trajectory(spacings, is_bins=False)
    print_validation_results(results)
