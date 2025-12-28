"""
Conditional Independence Tests for Causal Zeta.

Uses HSIC (Hilbert-Schmidt Independence Criterion) to test
whether two variables are independent given a conditioning set.

Key tests for graph validation:
1. Y_t _||_ S_{t-1} | Z_t  (Does phase screen off local effect?)
2. R_t _||_ S_{t-1} | Z_t  (Is rigidity global, not local?)
3. Y_t _||_ Z_t | H_t      (Is phase redundant given full hidden?)
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gamma
from dataclasses import dataclass
from typing import Optional
from .variables import CausalState


@dataclass
class CITestResult:
    """Result of a conditional independence test."""
    var_x: str
    var_y: str
    conditioning_set: list[str]
    hsic_value: float
    p_value: float
    independent: bool  # True if p > threshold
    threshold: float = 0.05


class HSICTest:
    """
    HSIC-based independence test.

    HSIC measures dependence between two random variables using kernel methods.
    HSIC = 0 iff X and Y are independent.
    """

    def __init__(self, sigma: Optional[float] = None):
        """
        Args:
            sigma: Kernel bandwidth. If None, use median heuristic.
        """
        self.sigma = sigma

    def _rbf_kernel(self, X: np.ndarray, sigma: float) -> np.ndarray:
        """Compute RBF (Gaussian) kernel matrix."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        pairwise_sq = squareform(pdist(X, 'sqeuclidean'))
        return np.exp(-pairwise_sq / (2 * sigma ** 2))

    def _median_heuristic(self, X: np.ndarray) -> float:
        """Compute median of pairwise distances."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        dists = pdist(X, 'euclidean')
        return np.median(dists) if len(dists) > 0 else 1.0

    def _center_kernel(self, K: np.ndarray) -> np.ndarray:
        """Center the kernel matrix."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def hsic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute HSIC between X and Y.

        Args:
            X: [n, dx] or [n,] array
            Y: [n, dy] or [n,] array

        Returns:
            HSIC value (higher = more dependent)
        """
        n = len(X)
        if n < 5:
            return 0.0

        # Bandwidth selection
        sigma_x = self.sigma or self._median_heuristic(X)
        sigma_y = self.sigma or self._median_heuristic(Y)

        # Kernel matrices
        Kx = self._rbf_kernel(X, sigma_x)
        Ky = self._rbf_kernel(Y, sigma_y)

        # Center and compute HSIC
        Kxc = self._center_kernel(Kx)
        Kyc = self._center_kernel(Ky)

        hsic_val = np.trace(Kxc @ Kyc) / (n - 1) ** 2
        return hsic_val

    def test_independence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_permutations: int = 100,
        threshold: float = 0.05,
    ) -> tuple[float, float, bool]:
        """
        Test if X and Y are independent using permutation test.

        Args:
            X, Y: Data arrays
            n_permutations: Number of permutations for p-value
            threshold: Significance level

        Returns:
            (hsic_value, p_value, is_independent)
        """
        observed_hsic = self.hsic(X, Y)

        # Permutation test
        null_hsic = []
        for _ in range(n_permutations):
            Y_perm = np.random.permutation(Y)
            null_hsic.append(self.hsic(X, Y_perm))

        null_hsic = np.array(null_hsic)
        p_value = np.mean(null_hsic >= observed_hsic)

        return observed_hsic, p_value, p_value > threshold


class ConditionalHSIC(HSICTest):
    """
    Conditional HSIC for testing X _||_ Y | Z.

    Uses residualization: test independence of residuals after
    regressing out Z.
    """

    def _residualize(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Compute residuals of X after regressing on Z.

        Uses kernel ridge regression.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        n = len(X)
        sigma_z = self._median_heuristic(Z)
        Kz = self._rbf_kernel(Z, sigma_z)

        # Kernel ridge regression
        alpha = 1e-3  # Regularization
        fitted = Kz @ np.linalg.solve(Kz + alpha * n * np.eye(n), X)

        return X - fitted

    def conditional_hsic(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> float:
        """
        Compute conditional HSIC: HSIC(X, Y | Z).
        """
        X_res = self._residualize(X, Z)
        Y_res = self._residualize(Y, Z)
        return self.hsic(X_res.flatten(), Y_res.flatten())

    def test_conditional_independence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        n_permutations: int = 100,
        threshold: float = 0.05,
    ) -> tuple[float, float, bool]:
        """
        Test X _||_ Y | Z.
        """
        X_res = self._residualize(X, Z)
        Y_res = self._residualize(Y, Z)
        return self.test_independence(
            X_res.flatten(), Y_res.flatten(),
            n_permutations, threshold
        )


def extract_variable(states: list[CausalState], var_name: str) -> np.ndarray:
    """Extract a variable from list of CausalState.

    Variables:
        S_{t-1}: Current spacing (input to model)
        S_deep: Spacing OUTSIDE R_t window (for CI-1 test)
        Z_t: Latent mode [2,] from PCA
        R_t: Rigidity proxy
        Y_t: Target spacing
    """
    if var_name in ("S_deep", "S_{t-2}"):  # S_deep is stored in S_t_prev
        return np.array([s.S_t_prev if hasattr(s, 'S_t_prev') and s.S_t_prev is not None else s.S_t for s in states])
    elif var_name == "S_{t-1}":
        return np.array([s.S_t for s in states])
    elif var_name == "Z_t":
        return np.array([s.Z_t for s in states])
    elif var_name == "R_t":
        return np.array([s.R_t for s in states])
    elif var_name == "Y_t":
        return np.array([s.Y_t for s in states])
    else:
        raise ValueError(f"Unknown variable: {var_name}")


def run_ci_tests(
    states: list[CausalState],
    tests: list[tuple[str, str, list[str]]],
    n_permutations: int = 100,
    threshold: float = 0.05,
) -> list[CITestResult]:
    """
    Run a batch of CI tests.

    Args:
        states: List of CausalState objects
        tests: List of (X, Y, [Z1, Z2, ...]) tuples
        n_permutations: Permutations for p-value
        threshold: Significance level

    Returns:
        List of CITestResult
    """
    results = []
    hsic_test = ConditionalHSIC()

    for x_name, y_name, z_names in tests:
        X = extract_variable(states, x_name)
        Y = extract_variable(states, y_name)

        if z_names:
            Z_list = [extract_variable(states, z) for z in z_names]
            Z = np.column_stack(Z_list) if len(Z_list) > 1 else Z_list[0]
            hsic_val, p_val, is_indep = hsic_test.test_conditional_independence(
                X, Y, Z, n_permutations, threshold
            )
        else:
            hsic_val, p_val, is_indep = hsic_test.test_independence(
                X, Y, n_permutations, threshold
            )

        results.append(CITestResult(
            var_x=x_name,
            var_y=y_name,
            conditioning_set=z_names,
            hsic_value=hsic_val,
            p_value=p_val,
            independent=is_indep,
            threshold=threshold,
        ))

    return results


# Pre-defined test suites

def graph_validation_tests_v01() -> list[tuple[str, str, list[str]]]:
    """
    Tests for DAG v0.1 (DEPRECATED after Round 003).

    CI-A: R_t ⊥ S_{t-1} | Z_t → FAILED in all rounds → added edge S→R
    """
    return [
        ("R_t", "S_{t-1}", ["Z_t"]),
        ("Y_t", "Z_t", []),
    ]


def graph_validation_tests() -> list[tuple[str, str, list[str]]]:
    """
    Tests to validate DAG v0.2 (with S_{t-1} → R_t edge).

    Returns list of (X, Y, Z) tuples for CI tests.

    CI-1: R_t ⊥ S_deep | (S_{t-1}, Z_t)
        S_deep = spacing OUTSIDE the R_t window (at position t - L_R - 1).
        Tests if rigidity is explained by immediate parents.

        If PASS: rigidity depends only on immediate context (DAG is sufficient)
        If FAIL: R_t has longer memory, need to expand R definition or window

    CI-2: Y_t ⊥ Z_t (unconditional independence test)
        Tests if Z_t is informative about Y_t.
        Expected: FAIL (i.e., they ARE dependent) → Z_t is meaningful
        NOTE: Independence test FAIL = variables ARE dependent!
    """
    return [
        # CI-1: Screening test for new DAG
        # R_t ⊥ S_deep | (S_{t-1}, Z_t) where S_deep is OUTSIDE R_t window
        # If PASS: immediate parents sufficient
        # If FAIL: R_t has longer memory
        ("R_t", "S_deep", ["S_{t-1}", "Z_t"]),

        # CI-2: Z_t informativeness (independence test)
        # FAIL = dependent (expected!) → Z_t is meaningful
        ("Y_t", "Z_t", []),
    ]


def graph_validation_tests_extended() -> list[tuple[str, str, list[str]]]:
    """
    Extended tests for deeper analysis.
    """
    return [
        # Basic screening tests
        ("R_t", "S_{t-2}", ["S_{t-1}", "Z_t"]),
        ("Y_t", "Z_t", []),

        # Additional tests
        ("Y_t", "S_{t-2}", ["S_{t-1}", "Z_t"]),  # Long-range Y dependence
        ("Y_t", "R_t", []),  # R→Y edge check
    ]


def extended_ci_tests() -> list[tuple[str, str, list[str]]]:
    """
    Extended tests for deeper analysis (after MVP).

    Returns list of (X, Y, Z) tuples for CI tests.
    """
    return [
        # Does Z_t screen off S_{t-1} from Y_t?
        # Note: DAG has direct edge S→Y, so this MAY fail!
        ("Y_t", "S_{t-1}", ["Z_t"]),

        # Is Y_t dependent on R_t?
        ("Y_t", "R_t", []),

        # Is R_t dependent on Z_t?
        ("R_t", "Z_t", []),
    ]


def print_ci_results(results: list[CITestResult]):
    """Pretty print CI test results."""
    print("\n" + "=" * 60)
    print("CONDITIONAL INDEPENDENCE TEST RESULTS")
    print("=" * 60)

    for r in results:
        z_str = ", ".join(r.conditioning_set) if r.conditioning_set else "{}"
        status = "[PASS]" if r.independent else "[FAIL]"
        status_color = "\033[92m" if r.independent else "\033[91m"
        reset = "\033[0m"

        print(f"\n{status_color}{status}{reset} {r.var_x} _||_ {r.var_y} | {{{z_str}}}")
        print(f"    HSIC = {r.hsic_value:.6f}")
        print(f"    p-value = {r.p_value:.4f}")
        print(f"    threshold = {r.threshold}")

    # Summary
    n_pass = sum(1 for r in results if r.independent)
    print(f"\n{'=' * 60}")
    print(f"Summary: {n_pass}/{len(results)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 500

    # Generate synthetic causal data
    # Z -> S, Z -> Y, R -> Y
    Z = np.random.randn(n, 2)
    S = Z[:, 0] + 0.5 * np.random.randn(n)
    R = 0.3 * Z[:, 1] + 0.7 * np.random.randn(n)
    Y = 0.4 * Z[:, 0] + 0.3 * R + 0.3 * np.random.randn(n)

    states = [
        CausalState(t=i, S_t=S[i], Z_t=Z[i], R_t=R[i], Y_t=Y[i])
        for i in range(n)
    ]

    print("Testing CI on synthetic data...")
    tests = graph_validation_tests()
    results = run_ci_tests(states, tests, n_permutations=50)
    print_ci_results(results)
