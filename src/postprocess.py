"""Post-processing module enforcing physical laws of soil mechanics.

Cumulative Distribution Functions (CDFs) must strictly satisfy:
1. Non-negativity and upper bound: F(x) in [0.0, 100.0]
2. Monotonic non-decreasing law: F(x_i) <= F(x_{i+1})
3. Boundary condition: F(200 mm) == 100.0 (all soil passes 200 mm sieve)
"""

import numpy as np


def enforce_physical_cdf_axioms(predictions: np.ndarray) -> np.ndarray:
    """Enforces monotonicity, range bounds, and boundary conditions.

    Args:
        predictions: Unconstrained regression output of shape (N, 11).

    Returns:
        Physically valid cumulative distribution array of shape (N, 11).
    """
    # 1. Clip bounds to physically valid range [0, 100]%
    clipped = np.clip(predictions, 0.0, 100.0)

    # 2. Enforce monotonic non-decreasing law
    monotonic = np.sort(clipped, axis=1)

    # 3. Enforce boundary axiom: 100% of particles pass the 200mm sieve
    monotonic[:, -1] = 100.0

    return monotonic
