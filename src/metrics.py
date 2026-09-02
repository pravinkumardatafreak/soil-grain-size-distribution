"""Evaluation metrics for Soil Grain Size Distribution (GSD) Modeling.

Implements the official competition metric:
Weighted Earth Mover's Distance (EMD), Median Absolute Error (MedAE),
and granular per-sieve physical diagnostic errors.
"""

from typing import Dict
import numpy as np
from src.config import INTERVAL_WEIGHTS, SIEVE_COLUMNS


def calculate_weighted_emd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the Weighted Earth Mover's Distance between cumulative curves.

    Mathematical Formulation:
        EMD = mean( sum_{i=1}^{10} |y_pred_i - y_true_i| * delta_log10(x_i) )

    Args:
        y_true: Ground-truth cumulative passing percentage matrix (N, 11).
        y_pred: Predicted cumulative passing percentage matrix (N, 11).

    Returns:
        Scalar mean Weighted EMD across all soil samples.
    """
    abs_diff = np.abs(y_pred[:, :10] - y_true[:, :10])
    weighted_diff = abs_diff * INTERVAL_WEIGHTS
    return float(np.mean(np.sum(weighted_diff, axis=1)))


def calculate_median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes median absolute error across all sieve predictions."""
    abs_diff = np.abs(y_pred[:, :10] - y_true[:, :10])
    return float(np.median(abs_diff))


def calculate_per_sieve_mae(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Provides a granular per-sieve diagnostic error breakdown."""
    abs_diff = np.abs(y_pred - y_true)
    mean_errors = np.mean(abs_diff, axis=0)
    return {col: float(err) for col, err in zip(SIEVE_COLUMNS, mean_errors)}
