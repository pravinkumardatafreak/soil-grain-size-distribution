"""
Utility functions for Soil Grain Size Distribution (GSD) prediction.

Includes:
    - GroupKFold partitioning by soil sample_id (Pillar 5)
    - Professional Geotechnical GSD curve plotting (Semi-logarithmic DIN EN ISO 14688-1)
    - Submission file generation with integrity validation
"""

import os
import sys
from typing import Generator, List, Optional, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.dataset import TARGET_COLUMNS
from src.metrics import DIAMETERS_MM, compute_weighted_emd


def get_group_kfold_splits(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generates GroupKFold cross-validation splits grouped strictly by sample_id (Pillar 5).

    Args:
        df (pd.DataFrame): Training metadata dataframe containing 'sample_id'.
        n_splits (int): Number of folds (default 5).

    Yields:
        Tuple[np.ndarray, np.ndarray]: (train_indices, val_indices)
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["sample_id"].values
    return gkf.split(df, groups=groups)


def plot_gsd_curves(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    sample_title: str = "Soil Grain Size Distribution",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots professional geotechnical cumulative grain size distribution curves on a semi-log scale.

    Args:
        y_true (np.ndarray): Ground truth cumulative mass percentage vector (11,).
        y_pred (Optional[np.ndarray]): Predicted cumulative mass percentage vector (11,).
        sample_title (str): Title for the plot.
        save_path (Optional[str]): Filepath to save the plot image.
    """
    plt.figure(figsize=(10, 6))

    # Plot Ground Truth
    plt.plot(
        DIAMETERS_MM,
        y_true,
        "o-",
        color="#1f77b4",
        linewidth=2.5,
        markersize=6,
        label="Ground Truth (Sieve Analysis)",
    )

    # Plot Prediction if provided
    if y_pred is not None:
        score = compute_weighted_emd(y_true, y_pred)
        plt.plot(
            DIAMETERS_MM,
            y_pred,
            "s--",
            color="#d62728",
            linewidth=2.0,
            markersize=6,
            label=f"Predicted GSD (Weighted EMD: {score:.3f})",
        )

    # Geotechnical soil fraction vertical division lines
    boundaries = [
        (0.002, "Clay / Silt"),
        (0.063, "Silt / Sand"),
        (2.0, "Sand / Gravel"),
        (63.0, "Gravel / Cobbles"),
    ]
    for x_val, label in boundaries:
        plt.axvline(
            x=x_val, color="gray", linestyle=":", alpha=0.6, linewidth=1.2
        )

    plt.xscale("log")
    plt.xlim(0.001, 250.0)
    plt.ylim(-2.0, 105.0)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.xlabel("Particle Diameter $d$ [mm] (Logarithmic Scale)", fontsize=12)
    plt.ylabel("Cumulative Mass Passing $F(d)$ [%]", fontsize=12)
    plt.title(f"{sample_title} (DIN EN ISO 14688-1)", fontsize=14, pad=12)
    plt.legend(loc="upper left", frameon=True, fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def create_submission_file(
    sample_ids: List[str],
    predictions: np.ndarray,
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """
    Creates and validates a Kaggle-compliant submission CSV file.

    Args:
        sample_ids (List[str]): List of 10 test sample IDs.
        predictions (np.ndarray): Predicted matrix of shape (10, 11).
        output_path (str): Destination CSV filepath.

    Returns:
        pd.DataFrame: Formatted submission dataframe.
    """
    # Verify shape
    assert predictions.shape == (
        len(sample_ids),
        11,
    ), f"Expected shape ({len(sample_ids)}, 11), got {predictions.shape}"

    # Verify physical validity
    for i, row in enumerate(predictions):
        assert np.all(
            np.diff(row) >= -1e-5
        ), f"Sample {sample_ids[i]} violates monotonicity!"
        assert (
            row.min() >= -1e-5 and row.max() <= 100.0 + 1e-5
        ), f"Sample {sample_ids[i]} out of [0, 100]% range!"
        assert (
            abs(row[-1] - 100.0) < 1e-3
        ), f"Sample {sample_ids[i]} last bin is not 100%!"

    sub_df = pd.DataFrame(predictions, columns=TARGET_COLUMNS)
    sub_df.insert(0, "sample_id", sample_ids)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to: {output_path}")
    return sub_df


if __name__ == "__main__":
    from src.dataset import build_metadata_dataframe

    train_df, _ = build_metadata_dataframe()

    # Test GroupKFold split
    splits = list(get_group_kfold_splits(train_df, n_splits=5))
    print(f"Generated {len(splits)} GroupKFold splits.")
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_samples = set(train_df.iloc[train_idx]["sample_id"])
        val_samples = set(train_df.iloc[val_idx]["sample_id"])
        overlap = train_samples.intersection(val_samples)
        print(
            f"Fold {fold+1}: Train samples = {len(train_samples)}, Val samples = {len(val_samples)}, Leakage Overlap = {len(overlap)}"
        )
        assert len(overlap) == 0, "Catastrophic data leakage detected in fold!"

    # Test GSD plotting
    sample_true = train_df.iloc[0][TARGET_COLUMNS].values.astype(np.float64)
    sample_pred = sample_true + np.random.uniform(
        -2.0, 2.0, size=sample_true.shape
    )
    sample_pred = np.maximum.accumulate(np.clip(sample_pred, 0.0, 100.0))
    sample_pred[-1] = 100.0

    plot_gsd_curves(
        sample_true,
        sample_pred,
        sample_title="Soil Sample F827",
        save_path="notebooks/sample_gsd_plot.png",
    )
    print(
        "Plot generated and saved to notebooks/sample_gsd_plot.png successfully!"
    )
