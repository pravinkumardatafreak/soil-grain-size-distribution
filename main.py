"""Command-line interface for the Soil Grain Size Distribution Pipeline.

Usage:
    # 1. Run local cross-validation to get the exact local score without Kaggle:
    python main.py --evaluate

    # 2. Run inference to generate submission.csv:
    python main.py --predict

    # 3. Run complete evaluation + prediction pipeline:
    python main.py --all
"""

import argparse
import sys
import numpy as np
import pandas as pd
import torch

from src.config import (
    TRAIN_LABELS_PATH,
    TEST_SAMPLE_SUB_PATH,
    TRAIN_PHOTOS_DIR,
    TEST_PHOTOS_DIR,
    SUBMISSION_PATH,
    SIEVE_COLUMNS,
    CACHE_DIR,
)
from src.data_pipeline import build_image_map
from src.feature_extractor import DualBackboneExtractor, extract_sample_level_features
from src.models import CoordinatedSoilRegressor
from src.metrics import calculate_per_sieve_mae


def load_data():
    """Loads training ground-truth labels and sample submission template."""
    train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    sample_sub_df = pd.read_csv(TEST_SAMPLE_SUB_PATH)
    return train_labels_df, sample_sub_df


def extract_features(train_labels_df, sample_sub_df, device="cpu"):
    """Extracts or loads cached sample-level dual-backbone feature matrices."""
    print("\n[Step 1/3] Initializing Dual-Backbone Feature Extractor...")
    extractor = DualBackboneExtractor(device=device)

    train_img_map = build_image_map(TRAIN_PHOTOS_DIR)
    test_img_map = build_image_map(TEST_PHOTOS_DIR)

    train_cache = CACHE_DIR / "train_features.parquet"
    test_cache = CACHE_DIR / "test_features.parquet"

    X_train_df = extract_sample_level_features(
        sample_ids=list(train_labels_df["sample_id"]),
        img_map=train_img_map,
        extractor=extractor,
        desc="Train Set (24 soils)",
        cache_file=train_cache,
    )

    X_test_df = extract_sample_level_features(
        sample_ids=list(sample_sub_df["sample_id"]),
        img_map=test_img_map,
        extractor=extractor,
        desc="Test Set (10 soils)",
        cache_file=test_cache,
    )

    common_cols = [
        c for c in X_train_df.columns if c in X_test_df.columns and c != "sample_id"
    ]

    X_train = X_train_df[common_cols].values.astype(np.float32)
    y_train = train_labels_df[SIEVE_COLUMNS].values.astype(np.float32)
    X_test = X_test_df[common_cols].values.astype(np.float32)

    print(f"Features ready! Extracted {len(common_cols)} visual and physical features.")
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
    return X_train, y_train, X_test, common_cols


def run_evaluation(X_train, y_train):
    """Executes 5-fold cross-validation locally without requiring Kaggle."""
    print("\n[Step 2/3] Executing Local Cross-Validation (Evaluating Without Kaggle)...")
    regressor = CoordinatedSoilRegressor()
    total_emd, total_medae, oof_preds = regressor.cross_validate(X_train, y_train)

    print("\nGranular Per-Sieve Mean Absolute Error Breakdown:")
    per_sieve = calculate_per_sieve_mae(y_train, oof_preds)
    for sieve, err in per_sieve.items():
        print(f"  - Sieve {sieve:>6s} mm : {err:5.2f}% average error")

    return total_emd, total_medae


def run_prediction(X_train, y_train, X_test, sample_sub_df):
    """Fits coordinated ensemble on full training set and writes submission.csv."""
    print("\n[Step 3/3] Generating Physical Cumulative Sieve Curves...")
    regressor = CoordinatedSoilRegressor()
    test_preds = regressor.fit_predict(X_train, y_train, X_test)

    sub_df = pd.DataFrame(test_preds, columns=SIEVE_COLUMNS)
    sub_df.insert(0, "sample_id", sample_sub_df["sample_id"])
    sub_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nSaved validated submission to: {SUBMISSION_PATH}")
    print(sub_df.to_string())
    return sub_df


def main():
    parser = argparse.ArgumentParser(
        description="Soil Grain Size Distribution Modeling Pipeline"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run local cross-validation"
    )
    parser.add_argument(
        "--predict", action="store_true", help="Generate test predictions"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run both evaluation and prediction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )

    args = parser.parse_args()

    # Default to evaluation if no flags are passed
    if not (args.evaluate or args.predict or args.all):
        args.evaluate = True

    print("=" * 65)
    print(" [+] SOIL GRAIN SIZE DISTRIBUTION (GSD) PIPELINE")
    print(" Architecture: Multi-Backbone + Coordinated MultiTaskElasticNet")
    print("=" * 65)

    train_labels_df, sample_sub_df = load_data()
    X_train, y_train, X_test, common_cols = extract_features(
        train_labels_df, sample_sub_df, device=args.device
    )

    if args.evaluate or args.all:
        run_evaluation(X_train, y_train)

    if args.predict or args.all:
        run_prediction(X_train, y_train, X_test, sample_sub_df)

    print("\n Pipeline execution complete!")


if __name__ == "__main__":
    main()
