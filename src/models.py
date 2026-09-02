"""Coordinated Regularized Regression Architecture for Soil GSD.

Combines:
1. MultiTaskElasticNet: Imposes joint L1 group-sparsity across all 11 sieves.
2. Ridge Regression: High-dimensional L2 variance shrinkage.
3. 10-Seed Ensemble: Mitigates fold-split variance on small sample size (N=24).
"""

from typing import Tuple, List
import numpy as np
from sklearn.linear_model import Ridge, MultiTaskElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.config import SEEDS
from src.metrics import calculate_weighted_emd, calculate_median_absolute_error
from src.postprocess import enforce_physical_cdf_axioms


class CoordinatedSoilRegressor:
    """Multi-Seed Coordinated Ensemble of Ridge and MultiTaskElasticNet."""

    def __init__(self, seeds: List[int] = None):
        self.seeds = seeds or SEEDS
        self.ridge_alphas = np.logspace(-1, 5, 40)
        self.models_by_seed = []
        self.scalers_by_seed = []

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5
    ) -> Tuple[float, float, np.ndarray]:
        """Runs 5-Fold cross-validation across 10 random seeds.

        Returns:
            mean_emd: Overall Out-Of-Fold Weighted Earth Mover's Distance.
            mean_medae: Overall Out-Of-Fold Median Absolute Error.
            oof_predictions: Out-Of-Fold predicted cumulative curves (N, 11).
        """
        accumulated_oof = np.zeros_like(y, dtype=np.float64)
        seed_scores_medae = []
        seed_scores_emd = []

        print(f"\nTraining {len(self.seeds)}-Seed Coordinated Multi-Task Ensemble...")

        for seed in self.seeds:
            best_alpha = None
            best_score = float("inf")

            # 1. Grid search optimal Ridge alpha for this seed
            for alpha in self.ridge_alphas:
                oof_ridge_temp = np.zeros_like(y)
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

                for train_idx, val_idx in kf.split(X, y):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X[train_idx])
                    X_va = scaler.transform(X[val_idx])

                    model = Ridge(alpha=alpha, random_state=seed)
                    model.fit(X_tr, y[train_idx])
                    oof_ridge_temp[val_idx] = model.predict(X_va)

                oof_ridge_temp = enforce_physical_cdf_axioms(oof_ridge_temp)
                score = calculate_median_absolute_error(y, oof_ridge_temp)
                if score < best_score:
                    best_score = score
                    best_alpha = alpha

            # 2. Fit MultiTaskElasticNet across 5 folds
            oof_enet = np.zeros_like(y)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

            for train_idx, val_idx in kf.split(X, y):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[train_idx])
                X_va = scaler.transform(X[val_idx])

                # MultiTaskElasticNet imposes group-L1 sparsity across all 11 sieves
                model_enet = MultiTaskElasticNet(
                    alpha=0.5, l1_ratio=0.1, max_iter=2000, random_state=seed
                )
                model_enet.fit(X_tr, y[train_idx])
                oof_enet[val_idx] = model_enet.predict(X_va)

            oof_enet = enforce_physical_cdf_axioms(oof_enet)

            # 3. Blend Ridge + MultiTaskElasticNet (50/50)
            seed_oof_blend = 0.5 * oof_ridge_temp + 0.5 * oof_enet
            seed_oof_blend = enforce_physical_cdf_axioms(seed_oof_blend)

            seed_medae = calculate_median_absolute_error(y, seed_oof_blend)
            seed_emd = calculate_weighted_emd(y, seed_oof_blend)

            seed_scores_medae.append(seed_medae)
            seed_scores_emd.append(seed_emd)
            accumulated_oof += seed_oof_blend / len(self.seeds)

            print(
                f"Seed {seed:02d} | Ridge Alpha: {best_alpha:8.2f} | "
                f"MedAE: {seed_medae:6.4f} | Weighted EMD: {seed_emd:6.4f}"
            )

        final_oof = enforce_physical_cdf_axioms(accumulated_oof)
        total_medae = calculate_median_absolute_error(y, final_oof)
        total_emd = calculate_weighted_emd(y, final_oof)

        print("\n" + "=" * 65)
        print(f"Global 10-Seed Out-Of-Fold MedAE:        {total_medae:.4f}")
        print(f"Global 10-Seed Out-Of-Fold Weighted EMD: {total_emd:.4f}")
        print("=" * 65)

        return total_emd, total_medae, final_oof

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """Trains coordinated models on full training set and predicts on test set."""
        test_preds_accumulated = np.zeros((len(X_test), y_train.shape[1]), dtype=np.float64)

        for seed in self.seeds:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            # Fit Ridge
            ridge = Ridge(alpha=58.78, random_state=seed)
            ridge.fit(X_tr_s, y_train)
            pred_ridge = ridge.predict(X_te_s)

            # Fit MultiTaskElasticNet
            enet = MultiTaskElasticNet(
                alpha=0.5, l1_ratio=0.1, max_iter=2000, random_state=seed
            )
            enet.fit(X_tr_s, y_train)
            pred_enet = enet.predict(X_te_s)

            seed_pred = 0.5 * pred_ridge + 0.5 * pred_enet
            test_preds_accumulated += seed_pred / len(self.seeds)

        return enforce_physical_cdf_axioms(test_preds_accumulated)
