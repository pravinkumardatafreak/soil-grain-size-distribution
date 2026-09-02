"""
Training and Validation Engine for Soil Grain Size Distribution Prediction.

Implements:
    - 5-Fold GroupKFold Cross-Validation (Pillar 5)
    - Differentiable Weighted EMD Loss Optimization (Pillar 4)
    - Monotonic Vision Model Training (Pillars 1, 2, 3)
    - Out-Of-Fold (OOF) Benchmark Evaluation
    - 5-Fold Ensemble Test Inference & Submission Generation
"""

import os
import sys
import time
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import (
    SoilGSDDataset,
    TARGET_COLUMNS,
    build_metadata_dataframe,
)
from src.metrics import WeightedEMDLoss, compute_weighted_emd
from src.models import SoilGSDModel
from src.utils import create_submission_file, get_group_kfold_splits


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
    """Evaluates the model on validation data."""
    model.eval()
    all_preds = []
    all_targets = []
    all_sample_ids = []

    for images, targets, meta in loader:
        images = images.to(device)
        preds = model(images)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())
        all_sample_ids.extend(meta["sample_id"])

    all_preds_arr = np.vstack(all_preds)
    all_targets_arr = np.vstack(all_targets)

    # Aggregate predictions by unique sample_id
    unique_ids = list(dict.fromkeys(all_sample_ids))
    sample_preds = []
    sample_targets = []

    for sid in unique_ids:
        indices = [i for i, x in enumerate(all_sample_ids) if x == sid]
        sample_preds.append(np.mean(all_preds_arr[indices], axis=0))
        sample_targets.append(all_targets_arr[indices[0]])

    sample_preds_arr = np.array(sample_preds)
    sample_targets_arr = np.array(sample_targets)

    # Compute official competition metric at the sample level
    mean_emd = compute_weighted_emd(sample_targets_arr, sample_preds_arr)
    return mean_emd, sample_preds_arr, sample_targets_arr, unique_ids


def train_cv(
    backbone_name: str = "resnet18",
    num_epochs: int = 15,
    batch_size: int = 8,
    lr: float = 3e-4,
    physical_fov_mm: float = 50.0,
    output_size: int = 224,
    n_splits: int = 5,
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, float]:
    """
    Executes full 5-Fold Cross-Validation training loop.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} | Backbone: {backbone_name}")

    train_df, test_df = build_metadata_dataframe()
    splits = list(get_group_kfold_splits(train_df, n_splits=n_splits))

    oof_predictions = {}
    oof_ground_truth = {}
    fold_best_scores = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*20} Fold {fold + 1} / {n_splits} {'='*20}")
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        train_ds = SoilGSDDataset(
            fold_train_df,
            physical_fov_mm=physical_fov_mm,
            output_size=output_size,
            is_train=True,
        )
        val_ds = SoilGSDDataset(
            fold_val_df,
            physical_fov_mm=physical_fov_mm,
            output_size=output_size,
            is_train=False,
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False
        )

        model = SoilGSDModel(
            backbone_name=backbone_name, pretrained=True, dropout_rate=0.2
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        criterion = WeightedEMDLoss().to(device)

        best_val_emd = float("inf")
        best_preds = None
        best_targets = None
        best_ids = None
        best_model_path = os.path.join(
            checkpoint_dir, f"{backbone_name}_fold_{fold+1}.pt"
        )

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_emd, val_preds, val_targets, val_ids = evaluate_model(
                model, val_loader, device
            )
            scheduler.step()

            if val_emd < best_val_emd:
                best_val_emd = val_emd
                best_preds = val_preds
                best_targets = val_targets
                best_ids = val_ids
                torch.save(model.state_dict(), best_model_path)

            if epoch % 5 == 0 or epoch == num_epochs:
                print(
                    f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val EMD: {val_emd:.4f} | Best Val EMD: {best_val_emd:.4f}"
                )

        fold_best_scores.append(best_val_emd)
        print(f"Fold {fold+1} Best Val Weighted EMD: {best_val_emd:.4f}")

        for sid, p, t in zip(best_ids, best_preds, best_targets):
            oof_predictions[sid] = p
            oof_ground_truth[sid] = t

    # Compute Overall Out-Of-Fold Score
    oof_ids = sorted(list(oof_predictions.keys()))
    oof_p = np.array([oof_predictions[k] for k in oof_ids])
    oof_t = np.array([oof_ground_truth[k] for k in oof_ids])
    overall_oof_emd = compute_weighted_emd(oof_t, oof_p)

    print("\n" + "=" * 50)
    print(f"Fold Scores: {[round(s, 4) for s in fold_best_scores]}")
    print(f"Mean Fold Weighted EMD: {np.mean(fold_best_scores):.4f}")
    print(
        f"⭐ OVERALL OUT-OF-FOLD (OOF) WEIGHTED EMD: {overall_oof_emd:.4f} ⭐"
    )
    print("=" * 50)

    return {
        "overall_oof_emd": overall_oof_emd,
        "fold_scores": fold_best_scores,
    }


@torch.no_grad()
def generate_test_submission(
    backbone_name: str = "resnet18",
    checkpoint_dir: str = "checkpoints",
    physical_fov_mm: float = 50.0,
    output_size: int = 224,
    n_splits: int = 5,
    submission_path: str = "submission.csv",
) -> pd.DataFrame:
    """
    Generates test predictions using 5-Fold ensemble averaging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_df = build_metadata_dataframe()

    test_ds = SoilGSDDataset(
        test_df,
        physical_fov_mm=physical_fov_mm,
        output_size=output_size,
        is_train=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False, drop_last=False
    )

    models_list = []
    for fold in range(1, n_splits + 1):
        ckpt = os.path.join(checkpoint_dir, f"{backbone_name}_fold_{fold}.pt")
        if os.path.exists(ckpt):
            model = SoilGSDModel(
                backbone_name=backbone_name, pretrained=False
            ).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            models_list.append(model)

    print(f"Loaded {len(models_list)} trained fold models for test ensemble.")

    all_test_preds = []
    all_test_ids = []

    for images, _, meta in test_loader:
        images = images.to(device)
        batch_preds = []
        for model in models_list:
            preds = model(images)
            batch_preds.append(preds.cpu().numpy())

        # Average across folds
        avg_fold_preds = np.mean(batch_preds, axis=0)
        all_test_preds.append(avg_fold_preds)
        all_test_ids.extend(meta["sample_id"])

    all_test_preds_arr = np.vstack(all_test_preds)

    # Read the required submission order
    df_sub = pd.read_csv("data/raw/sample_submission.csv")
    required_sample_ids = df_sub["sample_id"].tolist()

    final_sample_predictions = []
    for sid in required_sample_ids:
        indices = [i for i, x in enumerate(all_test_ids) if x == sid]
        sample_pred = np.mean(all_test_preds_arr[indices], axis=0)
        # Guarantee physical validity
        sample_pred = np.maximum.accumulate(np.clip(sample_pred, 0.0, 100.0))
        sample_pred[-1] = 100.0
        final_sample_predictions.append(sample_pred)

    final_preds_arr = np.array(final_sample_predictions)
    sub = create_submission_file(
        required_sample_ids, final_preds_arr, output_path=submission_path
    )
    return sub


if __name__ == "__main__":
    print("Starting 5-Fold Cross-Validation Training...")
    metrics = train_cv(
        backbone_name="resnet18",
        num_epochs=10,
        batch_size=8,
        lr=3e-4,
        physical_fov_mm=50.0,
        output_size=224,
        n_splits=5,
    )
    print("\nGenerating final test submission...")
    sub = generate_test_submission(
        backbone_name="resnet18",
        submission_path="submission.csv",
    )
    print(sub.head(10))
