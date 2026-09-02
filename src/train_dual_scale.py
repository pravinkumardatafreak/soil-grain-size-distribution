"""
Dual-Scale Multi-Resolution Cross-Validation & TTA Inference Engine.

Implements:
    - Pillar 1: Physical Scale Normalization (PPM calibration)
    - Pillar 2: Dual-Scale Multi-Resolution Feature Extraction (25mm micro + 100mm macro)
    - Pillar 3: Guaranteed Monotonic Distribution Head (Softmax + Cumsum)
    - Pillar 4: Direct Differentiable Weighted EMD Loss
    - Pillar 5: Leak-Proof 5-Fold GroupKFold
    - Multi-Crop Test-Time Augmentation (TTA)
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as TF

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import (
    DualScaleSoilGSDDataset,
    TARGET_COLUMNS,
    build_metadata_dataframe,
    _GLOBAL_IMAGE_CACHE,
)
from src.metrics import WeightedEMDLoss, compute_weighted_emd
from src.models import DualScaleSoilGSDModel
from src.utils import create_submission_file, get_group_kfold_splits


def train_dual_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Trains the dual-scale model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for micro_imgs, macro_imgs, targets, _ in loader:
        micro_imgs = micro_imgs.to(device)
        macro_imgs = macro_imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(micro_imgs, macro_imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        b_sz = micro_imgs.size(0)
        running_loss += loss.item() * b_sz
        total_samples += b_sz

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_dual_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Evaluates the dual-scale model on validation data."""
    model.eval()
    all_preds = []
    all_targets = []
    all_sample_ids = []

    for micro_imgs, macro_imgs, targets, meta in loader:
        micro_imgs = micro_imgs.to(device)
        macro_imgs = macro_imgs.to(device)
        preds = model(micro_imgs, macro_imgs)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())
        all_sample_ids.extend(meta["sample_id"])

    all_preds_arr = np.vstack(all_preds)
    all_targets_arr = np.vstack(all_targets)

    unique_ids = list(dict.fromkeys(all_sample_ids))
    sample_preds = []
    sample_targets = []

    for sid in unique_ids:
        indices = [i for i, x in enumerate(all_sample_ids) if x == sid]
        sample_preds.append(np.mean(all_preds_arr[indices], axis=0))
        sample_targets.append(all_targets_arr[indices[0]])

    sample_preds_arr = np.array(sample_preds)
    sample_targets_arr = np.array(sample_targets)

    mean_emd = compute_weighted_emd(sample_targets_arr, sample_preds_arr)
    return mean_emd, sample_preds_arr, sample_targets_arr, unique_ids


def train_dual_scale_cv(
    backbone_name: str = "resnet18",
    num_epochs: int = 15,
    batch_size: int = 8,
    lr: float = 3e-4,
    micro_fov_mm: float = 25.0,
    macro_fov_mm: float = 100.0,
    output_size: int = 224,
    n_splits: int = 5,
    checkpoint_dir: str = "checkpoints_dual",
):
    """Executes full 5-Fold Cross-Validation on Dual-Scale Architecture."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} | Dual-Scale Backbone: {backbone_name}")

    train_df, _ = build_metadata_dataframe()
    splits = list(get_group_kfold_splits(train_df, n_splits=n_splits))

    oof_predictions = {}
    oof_ground_truth = {}
    fold_best_scores = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*20} Dual-Scale Fold {fold + 1} / {n_splits} {'='*20}")
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        train_ds = DualScaleSoilGSDDataset(
            fold_train_df,
            micro_fov_mm=micro_fov_mm,
            macro_fov_mm=macro_fov_mm,
            output_size=output_size,
            is_train=True,
            preload=True,
        )
        val_ds = DualScaleSoilGSDDataset(
            fold_val_df,
            micro_fov_mm=micro_fov_mm,
            macro_fov_mm=macro_fov_mm,
            output_size=output_size,
            is_train=False,
            preload=True,
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False
        )

        model = DualScaleSoilGSDModel(
            backbone_name=backbone_name, pretrained=True, dropout_rate=0.25
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
        best_ckpt = os.path.join(
            checkpoint_dir, f"dual_{backbone_name}_fold_{fold+1}.pt"
        )

        for epoch in range(1, num_epochs + 1):
            train_loss = train_dual_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_emd, val_preds, val_targets, val_ids = evaluate_dual_model(
                model, val_loader, device
            )
            scheduler.step()

            if val_emd < best_val_emd:
                best_val_emd = val_emd
                best_preds = val_preds
                best_targets = val_targets
                best_ids = val_ids
                torch.save(model.state_dict(), best_ckpt)

            if epoch % 5 == 0 or epoch == num_epochs:
                print(
                    f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val EMD: {val_emd:.4f} | Best Val EMD: {best_val_emd:.4f}"
                )

        fold_best_scores.append(best_val_emd)
        print(f"Fold {fold+1} Best Val Weighted EMD: {best_val_emd:.4f}")

        for sid, p, t in zip(best_ids, best_preds, best_targets):
            oof_predictions[sid] = p
            oof_ground_truth[sid] = t

    oof_ids = sorted(list(oof_predictions.keys()))
    oof_p = np.array([oof_predictions[k] for k in oof_ids])
    oof_t = np.array([oof_ground_truth[k] for k in oof_ids])
    overall_oof_emd = compute_weighted_emd(oof_t, oof_p)

    print("\n" + "=" * 50)
    print(f"Dual-Scale Fold Scores: {[round(s, 4) for s in fold_best_scores]}")
    print(f"Mean Fold Weighted EMD: {np.mean(fold_best_scores):.4f}")
    print(f"OVERALL DUAL-SCALE OUT-OF-FOLD (OOF) WEIGHTED EMD: {overall_oof_emd:.4f}")
    print("=" * 50)

    return overall_oof_emd


def extract_tta_crops(
    image: Image.Image,
    fov_mm: float,
    ppm: float,
    output_size: int = 224,
) -> List[torch.Tensor]:
    """Extracts 5 spatial grid crops (Center + 4 Corners) with flips for TTA."""
    w, h = image.size
    crop_pixels = int(round(fov_mm * ppm))
    crop_pixels = min(crop_pixels, min(w, h))

    max_x = max(0, w - crop_pixels)
    max_y = max(0, h - crop_pixels)
    mid_x = max_x // 2
    mid_y = max_y // 2

    # 5 crop positions: Center, Top-Left, Top-Right, Bottom-Left, Bottom-Right
    coords = [
        (mid_x, mid_y),
        (0, 0),
        (max_x, 0),
        (0, max_y),
        (max_x, max_y),
    ]

    tensors = []
    for left, top in coords:
        crop = image.crop((left, top, left + crop_pixels, top + crop_pixels))
        crop = crop.resize(
            (output_size, output_size), Image.Resampling.BILINEAR
        )
        t = TF.to_tensor(crop)
        t = TF.normalize(
            t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        tensors.append(t)
        # Add horizontal flip
        tensors.append(TF.hflip(t))

    return tensors


@torch.no_grad()
def generate_dual_tta_submission(
    backbone_name: str = "resnet18",
    checkpoint_dir: str = "checkpoints_dual",
    micro_fov_mm: float = 25.0,
    macro_fov_mm: float = 100.0,
    output_size: int = 224,
    n_splits: int = 5,
    submission_path: str = "submission.csv",
) -> pd.DataFrame:
    """Generates test predictions using 5-Fold Dual-Scale Ensemble with Multi-Crop TTA."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_df = build_metadata_dataframe()

    # Load all trained fold models
    models_list = []
    for fold in range(1, n_splits + 1):
        ckpt = os.path.join(
            checkpoint_dir, f"dual_{backbone_name}_fold_{fold}.pt"
        )
        if os.path.exists(ckpt):
            model = DualScaleSoilGSDModel(
                backbone_name=backbone_name, pretrained=False
            ).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            models_list.append(model)

    print(
        f"Loaded {len(models_list)} trained Dual-Scale fold models for TTA ensemble."
    )

    df_sub = pd.read_csv("data/raw/sample_submission.csv")
    required_sample_ids = df_sub["sample_id"].tolist()

    final_predictions = []

    for sid in required_sample_ids:
        sample_rows = test_df[test_df["sample_id"] == sid]
        sample_crop_preds = []

        for _, row in sample_rows.iterrows():
            img_path = row["image_path"]
            ppm = float(row["ppm"])

            if img_path in _GLOBAL_IMAGE_CACHE:
                image = _GLOBAL_IMAGE_CACHE[img_path]
            else:
                image = Image.open(img_path).convert("RGB")
                _GLOBAL_IMAGE_CACHE[img_path] = image

            # Extract 10 micro & macro TTA crops per photo
            micro_crops = extract_tta_crops(
                image, micro_fov_mm, ppm, output_size
            )
            macro_crops = extract_tta_crops(
                image, macro_fov_mm, ppm, output_size
            )

            micro_batch = torch.stack(micro_crops).to(device)
            macro_batch = torch.stack(macro_crops).to(device)

            # Predict across all fold models
            for model in models_list:
                preds = model(micro_batch, macro_batch)
                sample_crop_preds.append(preds.cpu().numpy())

        # Average all TTA crops & fold models for this soil sample
        all_sample_preds = np.vstack(sample_crop_preds)
        avg_pred = np.mean(all_sample_preds, axis=0)

        # Enforce physical validity guarantees
        avg_pred = np.maximum.accumulate(np.clip(avg_pred, 0.0, 100.0))
        avg_pred[-1] = 100.0
        final_predictions.append(avg_pred)

    final_preds_arr = np.array(final_predictions)
    sub = create_submission_file(
        required_sample_ids, final_preds_arr, output_path=submission_path
    )
    return sub


if __name__ == "__main__":
    print("=== Training Dual-Scale (Micro 25mm + Macro 100mm) Architecture ===")
    oof_score = train_dual_scale_cv(
        backbone_name="resnet18",
        num_epochs=12,
        batch_size=8,
        lr=3e-4,
        micro_fov_mm=25.0,
        macro_fov_mm=100.0,
        n_splits=5,
    )
    print(f"\nGenerating High-Precision Multi-Crop TTA Submission...")
    sub = generate_dual_tta_submission(
        backbone_name="resnet18",
        submission_path="submission.csv",
    )
    print("\nTop Submission Rows:")
    print(sub.head(10).to_string(index=False))
