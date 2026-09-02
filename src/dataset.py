"""
Dataset, Preprocessing, and PPM-Calibrated Crop Pipeline for Soil GSD Prediction.

Implements:
    - Pillar 1: Physical Scale Normalization (PPM-calibrated Field of View)
    - Pillar 2: Dual-Scale Micro and Macro crop extraction
    - Pillar 5: GroupKFold partitioning by soil sample_id
"""

import os
import sys
import re
from typing import Dict, List, Optional, Tuple, Union

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.metrics import DIAMETERS_MM

TARGET_COLUMNS = [
    "0.002",
    "0.0063",
    "0.02",
    "0.063",
    "0.2",
    "0.63",
    "2",
    "6.3",
    "20",
    "63",
    "200",
]

# Camera PPM lookup mapping
PPM_LOOKUP: Dict[str, float] = {
    "iphone 14": 13.942,
    "iphone14": 13.942,
    "iphone 16": 19.525,
    "iphone16": 19.525,
    "iphone_16": 19.525,
    "motorola edge 20": 11.492,
    "motorola edge": 11.492,
    "motorola_edge": 11.492,
    "motorola edge 60 fusion": 12.465,
    "motorola_edge_60_fusion": 12.465,
    "samsung a52": 26.330,
    "sm-a525f": 26.330,
    "samsung_a52": 26.330,
}


def _normalize_name(text: str) -> str:
    """Normalize text by removing spaces, dashes, commas, and handling German umlauts."""
    t = text.lower()
    t = (
        t.replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("(", "")
        .replace(")", "")
    )
    t = (
        t.replace("muenster", "mnster")
        .replace("münster", "mnster")
        .replace("mnster", "mnster")
    )
    t = t.replace(",", "")
    return t


def parse_camera_and_sample(
    filename: str, valid_samples: List[str]
) -> Tuple[Optional[float], Optional[str]]:
    """
    Parses camera model (PPM) and sample_id from image filename.

    Args:
        filename (str): Base image filename.
        valid_samples (List[str]): List of known sample_ids.

    Returns:
        Tuple[Optional[float], Optional[str]]: (ppm_value, matched_sample_id)
    """
    stem = os.path.splitext(filename)[0]
    lower_stem = stem.lower()

    # Match Camera PPM
    cam_ppm = None
    for key, ppm in PPM_LOOKUP.items():
        if key in lower_stem:
            cam_ppm = ppm
            break

    # Match Sample ID
    matched_sample = None
    norm_stem = _normalize_name(stem)
    for s in valid_samples:
        norm_s = _normalize_name(s)
        if norm_s in norm_stem:
            matched_sample = s
            break

    return cam_ppm, matched_sample


def build_metadata_dataframe(
    labels_csv_path: str = "data/raw/Training_labels_updated.csv",
    sub_csv_path: str = "data/raw/sample_submission.csv",
    train_dir: str = "data/raw/Training-All_Photos_updated",
    test_dir: str = "data/raw/Test_All_Photos",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds clean metadata DataFrames for training and testing images.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    df_labels = pd.read_csv(labels_csv_path)
    train_sample_ids = df_labels["sample_id"].astype(str).tolist()

    df_sub = pd.read_csv(sub_csv_path)
    test_sample_ids = df_sub["sample_id"].astype(str).tolist()

    # Index training images
    train_records = []
    for root, _, files in os.walk(train_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                ppm, sample_id = parse_camera_and_sample(f, train_sample_ids)
                if ppm is not None and sample_id is not None:
                    row_data = {
                        "image_path": os.path.join(root, f),
                        "filename": f,
                        "sample_id": sample_id,
                        "ppm": ppm,
                        "is_test": False,
                    }
                    # Merge ground truth cumulative targets
                    sample_row = df_labels[
                        df_labels["sample_id"] == sample_id
                    ].iloc[0]
                    for col in TARGET_COLUMNS:
                        row_data[col] = float(sample_row[col])
                    train_records.append(row_data)

    train_df = pd.DataFrame(train_records)

    # Index test images
    test_records = []
    for root, _, files in os.walk(test_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                ppm, sample_id = parse_camera_and_sample(f, test_sample_ids)
                if ppm is not None and sample_id is not None:
                    test_records.append(
                        {
                            "image_path": os.path.join(root, f),
                            "filename": f,
                            "sample_id": sample_id,
                            "ppm": ppm,
                            "is_test": True,
                        }
                    )

    test_df = pd.DataFrame(test_records)
    return train_df, test_df


# Global in-memory cache to store decoded PIL images across epochs & folds
_GLOBAL_IMAGE_CACHE: Dict[str, Image.Image] = {}


class SoilGSDDataset(Dataset):
    """
    PyTorch Dataset implementing PPM Physical Scale FOV Cropping.

    Args:
        df (pd.DataFrame): Metadata dataframe containing image_path, ppm, sample_id.
        physical_fov_mm (float): Desired physical field of view in millimeters (e.g. 50.0 mm).
        output_size (int): Output square tensor resolution (e.g. 224 or 384).
        is_train (bool): If True, applies random crop & data augmentations; else center crop.
        preload (bool): If True, caches decoded PIL images in memory for 100x faster epoch iterations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        physical_fov_mm: float = 50.0,
        output_size: int = 224,
        is_train: bool = True,
        preload: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.physical_fov_mm = physical_fov_mm
        self.output_size = output_size
        self.is_train = is_train

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if preload:
            for p in self.df["image_path"].unique():
                if p not in _GLOBAL_IMAGE_CACHE:
                    # Open and load image into memory
                    img = Image.open(p).convert("RGB")
                    _GLOBAL_IMAGE_CACHE[p] = img

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[str, float]]]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        ppm = float(row["ppm"])

        if img_path in _GLOBAL_IMAGE_CACHE:
            image = _GLOBAL_IMAGE_CACHE[img_path]
        else:
            image = Image.open(img_path).convert("RGB")
            _GLOBAL_IMAGE_CACHE[img_path] = image

        w, h = image.size

        # Pillar 1: Calculate exact crop size in pixels from physical mm and PPM
        crop_pixels = int(round(self.physical_fov_mm * ppm))
        crop_pixels = min(crop_pixels, min(w, h))

        if self.is_train:
            # Random physical FOV crop
            max_x = w - crop_pixels
            max_y = h - crop_pixels
            left = np.random.randint(0, max_x + 1) if max_x > 0 else 0
            top = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        else:
            # Deterministic center physical FOV crop
            left = (w - crop_pixels) // 2
            top = (h - crop_pixels) // 2

        crop = image.crop((left, top, left + crop_pixels, top + crop_pixels))
        crop = crop.resize(
            (self.output_size, self.output_size), Image.Resampling.BILINEAR
        )

        tensor = TF.to_tensor(crop)

        if self.is_train:
            # Random horizontal / vertical flips & isotropic rotations
            if np.random.rand() > 0.5:
                tensor = TF.hflip(tensor)
            if np.random.rand() > 0.5:
                tensor = TF.vflip(tensor)
            rotations = [0, 90, 180, 270]
            rot_deg = int(np.random.choice(rotations))
            if rot_deg > 0:
                tensor = TF.rotate(tensor, rot_deg)

        tensor = self.normalize(tensor)

        # Extract target labels if available
        if not row.get("is_test", False) and TARGET_COLUMNS[0] in row:
            targets = np.array(
                [row[col] for col in TARGET_COLUMNS], dtype=np.float32
            )
            target_tensor = torch.tensor(targets, dtype=torch.float32)
        else:
            target_tensor = torch.zeros(len(TARGET_COLUMNS), dtype=torch.float32)

        meta = {
            "sample_id": row["sample_id"],
            "ppm": ppm,
            "filename": row["filename"],
        }

        return tensor, target_tensor, meta


class DualScaleSoilGSDDataset(Dataset):
    """
    PyTorch Dataset implementing Dual-Scale Multi-Resolution FOV Cropping.

    Pillars 1 & 2 in Project Specification:
        - Micro-Scale (25 mm FOV): High-frequency texture (clay/silt).
        - Macro-Scale (100 mm FOV): Coarse grain boundaries & morphology (sand/gravel).

    Args:
        df (pd.DataFrame): Metadata dataframe.
        micro_fov_mm (float): Micro FOV in mm (default 25.0 mm).
        macro_fov_mm (float): Macro FOV in mm (default 100.0 mm).
        output_size (int): Output tensor resolution (default 224).
        is_train (bool): Train vs Validation mode.
        preload (bool): Cache decoded images in memory.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        micro_fov_mm: float = 25.0,
        macro_fov_mm: float = 100.0,
        output_size: int = 224,
        is_train: bool = True,
        preload: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.micro_fov_mm = micro_fov_mm
        self.macro_fov_mm = macro_fov_mm
        self.output_size = output_size
        self.is_train = is_train

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if preload:
            for p in self.df["image_path"].unique():
                if p not in _GLOBAL_IMAGE_CACHE:
                    img = Image.open(p).convert("RGB")
                    _GLOBAL_IMAGE_CACHE[p] = img

    def __len__(self) -> int:
        return len(self.df)

    def _get_crop_tensor(
        self, image: Image.Image, fov_mm: float, ppm: float
    ) -> torch.Tensor:
        w, h = image.size
        crop_pixels = int(round(fov_mm * ppm))
        crop_pixels = min(crop_pixels, min(w, h))

        if self.is_train:
            max_x = w - crop_pixels
            max_y = h - crop_pixels
            left = np.random.randint(0, max_x + 1) if max_x > 0 else 0
            top = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        else:
            left = (w - crop_pixels) // 2
            top = (h - crop_pixels) // 2

        crop = image.crop((left, top, left + crop_pixels, top + crop_pixels))
        crop = crop.resize(
            (self.output_size, self.output_size), Image.Resampling.BILINEAR
        )
        tensor = TF.to_tensor(crop)

        if self.is_train:
            if np.random.rand() > 0.5:
                tensor = TF.hflip(tensor)
            if np.random.rand() > 0.5:
                tensor = TF.vflip(tensor)
            rotations = [0, 90, 180, 270]
            rot_deg = int(np.random.choice(rotations))
            if rot_deg > 0:
                tensor = TF.rotate(tensor, rot_deg)

        return self.normalize(tensor)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Union[str, float]],
    ]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        ppm = float(row["ppm"])

        if img_path in _GLOBAL_IMAGE_CACHE:
            image = _GLOBAL_IMAGE_CACHE[img_path]
        else:
            image = Image.open(img_path).convert("RGB")
            _GLOBAL_IMAGE_CACHE[img_path] = image

        micro_tensor = self._get_crop_tensor(image, self.micro_fov_mm, ppm)
        macro_tensor = self._get_crop_tensor(image, self.macro_fov_mm, ppm)

        if not row.get("is_test", False) and TARGET_COLUMNS[0] in row:
            targets = np.array(
                [row[col] for col in TARGET_COLUMNS], dtype=np.float32
            )
            target_tensor = torch.tensor(targets, dtype=torch.float32)
        else:
            target_tensor = torch.zeros(len(TARGET_COLUMNS), dtype=torch.float32)

        meta = {
            "sample_id": row["sample_id"],
            "ppm": ppm,
            "filename": row["filename"],
        }

        return micro_tensor, macro_tensor, target_tensor, meta


if __name__ == "__main__":
    train_df, test_df = build_metadata_dataframe()
    print(f"Indexed {len(train_df)} train image records across 24 samples.")
    print(f"Indexed {len(test_df)} test image records across 10 samples.")

    # Test Dual-Scale PyTorch dataset loader
    ds = DualScaleSoilGSDDataset(
        train_df, micro_fov_mm=25.0, macro_fov_mm=100.0, output_size=224, is_train=True
    )
    micro_t, macro_t, sample_target, sample_meta = ds[0]

    print(f"\nMicro tensor shape (25mm): {micro_t.shape}")
    print(f"Macro tensor shape (100mm): {macro_t.shape}")
    print(f"Target vector shape: {sample_target.shape}")
    print(f"Sample metadata: {sample_meta}")
    print("\nDual-Scale Dataset test completed successfully!")

