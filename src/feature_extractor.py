"""Dual-Backbone Vision Feature Extractor for Soil Granulometry.

Combines:
1. EfficientNet-B0 (1280 features): High sensitivity to fine micro-textures and particle colors.
2. ResNet-34 (512 features): Residual representations of particle shapes and structural arrangements.
3. Canny Edge Density: Handcrafted edge frequency indicating particle boundary transitions.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

from src.config import CACHE_DIR
from src.data_pipeline import (
    IMAGE_TRANSFORM,
    build_image_map,
    extract_normalized_patches,
    normalize_string,
)


class DualBackboneExtractor:
    """Pretrained feature extractor pooling EfficientNet-B0 and ResNet-34."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # 1. EfficientNet-B0 backbone (1280-dim)
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.effnet = nn.Sequential(effnet.features, effnet.avgpool, nn.Flatten()).to(self.device).eval()

        # 2. ResNet-34 backbone (512-dim)
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten()).to(self.device).eval()

    @torch.no_grad()
    def extract_patch_features(self, patches) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts 1280-dim EffNet and 512-dim ResNet embeddings for image patches."""
        if not patches:
            return np.zeros((1, 1280)), np.zeros((1, 512))

        batch = torch.stack([IMAGE_TRANSFORM(p) for p in patches]).to(self.device)
        eff_feats = self.effnet(batch).cpu().numpy()
        res_feats = self.resnet(batch).cpu().numpy()
        return eff_feats, res_feats


def extract_sample_level_features(
    sample_ids: list,
    img_map: Dict[str, str],
    extractor: DualBackboneExtractor,
    desc: str = "Dataset",
    cache_file: Path = None,
) -> pd.DataFrame:
    """Extracts and aggregates features into a single row per unique soil sample.

    Args:
        sample_ids: List of unique sample IDs.
        img_map: Mapping of normalized stems to full image paths.
        extractor: Initialized DualBackboneExtractor instance.
        desc: Progress bar label.
        cache_file: Path to optional disk cache.

    Returns:
        DataFrame of shape (N_samples, 1794) with columns sample_id, eff_*, res_*, edge_density, ppm.
    """
    if cache_file and cache_file.exists():
        print(f"Loading cached features from: {cache_file}")
        return pd.read_parquet(cache_file)

    records = []
    print(f"\nExtracting Dual-Backbone Physical Features for: {desc}")

    for sid in tqdm(sample_ids, desc=desc):
        norm_sid = normalize_string(sid)

        # Match all photos belonging to this soil sample
        matching_paths = [
            path
            for key, path in img_map.items()
            if (norm_sid in key or key in norm_sid or norm_sid.replace("hpc_", "") in key)
        ]

        if not matching_paths:
            # Fallback prefix matching
            prefix = norm_sid.split("_")[0]
            matching_paths = [path for key, path in img_map.items() if prefix in key]

        eff_embs_all, res_embs_all, edges_all, ppms_all = [], [], [], []

        for p_path in matching_paths:
            patches, ppm, edge_density = extract_normalized_patches(p_path)
            if patches:
                eff_f, res_f = extractor.extract_patch_features(patches)
                eff_embs_all.append(np.mean(eff_f, axis=0))
                res_embs_all.append(np.mean(res_f, axis=0))
                edges_all.append(edge_density)
                ppms_all.append(ppm)

        row_dict = {"sample_id": sid}

        if eff_embs_all:
            mean_eff = np.mean(eff_embs_all, axis=0)
            mean_res = np.mean(res_embs_all, axis=0)
            row_dict["edge_density"] = float(np.mean(edges_all))
            row_dict["ppm"] = float(np.mean(ppms_all))

            for i, val in enumerate(mean_eff):
                row_dict[f"eff_{i}"] = float(val)
            for i, val in enumerate(mean_res):
                row_dict[f"res_{i}"] = float(val)
        else:
            row_dict["edge_density"] = 0.0
            row_dict["ppm"] = 16.75
            for i in range(1280):
                row_dict[f"eff_{i}"] = 0.0
            for i in range(512):
                row_dict[f"res_{i}"] = 0.0

        records.append(row_dict)

    df_feats = pd.DataFrame(records)

    if cache_file:
        df_feats.to_parquet(cache_file, index=False)
        print(f"Saved extracted features to cache: {cache_file}")

    return df_feats
