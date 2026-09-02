"""Data pipeline for physical scale normalization and patch sampling.

Implements:
1. Robust filename stem parsing with German umlaut mapping (e.g., Münster -> Mnster).
2. Sensor calibration: Scaling raw photos to canonical 15.0 PPM resolution.
3. Multi-patch interior spatial sampling (5 safe interior crops per image).
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from src.config import PPM_LOOKUP, DEFAULT_PPM, TARGET_PPM


# Standard ImageNet normalization for pre-trained backbones
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def normalize_string(text: str) -> str:
    """Normalizes sample IDs and filenames across special characters and umlauts."""
    s = str(text).lower().strip()
    s = s.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("ß", "ss")
    for char in [",", "-", " ", "(", ")", "."]:
        s = s.replace(char, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def build_image_map(directory: Path) -> Dict[str, str]:
    """Recursively traverses a directory and maps normalized stems to file paths."""
    img_map = {}
    if not directory.exists():
        return img_map

    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                stem = Path(f).stem
                norm_key = normalize_string(stem)
                img_map[norm_key] = os.path.join(root, f)

    return img_map


def get_camera_ppm(filename: str) -> float:
    """Extracts the camera sensor calibration factor (PPM) from filename."""
    lower_name = filename.lower()
    for key, ppm in PPM_LOOKUP.items():
        if key in lower_name:
            return ppm
    return DEFAULT_PPM


def extract_normalized_patches(
    image_path: str,
    target_ppm: float = TARGET_PPM,
    patch_size: int = 224,
) -> Tuple[List[Image.Image], float, float]:
    """Scales photo to physical target PPM and extracts 5 interior spatial crops.

    Returns:
        patches: List of PIL Image patches.
        current_ppm: The calibrated PPM of the source image.
        edge_density: Canny edge density ratio representing grain boundary density.
    """
    try:
        img_bytes = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    except Exception:
        img = None

    if img is None:
        return [], DEFAULT_PPM, 0.0

    h, w = img.shape[0], img.shape[1]
    current_ppm = get_camera_ppm(os.path.basename(image_path))

    # Scale image to canonical physical resolution (e.g. 15.0 pixels per mm)
    scale_factor = target_ppm / current_ppm
    scaled_w = max(patch_size, int(w * scale_factor))
    scaled_h = max(patch_size, int(h * scale_factor))
    img_scaled = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # Compute Canny edge density (granulometric roughness indicator)
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edge_density = float(np.mean(edges > 0))

    # 5 interior safe shift coordinates centered in the photo
    sh, sw = img_scaled.shape[0], img_scaled.shape[1]
    cy, cx = sh // 2, sw // 2
    inner_shifts = [
        (cy, cx),
        (cy - 45, cx),
        (cy + 45, cx),
        (cy, cx - 45),
        (cy, cx + 45),
    ]

    img_rgb = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)
    patches = []

    for y_c, x_c in inner_shifts:
        y1 = max(0, min(sh - patch_size, y_c - patch_size // 2))
        x1 = max(0, min(sw - patch_size, x_c - patch_size // 2))
        patch = img_rgb[y1 : y1 + patch_size, x1 : x1 + patch_size]
        patches.append(Image.fromarray(patch))

    return patches, current_ppm, edge_density
