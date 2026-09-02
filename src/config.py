"""Configuration module for Soil Grain Size Distribution (GSD) Modeling.

This module defines domain-specific geotechnical constants, physical sieve
diameters, camera calibration parameters (Pixels Per Millimeter), and path
configurations adhering to PEP-8 standards.
"""

from pathlib import Path
import numpy as np

# Random seed for global reproducibility
RANDOM_SEED = 42
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Ground truth label and photo paths
TRAIN_LABELS_PATH = DATA_DIR / "Training_labels_updated.csv"
TEST_SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
TRAIN_PHOTOS_DIR = DATA_DIR / "Training-All_Photos_updated"
TEST_PHOTOS_DIR = DATA_DIR / "Test_All_Photos"

# Output submission and cache paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Geotechnical Sieve Sizes (mm)
SIEVE_COLUMNS = [
    "0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"
]

SIEVE_DIAMETERS = np.array(
    [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0],
    dtype=np.float32
)

# Interval logarithmic weights for Weighted Earth Mover's Distance (EMD)
# \Delta \log_{10}(x_i) \approx 0.5 for all 10 intervals
INTERVAL_WEIGHTS = np.diff(np.log10(SIEVE_DIAMETERS)).astype(np.float32)

# Physical Scale Normalization Target (PPM = Pixels Per Millimeter)
# Canonical physical resolution: 1 mm in real world = exactly 15 pixels
TARGET_PPM = 15.0

# Calibrated Camera Sensors Lookup (Pixels Per Millimeter)
PPM_LOOKUP = {
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
DEFAULT_PPM = 16.75
