# Project Specification & Ground Truth Reference
## Kaggle Competition: Predicting Soil Grain Size Distributions from Images

> **Single Source of Truth (SSOT)** for problem definition, dataset schemas, mathematical constraints, evaluation metric, and validation strategy.

---

## 1. Executive Overview
- **Competition Title**: Predicting Soil Grain Size Distributions from Images
- **Host**: Lukas Leibold, Enrico Soranzo (BOKU University, Vienna & EU Horizon research project **GRID** - *Geotechnical Resilience through Intelligent Design*)
- **Domain**: Geotechnical Engineering + Computer Vision + Distribution Regression
- **Objective**: Predict the **cumulative grain size distribution (GSD)** curve of soil samples based on surface photographs and camera calibration data (`ppm.csv`).
- **Core Value Proposition**: Traditional sieve/hydrometer testing takes days in a laboratory; image-based estimation enables real-time, automated soil characterization on construction and earthwork sites.

---

## 2. Target Variables & Grain Size Bins

The target is a cumulative distribution function (CDF) curve representing the **mass fraction (%)** of particles smaller than a given diameter, defined at **11 standard sieve diameters** per **DIN EN ISO 14688-1**:

| Index $i$ | Diameter $x_i$ (mm) | Soil Fraction Category | Column Header in CSV |
| :---: | :---: | :---: | :---: |
| 1 | $0.002$ | Clay / Silt boundary | `0.002` |
| 2 | $0.0063$ | Fine Silt | `0.0063` |
| 3 | $0.02$ | Medium Silt | `0.02` |
| 4 | $0.063$ | Coarse Silt / Sand boundary | `0.063` |
| 5 | $0.2$ | Fine Sand | `0.2` |
| 6 | $0.63$ | Medium Sand | `0.63` |
| 7 | $2.0$ | Coarse Sand / Gravel boundary | `2` |
| 8 | $6.3$ | Fine Gravel | `6.3` |
| 9 | $20.0$ | Medium Gravel | `20` |
| 10 | $63.0$ | Coarse Gravel / Cobbles boundary | `63` |
| 11 | $200.0$ | Cobbles / Boulders boundary | `200` |

---

## 3. Strict Mathematical & Physical Constraints

Every predicted cumulative curve $\mathbf{\hat{F}} = [\hat{F}_1, \hat{F}_2, \dots, \hat{F}_{11}]$ must satisfy:

1. **Monotonic Non-Decreasing**:
   $$\hat{F}_i \le \hat{F}_{i+1} \quad \forall i \in \{1, 2, \dots, 10\}$$
2. **Bounded Range**:
   $$0.0 \le \hat{F}_i \le 100.0 \quad \forall i \in \{1, 2, \dots, 11\}$$
3. **Upper Boundary Condition**:
   $$\hat{F}_{11} = \hat{F}(200\,\text{mm}) = 100.0\%$$

*Note: Any submission violating monotonicity or boundary bounds is automatically marked as invalid on Kaggle.*

---

## 4. Exact Evaluation Metric: Weighted Earth Mover's Distance (EMD)

The competition evaluates predictions using the **1D Wasserstein-1 Distance (Earth Mover's Distance)** weighted on a **logarithmic diameter scale**:

$$\text{EMD} = \sum_{i=1}^{10} |\hat{F}_i - F_i| \cdot \left(\log_{10}(x_{i+1}) - \log_{10}(x_i)\right)$$

### Python Ground-Truth Implementation:
```python
import numpy as np

# Exact diameters defined by DIN EN ISO 14688-1
DIAMETERS = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])

# Logarithmic weights for the 10 intervals between adjacent diameters
LOG_DIAMETERS = np.log10(DIAMETERS)
WEIGHTS = np.diff(LOG_DIAMETERS)  # delta log10(x_{i+1}) - log10(x_i) approx 0.5 per interval

def weighted_emd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the competition evaluation metric: Weighted Earth Mover's Distance.
    
    Parameters:
        y_true (np.ndarray): Ground truth cumulative distributions of shape (N, 11) or (11,)
        y_pred (np.ndarray): Predicted cumulative distributions of shape (N, 11) or (11,)
        
    Returns:
        float: Mean weighted EMD across samples.
    """
    # Absolute difference at the first 10 support points
    abs_diff = np.abs(y_true[..., :10] - y_pred[..., :10])
    
    # Weighted sum across intervals
    sample_emd = np.sum(abs_diff * WEIGHTS, axis=-1)
    return float(np.mean(sample_emd))
```

---

## 5. Dataset Architecture & File Descriptions

- `Training_labels_updated.csv`: Contains `sample_id` and 11 target columns for 24 training soil samples.
- `test.csv` / `sample_submission.csv`: Template with required column headers and 10 test sample IDs.
- `ppm_updated.csv`: Camera calibration specifications containing **Pixels Per Millimeter (PPM)** values for each device model used to shoot the photos.
- `Training-All_Photos_updated/`: 127 high-resolution image files for training samples.
- `Test_All_Photos/`: 35 high-resolution image files for test samples.

### Filename Schema & Camera Calibration
- Photographs are named following patterns such as: `<CameraModel>_<SampleID>_<PhotoIndex>.jpg` or `iPhone14_HPC_<SampleID> (<PhotoIndex>).JPG`
- The `ppm_updated.csv` allows physical pixel-to-millimeter scaling:
  $$\text{Physical Size (mm)} = \frac{\text{Pixel Dimension}}{\text{PPM}}$$

---

## 6. The 5 Strategic Engineering & Modeling Pillars

### 🏛️ Pillar 1: Physical Scale Normalization (PPM-Calibrated FOV)
- **Problem**: Raw pixel dimensions differ across cameras ($13.94$ PPM on iPhone 14 vs $26.33$ PPM on Samsung A52). Without normalization, a $1\,\text{mm}$ grain appears $1.88\times$ larger in pixels on Samsung.
- **Implementation**: Compute dynamic crop size in pixels:
  $$\text{Crop Size (pixels)} = \text{Target Physical FOV (mm)} \times \text{PPM}$$
- **Result**: Resizing this patch to $224 \times 224$ or $384 \times 384$ yields true physical scale invariance.

### 🏛️ Pillar 2: Dual-Scale Multi-Resolution Feature Extraction
- **Micro-Scale (e.g. $25\,\text{mm}$ FOV)**: High-resolution zoom focusing on fine clay/silt texture, sub-pixel matrix roughness, and inter-grain binder material.
- **Macro-Scale (e.g. $100\,\text{mm}$ FOV)**: Wide FOV capturing coarse sand/gravel boundaries, particle shape roundness/angularity, and void ratios.

### 🏛️ Pillar 3: Guaranteed Monotonic Distribution Head (Softmax + Cumsum)
- **Problem**: Standard linear regression heads predict independent unconstrained numbers $\hat{F}_i$, resulting in non-monotonic curves ($F(d_i) > F(d_{i+1})$) and illegal values ($<0$ or $>100$), leading to competition disqualification.
- **Solution**: Model the differential mass increments (PMF) $\Delta F \in \mathbb{R}^{11}$:
  $$p = \text{Softmax}(\mathbf{z}) \quad \text{such that} \quad p_i \ge 0, \; \sum_{i=1}^{11} p_i = 1.0$$
  $$\mathbf{\hat{F}} = 100 \times \text{cumsum}(p)$$
- **Guarantees**:
  1. $0 \le \hat{F}_i \le 100\%$
  2. $\hat{F}_i \le \hat{F}_{i+1}$ (Monotonicity)
  3. $\hat{F}_{11} \equiv 100.0\%$ (Boundary constraint)

### 🏛️ Pillar 4: Direct Differentiable Weighted EMD Loss Function
- **Problem**: Training with MSE/MAE does not align with the competition metric.
- **Solution**: The 1D Wasserstein-1 Distance between cumulative distributions is an exact $L_1$ norm weighted by logarithmic interval lengths:
  $$\mathcal{L}_{\text{EMD}} = \sum_{i=1}^{10} |\hat{F}_i - F_i| \cdot (\log_{10}(x_{i+1}) - \log_{10}(x_i))$$
- **Implementation**: Differentiable PyTorch module `WeightedEMDLoss` used directly for backpropagation.

### 🏛️ Pillar 5: Leak-Proof Validation Protocol (GroupKFold by Sample ID)
- **Problem**: 127 photos belong to only 24 unique soil samples. Random train-test splitting leaks sample-specific lighting, background, and color into validation folds, causing catastrophic overfitting.
- **Solution**: Group 5-Fold Cross-Validation (`GroupKFold(n_splits=5)`) strictly grouped on `sample_id`.

---

## 7. Competition Rules & Constraints
- **Submissions**: Max 5 submissions per day; 2 final selected submissions.
- **External Data**: Permitted if publicly accessible and open-source (standard pre-trained vision backbones like ConvNeXt, EfficientNet, Swin).
- **License**: CC BY 4.0.

