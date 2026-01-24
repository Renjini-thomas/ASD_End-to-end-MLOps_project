
# ASD Diagnoser – Machine Learning with MLOps

## Project Description
This project implements an Autism Spectrum Disorder (ASD) diagnosis system using machine learning on MRI images from the ABIDE-I dataset. The system uses mid-sagittal MRI slices and handcrafted feature extraction methods, followed by Random Forest–based feature selection.  
MLOps practices are integrated to ensure reproducibility, modularity, and version control.

---

## Dataset
- **Dataset**: ABIDE-I
- **Image Type**: Mid-sagittal MRI slices
- **Classes**:
  - autistic
  - control
- **Directory Structure**:

data/raw/abide1_data(sagittal_2)/
├── autistic/
└── control/


The dataset is versioned using **DVC**.

---

## Feature Extraction (Final Implementation)
Feature extraction logic is identical to the final implementation notebook.

### Extracted Features
1. **GLCM Features**
 - Contrast
 - Correlation
 - Energy
 - Homogeneity
 - Mean
 - Variance
 - Entropy
 - Higher-order statistical moments

2. **LBP Features**
 - Uniform Local Binary Pattern
 - Block-wise histogram extraction (32×32 blocks)

3. **GFCC-like Morphological Features**
 - Multi-Otsu thresholding
 - Mid-sagittal band analysis
 - Region properties (area, perimeter, axis lengths, solidity, extent)

**Total features per image**: 21

---

## Feature Selection
Feature selection is performed using a **Random Forest classifier**:

- `n_estimators = 300`
- `criterion = gini`
- Features ranked using `feature_importances_`
- Ranked indices sorted in descending order
- Ranked feature indices stored for reuse in later stages

This logic is identical to the final ML implementation.

---

## MLOps Components Implemented

### Component 1
- Git for code versioning
- DVC for data versioning
- Modular pipeline execution via `main.py`

### Component 2 (Partial)
- Reproducible feature artifacts
- Deterministic train–test split
- Persisted feature ranking for experiment reproducibility

---



