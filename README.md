
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

- **data/raw/abide1_data(sagittal_2)/**
- ├── autistic/
- └── control/
- 

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

### Data Augmentation
Data augmentation is applied **only to training images** before feature extraction to improve generalization.

**Augmentation techniques:**
- Horizontal flipping
- Rotation (±10 degrees)
- Intensity scaling

Augmented images follow the same handcrafted feature extraction pipeline.

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
## Model Training
Three machine learning models are trained using RF-ranked top-K features:

- **Decision Tree**
- **K-Nearest Neighbors (k = 3)**
- **Support Vector Machine (linear kernel)**

Training uses:
- 70–30 stratified train–test split
- Fixed random state for reproducibility
- Accuracy and ROC-AUC as evaluation metrics

---
## Model Evaluation & Best Model Selection
- All trained models are evaluated on the same test set
- Metrics used:
  - Accuracy
  - ROC-AUC (primary metric)
- The best-performing model is selected based on **ROC-AUC**
- The selected model is registered as the **best model**

---

## MLOps Components Implemented

### Component 1 – Version Control & Model Versioning (Completed)
- Git for code versioning
- DVC for data versioning
- MLflow for model versioning and artifact tracking
- Weights & Biases for experiment tracking

### Component 2 – Experiment Tracking (Completed)
- Hyperparameter tracking
- Metric tracking (accuracy, ROC-AUC)
- Model comparison (DT vs KNN vs SVM)
- Reproducible experiments

---



