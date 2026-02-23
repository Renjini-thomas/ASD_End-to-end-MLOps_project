# 🧠 ASD Diagnoser – Machine Learning with MLOps

## Project Overview
This project implements an **Autism Spectrum Disorder (ASD) diagnosis system** using **structural MRI (sMRI)** images from the **ABIDE-I dataset**.  
The system follows a **base-paper-aligned handcrafted feature pipeline** and integrates **MLOps practices** to ensure **reproducibility, traceability, and reliable evaluation**.

The work extends a traditional ML pipeline by incorporating:
- **DDPM-based data augmentation**
- **Feature-rich extraction (279 handcrafted features)**
- **Strict train–test separation**
- **Experiment tracking and model versioning**

---

## Dataset

### Source
- **Dataset**: ABIDE-I
- **Modality**: Structural MRI (sMRI)
- **Slice Used**: Full **mid-sagittal slice** extracted from 3D NIfTI volumes

### Classes
- `autistic`
- `non-autistic`

### Dataset Split (Predefined)
The dataset is organized into **explicit train and test folders** to avoid data leakage.

- data/ 
- └── Autism_split_valid_2D_mid_sagittal/
- ├── train/
- │ ├── autistic/
- │ └── non-autistic/
- └── test/
- ├── autistic/
- └── non-autistic/


- **Training set**: DDPM-augmented images
- **Test set**: Real (non-augmented) images only

The dataset is **versioned using DVC**, enabling full data reproducibility across experiments.

---

## Preprocessing

- MRI volumes are loaded from **NIfTI (.nii)** format
- The **mid-sagittal slice** is extracted along the anatomical midline
- Images are:
  - Rotated for correct orientation
  - Min–max normalized
  - Resized to **256 × 256**
  - Converted to grayscale PNG format

This preprocessing strictly follows the **base paper implementation**.

---

## Feature Extraction (Base-Paper Aligned)

Feature extraction is **deterministic** and applied **separately to train and test datasets**.

### Extracted Feature Groups

#### 1️⃣ GLCM (Texture Features)
- Contrast  
- Correlation  
- Energy  
- Homogeneity  
- Mean intensity  
- Variance  
- Entropy  
- Multi-distance and multi-orientation statistics  

#### 2️⃣ LBP (Local Texture Features)
- Uniform Local Binary Patterns  
- Histogram-based encoding across spatial regions  

#### 3️⃣ GFCC-like Morphological Features
- Multi-Otsu thresholding
- Region-based properties:
  - Area
  - Perimeter
  - Major axis length
  - Minor axis length
  - Shape descriptors (solidity, extent, etc.)
  - Eccentricity

📌 **Total features per image: 279**

This feature configuration matches the **base research paper**, enabling fair comparison.

---

## Data Augmentation

### Method
- **DDPM (Denoising Diffusion Probabilistic Model)**  
- Pretrained model applied **offline**
- Augmentation applied **only to training images**

### Purpose
- Increase data diversity
- Improve generalization
- Prevent test data leakage

Augmented images undergo the **same preprocessing and feature extraction pipeline** as real images.

---

## Feature Selection (Ranking Only)

Feature selection is implemented as **feature ranking**, not hard selection.

### Method
- **Random Forest–based feature ranking**
- Parameters:
  - `n_estimators = 300`
  - `criterion = gini`
- Feature importance computed using `feature_importances_`

### Key Design Choices
- Ranking performed **only on training data**
- No top-K selection at this stage
- Ranked feature indices reused consistently during training and evaluation

This ensures **leakage-free and reproducible feature selection**.

---

## Model Training

Three classical machine learning models are trained using **RF-ranked features**:

### Models
- **Decision Tree**
- **K-Nearest Neighbors (k = 3,5,7,9)**
- **Support Vector Machine (Linear Kernel, RBF    Kernel and Polynomial Kernel)**

### Training Strategy
- Training performed only on training data
- Model-specific top-K features (derived from prior CV experiments)
- Fixed random seed for reproducibility

### Metrics Logged During Training
- Accuracy
- ROC-AUC (Primary selection metric)
- Precision
- Recall
- F1-score

---

## Model Evaluation & Best Model Selection

### Evaluation Protocol
- Evaluation performed **only on the held-out test set**
- No retraining or resplitting during evaluation

### Metrics Used
- Accuracy  
- ROC-AUC (**primary selection metric**)  
- Precision  
- Recall  
- F1-score  

### Best Model Selection
- The best-performing model is selected based on **ROC-AUC**
- The selected model is registered as the **final production candidate**

---

## MLOps Components Implemented

### 1️⃣ Code Versioning
- **Git** is used for versioning all code
- Enables rollback, comparison, and traceability

### 2️⃣ Data Versioning
- **DVC** is used to version:
  - Original datasets
  - DDPM-augmented datasets
- Dataset versions are linked with Git commits

### 3️⃣ Pipeline Modularization
The pipeline is fully modular:


Each module performs a single, well-defined task.

### 4️⃣ Experiment Tracking
- **MLflow**
  - Parameter tracking
  - Metric logging
  - Model registry
- **Weights & Biases**
  - Training visualization
  - Experiment comparison

### 5️⃣ Pipeline Orchestration
- A single `main.py` orchestrates:
  1. Feature extraction
  2. Feature ranking
  3. Model training
  4. Model evaluation
  5. Model registration

This enables **one-command, end-to-end execution**.

---

## Key Outcomes

- Base-paper feature richness preserved (274 features)
- DDPM-based augmentation integrated safely
- Strict train–test separation maintained
- Honest generalization performance measured
- Full experiment and data reproducibility achieved
- Complete MLOps lifecycle implemented

---

## Conclusion
This project successfully transforms a **research-oriented machine learning solution** into a **reproducible, auditable, and production-ready MLOps pipeline** for Autism Spectrum Disorder diagnosis using structural MRI data.
