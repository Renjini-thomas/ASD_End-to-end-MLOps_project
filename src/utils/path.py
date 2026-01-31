import os

# =========================
# PROJECT ROOT
# =========================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# =========================
# DATA DIRECTORIES
# =========================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Raw data (original NIfTI / sagittal slices if needed)
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# DDPM augmented dataset
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, "augmented_train_500")

# Train / Test split (already created)
TRAIN_DIR = os.path.join(AUGMENTED_DATA_DIR, "train")
TEST_DIR = os.path.join(AUGMENTED_DATA_DIR, "test")

# Class directories
TRAIN_AUTISM_DIR = os.path.join(TRAIN_DIR, "Autism")
TRAIN_CONTROL_DIR = os.path.join(TRAIN_DIR, "Control")

TEST_AUTISM_DIR = os.path.join(TEST_DIR, "Autism")
TEST_CONTROL_DIR = os.path.join(TEST_DIR, "Control")

# =========================
# ARTIFACTS
# =========================
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

FEATURES_DIR = os.path.join(ARTIFACTS_DIR, "features")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
