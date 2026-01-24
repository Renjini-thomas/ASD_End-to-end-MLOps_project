import os

# Project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Your actual dataset folder
ABIDE_DIR = os.path.join(RAW_DATA_DIR, "abide1_data(sagittal_2)")

CONTROL_DIR = os.path.join(ABIDE_DIR, "control")
AUTISTIC_DIR = os.path.join(ABIDE_DIR, "autistic")

# Artifacts
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
FEATURES_DIR = os.path.join(ARTIFACTS_DIR, "features")
