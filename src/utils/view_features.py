import os
import numpy as np
import pandas as pd

from src.utils.path import FEATURES_DIR
from src.utils.logger import logger


def load_features_as_dataframe(save_csv=False):
    """
    Load extracted features and convert them into a Pandas DataFrame
    """

    X_path = os.path.join(FEATURES_DIR, "X.npy")
    y_path = os.path.join(FEATURES_DIR, "y.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Feature files not found. Run feature extraction first.")

    X = np.load(X_path)
    y = np.load(y_path)

    logger.info(f"Loaded features | X shape: {X.shape}, y shape: {y.shape}")

    # ---- Feature names (21 features total) ----
    feature_names = [
        # GLCM (7)
        "glcm_contrast",
        "glcm_correlation",
        "glcm_energy",
        "glcm_homogeneity",
        "mean_intensity",
        "variance",
        "entropy",

        # LBP (10)
        "lbp_bin_0",
        "lbp_bin_1",
        "lbp_bin_2",
        "lbp_bin_3",
        "lbp_bin_4",
        "lbp_bin_5",
        "lbp_bin_6",
        "lbp_bin_7",
        "lbp_bin_8",
        "lbp_bin_9",

        # GFCC / Morphological (4)
        "region_area",
        "region_perimeter",
        "major_axis_length",
        "minor_axis_length"
    ]

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    df["class"] = df["label"].map({0: "Control", 1: "Autism"})

    if save_csv:
        csv_path = os.path.join(FEATURES_DIR, "features_dataframe.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Feature DataFrame saved to {csv_path}")

    return df


if __name__ == "__main__":
    df = load_features_as_dataframe(save_csv=True)

    # Show basic info
    print("\n===== FEATURE DATAFRAME INFO =====")
    print(df.info())

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== CLASS DISTRIBUTION =====")
    print(df["class"].value_counts())
