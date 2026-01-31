import os
import numpy as np
import pandas as pd

from src.utils.path import FEATURES_DIR
from src.utils.logger import logger


# --------------------------------------------------
# Feature names (21 total – SAME as final implementation)
# --------------------------------------------------
FEATURE_NAMES = [
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


def _load_split(split="train"):
    """
    Load features for a given split (train or test) and return DataFrame
    """
    X_path = os.path.join(FEATURES_DIR, f"X_{split}.npy")
    y_path = os.path.join(FEATURES_DIR, f"y_{split}.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Feature files for '{split}' split not found. "
            "Run feature extraction first."
        )

    X = np.load(X_path)
    y = np.load(y_path)

    logger.info(
        f"Loaded {split.upper()} features | "
        f"X shape: {X.shape}, y shape: {y.shape}"
    )

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df["class"] = df["label"].map({0: "Control", 1: "Autism"})
    df["split"] = split

    return df


def load_features_as_dataframe(save_csv=False, combined=True):
    """
    Load TRAIN and TEST features and convert them into Pandas DataFrames

    Args:
        save_csv (bool): Save DataFrame(s) as CSV
        combined (bool): If True, return combined DataFrame
    """
    df_train = _load_split("train")
    df_test = _load_split("test")

    if combined:
        df_all = pd.concat([df_train, df_test], ignore_index=True)

        if save_csv:
            csv_path = os.path.join(FEATURES_DIR, "features_dataframe_all.csv")
            df_all.to_csv(csv_path, index=False)
            logger.info(f"Combined feature DataFrame saved to {csv_path}")

        return df_all

    else:
        if save_csv:
            train_csv = os.path.join(FEATURES_DIR, "features_train.csv")
            test_csv = os.path.join(FEATURES_DIR, "features_test.csv")

            df_train.to_csv(train_csv, index=False)
            df_test.to_csv(test_csv, index=False)

            logger.info(f"Train features saved to {train_csv}")
            logger.info(f"Test features saved to {test_csv}")

        return df_train, df_test


# --------------------------------------------------
# Run as script
# --------------------------------------------------
if __name__ == "__main__":
    df = load_features_as_dataframe(save_csv=True, combined=True)

    print("\n===== FEATURE DATAFRAME INFO =====")
    print(df.info())

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== CLASS DISTRIBUTION (by split) =====")
    print(df.groupby("split")["class"].value_counts())
