import os
import cv2
import numpy as np

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops, label

from src.utils.logger import logger
from src.utils.path import (
    TRAIN_AUTISM_DIR,
    TRAIN_CONTROL_DIR,
    TEST_AUTISM_DIR,
    TEST_CONTROL_DIR,
    FEATURES_DIR
)


class FeatureExtractor:
    """
    Feature extractor for DDPM-augmented sMRI images
    """

    def extract_glcm(self, img):
        glcm = graycomatrix(
            img,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        return [
            graycoprops(glcm, 'contrast').mean(),
            graycoprops(glcm, 'correlation').mean(),
            graycoprops(glcm, 'energy').mean(),
            graycoprops(glcm, 'homogeneity').mean(),
            graycoprops(glcm, 'ASM').mean(),
            graycoprops(glcm, 'dissimilarity').mean(),
            np.mean(img),
            np.var(img),
            -np.sum(glcm * np.log2(glcm + 1e-10)),
            np.mean((img - np.mean(img)) ** 3),   # skewness proxy
            np.mean((img - np.mean(img)) ** 4),   # kurtosis proxy
            glcm.max()
        ]

    def extract_lbp(self, img):
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=256,
            range=(0, 256),
            density=True
        )
        return hist.tolist()

    def extract_gfcc(self, img):
        thresholds = threshold_multiotsu(img, classes=3)
        segmented = np.digitize(img, thresholds)

        labeled = label(segmented)
        regions = regionprops(labeled)

        if not regions:
            return [0]*6

        largest = max(regions, key=lambda r: r.area)

        return [
            largest.area,
            largest.perimeter,
            largest.major_axis_length,
            largest.minor_axis_length,
            largest.solidity,
            largest.extent    
        ]

    def extract_all_features(self, img):
        return (
            self.extract_glcm(img)
            + self.extract_lbp(img)
            + self.extract_gfcc(img)
        )


def _process_split(dataset, split_name):
    """
    Process a dataset split (train/test)
    """
    extractor = FeatureExtractor()
    X, y = [], []

    logger.info(f"Processing {split_name.upper()} dataset")

    for folder, label, name in dataset:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        logger.info(f" → {name}: {folder}")

        for fname in os.listdir(folder):
            img_path = os.path.join(folder, fname)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (256, 256))

            features = extractor.extract_all_features(img)
            X.append(features)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def run_feature_extraction():
    """
    Extract features from DDPM-augmented TRAIN and TEST datasets separately
    """
    logger.info("PIPELINE STARTED – STEP 1: FEATURE EXTRACTION")

    os.makedirs(FEATURES_DIR, exist_ok=True)

    # -----------------------
    # TRAIN SET
    # -----------------------
    train_dataset = [
        (TRAIN_CONTROL_DIR, 0, "Control"),
        (TRAIN_AUTISM_DIR, 1, "Autism")
    ]

    X_train, y_train = _process_split(train_dataset, "train")

    np.save(os.path.join(FEATURES_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(FEATURES_DIR, "y_train.npy"), y_train)

    logger.info(f"Train features saved | X_train shape: {X_train.shape}")

    # -----------------------
    # TEST SET
    # -----------------------
    test_dataset = [
        (TEST_CONTROL_DIR, 0, "Control"),
        (TEST_AUTISM_DIR, 1, "Autism")
    ]

    X_test, y_test = _process_split(test_dataset, "test")

    np.save(os.path.join(FEATURES_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(FEATURES_DIR, "y_test.npy"), y_test)

    logger.info(f"Test features saved | X_test shape: {X_test.shape}")

    logger.info("FEATURE EXTRACTION COMPLETED SUCCESSFULLY")
