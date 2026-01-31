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
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        return [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            np.mean(img),
            np.var(img),
            -np.sum(glcm * np.log2(glcm + 1e-10))
        ]

    def extract_lbp(self, img):
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=10,
            range=(0, 10),
            density=True
        )
        return hist.tolist()

    def extract_gfcc(self, img):
        thresholds = threshold_multiotsu(img, classes=3)
        segmented = np.digitize(img, thresholds)

        labeled = label(segmented)
        regions = regionprops(labeled)

        if not regions:
            return [0, 0, 0, 0]

        largest = max(regions, key=lambda r: r.area)

        return [
            largest.area,
            largest.perimeter,
            largest.major_axis_length,
            largest.minor_axis_length
        ]

    def extract_all_features(self, img):
        return (
            self.extract_glcm(img)
            + self.extract_lbp(img)
            + self.extract_gfcc(img)
        )


def run_feature_extraction():
    """
    Extract features from DDPM-augmented TRAIN dataset only
    """
    logger.info("Starting feature extraction (DDPM-augmented TRAIN set)")

    extractor = FeatureExtractor()
    X, y = [], []

    dataset = [
        (TRAIN_CONTROL_DIR, 0, "Control"),
        (TRAIN_AUTISM_DIR, 1, "Autism")
    ]

    for folder, label, name in dataset:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        logger.info(f"Processing {name} images from {folder}")

        for fname in os.listdir(folder):
            img_path = os.path.join(folder, fname)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (256, 256))

            features = extractor.extract_all_features(img)
            X.append(features)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    os.makedirs(FEATURES_DIR, exist_ok=True)

    np.save(os.path.join(FEATURES_DIR, "X.npy"), X)
    np.save(os.path.join(FEATURES_DIR, "y.npy"), y)

    logger.info(f"Feature extraction completed | Shape: {X.shape}")
    logger.info(f"Features saved to {FEATURES_DIR}")
    