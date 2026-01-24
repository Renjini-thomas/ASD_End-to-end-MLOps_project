import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_multiotsu
from skimage.morphology import opening, closing, disk
from skimage.measure import regionprops, label
from src.utils.logger import logger
from src.utils.path import CONTROL_DIR, AUTISTIC_DIR, FEATURES_DIR

class FeatureExtractor:

    def extract_glcm(self, img):
        img = cv2.resize(img, (256, 256))
        glcm = graycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
        mean = np.mean(img)

        return [
            graycoprops(glcm, 'contrast').mean(),
            graycoprops(glcm, 'correlation').mean(),
            graycoprops(glcm, 'energy').mean(),
            graycoprops(glcm, 'homogeneity').mean(),
            mean,
            np.var(img),
            -np.sum(glcm * np.log2(glcm + 1e-10))
        ]

    def extract_lbp(self, img):
        img = cv2.resize(img, (256, 256))
        lbp = local_binary_pattern(img, 8, 1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        return hist.tolist()

    def extract_gfcc(self, img):
        img = cv2.resize(img, (256, 256))
        thresholds = threshold_multiotsu(img, classes=3)
        segmented = np.digitize(img, thresholds)

        labeled = label(segmented)
        regions = regionprops(labeled)

        if not regions:
            return [0, 0, 0, 0]

        r = max(regions, key=lambda x: x.area)
        return [
            r.area,
            r.perimeter,
            r.major_axis_length,
            r.minor_axis_length
        ]

    def extract_all_features(self, img):
        return (
            self.extract_glcm(img)
            + self.extract_lbp(img)
            + self.extract_gfcc(img)
        )


def run_feature_extraction():
    logger.info("Starting feature extraction")

    extractor = FeatureExtractor()
    X, y = [], []

    for folder, label in [(CONTROL_DIR, 0), (AUTISTIC_DIR, 1)]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            features = extractor.extract_all_features(img)
            X.append(features)
            y.append(label)


    X = np.array(X)
    y = np.array(y)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.save(os.path.join(FEATURES_DIR, "X.npy"), X)
    np.save(os.path.join(FEATURES_DIR, "y.npy"), y)


    logger.info(f"Feature extraction completed. Shape: {X.shape}")
