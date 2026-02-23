import os
import cv2
from cv2.gapi import mask
from matplotlib.pyplot import hist
from networkx import radius
import numpy as np

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops, label
from skimage.morphology import closing, disk, opening

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
        glcm = graycomatrix(img,[1],[0],256,True,True)
        P = glcm[:,:,0,0]
        P = P/(P.sum()+1e-12)

        i,j = np.indices(P.shape)

        contrast = np.sum(P*(i-j)**2)
        dissimilarity = np.sum(P*np.abs(i-j))
        homogeneity = np.sum(P/(1+(i-j)**2))
        energy = np.sum(P**2)
        asm = energy
        correlation = graycoprops(glcm,'correlation')[0,0]

        mean_i = np.sum(i*P)
        mean_j = np.sum(j*P)

        var = (np.sum((i-mean_i)**2*P)+np.sum((j-mean_j)**2*P))/2
        entropy = -np.sum(P*np.log(P+1e-12))

        mu = mean_i+mean_j
        shade = np.sum(((i+j-mu)**3)*P)
        prominence = np.sum(((i+j-mu)**4)*P)

        autocorr = np.sum(i*j*P)

        return np.array([
        contrast, correlation, energy, homogeneity,
        asm, dissimilarity, entropy,
        (mean_i+mean_j)/2, var,
        shade, prominence, autocorr
    ])

    def extract_lbp(self, img):
        radius = 1
        points = 8

        lbp = local_binary_pattern(img, points, radius, method="default")

        hist,_ = np.histogram(lbp.ravel(),
                          bins=256,
                          range=(0,256))

        hist = hist / (hist.sum() + 1e-6)

        return hist

    def extract_gfcc(self, img):
        th = threshold_multiotsu(img,3)
        seg = np.digitize(img,bins=th)

        mask = seg==2
        mask = opening(mask,disk(3))
        mask = closing(mask,disk(5))

        lbl = label(mask)
        props = regionprops(lbl)

        if not props:
            return [0]*11

        r = max(props,key=lambda x:x.area)

        circularity = (4*np.pi*r.area)/(r.perimeter**2 + 1e-6)
        axis_ratio = r.major_axis_length/(r.minor_axis_length+1e-6)
        convex_ratio = r.area/(r.convex_area+1e-6)

        minr, minc, maxr, maxc = r.bbox
        bbox_ratio = (maxc-minc)/(maxr-minr+1e-6)

        return [
        r.area,
        r.perimeter,
        r.major_axis_length,
        r.minor_axis_length,
        r.eccentricity,
        r.solidity,
        r.extent,
        circularity,
        axis_ratio,
        convex_ratio,
        bbox_ratio
    ]

    def extract_all_features(self, img):
        glcm = self.extract_glcm(img)
        lbp = self.extract_lbp(img)
        gfcc = self.extract_gfcc(img)
        fusion = np.concatenate([glcm, lbp, gfcc])

        return fusion


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
