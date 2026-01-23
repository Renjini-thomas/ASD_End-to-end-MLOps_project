import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(img):
    img = cv2.resize(img, (256, 256))

    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    mean = np.mean(img)
    variance = np.var(img)
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    dissimilarity = contrast
    cluster_shade = np.mean((img - mean) ** 3)
    cluster_prominence = np.mean((img - mean) ** 4)

    return [
        contrast, correlation, energy, homogeneity,
        mean, variance, entropy, dissimilarity,
        cluster_shade, cluster_prominence, 0.0, 0.0
    ]
