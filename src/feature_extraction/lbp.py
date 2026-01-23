import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(img, block_size=32):
    img = cv2.resize(img, (256, 256))
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")

    features = []
    for y in range(0, 256, block_size):
        for x in range(0, 256, block_size):
            block = lbp[y:y+block_size, x:x+block_size]
            hist, _ = np.histogram(
                block.ravel(),
                bins=10,
                range=(0, 10),
                density=True
            )
            features.extend(hist)

    return features
