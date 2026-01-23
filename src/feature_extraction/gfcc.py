import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.morphology import opening, closing, disk
from skimage.measure import regionprops, label as sk_label

def extract_gfcc_features(img):
    img = cv2.resize(img, (256, 256))
    h, w = img.shape

    thresholds = threshold_multiotsu(img, classes=3)
    segmented = np.digitize(img, bins=thresholds)

    mid_band = np.zeros_like(img, dtype=bool)
    mid_band[:, w//2 - 20:w//2 + 20] = True

    best_region = None
    max_area = 0

    for lbl in np.unique(segmented):
        mask = (segmented == lbl) & mid_band
        mask = opening(mask, disk(3))
        mask = closing(mask, disk(3))

        labeled = sk_label(mask)
        for r in regionprops(labeled):
            if r.area > max_area:
                best_region = r
                max_area = r.area

    if best_region is None:
        return [0, 0, 0, 0, 0, 0]

    return [
        best_region.area,
        best_region.perimeter,
        best_region.major_axis_length,
        best_region.minor_axis_length,
        best_region.solidity,
        best_region.extent
    ]
