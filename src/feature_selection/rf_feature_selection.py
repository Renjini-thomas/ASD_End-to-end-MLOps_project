import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import logger
from src.utils.path import FEATURES_DIR


class RFFeatureSelector:
    """
    Random Forest based feature ranking.
    NOTE:
    - This class ONLY ranks features.
    - It does NOT select top-K features.
    """

    def __init__(self, n_estimators=300, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def rank_features(self, X_train, y_train):
        """
        Rank features using Random Forest importance.

        Args:
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training labels

        Returns:
            np.ndarray: Indices of features sorted by importance (descending)
        """
        logger.info("Starting RF feature ranking (TRAIN data only)")

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion="gini",
            random_state=self.random_state,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        importances = rf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        # Save ranking artifacts
        os.makedirs(FEATURES_DIR, exist_ok=True)

        np.save(
            os.path.join(FEATURES_DIR, "sorted_feature_indices.npy"),
            sorted_indices
        )

        np.save(
            os.path.join(FEATURES_DIR, "feature_importances.npy"),
            importances
        )

        logger.info(
            f"RF feature ranking completed | Total features ranked: {len(sorted_indices)}"
        )

        return sorted_indices
