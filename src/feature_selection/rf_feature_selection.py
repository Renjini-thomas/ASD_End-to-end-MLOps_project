import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import logger
from src.utils.path import FEATURES_DIR


class RFFeatureSelector:
    def __init__(self, n_estimators=300, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def rank_features(self, X_train, y_train):
        """
        Rank features using Random Forest importance.
        This method DOES NOT select top-K features.
        """

        logger.info("Starting RF feature ranking (TRAIN DATA ONLY)")

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion="gini",
            random_state=self.random_state,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        importances = rf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        # Save ranking for reuse in training
        os.makedirs(FEATURES_DIR, exist_ok=True)
        np.save(
            os.path.join(FEATURES_DIR, "sorted_feature_indices.npy"),
            sorted_indices
        )

        logger.info(
            f"Feature ranking completed | Total features ranked: {len(sorted_indices)}"
        )

        return sorted_indices
