import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import logger


class RFFeatureSelector:
    def __init__(self, n_estimators=300, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def rank_features(self, X_train, y_train):
        logger.info("Starting RF feature ranking (FINAL IMPLEMENTATION LOGIC)")

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion="gini",
            random_state=self.random_state,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        importances = rf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        logger.info("RF feature ranking completed")
        return sorted_indices
