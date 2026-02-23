import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.utils.logger import logger
from src.utils.path import FEATURES_DIR


class RFFeatureSelector:

    def __init__(self, k=40):

        self.k = k

    def select_features(self,X_train,y_train,X_test):

        logger.info("RF Selecting Top Features")

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train,y_train)

        importances = rf.feature_importances_

        sorted_idx = np.argsort(importances)[::-1][:self.k]

        # APPLY SELECTION
        X_train = X_train[:,sorted_idx]
        X_test = X_test[:,sorted_idx]

        # SAVE INDEX
        np.save(
            os.path.join(FEATURES_DIR,"rf_selected_indices.npy"),
            sorted_idx
        )

        # =====================
        # SCALING (COLAB STYLE)
        # =====================

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        joblib.dump(
            scaler,
            os.path.join(FEATURES_DIR,"fusion_scaler.pkl")
        )

        np.save(os.path.join(FEATURES_DIR,"X_train.npy"),X_train)
        np.save(os.path.join(FEATURES_DIR,"X_test.npy"),X_test)

        logger.info("RF Selection + Scaling Completed")

        return X_train,X_test