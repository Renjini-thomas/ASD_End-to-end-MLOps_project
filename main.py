import numpy as np
from sklearn.model_selection import train_test_split

from src.feature_extraction.extract_features import run_feature_extraction
from src.feature_selection.rf_feature_selection import RFFeatureSelector
from src.utils.logger import logger


if __name__ == "__main__":
    try:
        logger.info("PIPELINE STARTED – FEATURE EXTRACTION + SELECTION")

        # Step 1: Feature extraction
        run_feature_extraction()
        

        # Load features
        X = np.load("artifacts/features/X.npy")
        y = np.load("artifacts/features/y.npy")

        # Same split as final implementation
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=42
        )

        # Step 2: Feature selection
        selector = RFFeatureSelector()
        sorted_idx = selector.rank_features(X_train, y_train)

        np.save(
            "artifacts/features/sorted_feature_indices.npy",
            sorted_idx
        )

        logger.info("FEATURE EXTRACTION + SELECTION COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.exception(e)
        raise e
