import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.feature_extraction.extract_features import run_feature_extraction
from src.feature_selection.rf_feature_selection import RFFeatureSelector
from src.training.train_models import ModelTrainer
from src.evaluation.evaluate import ModelEvaluator
from src.utils.logger import logger
from src.utils.path import FEATURES_DIR, ARTIFACTS_DIR


if __name__ == "__main__":
    try:
        # ==================================================
        # STEP 1: FEATURE EXTRACTION
        # ==================================================
        logger.info("PIPELINE STARTED – STEP 1: FEATURE EXTRACTION")

        run_feature_extraction()

        X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))

        X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

        logger.info(
            f"Features loaded | "
            f"X_train: {X_train.shape}, y_train: {y_train.shape} | "
            f"X_test: {X_test.shape}, y_test: {y_test.shape}"
        )


        # ==================================================
        # STEP 2: TRAIN–TEST SPLIT (FREEZE TEST SET)
        # ==================================================
        logger.info("PIPELINE STARTED – STEP 2: Random Forest FEATURE SELECTION")
        X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
        logger.info(f"Loaded TRAIN features | X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        selector = RFFeatureSelector(n_estimators=300, random_state=42)
        sorted_idx = selector.rank_features(X_train, y_train)
        logger.info(f"RF feature ranking completed successfully | total features ranked: {len(sorted_idx)}")
    #     logger.info("Creating train–test split")

    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X,
    #         y,
    #         test_size=0.3,
    #         stratify=y,
    #         random_state=42
    #     )

    #     # 🔐 Persist test set for evaluation (CRITICAL FOR MLOPS)
    #     os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    #     np.save(os.path.join(ARTIFACTS_DIR, "X_test.npy"), X_test)
    #     np.save(os.path.join(ARTIFACTS_DIR, "y_test.npy"), y_test)

    #     logger.info("Train–test split completed and test set saved")

    # ==================================================
# # STEP 3: FEATURE SELECTION (RF RANKING)
# # ==================================================
#         logger.info("PIPELINE STARTED – STEP 3: FEATURE SELECTION")

#         selector = RFFeatureSelector(n_estimators=300, random_state=42)

#         sorted_idx = selector.rank_features(X_train, y_train)

#         logger.info(
#     f"RF feature ranking saved | Total features ranked: {len(sorted_idx)}"
#         )


        # ==================================================
        # STEP 4: MODEL TRAINING
        # ==================================================
        logger.info("PIPELINE STARTED – STEP 3: MODEL TRAINING")
        trainer = ModelTrainer(params_path="configs/params.yaml")
        trainer.train_decision_tree()
        trainer.train_knn()
        trainer.train_svm() 
        logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
    #     logger.info("PIPELINE STARTED – STEP 3: MODEL TRAINING")

    #     trainer = ModelTrainer()

    #     # Final frozen configuration (from CV)
    #     trainer.train_decision_tree(top_k=10)
    #     trainer.train_knn(top_k=10, n_neighbors=3)
    #     trainer.train_svm(top_k=40)

    #     logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")

    #     # ==================================================
    #     # STEP 4: MODEL EVALUATION & REGISTRATION
    #     # ==================================================
        logger.info("PIPELINE STARTED – STEP 4: MODEL EVALUATION")

        evaluator = ModelEvaluator()
        best_model, results = evaluator.evaluate_models()

        (
            best_model_name,
            best_accuracy,
            best_roc_auc,
            best_precision,
            best_recall,
            best_f1
        ) = best_model

        evaluator.register_best_model(
            best_model_name=best_model_name,
            accuracy=best_accuracy,
            roc_auc=best_roc_auc,
            precision=best_precision,
            recall=best_recall,
            f1=best_f1
        )

        logger.info("MODEL EVALUATION & REGISTRATION COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.exception(e)
        raise e
