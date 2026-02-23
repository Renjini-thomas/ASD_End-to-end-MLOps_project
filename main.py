import os
import numpy as np

from src.feature_extraction.extract_features import run_feature_extraction
from src.feature_selection.rf_feature_selection import RFFeatureSelector
from src.training.train_models import ModelTrainer
from src.evaluation.evaluate import ModelEvaluator

from src.utils.logger import logger
from src.utils.path import FEATURES_DIR


if __name__ == "__main__":

    try:

        # ==================================================
        # STEP 1 : FEATURE EXTRACTION
        # ==================================================
        logger.info(
            "PIPELINE STEP 1 : FEATURE EXTRACTION STARTED"
        )

        run_feature_extraction()

        logger.info(
            "FEATURE EXTRACTION COMPLETED"
        )

        # --------------------------------------------------
        # Load Extracted Features
        # --------------------------------------------------
        X_train = np.load(
            os.path.join(
                FEATURES_DIR,
                "X_train.npy"
            )
        )

        y_train = np.load(
            os.path.join(
                FEATURES_DIR,
                "y_train.npy"
            )
        )

        X_test = np.load(
            os.path.join(
                FEATURES_DIR,
                "X_test.npy"
            )
        )

        y_test = np.load(
            os.path.join(
                FEATURES_DIR,
                "y_test.npy"
            )
        )

        logger.info(

            f"Features Loaded | "
            f"Train={X_train.shape} "
            f"Test={X_test.shape}"
        )

        # ==================================================
        # STEP 2 : RF FEATURE SELECTION + SCALING
        # ==================================================
        logger.info(
            "PIPELINE STEP 2 : RF FEATURE SELECTION STARTED"
        )

        selector = RFFeatureSelector(
            k=40     # From Colab RF selection
        )

        X_train_selected, X_test_selected = selector.select_features(

            X_train,
            y_train,
            X_test
        )

        logger.info(

            f"RF Selection Completed | "
            f"Train={X_train_selected.shape} "
            f"Test={X_test_selected.shape}"
        )

        # ==================================================
        # STEP 3 : MODEL TRAINING
        # ==================================================
        logger.info(
            "PIPELINE STEP 3 : MODEL TRAINING STARTED"
        )

        trainer = ModelTrainer(
            params_path="configs/params.yaml"
        )

        # NEW SINGLE FUNCTION
        trainer.train_all_models()

        logger.info(
            "MODEL TRAINING COMPLETED"
        )

        # ==================================================
        # STEP 4 : MODEL EVALUATION
        # ==================================================
        logger.info(
            "PIPELINE STEP 4 : MODEL EVALUATION STARTED"
        )

        evaluator = ModelEvaluator()

        best_model, results = evaluator.evaluate_models()

        (
            best_model_name,
            best_accuracy,
            best_auc,
            best_f1,
            best_sensitivity,
            best_specificity
        ) = best_model

        logger.info(
            f"BEST MODEL SELECTED : {best_model_name}"
        )

        # ==================================================
        # STEP 5 : REGISTER BEST MODEL
        # ==================================================
        evaluator.register_best_model(

            best_model_name=best_model_name,

            accuracy=best_accuracy,

            roc_auc=best_auc,

            f1=best_f1,

            sensitivity=best_sensitivity,

            specificity=best_specificity
        )

        logger.info(

            "PIPELINE COMPLETED SUCCESSFULLY"
        )

    except Exception as e:

        logger.exception(
            "PIPELINE FAILED"
        )

        raise e