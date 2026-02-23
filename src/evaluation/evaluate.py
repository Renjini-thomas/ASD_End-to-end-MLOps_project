import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.utils.logger import logger
from src.utils.path import FEATURES_DIR, ARTIFACTS_DIR


class ModelEvaluator:
    """
    Leakage-free evaluation on HOLD-OUT TEST SET
    (RF Selected + Scaled Features Already)
    """

    def __init__(self):

        mlflow.set_tracking_uri("file:./mlruns")

        mlflow.set_experiment(
            "ASD_Diagnoser_Model_Evaluation"
        )

        self.models_dir = os.path.join(
            ARTIFACTS_DIR,
            "models"
        )

    # -------------------------------------------------
    # Load TEST Features
    # -------------------------------------------------
    def _load_test_data(self):

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

            f"Loaded TEST data | "
            f"X_test={X_test.shape} "
            f"y_test={y_test.shape}"
        )

        return X_test, y_test

    # -------------------------------------------------
    # Evaluate ALL Saved Models
    # -------------------------------------------------
    def evaluate_models(self):

        logger.info(
            "PIPELINE STEP 4 : MODEL EVALUATION"
        )

        X_test, y_test = self._load_test_data()

        results = []

        model_files = [

            f for f in os.listdir(
                self.models_dir
            )

            if f.endswith(".pkl")
        ]

        # -----------------------
        # Evaluate each model
        # -----------------------
        for model_file in model_files:

            model_path = os.path.join(

                self.models_dir,
                model_file
            )

            model = joblib.load(
                model_path
            )

            model_name = model_file.replace(
                ".pkl",
                ""
            )

            logger.info(
                f"Evaluating {model_name}"
            )

            y_pred = model.predict(
                X_test
            )

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

            # ------------------
            # Metrics
            # ------------------
            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            roc_auc = roc_auc_score(
                y_test,
                y_prob
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            sensitivity = recall_score(
                y_test,
                y_pred
            )

            tn, fp, fn, tp = confusion_matrix(

                y_test,
                y_pred

            ).ravel()

            specificity = tn / (tn + fp + 1e-9)

            results.append(

                (
                    model_name,
                    accuracy,
                    roc_auc,
                    f1,
                    sensitivity,
                    specificity
                )
            )

        # -----------------------
        # BEST MODEL BY AUC
        # -----------------------
        best_model = max(
            results,
            key=lambda x: x[2]
        )

        logger.info(
            "===== FINAL TEST RESULTS ====="
        )

        for r in results:

            logger.info(

                f"{r[0]} | "
                f"Acc={r[1]:.4f} | "
                f"AUC={r[2]:.4f} | "
                f"F1={r[3]:.4f} | "
                f"Sens={r[4]:.4f} | "
                f"Spec={r[5]:.4f}"
            )

        logger.info(

            f"BEST MODEL → "
            f"{best_model[0]} "
            f"(AUC={best_model[2]:.4f})"
        )

        return best_model, results

    # -------------------------------------------------
    # Register BEST Model in MLflow Registry
    # -------------------------------------------------
    def register_best_model(

        self,
        best_model_name,
        accuracy,
        roc_auc,
        f1,
        sensitivity,
        specificity
    ):

        logger.info(
            f"Registering BEST MODEL → {best_model_name}"
        )

        model_path = os.path.join(

            self.models_dir,
            f"{best_model_name}.pkl"
        )

        model = joblib.load(
            model_path
        )

        with mlflow.start_run(

            run_name=f"Register_{best_model_name}"

        ):

            mlflow.set_tag(
                "stage",
                "evaluation"
            )

            mlflow.set_tag(
                "best_model",
                best_model_name
            )

            # Dataset Info
            mlflow.log_param(

                "dataset_name",
                "Autism_split_valid_2D_mid_sagittal"
            )

            mlflow.log_param(
                "augmentation",
                "DDPM"
            )

            mlflow.log_param(
                "rf_selected_features",
                40
            )

            # Metrics
            mlflow.log_metric(
                "accuracy",
                accuracy
            )

            mlflow.log_metric(
                "roc_auc",
                roc_auc
            )

            mlflow.log_metric(
                "f1_score",
                f1
            )

            mlflow.log_metric(
                "sensitivity",
                sensitivity
            )

            mlflow.log_metric(
                "specificity",
                specificity
            )

            mlflow.sklearn.log_model(

                model,

                artifact_path="Best_ASD_Model",

                registered_model_name="ASD_Diagnoser_Best_Model"
            )

        logger.info(
            "BEST MODEL REGISTERED SUCCESSFULLY"
        )