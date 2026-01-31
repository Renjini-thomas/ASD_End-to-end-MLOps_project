import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from src.utils.logger import logger
from src.utils.path import FEATURES_DIR, ARTIFACTS_DIR


class ModelEvaluator:
    """
    Model evaluation on HOLD-OUT TEST SET (leakage-free)
    """

    def __init__(self):
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("ASD_Diagnoser_Model_Evaluation")

        self.models_dir = os.path.join(ARTIFACTS_DIR, "models")

        # Frozen feature counts (from CV / params.yaml)
        self.feature_config = {
            "DecisionTree": 20,
            "KNN": 10,
            "SVM": 20
        }

    # -------------------------------------------------
    # Load test features + RF ranking
    # -------------------------------------------------
    def _load_test_data(self):
        X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

        sorted_idx = np.load(
            os.path.join(FEATURES_DIR, "sorted_feature_indices.npy")
        )

        logger.info(
            f"Loaded TEST data | X_test={X_test.shape}, y_test={y_test.shape}"
        )

        return X_test, y_test, sorted_idx

    # -------------------------------------------------
    # Evaluate all trained models
    # -------------------------------------------------
    def evaluate_models(self):
        logger.info("PIPELINE STARTED – STEP 4: MODEL EVALUATION")

        X_test, y_test, sorted_idx = self._load_test_data()
        results = []

        # ============================
        # Decision Tree
        # ============================
        dt_model = joblib.load(
            os.path.join(self.models_dir, "decision_tree.pkl")
        )

        k = self.feature_config["DecisionTree"]
        X_dt = X_test[:, sorted_idx][:, :k]

        dt_pred = dt_model.predict(X_dt)
        dt_prob = dt_model.predict_proba(X_dt)[:, 1]

        results.append((
            "DecisionTree",
            accuracy_score(y_test, dt_pred),
            roc_auc_score(y_test, dt_prob),
            precision_score(y_test, dt_pred),
            recall_score(y_test, dt_pred),
            f1_score(y_test, dt_pred)
        ))

        # ============================
        # KNN
        # ============================
        knn_model = joblib.load(
            os.path.join(self.models_dir, "knn.pkl")
        )

        k = self.feature_config["KNN"]
        X_knn = X_test[:, sorted_idx][:, :k]

        knn_pred = knn_model.predict(X_knn)
        knn_prob = knn_model.predict_proba(X_knn)[:, 1]

        results.append((
            "KNN",
            accuracy_score(y_test, knn_pred),
            roc_auc_score(y_test, knn_prob),
            precision_score(y_test, knn_pred),
            recall_score(y_test, knn_pred),
            f1_score(y_test, knn_pred)
        ))

        # ============================
        # SVM
        # ============================
        svm_model = joblib.load(
            os.path.join(self.models_dir, "svm.pkl")
        )
        scaler = joblib.load(
            os.path.join(self.models_dir, "svm_scaler.pkl")
        )

        k = self.feature_config["SVM"]
        X_svm = X_test[:, sorted_idx][:, :k]
        X_svm = scaler.transform(X_svm)

        svm_pred = svm_model.predict(X_svm)
        svm_prob = svm_model.predict_proba(X_svm)[:, 1]

        results.append((
            "SVM",
            accuracy_score(y_test, svm_pred),
            roc_auc_score(y_test, svm_prob),
            precision_score(y_test, svm_pred),
            recall_score(y_test, svm_pred),
            f1_score(y_test, svm_pred)
        ))

        # ============================
        # Select Best Model (ROC-AUC)
        # ============================
        best_model = max(results, key=lambda x: x[2])

        logger.info("===== FINAL TEST RESULTS =====")
        for r in results:
            logger.info(
                f"{r[0]} | "
                f"Acc={r[1]:.4f} | "
                f"ROC-AUC={r[2]:.4f} | "
                f"Prec={r[3]:.4f} | "
                f"Recall={r[4]:.4f} | "
                f"F1={r[5]:.4f}"
            )

        logger.info(
            f"BEST MODEL → {best_model[0]} (ROC-AUC={best_model[2]:.4f})"
        )

        return best_model, results

    # -------------------------------------------------
    # Register best model in MLflow
    # -------------------------------------------------
    def register_best_model(
        self,
        best_model_name,
        accuracy,
        roc_auc,
        precision,
        recall,
        f1
    ):
        logger.info(f"Registering best model: {best_model_name}")

        model_map = {
            "DecisionTree": "decision_tree.pkl",
            "KNN": "knn.pkl",
            "SVM": "svm.pkl"
        }

        model_path = os.path.join(self.models_dir, model_map[best_model_name])
        model = joblib.load(model_path)

        with mlflow.start_run(run_name=f"Register_{best_model_name}"):
            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("best_model", best_model_name)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(
                model,
                artifact_path="Best_ASD_Model",
                registered_model_name="ASD_Diagnoser_Best_Model"
            )

        logger.info("Best model successfully registered in MLflow")
