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
    def __init__(self):
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("ASD_Diagnoser_Model_Evaluation")

        self.models_dir = os.path.join(ARTIFACTS_DIR, "models")

        # Final frozen feature counts (from CV)
        self.feature_config = {
            "DecisionTree": 10,
            "KNN": 10,
            "SVM": 40
        }

    # -------------------------------------------------
    # Load EXACT same test data used in training
    # -------------------------------------------------
    def _load_test_data(self):
        X_test = np.load(os.path.join(ARTIFACTS_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))

        sorted_idx = np.load(
            os.path.join(FEATURES_DIR, "sorted_feature_indices.npy")
        )

        return X_test, y_test, sorted_idx

    # -------------------------------------------------
    # Main evaluation logic
    # -------------------------------------------------
    def evaluate_models(self):
        logger.info("Starting model evaluation")

        X_test, y_test, sorted_idx = self._load_test_data()
        results = []

        # =================================================
        # Decision Tree
        # =================================================
        dt_model = joblib.load(
            os.path.join(self.models_dir, "decision_tree_model.pkl")
        )

        X_dt = X_test[:, sorted_idx][:, :self.feature_config["DecisionTree"]]

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

        # =================================================
        # KNN
        # =================================================
        knn_model = joblib.load(
            os.path.join(self.models_dir, "knn_model.pkl")
        )

        X_knn = X_test[:, sorted_idx][:, :self.feature_config["KNN"]]

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

        # =================================================
        # SVM
        # =================================================
        svm_model = joblib.load(
            os.path.join(self.models_dir, "svm_model.pkl")
        )
        scaler = joblib.load(
            os.path.join(self.models_dir, "svm_scaler.pkl")
        )

        X_svm = X_test[:, sorted_idx][:, :self.feature_config["SVM"]]
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

        # =================================================
        # Select Best Model (by ROC-AUC)
        # =================================================
        best_model = max(results, key=lambda x: x[2])

        logger.info("Evaluation Results:")
        for r in results:
            logger.info(
                f"{r[0]} | "
                f"Accuracy={r[1]:.4f} | "
                f"ROC-AUC={r[2]:.4f} | "
                f"Precision={r[3]:.4f} | "
                f"Recall={r[4]:.4f} | "
                f"F1={r[5]:.4f}"
            )

        logger.info(
            f"Best Model Selected: {best_model[0]} "
            f"(ROC-AUC={best_model[2]:.4f})"
        )

        return best_model, results

    # -------------------------------------------------
    # Register Best Model in MLflow
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
            "DecisionTree": "decision_tree_model.pkl",
            "KNN": "knn_model.pkl",
            "SVM": "svm_model.pkl"
        }

        model_path = os.path.join(
            self.models_dir,
            model_map[best_model_name]
        )

        with mlflow.start_run(run_name=f"Register_{best_model_name}"):
            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("best_model", best_model_name)

            model = joblib.load(model_path)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(
                model,
                name="Best_ASD_Model",
                registered_model_name="ASD_Diagnoser_Best_Model"
            )

        logger.info("Best model successfully registered in MLflow")
