import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.logger import logger
from src.utils.path import FEATURES_DIR, ARTIFACTS_DIR


class ModelEvaluator:
    def __init__(self):
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("ASD_Diagnoser_Model_Evaluation")

        self.models_dir = os.path.join(ARTIFACTS_DIR, "models")

    def _load_test_data(self):
        X = np.load(os.path.join(FEATURES_DIR, "X.npy"))
        y = np.load(os.path.join(FEATURES_DIR, "y.npy"))
        sorted_idx = np.load(
            os.path.join(FEATURES_DIR, "sorted_feature_indices.npy")
        )

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=42
        )

        return X_test, y_test, sorted_idx

    def evaluate_models(self):
        logger.info("Starting model evaluation")

        X_test, y_test, sorted_idx = self._load_test_data()

        results = []

        # ---------------------------
        # Decision Tree
        # ---------------------------
        dt_model = joblib.load(
            os.path.join(self.models_dir, "decision_tree_model.pkl")
        )
        X_dt = X_test[:, sorted_idx][:, :30]

        dt_pred = dt_model.predict(X_dt)
        dt_prob = dt_model.predict_proba(X_dt)[:, 1]

        dt_acc = accuracy_score(y_test, dt_pred)
        dt_roc = roc_auc_score(y_test, dt_prob)

        results.append(("DecisionTree", dt_acc, dt_roc))

        # ---------------------------
        # KNN
        # ---------------------------
        knn_model = joblib.load(
            os.path.join(self.models_dir, "knn_model.pkl")
        )
        X_knn = X_test[:, sorted_idx][:, :10]

        knn_pred = knn_model.predict(X_knn)
        knn_prob = knn_model.predict_proba(X_knn)[:, 1]

        knn_acc = accuracy_score(y_test, knn_pred)
        knn_roc = roc_auc_score(y_test, knn_prob)

        results.append(("KNN", knn_acc, knn_roc))

        # ---------------------------
        # SVM
        # ---------------------------
        svm_model = joblib.load(
            os.path.join(self.models_dir, "svm_model.pkl")
        )
        scaler = joblib.load(
            os.path.join(self.models_dir, "svm_scaler.pkl")
        )

        X_svm = X_test[:, sorted_idx][:, :50]
        X_svm = scaler.transform(X_svm)

        svm_pred = svm_model.predict(X_svm)
        svm_prob = svm_model.predict_proba(X_svm)[:, 1]

        svm_acc = accuracy_score(y_test, svm_pred)
        svm_roc = roc_auc_score(y_test, svm_prob)

        results.append(("SVM", svm_acc, svm_roc))

        # ---------------------------
        # Select Best Model
        # ---------------------------
        best_model = max(results, key=lambda x: x[2])

        logger.info("Evaluation Results:")
        for r in results:
            logger.info(
                f"{r[0]} | Accuracy={r[1]:.4f} | ROC-AUC={r[2]:.4f}"
            )

        logger.info(
            f"Best Model: {best_model[0]} "
            f"(ROC-AUC={best_model[2]:.4f})"
        )

        return best_model, results

    def register_best_model(self, best_model_name, best_accuracy, best_roc_auc):
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
            model = joblib.load(model_path)

        # 🔹 Log metrics explicitly
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("accuracy", best_accuracy)
            mlflow.log_metric("roc_auc", best_roc_auc)

        # 🔹 Register model
            mlflow.sklearn.log_model(
            model,
            name="Best_ASD_Model",
            registered_model_name="ASD_Diagnoser_Best_Model"
        )

        logger.info("Best model registered with metrics in MLflow")

