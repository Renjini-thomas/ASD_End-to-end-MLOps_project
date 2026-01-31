import os
import yaml
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import wandb

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from src.utils.logger import logger
from src.utils.path import FEATURES_DIR, ARTIFACTS_DIR


class ModelTrainer:
    def __init__(self, params_path="params.yaml"):
        # Load params.yaml
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)

        # MLflow config
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("ASD_Diagnoser_Model_Training")

        # Artifact dirs
        self.models_dir = os.path.join(ARTIFACTS_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Params
        self.ct_k = self.params["features"]["ct_best"]
        self.knn_k = self.params["features"]["knn_best"]
        self.svm_k = self.params["features"]["svm_best"]

        self.knn_neighbors = self.params["models"]["knn_neighbors"]
        self.svm_kernel = self.params["models"]["svm_kernel"]

    def _load_data(self):
        """
        Load predefined train/test split and RF ranking
        """
        X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
        X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

        sorted_idx = np.load(
            os.path.join(FEATURES_DIR, "sorted_feature_indices.npy")
        )

        logger.info(
            f"Loaded data | "
            f"X_train={X_train.shape}, X_test={X_test.shape}"
        )

        return X_train, X_test, y_train, y_test, sorted_idx

    # ==================================================
    # Decision Tree
    # ==================================================
    def train_decision_tree(self):
        logger.info(f"Training Decision Tree | Top-{self.ct_k} features")

        Xtr, Xte, ytr, yte, idx = self._load_data()

        Xtr = Xtr[:, idx][:, :self.ct_k]
        Xte = Xte[:, idx][:, :self.ct_k]

        with mlflow.start_run(run_name="DecisionTree"):
            wandb.init(project="ASD-Diagnoser", name="DecisionTree")

            model = DecisionTreeClassifier(
                criterion="gini",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )

            model.fit(Xtr, ytr)

            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:, 1]

            metrics = {
                "accuracy": accuracy_score(yte, y_pred),
                "roc_auc": roc_auc_score(yte, y_prob),
                # "precision": precision_score(yte, y_pred),
                # "recall": recall_score(yte, y_pred),
                # "f1_score": f1_score(yte, y_pred)
            }

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.log_param("top_k_features", self.ct_k)
            mlflow.log_param("model", "DecisionTree")

            mlflow.sklearn.log_model(model, "decision_tree")
            joblib.dump(model, os.path.join(self.models_dir, "decision_tree.pkl"))

            wandb.log(metrics)
            wandb.finish()

        logger.info(f"Decision Tree metrics: {metrics}")

    # ==================================================
    # KNN
    # ==================================================
    def train_knn(self):
        logger.info(f"Training KNN | Top-{self.knn_k} features")

        Xtr, Xte, ytr, yte, idx = self._load_data()

        Xtr = Xtr[:, idx][:, :self.knn_k]
        Xte = Xte[:, idx][:, :self.knn_k]

        with mlflow.start_run(run_name="KNN"):
            wandb.init(project="ASD-Diagnoser", name="KNN")

            model = KNeighborsClassifier(
                n_neighbors=self.knn_neighbors
            )

            model.fit(Xtr, ytr)

            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:, 1]

            metrics = {
                "accuracy": accuracy_score(yte, y_pred),
                "roc_auc": roc_auc_score(yte, y_prob),
                # "precision": precision_score(yte, y_pred),
                # "recall": recall_score(yte, y_pred),
                # "f1_score": f1_score(yte, y_pred)
            }

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.log_param("top_k_features", self.knn_k)
            mlflow.log_param("n_neighbors", self.knn_neighbors)

            mlflow.sklearn.log_model(model, "knn")
            joblib.dump(model, os.path.join(self.models_dir, "knn.pkl"))

            wandb.log(metrics)
            wandb.finish()

        logger.info(f"KNN metrics: {metrics}")

    # ==================================================
    # SVM
    # ==================================================
    def train_svm(self):
        logger.info(f"Training SVM | Top-{self.svm_k} features")

        Xtr, Xte, ytr, yte, idx = self._load_data()

        Xtr = Xtr[:, idx][:, :self.svm_k]
        Xte = Xte[:, idx][:, :self.svm_k]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        with mlflow.start_run(run_name="SVM"):
            wandb.init(project="ASD-Diagnoser", name="SVM")

            model = SVC(
                kernel=self.svm_kernel,
                probability=True,
                random_state=42
            )

            model.fit(Xtr, ytr)

            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:, 1]

            metrics = {
                "accuracy": accuracy_score(yte, y_pred),
                "roc_auc": roc_auc_score(yte, y_prob),
                # "precision": precision_score(yte, y_pred),
                # "recall": recall_score(yte, y_pred),
                # "f1_score": f1_score(yte, y_pred)
            }

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.log_param("top_k_features", self.svm_k)
            mlflow.log_param("kernel", self.svm_kernel)

            mlflow.sklearn.log_model(model, "svm")
            joblib.dump(model, os.path.join(self.models_dir, "svm.pkl"))
            joblib.dump(scaler, os.path.join(self.models_dir, "svm_scaler.pkl"))

            wandb.log(metrics)
            wandb.finish()

        logger.info(f"SVM metrics: {metrics}")
