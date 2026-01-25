import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import wandb

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.logger import logger
from src.utils.path import FEATURES_DIR, ARTIFACTS_DIR


class ModelTrainer:
    def __init__(self):
        # MLflow configuration (local)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("ASD_Diagnoser_Model_Training")

        # Artifacts directory
        self.models_dir = os.path.join(ARTIFACTS_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def _load_data(self):
        """
        Loads features, labels, and RF-ranked feature indices.
        Uses the same split as the final implementation.
        """
        X = np.load(os.path.join(FEATURES_DIR, "X.npy"))
        y = np.load(os.path.join(FEATURES_DIR, "y.npy"))
        sorted_idx = np.load(
            os.path.join(FEATURES_DIR, "sorted_feature_indices.npy")
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=42
        )

        return X_train, X_test, y_train, y_test, sorted_idx

    # ------------------------------------------------------------------
    # Decision Tree (Final implementation)
    # ------------------------------------------------------------------
    def train_decision_tree(self, top_k=30):
        logger.info(f"Training Decision Tree with top-{top_k} features")

        Xtr, Xte, ytr, yte, idx = self._load_data()

        Xtr = Xtr[:, idx][:, :top_k]
        Xte = Xte[:, idx][:, :top_k]

        with mlflow.start_run(run_name=f"DT_top{top_k}"):
            wandb.init(
                project="ASD-Diagnoser",
                name=f"DT_top{top_k}",
                reinit=True
            )

            model = DecisionTreeClassifier(random_state=42)
            model.fit(Xtr, ytr)

            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:, 1]

            acc = accuracy_score(yte, y_pred)
            roc = roc_auc_score(yte, y_prob)

            # Logging
            mlflow.log_param("model", "DecisionTree")
            mlflow.log_param("top_k_features", top_k)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", roc)
            mlflow.sklearn.log_model(model, "decision_tree_model")

            wandb.log({
                "accuracy": acc,
                "roc_auc": roc
            })

            joblib.dump(
                model,
                os.path.join(self.models_dir, "decision_tree_model.pkl")
            )

            wandb.finish()

        logger.info(
            f"Decision Tree completed | Accuracy={acc:.4f}, ROC-AUC={roc:.4f}"
        )

    # ------------------------------------------------------------------
    # KNN (Final implementation)
    # ------------------------------------------------------------------
    def train_knn(self, top_k=10, n_neighbors=3):
        logger.info(
            f"Training KNN with top-{top_k} features, k={n_neighbors}"
        )

        Xtr, Xte, ytr, yte, idx = self._load_data()

        Xtr = Xtr[:, idx][:, :top_k]
        Xte = Xte[:, idx][:, :top_k]

        with mlflow.start_run(run_name=f"KNN_top{top_k}_k{n_neighbors}"):
            wandb.init(
                project="ASD-Diagnoser",
                name=f"KNN_top{top_k}",
                reinit=True
            )

            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(Xtr, ytr)

            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:, 1]

            acc = accuracy_score(yte, y_pred)
            roc = roc_auc_score(yte, y_prob)

            # Logging
            mlflow.log_param("model", "KNN")
            mlflow.log_param("top_k_features", top_k)
            mlflow.log_param("n_neighbors", n_neighbors)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", roc)
            mlflow.sklearn.log_model(model, "knn_model")

            wandb.log({
                "accuracy": acc,
                "roc_auc": roc
            })

            joblib.dump(
                model,
                os.path.join(self.models_dir, "knn_model.pkl")
            )

            wandb.finish()

        logger.info(
            f"KNN completed | Accuracy={acc:.4f}, ROC-AUC={roc:.4f}"
        )

    # ------------------------------------------------------------------
    # SVM (Final implementation – Linear kernel)
    # ------------------------------------------------------------------
    def train_svm(self, top_k=50):
        logger.info(f"Training SVM (linear) with top-{top_k} features")

        Xtr, Xte, ytr, yte, idx = self._load_data()

        Xtr = Xtr[:, idx][:, :top_k]
        Xte = Xte[:, idx][:, :top_k]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        with mlflow.start_run(run_name=f"SVM_top{top_k}_linear"):
            wandb.init(
                project="ASD-Diagnoser",
                name=f"SVM_top{top_k}",
                reinit=True
            )

            model = SVC(
                kernel="linear",
                probability=True,
                random_state=42
            )
            model.fit(Xtr, ytr)

            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:, 1]

            acc = accuracy_score(yte, y_pred)
            roc = roc_auc_score(yte, y_prob)

            # Logging
            mlflow.log_param("model", "SVM")
            mlflow.log_param("kernel", "linear")
            mlflow.log_param("top_k_features", top_k)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", roc)
            mlflow.sklearn.log_model(model, "svm_model")

            wandb.log({
                "accuracy": acc,
                "roc_auc": roc
            })

            joblib.dump(
                model,
                os.path.join(self.models_dir, "svm_model.pkl")
            )
            joblib.dump(
                scaler,
                os.path.join(self.models_dir, "svm_scaler.pkl")
            )

            wandb.finish()

        logger.info(
            f"SVM completed | Accuracy={acc:.4f}, ROC-AUC={roc:.4f}"
        )
