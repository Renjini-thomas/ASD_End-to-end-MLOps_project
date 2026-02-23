import os
import yaml
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import wandb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix
)

from src.utils.logger import logger
from src.utils.path import FEATURES_DIR, ARTIFACTS_DIR


class ModelTrainer:

    def __init__(self, params_path="params.yaml"):

        # -----------------------
        # Load Params
        # -----------------------
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)

        # -----------------------
        # MLflow Setup
        # -----------------------
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(
            "ASD_Diagnoser_Model_Training"
        )

        # -----------------------
        # Model save directory
        # -----------------------
        self.models_dir = os.path.join(
            ARTIFACTS_DIR,
            "models"
        )

        os.makedirs(
            self.models_dir,
            exist_ok=True
        )

    # ======================================================
    # LOAD DATA
    # ======================================================
    def _load_data(self):

        X_train = np.load(
            os.path.join(FEATURES_DIR, "X_train.npy")
        )

        y_train = np.load(
            os.path.join(FEATURES_DIR, "y_train.npy")
        )

        X_test = np.load(
            os.path.join(FEATURES_DIR, "X_test.npy")
        )

        y_test = np.load(
            os.path.join(FEATURES_DIR, "y_test.npy")
        )

        logger.info(
            f"Loaded Features | "
            f"Train={X_train.shape} "
            f"Test={X_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    # ======================================================
    # METRICS FUNCTION (COLAB STYLE)
    # ======================================================
    def _evaluate_metrics(
        self,
        y_true,
        y_pred,
        y_prob
    ):

        acc = accuracy_score(
            y_true,
            y_pred
        )

        auc = roc_auc_score(
            y_true,
            y_prob
        )

        f1 = f1_score(
            y_true,
            y_pred
        )

        sensitivity = recall_score(
            y_true,
            y_pred
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred
        ).ravel()

        specificity = tn / (tn + fp + 1e-9)

        return {

            "accuracy": acc,
            "roc_auc": auc,
            "f1_score": f1,
            "sensitivity": sensitivity,
            "specificity": specificity
        }

    # ======================================================
    # TRAIN ALL MODELS
    # ======================================================
    def train_all_models(self):

        logger.info(
            "PIPELINE STEP 3 : MODEL TRAINING STARTED"
        )

        Xtr, Xte, ytr, yte = self._load_data()

        # ===============================
        # EXACT COLAB MODELS
        # ===============================
        models = {

            "KNN3":
            KNeighborsClassifier(3),

            "KNN5":
            KNeighborsClassifier(5),

            "KNN7":
            KNeighborsClassifier(7),

            "KNN9":
            KNeighborsClassifier(9),

            "SVM_linear":
            SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=42
            ),

            "SVM_rbf":
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            ),

            "SVM_poly":
            SVC(
                kernel="poly",
                degree=3,
                probability=True,
                random_state=42
            ),

            "Tree":
            DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=10,
                min_samples_leaf=5,
                random_state=42
            )
        }

        # ===============================
        # TRAIN LOOP
        # ===============================
        for name, model in models.items():

            logger.info(
                f"Training Model : {name}"
            )

            with mlflow.start_run(
                run_name=name
            ):

                # WandB tracking
                wandb.init(
                    project="ASD-Diagnoser",
                    name=name,
                    reinit=True
                )

                # -------------------
                # TRAIN
                # -------------------
                model.fit(
                    Xtr,
                    ytr
                )

                # -------------------
                # TEST PREDICTION
                # -------------------
                y_pred = model.predict(Xte)

                y_prob = model.predict_proba(
                    Xte
                )[:, 1]

                metrics = self._evaluate_metrics(

                    yte,
                    y_pred,
                    y_prob
                )

                # -------------------
                # MLflow Logging
                # -------------------
                for k, v in metrics.items():

                    mlflow.log_metric(
                        k,
                        float(v)
                    )

                mlflow.log_param(
                    "model_name",
                    name
                )

                mlflow.log_param(
                    "dataset",
                    self.params["dataset"]["name"]
                )

                mlflow.log_param(
                    "augmentation",
                    "ddpm_augmented"
                )

                # -------------------
                # SAVE MODEL
                # -------------------
                model_path = os.path.join(

                    self.models_dir,
                    f"{name}.pkl"
                )

                joblib.dump(
                    model,
                    model_path
                )

                # MLflow model log
                mlflow.sklearn.log_model(
                    model,
                    artifact_path=name
                )

                # WandB log
                wandb.log(metrics)

                wandb.finish()

                logger.info(
                    f"{name} Training Completed "
                    f"| Metrics : {metrics}"
                )

        logger.info(
            "ALL MODELS TRAINED SUCCESSFULLY"
        )