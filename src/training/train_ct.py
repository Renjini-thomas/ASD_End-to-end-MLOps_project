import os
import cv2
import yaml
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.feature_extraction.glcm import extract_glcm_features
from src.feature_extraction.lbp import extract_lbp_features
from src.feature_extraction.gfcc import extract_gfcc_features
from src.feature_selection.rf_ranking import rank_features

params = yaml.safe_load(open("configs/params.yaml"))
DATA_PATH = "data/raw/abide1_data(sagittal_2)"
BEST_FEATURES = params["features"]["ct_best"]

def load_data():
    X, y = [], []
    classes = sorted(os.listdir(DATA_PATH))
    label_map = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        for file in os.listdir(os.path.join(DATA_PATH, cls)):
            img = cv2.imread(os.path.join(DATA_PATH, cls, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(
                extract_glcm_features(img) +
                extract_lbp_features(img) +
                extract_gfcc_features(img)
            )
            y.append(label_map[cls])
    return np.array(X), np.array(y)

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["dataset"]["split_ratio"],
    stratify=y,
    random_state=params["dataset"]["random_state"]
)

sorted_idx = rank_features(X_train, y_train)
X_train = X_train[:, sorted_idx][:, :BEST_FEATURES]
X_test = X_test[:, sorted_idx][:, :BEST_FEATURES]

mlflow.set_experiment("ASD_Diagnosis_ABIDE")

with mlflow.start_run(run_name="DecisionTree_CT"):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    mlflow.log_param("model", "DecisionTree")
    mlflow.log_param("top_k_features", BEST_FEATURES)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc)

    mlflow.sklearn.log_model(model, "ct_model")

    print("CT Accuracy:", acc)
    print("CT ROC-AUC :", roc)
