import numpy as np
from sklearn.ensemble import RandomForestClassifier

def rank_features(X, y, n_estimators=300, random_state=42):
    """
    Rank features using Random Forest feature importance.

    Parameters:
    - X: Feature matrix (samples x features)
    - y: Labels
    - n_estimators: Number of trees in RF
    - random_state: Seed for reproducibility

    Returns:
    - sorted_idx: Indices of features sorted by importance (descending)
    """

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        random_state=random_state,
        n_jobs=-1
    )

    rf.fit(X, y)

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    return sorted_idx
