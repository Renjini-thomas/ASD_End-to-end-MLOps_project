import numpy as np
from sklearn.model_selection import train_test_split

from src.feature_extraction.extract_features import run_feature_extraction
from src.feature_selection.rf_feature_selection import RFFeatureSelector
from src.utils.logger import logger
from src.training.train_models import ModelTrainer
from src.evaluation.evaluate import ModelEvaluator

if __name__ == "__main__":
    try:
        logger.info("PIPELINE STARTED – FEATURE EXTRACTION + SELECTION")

        # Step 1: Feature extraction
        run_feature_extraction()
        

        # Load features
        X = np.load("artifacts/features/X.npy")
        y = np.load("artifacts/features/y.npy")

        # Same split as final implementation
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=42
        )

        # Step 2: Feature selection
        selector = RFFeatureSelector()
        sorted_idx = selector.rank_features(X_train, y_train)

        np.save(
            "artifacts/features/sorted_feature_indices.npy",
            sorted_idx
        )

        logger.info("FEATURE EXTRACTION + SELECTION COMPLETED SUCCESSFULLY")
        # -------------------------------
        # STEP 2: Model Training
        # -------------------------------
        logger.info("PIPELINE STARTED – STEP 2: MODEL TRAINING")

        trainer = ModelTrainer()

        # IMPORTANT: run ONE model first
        trainer.train_decision_tree(top_k=30)

        # Uncomment only after DT works
        trainer.train_knn(top_k=10, n_neighbors=3)
        trainer.train_svm(top_k=50)

        logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
        # -------------------------------
        # STEP 3: Model Evaluation      
        # -------------------------------
        logger.info("PIPELINE STARTED – STEP 3: MODEL EVALUATION")
        evaluator = ModelEvaluator()
        best_model, results = evaluator.evaluate_models()

        best_model_name = best_model[0]
        best_accuracy = best_model[1]
        best_roc_auc = best_model[2]

        evaluator.register_best_model(
    best_model_name,
    best_accuracy,
    best_roc_auc
)
        logger.info("MODEL EVALUATION COMPLETED SUCCESSFULLY")

        

    except Exception as e:
        logger.exception(e)
        raise e
