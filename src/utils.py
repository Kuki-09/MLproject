import os
import sys
import pickle
import numpy as np

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    models,
    param,
    cv=5
):
    try:

        report = {}

        print("\nModel Performance Comparison:\n")

        for model_name, model in models.items():

            print(f"{model_name}")

            # ------------------------------------------------
            # 1. BASELINE CV SCORE
            # ------------------------------------------------

            baseline_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="r2"
            )

            baseline_cv_score = baseline_scores.mean()

            print(
                f"Baseline CV R2: {baseline_cv_score:.4f}"
            )

            # ------------------------------------------------
            # 2. HYPERPARAMETER TUNING
            # ------------------------------------------------

            para = param[model_name]

            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=cv,
                scoring="r2",
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            # ------------------------------------------------
            # 3. BEST CV SCORE AFTER TUNING
            # ------------------------------------------------

            best_cv_score = gs.best_score_

            best_model = gs.best_estimator_

            improvement = best_cv_score - baseline_cv_score

            print(
                f"Best CV R2: {best_cv_score:.4f}"
            )

            print(
                f"Improvement: {improvement:.4f}"
            )

            print(
                f"Best Params: {gs.best_params_}"
            )

            print("-" * 50)

            report[model_name] = {
                "baseline_cv_score": baseline_cv_score,
                "best_cv_score": best_cv_score,
                "improvement": improvement,
                "best_params": gs.best_params_,
                "model": best_model
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)