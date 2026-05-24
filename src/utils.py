import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        print("\nModel Performance Comparison:\n")

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            # 🔴 BEFORE TUNING
            model.fit(X_train, y_train)
            y_pred_before = model.predict(X_test)
            before_score = r2_score(y_test, y_pred_before)

            # 🟢 GRID SEARCH
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # 🔴 AFTER TUNING
            best_model = gs.best_estimator_
            y_pred_after = best_model.predict(X_test)
            after_score = r2_score(y_test, y_pred_after)

            improvement = after_score - before_score

            # Store final score
            report[model_name] = {
                "before_score": before_score,
                "after_score": after_score,
                "improvement": improvement,
                "best_params": gs.best_params_,
                "model": gs.best_estimator_
            }

            # 🖨️ PRINT EVERYTHING
            print(f"{model_name}")
            print(f"Before Tuning R2: {before_score:.4f}")
            print(f"After Tuning R2:  {after_score:.4f}")
            print(f"Improvement:      {improvement:.4f}")
            print(f"Best Params:      {gs.best_params_}")
            print("-" * 50)

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)