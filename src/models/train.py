import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

from src.models.pipeline import create_pipeline

RANDOM_STATE = 42


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(Path(data_dir) / "train.csv")
    test = pd.read_csv(Path(data_dir) / "test.csv")
    X_train, y_train = train.drop(columns=["default"]), train["default"]
    X_test, y_test = test.drop(columns=["default"]), test["default"]
    return X_train, y_train, X_test, y_test


def get_param_distributions(model_type: str) -> dict:
    if model_type == "logreg":
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs", "liblinear"],
        }
    elif model_type == "rf":
        return {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [5, 10, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    elif model_type == "gbc":
        return {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [3, 5, 7],
            "classifier__subsample": [0.8, 1.0],
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_and_evaluate(
    model_type: str,
    data_dir: str = "data/processed",
    reports_dir: str = "reports",
    models_dir: str = "models",
    n_iter: int = 20,
) -> dict:
    X_train, y_train, X_test, y_test = load_data(data_dir)
    pipeline = create_pipeline(model_type)
    param_dist = get_param_distributions(model_type)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_type": model_type,
        "best_params": search.best_params_,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    models = Path(models_dir)
    models.mkdir(parents=True, exist_ok=True)

    with open(reports / f"metrics_{model_type}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"ROC-кривая - {model_type} (AUC = {metrics['roc_auc']})")
    plt.tight_layout()
    plt.savefig(reports / f"roc_curve_{model_type}.png", dpi=150)
    plt.close()

    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"Confusion Matrix - {model_type}")
    plt.tight_layout()
    plt.savefig(reports / f"confusion_matrix_{model_type}.png", dpi=150)
    plt.close()

    joblib.dump(best_model, models / f"model_{model_type}.pkl")

    print(f"\n[{model_type}] ROC-AUC: {metrics['roc_auc']:.4f}")
    print(classification_report(y_test, y_pred, target_names=["no default", "default"]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["logreg", "rf", "gbc"], required=True)
    args = parser.parse_args()
    train_and_evaluate(args.model_type)


if __name__ == "__main__":
    main()
