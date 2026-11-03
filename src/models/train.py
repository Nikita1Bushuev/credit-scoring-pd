import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
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

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "credit_pd"


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
    raise ValueError(f"Unknown model_type: {model_type}")


def train_and_evaluate(
    model_type: str,
    threshold: float = 0.5,
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
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    models = Path(models_dir)
    models.mkdir(parents=True, exist_ok=True)

    suffix = f"{model_type}_th{threshold}"

    roc_path = reports / f"roc_curve_{suffix}.png"
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"ROC - {model_type} (threshold={threshold}, AUC={metrics['roc_auc']:.3f})")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    cm_path = reports / f"confusion_matrix_{suffix}.png"
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"Confusion Matrix - {model_type} (threshold={threshold})")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    model_path = models / f"model_{suffix}.pkl"
    joblib.dump(best_model, model_path)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{model_type}_th{threshold}"):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", RANDOM_STATE)
        for k, v in search.best_params_.items():
            mlflow.log_param(k.replace("classifier__", ""), v)

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(str(roc_path), artifact_path="plots")
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=f"credit_pd_{model_type}",
        )

    print(f"\n[{model_type} th={threshold}] ROC-AUC: {metrics['roc_auc']:.4f}")
    print(classification_report(y_test, y_pred, target_names=["no default", "default"]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["logreg", "rf", "gbc"], required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    args = parser.parse_args()
    train_and_evaluate(args.model_type, threshold=args.threshold)


if __name__ == "__main__":
    main()
