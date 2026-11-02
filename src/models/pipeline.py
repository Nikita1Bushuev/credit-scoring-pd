from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "max_delay",
    "late_count",
    "avg_delay",
    "total_bill",
    "avg_bill",
    "bill_trend",
    "total_pay",
    "avg_pay",
    "pay_ratio",
]

CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


def create_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-999)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def create_pipeline(model_type: str = "logreg") -> Pipeline:
    preprocessor = create_preprocessor()

    if model_type == "logreg":
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    elif model_type == "rf":
        classifier = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_type == "gbc":
        classifier = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
