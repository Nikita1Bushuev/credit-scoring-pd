# Загрузка и первичная очистка датасета
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "default"
RENAME_COLS = {"default.payment.next.month": TARGET_COL}
EDUCATION_MAP = {0: 4, 5: 4, 6: 4}
MARRIAGE_MAP = {0: 3}
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.rename(columns=RENAME_COLS)
    return data


def fix_categorical_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["EDUCATION"] = data["EDUCATION"].replace(EDUCATION_MAP)
    data["MARRIAGE"] = data["MARRIAGE"].replace(MARRIAGE_MAP)
    return data


def drop_id(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(columns=["ID"], errors="ignore")


def split_data(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def main(raw_path: str, output_dir: str) -> None:
    print(f"[prepare] Reading: {raw_path}")
    data = load_raw_data(raw_path)
    print(f"[prepare] Loaded: {data.shape}")

    data = fix_categorical_anomalies(data)
    data = drop_id(data)
    print(f"[prepare] After cleaning: {data.shape}")

    X_train, X_test, y_train, y_test = split_data(data)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    print(f"[prepare] Saved: train={train.shape}, test={test.shape}")
    print(f"[prepare] Target balance in train: {y_train.mean()}")
    print(f"[prepare] Target balance in test:  {y_test.mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare UCI Credit Card data.")
    parser.add_argument("raw_path", type=str, help="Path to raw CSV")
    parser.add_argument("output_dir", type=str, help="Output directory")
    args = parser.parse_args()
    main(args.raw_path, args.output_dir)
