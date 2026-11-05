import pandas as pd

from src.data.make_dataset import drop_id, fix_categorical_anomalies, load_raw_data, split_data


def test_fix_categorical_anomalies(
    sample_raw_df,
):  # проверка того, что EDUCATION {0,5,6} мапится в 4, MARRIAGE 0 в 3
    data = fix_categorical_anomalies(sample_raw_df)
    assert set(data["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(data["MARRIAGE"].unique()).issubset({1, 2, 3})


def test_drop_id(sample_raw_df):
    data = drop_id(sample_raw_df)
    assert "ID" not in data.columns


def test_split_data_stratify(sample_raw_df):  # проверка того, что сплит сохраняет пропорции классов
    data = sample_raw_df.rename(columns={"default.payment.next.month": "default"})
    X_train, X_test, y_train, y_test = split_data(data)
    assert len(y_test) == 2
    assert y_test.sum() == 1, "В тесте должен быть ровно 1 дефолт"
    assert (y_test == 0).sum() == 1, "В тесте должен быть ровно 1 недефолт"
