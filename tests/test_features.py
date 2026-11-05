from src.features.build_features import build_features


def test_build_features_adds_columns(sample_raw_df):  # проверка FE
    data = sample_raw_df.rename(columns={"default.payment.next.month": "default"})
    data_fe = build_features(data)
    expected_cols = [
        "max_delay",
        "late_count",
        "avg_delay",
        "total_bill",
        "avg_bill",
        "bill_trend",
        "total_pay",
        "avg_pay",
        "pay_ratio",
        "age_bin",
    ]
    for col in expected_cols:
        assert col in data_fe.columns, f"Отсутствует колонка {col}"


def test_late_count_logic(sample_raw_df):  # проверка логики late_count
    data = sample_raw_df.rename(columns={"default.payment.next.month": "default"})
    data_fe = build_features(data)
    # для первой строки: PAY_0=2, PAY_2=2, остальные < 1 => late_count = 2
    assert data_fe.loc[0, "late_count"] == 2
