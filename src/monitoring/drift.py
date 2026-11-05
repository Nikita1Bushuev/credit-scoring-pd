import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(expected, actual, buckets=10):
    def scale_range(input_data, min_val, max_val):
        input_data += -(np.min(input_data))
        input_data /= np.max(input_data) / (max_val - min_val)
        input_data += min_val
        return input_data

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    psi_value = np.sum(
        (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    )
    return psi_value


def check_drift(train_path: str, new_data_path: str):
    train_df = pd.read_csv(train_path)
    new_df = pd.read_csv(new_data_path)
    report = {}
    continuous_cols = ["LIMIT_BAL", "AGE", "total_bill", "pay_ratio"]
    for col in continuous_cols:
        if col in train_df.columns and col in new_df.columns:
            ks_stat, p_value = stats.ks_2samp(train_df[col], new_df[col])
            psi = calculate_psi(train_df[col].values, new_df[col].values)

            report[col] = {
                "type": "continuous",
                "ks_pvalue": round(float(p_value), 4),
                "psi": round(float(psi), 4),
                "drift_detected": bool(psi > 0.25 or p_value < 0.05),
            }

    cat_cols = ["EDUCATION", "MARRIAGE"]
    for col in cat_cols:
        if col in train_df.columns and col in new_df.columns:
            train_counts = train_df[col].value_counts()
            new_counts = new_df[col].value_counts()
            all_cats = set(train_counts.index).union(set(new_counts.index))
            train_counts = train_counts.reindex(all_cats, fill_value=0)
            new_counts = new_counts.reindex(all_cats, fill_value=0)

            chi2, p_value, _, _ = stats.chi2_contingency([train_counts, new_counts])

            report[col] = {
                "type": "categorical",
                "chi2_pvalue": round(float(p_value), 4),
                "drift_detected": bool(p_value < 0.05),
            }

    out_path = Path("reports/drift_report.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Drift] Отчёт сохранен в {out_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    check_drift("data/processed/train.csv", "data/processed/test.csv")
