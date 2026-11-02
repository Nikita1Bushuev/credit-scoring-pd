# Feature Engineering
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def add_delay_features(data: pd.DataFrame) -> pd.DataFrame:  # агрегация по статусам платежей
    data = data.copy()
    pay = data[PAY_COLS]
    data["max_delay"] = pay.max(axis=1)  # максимальная задержка
    data["late_count"] = (pay >= 1).sum(
        axis=1
    )  # сколько месяцев из 6 платёж был просрочен хотя бы на 1 месяц
    data["avg_delay"] = pay.clip(lower=0).mean(axis=1)  # средняя задержка по просрочкам
    return data


def add_bill_features(data: pd.DataFrame) -> pd.DataFrame:  # агрегация по суммам счетов
    data = data.copy()
    bills = data[BILL_COLS]
    data["total_bill"] = bills.sum(axis=1)
    data["avg_bill"] = bills.mean(axis=1)
    data["bill_trend"] = data["BILL_AMT1"] - data["BILL_AMT6"]
    return data


def add_payment_features(
    data: pd.DataFrame,
) -> pd.DataFrame:  # агрегация по суммам платежей  их отношение к долгу
    data = data.copy()
    pay_amt = data[PAY_AMT_COLS]
    data["total_pay"] = pay_amt.sum(axis=1)
    data["avg_pay"] = pay_amt.mean(axis=1)
    data["pay_ratio"] = data["total_pay"] / (data["total_bill"] + 1)
    return data


def add_age_features(data: pd.DataFrame) -> pd.DataFrame:  # биннинг возраста
    data = data.copy()
    data["age_bin"] = pd.cut(
        data["AGE"], bins=[0, 30, 45, 100], labels=["young", "middle", "senior"]
    ).astype(str)
    return data


def build_features(data: pd.DataFrame) -> pd.DataFrame:  # пайплайн Feature Engineering
    data = add_delay_features(data)
    data = add_bill_features(data)
    data = add_payment_features(data)
    data = add_age_features(data)
    return data


def main(input_dir: str, output_dir: str) -> None:
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        data = pd.read_csv(inp / f"{split}.csv")
        data_fe = build_features(data)
        data_fe.to_csv(out / f"{split}.csv", index=False)
        print(f"[features] {split}: {data.shape} -> {data_fe.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features for UCI dataset.")
    parser.add_argument("input_dir", type=str, help="Input directory with train/test")
    parser.add_argument("output_dir", type=str, help="Output directory")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
