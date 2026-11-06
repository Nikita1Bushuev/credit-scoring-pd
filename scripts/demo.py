import joblib
import pandas as pd


def run_demo():
    data = pd.read_csv("data/processed/test.csv").head(1)
    y_true = data["default"].values[0]
    X_raw = data.drop(columns=["default"])

    model = joblib.load("models/model_gbc_th0.3.pkl")
    proba = model.predict_proba(X_raw)[0, 1]
    prediction = int(proba >= 0.3)

    print(f"Истинное значение: {y_true}")
    print(f"Предсказанная вероятность: {proba}")
    print(f"Предсказанный класс (порог 0.3): {prediction}")

    if prediction == y_true:
        print("Модель дала верный ответ")
    else:
        print("Модель ошиблась (это нормально, accuracy не 100%)")


if __name__ == "__main__":
    run_demo()
