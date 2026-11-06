# Credit Scoring - PD

End-to-end автоматизированный ML-пайплайн для предсказания вероятности дефолта клиента (Probability of Default, PD).

## Описание проекта
Реализован полный жизненный цикл ML-модели по методологии CRISP-DM: от бизнес-анализа до мониторинга дрифта в продакшене.

**Датасет:** [Default of Credit Card Clients Dataset (UCI)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?spm=a2ty_o01.29997173.0.0.532fc921UIOjfa).

## Бизнес-задача
Кредитный отдел банка хочет автоматически оценивать вероятность дефолта клиента в следующем месяце.

## Методология
1. **Анализ данных:** `notebooks/01_eda.ipynb` (исследование распределений, корреляций).
2. **Подготовка данных:** `src/data/make_dataset.py` (очистка, сплит), `src/features/build_features.py` (Feature Engineering: агрегаты по платежам).
3. **Моделирование:** `src/models/train.py` (Sklearn Pipeline: Imputer, Scaler, OHE, GBC/LogReg/RF).
4. **Оценка:** `RandomizedSearchCV`, логирование 5 экспериментов в MLflow.
5. **Внедрение:** FastAPI (`src/api/app.py`) + Docker, мониторинг дрифта (`src/monitoring/drift.py`).

## Стек технологий
*   **Данные:** Pandas, Numpy, Scikit-learn.
*   **Трекинг:** MLflow (Tracking, Models, Registry).
*   **Валидация:** Great Expectations.
*   **Версионирование:** Git, DVC (Data Version Control).
*   **Тестирование:** Pytest, GitHub Actions (CI).
*   **API:** FastAPI, Uvicorn, Pydantic.
*   **Деплой:** Docker.
*   **Мониторинг:** PSI, KS-test, Chi-square.

## Как использовать
```bash
# 1. Установка зависимостей
pip install -r requirements.txt
# 2. Воспроизведение пайплайна (DVC)
dvc repro

# 3. Запуск API
uvicorn src.api.app:app --reload
# Swagger UI доступен по адресу http://127.0.0.1:8000/docs

# 4. Проверка дрифта
python -m src.monitoring.drift

# 5. Демо-скрипт (проверка модели)
python scripts/demo.py
```
## Тестирование и CI
При каждом пуше в GitHub Actions запускаются:
- **Линтинг** (flake8, black).
- **Unit-тесты** (pytest).
- **Валидация данных** (Great Expectations).

## Результаты экспериментов (MLflow)

![alt text](<Screenshot 2026-11-03 040853.png>)

**Вывод**: Модель Gradient Boosting с порогом 0.3 выбрана как оптимальная (баланс Precision/Recall для банка).

## Мониторинг дрифта

Скрипт `src/monitoring/drift.py` рассчитывает метрики PSI, KS-test и Chi-square.
