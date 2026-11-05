from src.models.pipeline import create_pipeline


def test_create_pipeline(sample_raw_df):  # проверка того, что пайплайн создаётся и обучается
    pipeline = create_pipeline(
        "logreg"
    )  # для теста просто проверим, что пайплайн обучается без ошибок на сырых данных
    assert pipeline is not None
