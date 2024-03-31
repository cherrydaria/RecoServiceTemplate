from typing import Any  # Импорт необходимых типов

import pandas as pd  # Импорт библиотеки pandas
from rectools.dataset import Dataset, Interactions  # Импорт необходимых классов из rectools
from rectools.models.base import ModelBase  # Импорт базового класса модели


def metrics_full_visualization(report: dict[str, Any]) -> pd.DataFrame:
    """
    Визуализация метрик.
    
    Args:
        report (dict[str, Any]): Словарь с метриками.

    Returns:
        pd.DataFrame: DataFrame с визуализированными метриками.
    """
    # Создание DataFrame из отчета и удаление столбца 'fold'
    df = pd.DataFrame(report).drop(columns="fold").groupby(["model"], sort=False).agg(["mean"])
    df.columns = df.columns.droplevel(level=1)  # Удаление мультииндекса
    df_report = df.drop(columns=["train time (sec)"])  # Удаление столбца с временем обучения

    # Переименование столбцов с использованием мультииндекса
    df_report.columns = pd.MultiIndex.from_tuples(
        list(df_report.columns.str.extract(r"(top@\d+)_(\w+)", expand=True).itertuples(index=False))
    )
    df_report.columns = df_report.sort_index(axis=1, level=[0, 1], ascending=[True, True]).columns

    # Добавление столбца с временем обучения и форматирование стилей DataFrame
    df_report["train time (sec)"] = df["train time (sec)"]
    mean_metric_subset = list(df_report.columns)
    df_report = (
        df_report.style.set_table_styles(
            [
                {"selector": "thead th", "props": [("text-align", "center")]},  # Стиль заголовков таблицы
            ]
        )
        .highlight_min(subset=mean_metric_subset, color="darkred", axis=0)  # Выделение минимальных значений
        .highlight_max(subset=mean_metric_subset, color="green", axis=0)  # Выделение максимальных значений
    )
    return df_report


def training_result_full_visualization(
    model: ModelBase,
    dataset: Dataset,
    user_ids: list[int],
    item_data: dict[str, str],
    k_recos: int,
    interactions: Interactions,
    history_size_per_user: int = 10,
) -> pd.DataFrame:
    """
    Визуализация результатов обучения.

    Args:
        model (ModelBase): Объект модели.
        dataset (Dataset): Датасет.
        user_ids (list[int]): Список идентификаторов пользователей.
        item_data (dict[str, str]): Данные об объектах.
        k_recos (int): Количество рекомендаций.
        interactions (Interactions): Взаимодействия пользователей с объектами.
        history_size_per_user (int, optional): Количество предыдущих взаимодействий для учета истории. По умолчанию 10.

    Returns:
        pd.DataFrame: DataFrame с визуализированными результатами обучения.
    """
    # Получение DataFrame взаимодействий и рекомендаций модели
    df = dataset.interactions.df
    recos = model.recommend(users=user_ids, k=k_recos, dataset=dataset, filter_viewed=True)
    recos["type"] = "reco"
    recos.drop("score", axis=1, inplace=True)

    # Получение истории взаимодействий пользователей
    history = (
        df[df["user_id"].isin(user_ids)]
        .sort_values(["user_id", "datetime"], ascending=[True, False])
        .groupby("user_id")
        .head(history_size_per_user)
    )
    history["rank"] = history.sort_values("datetime").groupby(["user_id"]).datetime.rank().astype("int")
    history["type"] = "history"
    history.drop(["datetime", "weight"], axis=1, inplace=True)

    # Объединение рекомендаций и истории взаимодействий
    report = pd.concat([recos, history])
    count_views = interactions.df.groupby("item_id").count()["user_id"]
    report = report.merge(item_data, how="inner", on="item_id")
    count_views.name = "count_views"
    report = report.merge(count_views, how="inner", on="item_id")

    # Сортировка и форматирование стилей DataFrame
    report.sort_values(["user_id", "type"], inplace=True)
    report.set_index(["user_id", "type"], inplace=True)
    report = report.style.set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    return report