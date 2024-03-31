from typing import Any

import pandas as pd
from rectools.dataset import Dataset, Interactions
from rectools.models.base import ModelBase


def full_metrics_visualization(report: dict[str, Any]) -> pd.DataFrame:
    """
    Визуализирует полные метрики из отчета.

    Args:
        report (dict[str, Any]): Отчет с метриками.

    Returns:
        pd.DataFrame: DataFrame с визуализацией метрик.
    """
    df = pd.DataFrame(report).drop(columns="fold").groupby(["model"], sort=False).agg(["mean"])
    df.columns = df.columns.droplevel(level=1)
    df_report = df.drop(columns=["train time (sec)"])

    df_report.columns = pd.MultiIndex.from_tuples(
        list(df_report.columns.str.extract(r"(top@\d+)_(\w+)", expand=True).itertuples(index=False))
    )
    df_report.columns = df_report.sort_index(axis=1, level=[0, 1], ascending=[True, True]).columns
    df_report["train time (sec)"] = df["train time (sec)"]
    mean_metric_subset = list(df_report.columns)
    df_report = (
        df_report.style.set_table_styles(
            [
                {"selector": "thead th", "props": [("text-align", "center")]},
            ]
        )
        .highlight_min(subset=mean_metric_subset, color="darkred", axis=0)
        .highlight_max(subset=mean_metric_subset, color="green", axis=0)
    )
    return df_report


def full_training_result_visualization(
    model: ModelBase,
    dataset: Dataset,
    user_ids: list[int],
    item_data: dict[str, str],
    k_recos: int,
    interactions: Interactions,
    history_size_per_user: int = 10,
) -> pd.DataFrame:
    """
    Визуализирует полный результат обучения модели.

    Args:
        model (ModelBase): Обученная модель.
        dataset (Dataset): Набор данных.
        user_ids (list[int]): Идентификаторы пользователей для визуализации.
        item_data (dict[str, str]): Данные о элементах.
        k_recos (int): Количество рекомендаций.
        interactions (Interactions): Взаимодействия пользователей с элементами.
        history_size_per_user (int, optional): Количество предыдущих взаимодействий для каждого пользователя. По умолчанию 10.

    Returns:
        pd.DataFrame: DataFrame с визуализацией результатов обучения модели.
    """
    df = dataset.interactions.df
    recos = model.recommend(users=user_ids, k=k_recos, dataset=dataset, filter_viewed=True)
    recos["type"] = "reco"
    recos.drop("score", axis=1, inplace=True)
    history = (
        df[df["user_id"].isin(user_ids)]
        .sort_values(["user_id", "datetime"], ascending=[True, False])
        .groupby("user_id")
        .head(history_size_per_user)
    )
    history["rank"] = history.sort_values("datetime").groupby(["user_id"]).datetime.rank().astype("int")
    history["type"] = "history"
    history.drop(["datetime", "weight"], axis=1, inplace=True)

    report = pd.concat([recos, history])
    count_views = interactions.df.groupby("item_id").count()["user_id"]
    report = report.merge(item_data, how="inner", on="item_id")
    count_views.name = "count_views"
    report = report.merge(count_views, how="inner", on="item_id")

    report.sort_values(["user_id", "type"], inplace=True)
    report.set_index(["user_id", "type"], inplace=True)
    report = report.style.set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    return report