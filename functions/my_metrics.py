from copy import deepcopy
from time import time
from typing import Any, Iterator, Tuple, TypeAlias

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset, Interactions
from rectools.metrics import MAP, NDCG, MeanInvUserFreq, Precision, Recall, Serendipity, calc_metrics
from rectools.metrics.base import MetricAtK
from rectools.model_selection import Splitter, TimeRangeSplitter
from rectools.models.base import ModelBase

# Типы данных
ModelMetrics: TypeAlias = list[dict[str, Any]]
InteractionFold: TypeAlias = Tuple[np.ndarray, np.ndarray, dict[str, Any]]
InteractionFolds: TypeAlias = Iterator[InteractionFold]
_N_SPLITS = 3

def _split_dataset(splitter: Splitter, interactions: Interactions) -> InteractionFolds:
    """Разбивает взаимодействия на фолды с помощью разбивателя."""
    return splitter.split(interactions, collect_fold_stats=True)

def _calculate_model_metrics(
    model: ModelBase, metrics: dict[str, MetricAtK], df_train: pd.DataFrame, df_test: pd.DataFrame, k_recos: int
) -> Tuple[dict[str, float], float]:
    """Вычисляет метрики модели на тестовом наборе данных."""
    dataset = Dataset.construct(df_train)
    start_train_time = time()
    model.fit(dataset)
    train_time = time() - start_train_time
    recos = model.recommend(
        users=np.unique(df_test[Columns.User]),
        dataset=dataset,
        k=k_recos,
        filter_viewed=True,
    )
    metric_values = calc_metrics(
        metrics,
        reco=recos,
        interactions=df_test,
        prev_interactions=df_train,
        catalog=df_train[Columns.Item].unique(),
    )
    return metric_values, train_time

def metrics_culc(
    interactions: Interactions,
    model: ModelBase,
    k_recos: int,
) -> ModelMetrics:
    """Вычисляет метрики модели на валидационном наборе данных."""
    if not model:
        raise ValueError("Модель не должна быть пустой")
    if not interactions:
        raise ValueError("Набор взаимодействий не должен быть пустым")

    results = []

    for train_ids, test_ids, fold_info in _split_dataset(splitter=splitter, interactions=interactions):
        df_train = interactions.df.iloc[train_ids]
        df_test = interactions.df.iloc[test_ids][Columns.UserItem]
        metric_values, train_time = _calculate_model_metrics(
            model=deepcopy(model), metrics=metrics, df_train=df_train, df_test=df_test, k_recos=k_recos
        )
        fold_result = {"fold": fold_info["i_split"], "model": model.__class__.__name__, "train time (sec)": train_time}
        fold_result.update(metric_values)
        results.append(fold_result)
    return results