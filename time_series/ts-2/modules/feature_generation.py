from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .index_slicing import get_cols_idx, get_slice


def get_features_df_and_targets(
    df: pd.DataFrame,
    features_ids,
    targets_ids,
    id_column: Union[str, Sequence[str]] = "id",
    date_column: Union[str, Sequence[str]] = "datetime",
    target_column: str = "target",
    rolling_windows: Sequence[int] = (3, 7, 14),
):
    df = df.copy()
    # Формируем календарные признаки из признаков времени
    df["minute"] = df[date_column].dt.minute
    df["hour"] = df[date_column].dt.hour
    df["day"] = df[date_column].dt.day
    df["month"] = df[date_column].dt.month
    df["quarter"] = df[date_column].dt.quarter
    df["year"] = df[date_column].dt.year
    date_columns = ["minute", "hour", "day", "month", "quarter", "year"]

    # Признаки идентификатора ряда и времени возьмем из таргетных индексов
    features_df_id = get_slice(df, (targets_ids, get_cols_idx(df, id_column)))
    features_time = get_slice(df, (targets_ids, get_cols_idx(df, date_columns)))

    # Лаговые признаки возьмем из индексов признаков. Все доступные из истории
    features_lags = get_slice(
        df,
        (features_ids, get_cols_idx(df, target_column)),
    )

    df_sorted = df.sort_values(by=[id_column, date_column])

    print("Добавил признаки")
    window_features = []
    for window in rolling_windows:
        df_sorted[f"rolling_mean_{window}"] = (
            df_sorted.groupby(id_column)[target_column]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df_sorted[f"rolling_std_{window}"] = (
            df_sorted.groupby(id_column)[target_column]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
        window_features.extend([f"rolling_mean_{window}", f"rolling_std_{window}"])

    features_window = get_slice(
        df_sorted,
        (targets_ids, get_cols_idx(df_sorted, window_features))
    )

    # Объединим все признаки в один массив
    features = np.hstack([features_df_id, features_time, features_lags])
    categorical_features_idx = np.arange(
        features_df_id.shape[1] + features_time.shape[1]
    )  # Отметим категориальные признаки для CatBoost

    # Иначе cb.Pool не работает
    features_obj = features.astype(object)
    for j in categorical_features_idx:
        features_obj[:, j] = features_obj[:, j].astype(str)

    # Сформируем таргеты
    targets = get_slice(df, (targets_ids, get_cols_idx(df_sorted, target_column)))
    return features_obj, targets, categorical_features_idx
