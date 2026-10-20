from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
HORIZONS = [6, 12, 24]

def build_feature_matrix(
    df_pm25: pd.DataFrame,
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constructs unified feature matrix aligned strictly at issue time t.
    Builds autoregressive lags (1, 3, 6, 12, 24, 48h) and merges external atmospheric/fire signals.
    """
    df_pm25 = df_pm25.sort_values(by=["city", "ts_utc"]).copy()
    feature_dfs = []

    for city, group in df_pm25.groupby("city"):
        group = group.set_index("ts_utc").copy()
        s = group["pm25_cal"]

        # Lag features known at time t
        for lag in [1, 3, 6, 12, 24, 48]:
            group[f"pm25_lag_{lag}"] = s.shift(lag)

        # Rolling statistics
        group["pm25_roll_mean_6h"] = s.rolling(window=6, min_periods=1).mean()
        group["pm25_roll_mean_24h"] = s.rolling(window=24, min_periods=1).mean()
        group["pm25_roll_std_24h"] = s.rolling(window=24, min_periods=1).std().fillna(0.0)

        # Targets for horizons h
        for h in HORIZONS:
            group[f"target_h{h}"] = s.shift(-h)

        group = group.reset_index()
        feature_dfs.append(group)

    df_matrix = pd.concat(feature_dfs, ignore_index=True)

    # Merge atmospheric features if present
    if not df_features.empty:
        merge_cols = [
            "cams_pm25", "cams_pm10", "aod", "temp", "rh", "wind_u", "wind_v", "precip",
            "fire_count_lag1", "fire_count_lag2", "fire_count_lag3",
            "fire_frp_lag1", "fire_frp_lag2", "fire_frp_lag3",
        ]
        available_cols = [c for c in merge_cols if c in df_features.columns]
        df_matrix = df_matrix.merge(
            df_features[["city", "ts_utc"] + available_cols],
            on=["city", "ts_utc"],
            how="left",
        )

    # Calendar & Spatial
    df_matrix["hour"] = df_matrix["ts_utc"].dt.hour
    df_matrix["dayofweek"] = df_matrix["ts_utc"].dt.dayofweek
    df_matrix["month"] = df_matrix["ts_utc"].dt.month
    df_matrix["city_cat"] = df_matrix["city"].astype("category")

    return df_matrix

def train_quantile_models(
    df_matrix: pd.DataFrame,
    train_end: str = "2024-06-30T23:59:59Z",
    val_start: str = "2024-07-02T00:00:00Z",
    models_dir: Path = Path("data/gold/models"),
) -> Dict[str, lgb.Booster]:
    """
    Trains 15 LightGBM quantile regression models (5 quantiles x 3 horizons)
    strictly on the Train split.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    trained_models = {}

    feature_cols = [
        "pm25_cal", "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24", "pm25_lag_48",
        "pm25_roll_mean_6h", "pm25_roll_mean_24h", "pm25_roll_std_24h",
        "hour", "dayofweek", "month", "city_cat",
    ]
    # Add external features if populated
    for ext in ["cams_pm25", "cams_pm10", "aod", "temp", "rh", "wind_u", "wind_v", "precip",
                "fire_count_lag1", "fire_count_lag2", "fire_count_lag3",
                "fire_frp_lag1", "fire_frp_lag2", "fire_frp_lag3"]:
        if ext in df_matrix.columns:
            feature_cols.append(ext)

    train_mask = df_matrix["ts_utc"] <= pd.to_datetime(train_end, utc=True)
    df_train = df_matrix[train_mask]

    for h in HORIZONS:
        target_col = f"target_h{h}"
        valid_train = df_train.dropna(subset=[target_col])
        if len(valid_train) < 50:
            print(f"[Model Warning] Low training samples for horizon +{h}h ({len(valid_train)} rows).")

        X_train = valid_train[feature_cols]
        y_train = valid_train[target_col]

        for q in QUANTILES:
            model_name = f"lgbm_h{h}_q{int(q*100)}"
            params = {
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "verbosity": -1,
                "random_state": 42,
            }

            dtrain = lgb.Dataset(X_train, label=y_train)
            booster = lgb.train(params, dtrain, num_boost_round=100)
            
            save_path = models_dir / f"{model_name}.txt"
            booster.save_model(str(save_path))
            trained_models[model_name] = booster

    print(f"[Model] Successfully trained {len(trained_models)} LightGBM quantile models.")
    return trained_models
