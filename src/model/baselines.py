from datetime import timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def compute_persistence_baseline(
    df_city_pm25: pd.DataFrame,
    horizons: List[int] = [6, 12, 24],
) -> pd.DataFrame:
    """
    Computes Baseline A (Persistence): PM2.5 at issue time t projected to t + h.
    Persistence prediction at horizon h equals value at t.
    """
    df = df_city_pm25.sort_values(by=["city", "ts_utc"]).copy()
    records = []

    for city, group in df.groupby("city"):
        group = group.set_index("ts_utc")
        for h in horizons:
            # Shift backwards to align prediction for t+h known at t
            pred_series = group["pm25_cal"].shift(0)  # value at t
            actual_series = group["pm25_cal"].shift(-h)  # value at t+h

            valid = pred_series.notna() & actual_series.notna()
            ts_valid = group.index[valid]
            preds = pred_series[valid].values
            trues = actual_series[valid].values

            for ts, p, y in zip(ts_valid, preds, trues):
                records.append({
                    "city": city,
                    "issued_at_utc": ts,
                    "target_ts_utc": ts + timedelta(hours=h),
                    "horizon": h,
                    "baseline_type": "persistence",
                    "y_pred": float(p),
                    "y_true": float(y),
                })

    return pd.DataFrame(records)

def compute_seasonal_hour_baseline(
    df_city_pm25: pd.DataFrame,
    horizons: List[int] = [6, 12, 24],
    window_days: int = 7,
) -> pd.DataFrame:
    """
    Computes Baseline B (Seasonal-Hour) using fast vectorization:
    Mean of the same target hour over the previous 7 days (known at time t).
    For target at t+h, previous occurrences are at (t+h) - 24*k for k in 1..7.
    """
    df = df_city_pm25.sort_values(by=["city", "ts_utc"]).copy()
    records = []

    for city, group in df.groupby("city"):
        group = group.set_index("ts_utc").copy()
        s = group["pm25_cal"]

        for h in horizons:
            # Target is at t + h
            actual = s.shift(-h)
            
            # Historical same-hour occurrences known at or before issue time t:
            # The target hour is (t + h). Prior instances are (t + h) - 24*k
            # known at t implies (t + h - 24*k) <= t, i.e., 24*k >= h.
            k_min = int(np.ceil(h / 24.0))
            k_range = range(max(1, k_min), max(1, k_min) + window_days)

            past_shifts = [s.shift(-(h - 24 * k)) for k in k_range]
            seasonal_mean = pd.concat(past_shifts, axis=1).mean(axis=1)

            valid_mask = actual.notna() & seasonal_mean.notna()
            ts_valid = group.index[valid_mask]
            preds = seasonal_mean[valid_mask].values
            trues = actual[valid_mask].values

            for ts, p, y in zip(ts_valid, preds, trues):
                records.append({
                    "city": city,
                    "issued_at_utc": ts,
                    "target_ts_utc": ts + timedelta(hours=h),
                    "horizon": h,
                    "baseline_type": "seasonal_hour",
                    "y_pred": float(p),
                    "y_true": float(y),
                })

    return pd.DataFrame(records)
