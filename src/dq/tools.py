from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema

# Pandera schema for Gate 1 raw sensor readings
SENSOR_READINGS_SCHEMA = DataFrameSchema(
    {
        "sensor_id": Column(pa.String, nullable=False),
        "city": Column(pa.String, nullable=False),
        "ts_utc": Column(pa.DateTime, nullable=False),
        "pm25_raw": Column(pa.Float, nullable=True),
    },
    coerce=True,
    strict=False,
)

def detect_sentinels(df: pd.DataFrame, pm25_col: str = "pm25_raw") -> pd.Series:
    """
    Identifies sentinel values (e.g. -999, -99, or any negative values).
    Returns boolean Series where True indicates a sentinel value.
    """
    if pm25_col not in df.columns:
        return pd.Series(False, index=df.index)
    return (df[pm25_col] < 0) | (df[pm25_col] == -999.0) | (df[pm25_col] == -99.0)

def detect_stuck_values(
    df: pd.DataFrame,
    sensor_col: str = "sensor_id",
    pm25_col: str = "pm25_raw",
    window_hours: int = 6,
) -> pd.Series:
    """
    Detects sensor readings where the identical non-null value persists for
    at least window_hours (default 6h) consecutively using fast vectorization.
    """
    if pm25_col not in df.columns or sensor_col not in df.columns:
        return pd.Series(False, index=df.index)

    df_sorted = df.sort_values(by=[sensor_col, "ts_utc"])
    s = df_sorted[pm25_col]
    sensors = df_sorted[sensor_col]

    valid = s.notna()
    # Identical to previous row of the same sensor
    same_sensor = sensors == sensors.shift(1)
    same_val = s == s.shift(1)
    new_block = ~(same_sensor & same_val & valid)
    block_id = new_block.cumsum()

    # Fast count per block using transform
    block_sizes = df_sorted.groupby(block_id)[pm25_col].transform("count")
    is_stuck_sorted = valid & (block_sizes >= window_hours)

    return is_stuck_sorted.reindex(df.index, fill_value=False)

def detect_extreme_values(
    df: pd.DataFrame,
    pm25_col: str = "pm25_raw",
    quarantine_limit: float = 1000.0,
    adjudicate_limit: float = 500.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns two boolean masks:
    1. must_quarantine: PM2.5 > 1000 (physically implausible)
    2. needs_adjudication: 500 < PM2.5 <= 1000 (extreme, check neighbours)
    """
    if pm25_col not in df.columns:
        falses = pd.Series(False, index=df.index)
        return falses, falses

    s = df[pm25_col]
    must_quarantine = (s > quarantine_limit)
    needs_adjudication = (s > adjudicate_limit) & (s <= quarantine_limit)
    return must_quarantine, needs_adjudication
