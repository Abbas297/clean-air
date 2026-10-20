from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.dq.agent import run_gate1_quality_agent
from src.dq.tools import detect_sentinels, detect_stuck_values, detect_extreme_values

def test_sentinel_filtering():
    data = {
        "sensor_id": ["s1", "s1", "s1"],
        "city": ["lahore", "lahore", "lahore"],
        "ts_utc": pd.date_range("2026-10-20", periods=3, freq="h", tz="UTC"),
        "pm25_raw": [45.0, -999.0, -10.0],
    }
    df = pd.DataFrame(data)
    sentinels = detect_sentinels(df, pm25_col="pm25_raw")
    assert sentinels.sum() == 2
    assert not sentinels.iloc[0]
    assert sentinels.iloc[1]
    assert sentinels.iloc[2]

def test_stuck_sensor_detection():
    # 6 consecutive identical values
    data = {
        "sensor_id": ["s1"] * 7,
        "city": ["lahore"] * 7,
        "ts_utc": pd.date_range("2026-10-20", periods=7, freq="h", tz="UTC"),
        "pm25_raw": [55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 80.0],
    }
    df = pd.DataFrame(data)
    stuck = detect_stuck_values(df, sensor_col="sensor_id", pm25_col="pm25_raw", window_hours=6)
    assert stuck.sum() == 6
    assert not stuck.iloc[6]

def test_extreme_value_quarantine():
    data = {
        "sensor_id": ["s1", "s1", "s1"],
        "city": ["lahore", "lahore", "lahore"],
        "ts_utc": pd.date_range("2026-10-20", periods=3, freq="h", tz="UTC"),
        "pm25_raw": [85.0, 1450.0, 620.0],
    }
    df = pd.DataFrame(data)
    quarantine_mask, adj_mask = detect_extreme_values(df, pm25_col="pm25_raw")
    assert quarantine_mask.sum() == 1
    assert df.loc[quarantine_mask, "pm25_raw"].iloc[0] == 1450.0
    assert adj_mask.sum() == 1
    assert df.loc[adj_mask, "pm25_raw"].iloc[0] == 620.0

def test_gate1_agent_execution(tmp_path):
    data = {
        "sensor_id": ["s1", "s1", "s1", "s1", "s1", "s1", "s1"],
        "city": ["lahore"] * 7,
        "ts_utc": pd.date_range("2026-10-20", periods=7, freq="h", tz="UTC"),
        "pm25_raw": [45.0, -999.0, 1200.0, 60.0, 60.0, 60.0, 60.0],
    }
    df = pd.DataFrame(data)
    q_dir = tmp_path / "quarantine"
    log_path = tmp_path / "quality_log.parquet"

    df_clean, df_q = run_gate1_quality_agent(
        df=df,
        run_id="test_run",
        pm25_col="pm25_raw",
        quarantine_dir=q_dir,
        quality_log_path=log_path,
    )

    # Sentinel (-999 at index 1) must be converted to NaN
    assert pd.isna(df_clean.loc[1, "pm25_raw"])
    assert df_clean.loc[1, "qc_flag"] == "SENTINEL_FIXED"
    
    # 1200 must be quarantined
    assert len(df_q) >= 1
    assert (df_q["pm25_raw"] == 1200.0).any()
    assert (df_clean["pm25_raw"] == 1200.0).sum() == 0

    # quality_log.parquet must exist and contain log rows
    assert log_path.exists()
    df_log = pd.read_parquet(log_path)
    assert len(df_log) >= 2
    assert "GATE_1" in df_log["gate"].values
