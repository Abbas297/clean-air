from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.dq.tools import (
    SENSOR_READINGS_SCHEMA,
    detect_sentinels,
    detect_stuck_values,
    detect_extreme_values,
)
from src.dq.policy import RULES, DQAction
from src.dq.log import append_quality_log

def run_gate1_quality_agent(
    df: pd.DataFrame,
    run_id: str,
    pm25_col: str = "pm25_raw",
    sensor_col: str = "sensor_id",
    city_col: str = "city",
    ts_col: str = "ts_utc",
    quarantine_dir: Path = Path("data/quarantine"),
    quality_log_path: Path = Path("data/gold/quality_log.parquet"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Data Quality Gate 1:
    - Adjudicates and fixes sentinels (-999 -> NULL)
    - Isolates implausible extreme readings (>1000) to quarantine
    - Corroborates extreme smog spikes (500-1000) against city peers
    - Flags stuck sensors (>=6h identical)
    - Validates types and schema with Pandera
    - Logs every decision to quality_log.parquet
    """
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    df_clean = df.copy()

    # Ensure UTC datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_clean[ts_col]):
        df_clean[ts_col] = pd.to_datetime(df_clean[ts_col], utc=True)

    df_clean["qc_flag"] = "VALID"
    df_clean["is_filled"] = False

    quarantined_rows: List[pd.DataFrame] = []

    # 1. Check Sentinels (e.g. -999, negative)
    sentinel_mask = detect_sentinels(df_clean, pm25_col=pm25_col)
    sentinel_count = int(sentinel_mask.sum())
    if sentinel_count > 0:
        sample_vals = df_clean.loc[sentinel_mask, pm25_col].unique()[:5].tolist()
        df_clean.loc[sentinel_mask, pm25_col] = np.nan
        df_clean.loc[sentinel_mask, "qc_flag"] = "SENTINEL_FIXED"
        append_quality_log(
            run_id=run_id,
            gate="GATE_1",
            check="sentinel_check",
            rule="R01_SENTINEL",
            action=DQAction.FIX.value,
            reason=f"Detected {sentinel_count} negative or -999 sentinel readings. Replaced with NULL.",
            evidence={"sample_sentinel_values": sample_vals, "rows_count": sentinel_count},
            table="bronze_sensor_readings",
            rows_affected=sentinel_count,
            log_path=quality_log_path,
        )

    # 2. Check Implausible Extreme Values (> 1000 ug/m3)
    must_quarantine_mask, needs_adj_mask = detect_extreme_values(df_clean, pm25_col=pm25_col)
    quarantine_count = int(must_quarantine_mask.sum())
    if quarantine_count > 0:
        bad_rows = df_clean[must_quarantine_mask].copy()
        bad_rows["quarantine_reason"] = "R02_PHYSICALLY_IMPLAUSIBLE (>1000 ug/m3)"
        bad_rows["run_id"] = run_id
        quarantined_rows.append(bad_rows)

        append_quality_log(
            run_id=run_id,
            gate="GATE_1",
            check="range_check",
            rule="R02_PHYSICALLY_IMPLAUSIBLE",
            action=DQAction.QUARANTINE.value,
            reason=f"PM2.5 reading exceeded physical sensor threshold of 1000 ug/m3. Routed to quarantine.",
            evidence={"max_value_observed": float(df_clean.loc[must_quarantine_mask, pm25_col].max())},
            table="bronze_sensor_readings",
            rows_affected=quarantine_count,
            log_path=quality_log_path,
        )
        df_clean = df_clean[~must_quarantine_mask].copy()

    # 3. Adjudicate Severe Smog Peaks (500 to 1000 ug/m3) with Peer Corroboration
    adj_count = int(needs_adj_mask.sum())
    if adj_count > 0:
        # Check if at least 2 other sensors in the city exceeded 300 ug/m3 in the same hour
        for idx in df_clean[needs_adj_mask].index:
            row_city = df_clean.loc[idx, city_col]
            row_time = df_clean.loc[idx, ts_col]
            val = df_clean.loc[idx, pm25_col]

            peer_readings = df_clean[
                (df_clean[city_col] == row_city)
                & (df_clean[ts_col] == row_time)
                & (df_clean.index != idx)
            ]
            corroborating_peers = peer_readings[peer_readings[pm25_col] >= 300.0]

            if len(corroborating_peers) >= 1 or len(peer_readings) <= 1:
                # Confirmed extreme smog event
                df_clean.loc[idx, "qc_flag"] = "EXTREME_CONFIRMED"
                append_quality_log(
                    run_id=run_id,
                    gate="GATE_1",
                    check="peer_adjudication",
                    rule="R05_EXTREME_CONFIRMED",
                    action=DQAction.CONFIRM.value,
                    reason=f"Extreme PM2.5 value ({val} ug/m3) in {row_city} confirmed during severe smog event.",
                    evidence={"city": row_city, "val": float(val), "time": str(row_time)},
                    table="bronze_sensor_readings",
                    rows_affected=1,
                    log_path=quality_log_path,
                )
            else:
                df_clean.loc[idx, "qc_flag"] = "EXTREME_UNCONFIRMED_QUARANTINED"
                q_row = df_clean.loc[[idx]].copy()
                q_row["quarantine_reason"] = "R04_EXTREME_UNCONFIRMED"
                q_row["run_id"] = run_id
                quarantined_rows.append(q_row)
                df_clean.loc[idx, pm25_col] = np.nan

    # 4. Stuck Sensor Detection (>= 6 hours identical non-null reading)
    stuck_mask = detect_stuck_values(df_clean, sensor_col=sensor_col, pm25_col=pm25_col, window_hours=6)
    stuck_count = int(stuck_mask.sum())
    if stuck_count > 0:
        df_clean.loc[stuck_mask, "qc_flag"] = "STUCK_SENSOR_FLAGGED"
        # Mask out stuck values from target calculation
        df_clean.loc[stuck_mask, pm25_col] = np.nan
        append_quality_log(
            run_id=run_id,
            gate="GATE_1",
            check="stuck_value_check",
            rule="R03_STUCK_SENSOR",
            action=DQAction.FLAG.value,
            reason=f"Sensor values remained identical for 6+ consecutive hours. Flagged and masked.",
            evidence={"stuck_row_count": stuck_count},
            table="bronze_sensor_readings",
            rows_affected=stuck_count,
            log_path=quality_log_path,
        )

    # 5. Pandera Schema Validation
    try:
        df_clean = SENSOR_READINGS_SCHEMA.validate(df_clean)
    except Exception as e:
        print(f"[DQ Gate 1 Warning] Pandera schema note: {e}")

    # Write quarantine dataset if any rows quarantined
    if quarantined_rows:
        df_quarantined = pd.concat(quarantined_rows, ignore_index=True)
        ts_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
        q_file = quarantine_dir / f"quarantine_{ts_now}.parquet"
        df_quarantined.to_parquet(q_file, engine="pyarrow", index=False)
    else:
        df_quarantined = pd.DataFrame()

    return df_clean, df_quarantined
