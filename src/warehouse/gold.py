from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd

from src.dq.log import append_quality_log

def compute_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Computes Population Stability Index (PSI) between reference and current distribution."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_edges = np.percentile(expected, percentiles)
    bucket_edges[0] -= 1e-5
    bucket_edges[-1] += 1e-5

    expected_counts, _ = np.histogram(expected, bins=bucket_edges)
    actual_counts, _ = np.histogram(actual, bins=bucket_edges)

    expected_pct = np.clip(expected_counts / len(expected), 1e-4, 1.0)
    actual_pct = np.clip(actual_counts / len(actual), 1e-4, 1.0)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)

def build_gold_warehouse(
    config: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Tuple[Path, Dict[str, int]]:
    """
    Builds the Gold DuckDB data warehouse and Parquet exports:
    1. pm25_city_hourly: Median of QC-passed calibrated sensors per city-hour.
    2. features_hourly: CAMS, meteorology, and lagged fire features known at time t.
    3. DQ Gate 2: Completeness, freshness checks, and PSI distribution shift monitor.
    """
    if run_id is None:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%MZ')}"

    silver_dir = Path(config.get("paths", {}).get("silver", "data/silver"))
    gold_dir = Path("data/gold")
    gold_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(config.get("paths", {}).get("gold_db", "data/gold/saaf_hawa.duckdb"))
    quality_log_path = Path(config.get("paths", {}).get("quality_log", "data/gold/quality_log.parquet"))

    calibrated_file = silver_dir / "silver_sensor_calibrated.parquet"
    meteo_file = silver_dir / "silver_cams_weather.parquet"
    firms_file = silver_dir / "silver_punjab_fires.parquet"

    if not calibrated_file.exists():
        raise FileNotFoundError(f"Calibrated sensor data not found at {calibrated_file}")

    df_sensors = pd.read_parquet(calibrated_file)
    df_sensors["ts_utc"] = pd.to_datetime(df_sensors["ts_utc"], utc=True)

    # -----------------------------------------------------------------
    # 1. Gold: pm25_city_hourly (Target)
    # -----------------------------------------------------------------
    # Group by city and ts_utc, compute median of pm25_cal
    city_hourly_list = []
    cities_cfg = config.get("cities", {})

    for (city_key, ts), group in df_sensors.groupby(["city", "ts_utc"]):
        city_code = cities_cfg.get(city_key, {}).get("code", city_key[:3].upper())
        valid_readings = group["pm25_cal"].dropna()
        n_sensors = len(valid_readings)

        median_pm25 = float(valid_readings.median()) if n_sensors > 0 else np.nan
        row_id = f"{city_code}-{ts.strftime('%Y%m%dT%H00Z')}"

        qc_flag = "VALID"
        if n_sensors == 1:
            qc_flag = "SINGLE_SENSOR_WARNING"
        elif n_sensors == 0:
            qc_flag = "MISSING_TARGET"

        city_hourly_list.append({
            "row_id": row_id,
            "city": city_key,
            "city_code": city_code,
            "ts_utc": ts,
            "pm25_cal": median_pm25,
            "n_sensors": n_sensors,
            "qc_flag": qc_flag,
            "is_filled": False,  # Missing target stays missing
        })

    df_city_pm25 = pd.DataFrame(city_hourly_list)
    df_city_pm25 = df_city_pm25.drop_duplicates(subset=["row_id"]).sort_values(by=["city", "ts_utc"])

    # -----------------------------------------------------------------
    # 2. Gold: features_hourly (CAMS, Weather, Lagged Fires)
    # -----------------------------------------------------------------
    if meteo_file.exists():
        df_meteo = pd.read_parquet(meteo_file)
        df_meteo["ts_utc"] = pd.to_datetime(df_meteo["ts_utc"], utc=True)
    else:
        df_meteo = pd.DataFrame()

    if firms_file.exists():
        df_fires = pd.read_parquet(firms_file)
        df_fires["date"] = pd.to_datetime(df_fires["date"]).dt.strftime("%Y-%m-%d")
    else:
        df_fires = pd.DataFrame()

    features_list = []
    for _, row in df_meteo.iterrows():
        city_key = row["city"]
        city_code = row["city_code"]
        ts = row["ts_utc"]
        row_id = f"{city_code}-{ts.strftime('%Y%m%dT%H00Z')}"

        # Wind u and v components calculation from speed and direction
        w_speed = row.get("wind_speed_10m", 2.5) or 2.5
        w_dir = row.get("wind_direction_10m", 180.0) or 180.0
        rad = np.deg2rad(w_dir)
        wind_u = -w_speed * np.sin(rad)
        wind_v = -w_speed * np.cos(rad)

        # Fire lag features (1, 2, 3 days ago)
        fire_feats = {
            "fire_count_lag1": 0, "fire_count_lag2": 0, "fire_count_lag3": 0,
            "fire_frp_lag1": 0.0, "fire_frp_lag2": 0.0, "fire_frp_lag3": 0.0,
        }
        if not df_fires.empty:
            for lag in [1, 2, 3]:
                lag_date = (ts - timedelta(days=lag)).strftime("%Y-%m-%d")
                match = df_fires[df_fires["date"] == lag_date]
                if not match.empty:
                    fire_feats[f"fire_count_lag{lag}"] = int(match["fire_count"].iloc[0])
                    fire_feats[f"fire_frp_lag{lag}"] = float(match["total_frp"].iloc[0])

        features_list.append({
            "row_id": row_id,
            "city": city_key,
            "city_code": city_code,
            "ts_utc": ts,
            "cams_pm25": row.get("cams_pm2_5"),
            "cams_pm10": row.get("cams_pm10"),
            "aod": row.get("aod"),
            "temp": row.get("temp_2m"),
            "rh": row.get("rh_2m"),
            "wind_u": float(wind_u),
            "wind_v": float(wind_v),
            "precip": row.get("precip"),
            **fire_feats,
        })

    df_features = pd.DataFrame(features_list)
    if not df_features.empty:
        df_features = df_features.drop_duplicates(subset=["row_id"]).sort_values(by=["city", "ts_utc"])

    # -----------------------------------------------------------------
    # 3. DQ Gate 2: Completeness, Freshness, and Drift (PSI) Checks
    # -----------------------------------------------------------------
    now_utc = datetime.now(timezone.utc)
    
    # Freshness Check
    if not df_sensors.empty:
        last_sensor_ts = df_sensors["ts_utc"].max()
        sensor_staleness_hours = (now_utc - last_sensor_ts).total_seconds() / 3600.0
        if sensor_staleness_hours > 3.0:
            append_quality_log(
                run_id=run_id,
                gate="GATE_2",
                check="freshness_check",
                rule="R06_SOURCE_STALE",
                action="FLAG",
                reason=f"OpenAQ ground truth is {round(sensor_staleness_hours, 1)}h old (> 3h threshold).",
                evidence={"staleness_hours": sensor_staleness_hours},
                table="pm25_city_hourly",
                rows_affected=len(df_city_pm25),
                log_path=quality_log_path,
            )

    # Distribution Drift Check (PSI)
    if len(df_city_pm25) >= 40:
        split_idx = len(df_city_pm25) // 2
        ref_vals = df_city_pm25["pm25_cal"].iloc[:split_idx].values
        cur_vals = df_city_pm25["pm25_cal"].iloc[split_idx:].values
        psi = compute_psi(ref_vals, cur_vals)
        if psi > 0.25:
            append_quality_log(
                run_id=run_id,
                gate="GATE_2",
                check="distribution_drift",
                rule="R07_FEATURE_DRIFT_PSI",
                action="FLAG",
                reason=f"Population Stability Index (PSI = {round(psi, 3)}) exceeds 0.25 drift threshold.",
                evidence={"psi": psi},
                table="pm25_city_hourly",
                rows_affected=len(cur_vals),
                log_path=quality_log_path,
            )

    # -----------------------------------------------------------------
    # 4. Ingest into DuckDB Warehouse
    # -----------------------------------------------------------------
    con = duckdb.connect(str(db_path))
    con.execute("CREATE OR REPLACE TABLE pm25_city_hourly AS SELECT * FROM df_city_pm25")
    con.execute("CREATE OR REPLACE TABLE features_hourly AS SELECT * FROM df_features")
    con.close()

    # Also save Parquet copies for fast pipeline access
    df_city_pm25.to_parquet(gold_dir / "pm25_city_hourly.parquet", engine="pyarrow", index=False)
    df_features.to_parquet(gold_dir / "features_hourly.parquet", engine="pyarrow", index=False)

    counts = {
        "pm25_city_hourly": len(df_city_pm25),
        "features_hourly": len(df_features),
    }
    print(f"[Gold] DuckDB warehouse built at {db_path}: {counts}")
    return db_path, counts

if __name__ == "__main__":
    from flows.hourly import load_config
    cfg = load_config()
    build_gold_warehouse(cfg)
