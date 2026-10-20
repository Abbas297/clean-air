from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.dq.agent import run_gate1_quality_agent

def build_silver_layer(
    config: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Transforms Bronze raw data into typed, UTC-normalized, de-duplicated Silver Parquet files.
    Applies Data Quality Gate 1 to ground truth sensor feeds.
    """
    if run_id is None:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%MZ')}"

    silver_dir = Path(config.get("paths", {}).get("silver", "data/silver"))
    silver_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = Path(config.get("paths", {}).get("quarantine", "data/quarantine"))
    quality_log_path = Path(config.get("paths", {}).get("quality_log", "data/gold/quality_log.parquet"))

    bronze_openaq_dir = Path(config.get("paths", {}).get("bronze_openaq", "data/bronze/openaq"))
    bronze_meteo_dir = Path(config.get("paths", {}).get("bronze_openmeteo", "data/bronze/openmeteo"))
    bronze_airnow_dir = Path(config.get("paths", {}).get("bronze_airnow", "data/bronze/airnow"))
    bronze_firms_dir = Path(config.get("paths", {}).get("bronze_firms", "data/bronze/firms"))

    output_paths = {}

    # -------------------------------------------------------------
    # 1. OpenAQ Community Sensors -> Silver Sensor Readings
    # -------------------------------------------------------------
    openaq_files = list(bronze_openaq_dir.glob("*.parquet"))
    if openaq_files:
        sensor_frames = []
        cities_cfg = config.get("cities", {})
        for f in openaq_files:
            df_part = pd.read_parquet(f)
            if "pm25_raw" in df_part.columns and "ts_utc" in df_part.columns and not df_part.empty:
                sensor_frames.append(df_part[["sensor_id", "city", "city_code", "latitude", "longitude", "ts_utc", "pm25_raw", "licences"]])
        df_sensors = pd.concat(sensor_frames, ignore_index=True) if sensor_frames else pd.DataFrame()

        # Run DQ Gate 1
        df_clean_sensors, _ = run_gate1_quality_agent(
            df=df_sensors,
            run_id=run_id,
            pm25_col="pm25_raw",
            sensor_col="sensor_id",
            city_col="city",
            ts_col="ts_utc",
            quarantine_dir=quarantine_dir,
            quality_log_path=quality_log_path,
        )

        # De-duplicate by sensor and UTC timestamp
        df_clean_sensors = df_clean_sensors.drop_duplicates(subset=["sensor_id", "ts_utc"], keep="last")
        sensor_silver_path = silver_dir / "silver_sensor_readings.parquet"
        df_clean_sensors.to_parquet(sensor_silver_path, engine="pyarrow", index=False)
        output_paths["sensors"] = sensor_silver_path
        print(f"[Silver] Clean sensor readings written: {len(df_clean_sensors)} rows -> {sensor_silver_path}")

    # -------------------------------------------------------------
    # 2. Open-Meteo -> Silver CAMS & Weather
    # -------------------------------------------------------------
    meteo_files = list(bronze_meteo_dir.glob("*.parquet"))
    if meteo_files:
        df_meteo = pd.concat([pd.read_parquet(f) for f in meteo_files], ignore_index=True)
        df_meteo["ts_utc"] = pd.to_datetime(df_meteo["ts_utc"], utc=True)
        # De-duplicate
        df_meteo = df_meteo.drop_duplicates(subset=["city", "ts_utc"], keep="last")
        # Ensure row ID format: {CITYCODE}-{YYYYMMDDTHHZ}
        df_meteo["row_id"] = (
            df_meteo["city_code"] + "-" + df_meteo["ts_utc"].dt.strftime("%Y%m%dT%H00Z")
        )
        meteo_silver_path = silver_dir / "silver_cams_weather.parquet"
        df_meteo.to_parquet(meteo_silver_path, engine="pyarrow", index=False)
        output_paths["meteo"] = meteo_silver_path
        print(f"[Silver] CAMS & Weather variables written: {len(df_meteo)} rows -> {meteo_silver_path}")

    # -------------------------------------------------------------
    # 3. AirNow Archive -> Silver Historical Embassy Reference
    # -------------------------------------------------------------
    airnow_files = list(bronze_airnow_dir.glob("*.parquet"))
    if airnow_files:
        df_airnow = pd.concat([pd.read_parquet(f) for f in airnow_files], ignore_index=True)
        df_airnow["ts_utc"] = pd.to_datetime(df_airnow["ts_utc"], utc=True)
        df_airnow = df_airnow.drop_duplicates(subset=["city_code", "ts_utc"], keep="last")
        airnow_silver_path = silver_dir / "silver_embassy_reference.parquet"
        df_airnow.to_parquet(airnow_silver_path, engine="pyarrow", index=False)
        output_paths["airnow"] = airnow_silver_path
        print(f"[Silver] Embassy calibration reference written: {len(df_airnow)} rows -> {airnow_silver_path}")

    # -------------------------------------------------------------
    # 4. NASA FIRMS -> Silver Punjab Transboundary Fires
    # -------------------------------------------------------------
    firms_files = list(bronze_firms_dir.glob("*.parquet"))
    if firms_files:
        df_firms = pd.concat([pd.read_parquet(f) for f in firms_files], ignore_index=True)
        # Aggregate daily fire detections and total FRP
        df_firms["date"] = pd.to_datetime(df_firms["acq_date"]).dt.strftime("%Y-%m-%d")
        daily_fires = df_firms.groupby("date").agg(
            fire_count=("frp", "count"),
            total_frp=("frp", "sum"),
            mean_frp=("frp", "mean"),
        ).reset_index()
        
        firms_silver_path = silver_dir / "silver_punjab_fires.parquet"
        daily_fires.to_parquet(firms_silver_path, engine="pyarrow", index=False)
        output_paths["firms"] = firms_silver_path
        print(f"[Silver] Aggregated fire counts written: {len(daily_fires)} days -> {firms_silver_path}")

    return output_paths

if __name__ == "__main__":
    from flows.hourly import load_config
    cfg = load_config()
    build_silver_layer(cfg)
