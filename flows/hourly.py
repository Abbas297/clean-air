import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prefect import flow, task, get_run_logger

from src.ingest.openaq import fetch_latest as fetch_openaq, fetch_historical_community_sensors
from src.ingest.openmeteo import fetch_latest as fetch_openmeteo, fetch_historical_archive as fetch_meteo_hist
from src.ingest.firms import fetch_latest_fires
from src.ingest.airnow_archive import fetch_historical_archive as fetch_airnow_hist
from src.warehouse.silver import build_silver_layer
from src.warehouse.calibrate import fit_and_apply_calibration
from src.warehouse.gold import build_gold_warehouse

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    p = Path(config_path)
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            json_fallback = p.with_suffix(".json")
            if json_fallback.exists():
                with open(json_fallback, "r", encoding="utf-8") as f:
                    return json.load(f)
            raise
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

@task(name="task-ingest-openaq")
def task_ingest_openaq(config: Dict[str, Any]) -> str:
    path = fetch_openaq(config)
    return str(path)

@task(name="task-ingest-openmeteo")
def task_ingest_openmeteo(config: Dict[str, Any]) -> str:
    path = fetch_openmeteo(config)
    return str(path)

@task(name="task-ingest-firms")
def task_ingest_firms(config: Dict[str, Any]) -> str:
    path = fetch_latest_fires(config)
    return str(path)

@task(name="task-build-silver")
def task_build_silver(config: Dict[str, Any], run_id: str) -> Dict[str, str]:
    silver_paths = build_silver_layer(config, run_id=run_id)
    return {k: str(v) for k, v in silver_paths.items()}

@task(name="task-calibrate-sensors")
def task_calibrate_sensors(config: Dict[str, Any]) -> str:
    cal_path, _ = fit_and_apply_calibration(config)
    return str(cal_path)

@task(name="task-build-gold")
def task_build_gold(config: Dict[str, Any], run_id: str) -> Dict[str, int]:
    _, counts = build_gold_warehouse(config, run_id=run_id)
    return counts

@flow(name="saaf-hawa-hourly-heartbeat")
def saaf_hawa_hourly_flow() -> Dict[str, Any]:
    """
    Executes the genuine Prefect orchestrated Saaf Hawa hourly pipeline:
    1. Loads configuration
    2. Ingests OpenAQ & Open-Meteo Bronze Parquet + updates manifest
    3. Ingests NASA FIRMS Bronze Parquet + updates manifest
    4. Executes DQ Gate 1 and builds Silver layer records
    5. Fits and applies sensor calibration on the Train split
    6. Builds Gold DuckDB warehouse and Parquet exports
    7. Records status.json with pipeline health and audited counts
    """
    config = load_config()
    start_time = datetime.now(timezone.utc)
    run_id = f"run_{start_time.strftime('%Y%m%dT%H%MZ')}"
    print(f"[{start_time.isoformat()}] Starting Saaf Hawa Prefect pipeline (Run ID: {run_id})...", flush=True)

    # 1. Bronze Ingestion Tasks
    openaq_path = task_ingest_openaq(config)
    openmeteo_path = task_ingest_openmeteo(config)
    firms_path = task_ingest_firms(config)

    # 2. Silver Layer & DQ Gate 1
    silver_files = task_build_silver(config, run_id=run_id)

    # 3. Calibration on Train Split
    cal_file = task_calibrate_sensors(config)

    # 4. Gold Warehouse & DQ Gate 2
    gold_counts = task_build_gold(config, run_id=run_id)

    # 5. Manifest & Audit Status
    manifest_path = Path(config.get("paths", {}).get("manifest", "data/manifest.csv"))
    manifest_rows = 0
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_rows = sum(1 for _ in f) - 1

    quality_log_path = Path(config.get("paths", {}).get("quality_log", "data/gold/quality_log.parquet"))
    quality_events = 0
    if quality_log_path.exists():
        import pandas as pd
        df_ql = pd.read_parquet(quality_log_path)
        quality_events = len(df_ql)

    end_time = datetime.now(timezone.utc)
    duration_sec = (end_time - start_time).total_seconds()

    status_data = {
        "last_heartbeat_utc": end_time.isoformat(),
        "run_id": run_id,
        "run_status": "HEALTHY",
        "duration_seconds": round(duration_sec, 2),
        "manifest_rows_total": manifest_rows,
        "quality_events_logged": quality_events,
        "silver_files": silver_files,
        "gold_tables": gold_counts,
        "pipeline_version": "v2.0-prefect-orchestrated",
    }

    status_file = Path(config.get("paths", {}).get("status_json", "status.json"))
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    print(f"[{end_time.isoformat()}] Prefect Pipeline successful. Status written to {status_file}", flush=True)
    return status_data

if __name__ == "__main__":
    saaf_hawa_hourly_flow()
