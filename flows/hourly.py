import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.openaq import fetch_latest as fetch_openaq
from src.ingest.openmeteo import fetch_latest as fetch_openmeteo
from src.ingest.firms import fetch_latest_fires
from src.ingest.airnow_archive import fetch_historical_archive
from src.warehouse.silver import build_silver_layer

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    p = Path(config_path)
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            # Fallback to json equivalent if yaml module is not present
            json_fallback = p.with_suffix(".json")
            if json_fallback.exists():
                with open(json_fallback, "r", encoding="utf-8") as f:
                    return json.load(f)
            raise
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def run_hourly_pipeline() -> Dict[str, Any]:
    """
    Executes the ingestion, DQ Gate 1, and Silver layer pipeline:
    1. Loads configuration
    2. Ingests OpenAQ & Open-Meteo Bronze Parquet + updates manifest
    3. Ingests NASA FIRMS & AirNow archive Bronze Parquet (if needed) + updates manifest
    4. Executes DQ Gate 1 and builds Silver layer records
    5. Records health, freshness, and quality log summary in status.json
    """
    config = load_config()
    start_time = datetime.now(timezone.utc)
    run_id = f"run_{start_time.strftime('%Y%m%dT%H%MZ')}"
    print(f"[{start_time.isoformat()}] Starting Saaf Hawa hourly pipeline (Run ID: {run_id})...")

    # 1. Bronze Ingestions
    openaq_path = fetch_openaq(config)
    print(f"-> OpenAQ Bronze landed: {openaq_path}")

    openmeteo_path = fetch_openmeteo(config)
    print(f"-> Open-Meteo Bronze landed: {openmeteo_path}")

    # FIRMS fires
    firms_path = fetch_latest_fires(config)
    print(f"-> NASA FIRMS Bronze landed: {firms_path}")

    # Historical AirNow embassy reference (backfill if not existing)
    bronze_airnow_dir = Path(config.get("paths", {}).get("bronze_airnow", "data/bronze/airnow"))
    if not list(bronze_airnow_dir.glob("*.parquet")):
        airnow_path = fetch_historical_archive(config)
        print(f"-> AirNow Archive Bronze landed: {airnow_path}")

    # 2. Data Quality Gate 1 & Silver Layer Generation
    silver_paths = build_silver_layer(config, run_id=run_id)
    print(f"-> Silver layer built: {list(silver_paths.keys())}")

    # 3. Check Manifest and Quality Log
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
        "silver_files": {k: str(v) for k, v in silver_paths.items()},
        "pipeline_version": "v2.0-silver-ready",
    }

    status_file = Path(config.get("paths", {}).get("status_json", "status.json"))
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    print(f"[{end_time.isoformat()}] Pipeline successful. Status written to {status_file}")
    return status_data

if __name__ == "__main__":
    run_hourly_pipeline()
