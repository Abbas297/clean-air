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
    Executes the minimal hourly heartbeat pipeline:
    1. Loads configuration
    2. Ingests OpenAQ Bronze Parquet + updates manifest
    3. Ingests Open-Meteo Bronze Parquet + updates manifest
    4. Records health and freshness in status.json
    """
    config = load_config()
    start_time = datetime.now(timezone.utc)
    print(f"[{start_time.isoformat()}] Starting Saaf Hawa hourly pipeline...")

    # Bronze Ingestions
    openaq_path = fetch_openaq(config)
    print(f"-> OpenAQ Bronze landed: {openaq_path}")

    openmeteo_path = fetch_openmeteo(config)
    print(f"-> Open-Meteo Bronze landed: {openmeteo_path}")

    # Check manifest
    manifest_path = Path(config.get("paths", {}).get("manifest", "data/manifest.csv"))
    manifest_rows = 0
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_rows = sum(1 for _ in f) - 1

    end_time = datetime.now(timezone.utc)
    duration_sec = (end_time - start_time).total_seconds()

    status_data = {
        "last_heartbeat_utc": end_time.isoformat(),
        "run_status": "HEALTHY",
        "duration_seconds": round(duration_sec, 2),
        "manifest_rows_total": manifest_rows,
        "bronze_openaq_latest": str(openaq_path),
        "bronze_openmeteo_latest": str(openmeteo_path),
        "pipeline_version": "v2.0-heartbeat",
    }

    status_file = Path(config.get("paths", {}).get("status_json", "status.json"))
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    print(f"[{end_time.isoformat()}] Heartbeat successful. Status written to {status_file}")
    return status_data

if __name__ == "__main__":
    run_hourly_pipeline()
