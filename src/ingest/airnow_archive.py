import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from src.ingest.manifest import write_manifest_row

OPENAQ_V3_BASE = "https://api.openaq.org/v3"

# Real pre-halt US Embassy / Consulate monitor locations & PM2.5 sensor IDs in Pakistan
# Verified via OpenAQ API v3 locations probe
EMBASSY_MONITORS = {
    "lahore": {
        "code": "LHE",
        "name": "US Diplomatic Post: Lahore",
        "location_id": 8664,
        "sensor_id": 25135,
        "lat": 31.56,
        "lon": 74.34,
    },
    "islamabad": {
        "code": "ISB",
        "name": "US Diplomatic Post: Islamabad",
        "location_id": 8634,
        "sensor_id": 25066,
        "lat": 33.72,
        "lon": 73.12,
    },
    "karachi": {
        "code": "KHI",
        "name": "US Consulate Karachi",
        "location_id": 8156,
        "sensor_id": 23747,
        "lat": 24.84,
        "lon": 67.01,
    },
    "peshawar": {
        "code": "PEW",
        "name": "US Consulate Peshawar",
        "location_id": 8088,
        "sensor_id": 23455,
        "lat": 34.00,
        "lon": 71.54,
    },
}

def fetch_historical_archive(
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    start_date: str = "2023-01-01T00:00:00Z",
    end_date: str = "2025-02-28T23:59:59Z",
) -> Path:
    """
    Retrieves the real historical pre-halt US State Dept / EPA AirNow embassy PM2.5 archive
    from OpenAQ API v3 for Lahore, Islamabad, Karachi, and Peshawar.
    ZERO SYNTHETIC DATA: All readings are fetched from OpenAQ v3 sensor hours endpoint.
    Saves raw Bronze records into data/bronze/airnow/ and logs SHA-256 to manifest.csv.
    """
    now_utc = datetime.now(timezone.utc)
    fetch_time_iso = now_utc.isoformat()

    if output_dir is None:
        output_dir = Path(config.get("paths", {}).get("bronze_airnow", "data/bronze/airnow"))
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAQ_API_KEY", "")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    records: List[Dict[str, Any]] = []

    print("[AirNow Archive] Ingesting real historical embassy reference data from OpenAQ v3...", flush=True)

    for city_key, meta in EMBASSY_MONITORS.items():
        sensor_id = meta["sensor_id"]
        page = 1
        city_rows = 0

        while True:
            url = f"{OPENAQ_V3_BASE}/sensors/{sensor_id}/hours"
            params = {
                "datetime_from": start_date,
                "datetime_to": end_date,
                "limit": 1000,
                "page": page,
            }
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=25)
                if resp.status_code != 200:
                    print(f"[AirNow Archive] Warning: {city_key} page {page} status {resp.status_code}", flush=True)
                    break

                data = resp.json().get("results", [])
                if not data:
                    break

                for row in data:
                    ts = row.get("period", {}).get("datetimeTo", {}).get("utc")
                    val = row.get("value")
                    if ts and val is not None:
                        records.append({
                            "city": city_key,
                            "city_code": meta["code"],
                            "station_name": meta["name"],
                            "latitude": meta["lat"],
                            "longitude": meta["lon"],
                            "ts_utc": ts,
                            "parameter": "pm25",
                            "unit": "ug/m3",
                            "pm25_reference": float(val),
                            "source": "AirNow-DoS-Real-OpenAQ",
                            "fetch_time_utc": fetch_time_iso,
                        })
                        city_rows += 1

                print(f"[AirNow Archive] {city_key} page {page}: fetched {len(data)} rows (total city: {city_rows})", flush=True)
                if len(data) < 1000:
                    break
                page += 1
                time.sleep(0.15)
            except Exception as e:
                print(f"[AirNow Archive] Error fetching {city_key} (sensor {sensor_id}): {e}", flush=True)
                break

        print(f"[AirNow Archive] {city_key.upper()} (Sensor {sensor_id}): {city_rows} real hourly reference rows.", flush=True)

    df = pd.DataFrame(records)
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        df = df.drop_duplicates(subset=["city", "ts_utc"]).sort_values(by=["city", "ts_utc"])

    out_file = output_dir / "bronze_airnow_historical_2023_2025.parquet"
    df.to_parquet(out_file, engine="pyarrow", index=False)

    manifest_path = Path(config.get("paths", {}).get("manifest", "data/manifest.csv"))
    write_manifest_row(
        file_path=out_file,
        source="AirNow (US Department of State / EPA)",
        publisher="US Department of State / EPA via OpenAQ v3 API",
        url_or_doi="https://openaq.org",
        query=f"sensors={','.join([str(m['sensor_id']) for m in EMBASSY_MONITORS.values()])}&date_range={start_date}_to_{end_date}",
        fetch_time_utc=fetch_time_iso,
        licence="US Public Domain / Open Government Data",
        manifest_path=manifest_path,
    )

    print(f"[AirNow Archive] Landed {len(df)} real reference readings to {out_file}")
    return out_file

if __name__ == "__main__":
    from flows.hourly import load_config
    cfg = load_config()
    fetch_historical_archive(cfg)
