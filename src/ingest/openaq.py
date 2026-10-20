import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from src.ingest.manifest import write_manifest_row

OPENAQ_V3_BASE = "https://api.openaq.org/v3"

def fetch_latest(
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Fetches latest PM2.5 readings for Pakistani sensors from OpenAQ API v3
    and writes raw data to Bronze Parquet.
    """
    now_utc = datetime.now(timezone.utc)
    ts_str = now_utc.strftime("%Y%m%dT%H%MZ")
    fetch_time_iso = now_utc.isoformat()

    if output_dir is None:
        output_dir = Path(config.get("paths", {}).get("bronze_openaq", "data/bronze/openaq"))
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAQ_API_KEY", config.get("api_keys", {}).get("openaq", ""))
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    records: List[Dict[str, Any]] = []
    licences_observed: List[str] = []

    try:
        # Query locations in Pakistan with PM2.5 (parameter id 2)
        url = f"{OPENAQ_V3_BASE}/locations"
        params = {"countries_id": "PK", "parameters_id": 2, "limit": 100}
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json().get("results", [])
            for loc in data:
                loc_id = loc.get("id")
                loc_name = loc.get("name")
                loc_licences = loc.get("licenses") or []
                lic_str = "; ".join([l.get("name", "Unknown") for l in loc_licences]) if isinstance(loc_licences, list) else "Unknown"
                if lic_str and lic_str not in licences_observed:
                    licences_observed.append(lic_str)

                coords = loc.get("coordinates", {})
                sensors = loc.get("sensors", [])

                for s in sensors:
                    param = s.get("parameter", {})
                    if param.get("name") == "pm25" or param.get("id") == 2:
                        records.append({
                            "location_id": loc_id,
                            "location_name": loc_name,
                            "sensor_id": s.get("id"),
                            "latitude": coords.get("latitude"),
                            "longitude": coords.get("longitude"),
                            "datetime_last": loc.get("datetimeLast", {}).get("utc") if isinstance(loc.get("datetimeLast"), dict) else None,
                            "licences": lic_str,
                            "raw_sensor_meta": str(s),
                            "fetch_time_utc": fetch_time_iso,
                        })
    except Exception as e:
        records.append({
            "location_id": -1,
            "location_name": "FETCH_ERROR",
            "sensor_id": -1,
            "latitude": None,
            "longitude": None,
            "datetime_last": None,
            "licences": "None",
            "raw_sensor_meta": str(e),
            "fetch_time_utc": fetch_time_iso,
        })

    if not records:
        records.append({
            "location_id": 0,
            "location_name": "EMPTY_OR_UNAVAILABLE",
            "sensor_id": 0,
            "latitude": None,
            "longitude": None,
            "datetime_last": None,
            "licences": "Unknown",
            "raw_sensor_meta": "No sensors returned",
            "fetch_time_utc": fetch_time_iso,
        })

    df = pd.DataFrame(records)
    out_file = output_dir / f"bronze_openaq_{ts_str}.parquet"
    df.to_parquet(out_file, engine="pyarrow", index=False)

    manifest_path = Path(config.get("paths", {}).get("manifest", "data/manifest.csv"))
    write_manifest_row(
        file_path=out_file,
        source="OpenAQ",
        publisher="Pakistan Air Quality Initiative (PAQI) / Community via OpenAQ",
        url_or_doi="https://openaq.org",
        query="countries_id=PK&parameters_id=2",
        fetch_time_utc=fetch_time_iso,
        licence="; ".join(licences_observed) if licences_observed else "CC BY 4.0 / Community",
        manifest_path=manifest_path,
    )

    return out_file
