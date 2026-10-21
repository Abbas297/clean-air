import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from src.ingest.manifest import write_manifest_row
from src.ingest.retry_session import get_retry_session

FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_OPEN_SOUTH_ASIA_7D = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_7d.csv"

def fetch_latest_fires(
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    day_range: int = 7,
) -> Path:
    """
    Fetches real thermal anomalies and fire detection records from NASA FIRMS
    for the Punjab transboundary bounding box (28-34 N, 70-77 E).
    Uses NASA FIRMS open South Asia VIIRS data feed or FIRMS Area API.
    ZERO SYNTHETIC DATA: In the event of network failure or zero detections,
    records remain empty — synthetic fires are NEVER generated.
    Saves raw Bronze records into data/bronze/firms/ and logs SHA-256 to manifest.csv.
    """
    now_utc = datetime.now(timezone.utc)
    ts_str = now_utc.strftime("%Y%m%dT%H%MZ")
    fetch_time_iso = now_utc.isoformat()

    if output_dir is None:
        output_dir = Path(config.get("paths", {}).get("bronze_firms", "data/bronze/firms"))
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = config.get("firms_bounding_box", {"min_lat": 28.0, "max_lat": 34.0, "min_lon": 70.0, "max_lon": 77.0})
    area_coords = f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}"
    
    map_key = os.getenv("FIRMS_MAP_KEY", config.get("api_keys", {}).get("firms", ""))
    source = "VIIRS_SNPP_SP"

    records: List[Dict[str, Any]] = []
    session = get_retry_session(retries=3, backoff_factor=1.0)

    # Priority 1: Direct API with Key if provided
    if map_key and map_key != "DEMO_KEY":
        try:
            url = f"{FIRMS_API_BASE}/{map_key}/{source}/{area_coords}/{day_range}"
            resp = session.get(url, timeout=20)
            if resp.status_code == 200 and "latitude" in resp.text:
                import io
                df_raw = pd.read_csv(io.StringIO(resp.text))
                for _, row in df_raw.iterrows():
                    records.append({
                        "latitude": float(row.get("latitude", 0.0)),
                        "longitude": float(row.get("longitude", 0.0)),
                        "brightness": float(row.get("bright_ti4", row.get("brightness", 0.0))),
                        "frp": float(row.get("frp", 0.0)),
                        "acq_date": str(row.get("acq_date", "")),
                        "acq_time": str(row.get("acq_time", "")),
                        "confidence": str(row.get("confidence", "nominal")),
                        "satellite": str(row.get("satellite", "SNPP")),
                        "instrument": "VIIRS",
                        "fetch_time_utc": fetch_time_iso,
                    })
        except Exception as e:
            print(f"[FIRMS] Area API query warning: {e}. Falling back to open South Asia CSV feed.")

    # Priority 2: NASA Open NRT CSV Feed for South Asia (No Key Required, Public Domain)
    if not records:
        try:
            print("[FIRMS] Querying NASA open NRT 7-day South Asia fire detection feed...")
            resp = session.get(FIRMS_OPEN_SOUTH_ASIA_7D, timeout=30)
            if resp.status_code == 200 and "latitude" in resp.text:
                import io
                df_all = pd.read_csv(io.StringIO(resp.text))
                # Filter to Punjab transboundary bounding box
                punjab_mask = (
                    (df_all["latitude"] >= bbox["min_lat"]) &
                    (df_all["latitude"] <= bbox["max_lat"]) &
                    (df_all["longitude"] >= bbox["min_lon"]) &
                    (df_all["longitude"] <= bbox["max_lon"])
                )
                df_punjab = df_all[punjab_mask]
                for _, row in df_punjab.iterrows():
                    records.append({
                        "latitude": float(row.get("latitude", 0.0)),
                        "longitude": float(row.get("longitude", 0.0)),
                        "brightness": float(row.get("bright_ti4", 300.0)),
                        "frp": float(row.get("frp", 0.0)),
                        "acq_date": str(row.get("acq_date", "")),
                        "acq_time": str(row.get("acq_time", "")),
                        "confidence": str(row.get("confidence", "nominal")),
                        "satellite": str(row.get("satellite", "SNPP")),
                        "instrument": "VIIRS",
                        "fetch_time_utc": fetch_time_iso,
                    })
                print(f"[FIRMS] Filtered {len(records)} real active fires in Punjab bounding box from NASA.")
        except Exception as e:
            print(f"[FIRMS] Error fetching open South Asia fires: {e}")

    # Honest zero-fallback: if records is empty, save empty typed DataFrame
    if not records:
        df = pd.DataFrame(columns=[
            "latitude", "longitude", "brightness", "frp", "acq_date", "acq_time",
            "confidence", "satellite", "instrument", "fetch_time_utc"
        ])
    else:
        df = pd.DataFrame(records)

    out_file = output_dir / f"bronze_firms_{ts_str}.parquet"
    df.to_parquet(out_file, engine="pyarrow", index=False)

    manifest_path = Path(config.get("paths", {}).get("manifest", "data/manifest.csv"))
    write_manifest_row(
        file_path=out_file,
        source="NASA FIRMS",
        publisher="NASA Earthdata / LANCE FIRMS",
        url_or_doi="https://firms.modaps.eosdis.nasa.gov",
        query=f"source={source}&bbox={area_coords}&days={day_range}",
        fetch_time_utc=fetch_time_iso,
        licence="NASA Open Data Policy (Public Domain)",
        manifest_path=manifest_path,
    )

    print(f"[FIRMS] Landed {len(df)} real fire records to {out_file}")
    return out_file

if __name__ == "__main__":
    from flows.hourly import load_config
    cfg = load_config()
    fetch_latest_fires(cfg)
