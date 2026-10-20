from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from src.ingest.manifest import write_manifest_row

OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENMETEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_latest(
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Fetches the latest CAMS air quality and weather variables for monitored Pakistani cities
    from Open-Meteo and stores raw records in a Bronze Parquet file.
    """
    now_utc = datetime.now(timezone.utc)
    ts_str = now_utc.strftime("%Y%m%dT%H%MZ")
    fetch_time_iso = now_utc.isoformat()

    if output_dir is None:
        output_dir = Path(config.get("paths", {}).get("bronze_openmeteo", "data/bronze/openmeteo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cities = config.get("cities", {})
    all_city_records: List[Dict[str, Any]] = []

    for city_key, city_data in cities.items():
        lat = city_data["latitude"]
        lon = city_data["longitude"]
        city_code = city_data["code"]

        # Fetch Air Quality (CAMS)
        aq_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5,pm10,aerosol_optical_depth",
            "past_days": 1,
            "forecast_days": 1,
            "timezone": "UTC",
        }
        aq_resp = requests.get(OPENMETEO_AQ_URL, params=aq_params, timeout=15)
        aq_resp.raise_for_status()
        aq_json = aq_resp.json()

        # Fetch Weather
        wx_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m",
            "past_days": 1,
            "forecast_days": 1,
            "timezone": "UTC",
        }
        wx_resp = requests.get(OPENMETEO_WEATHER_URL, params=wx_params, timeout=15)
        wx_resp.raise_for_status()
        wx_json = wx_resp.json()

        aq_hourly = aq_json.get("hourly", {})
        wx_hourly = wx_json.get("hourly", {})
        times = aq_hourly.get("time", [])

        for i, t in enumerate(times):
            rec = {
                "city": city_key,
                "city_code": city_code,
                "ts_utc": t,
                "cams_pm2_5": aq_hourly.get("pm2_5", [None])[i] if i < len(aq_hourly.get("pm2_5", [])) else None,
                "cams_pm10": aq_hourly.get("pm10", [None])[i] if i < len(aq_hourly.get("pm10", [])) else None,
                "aod": aq_hourly.get("aerosol_optical_depth", [None])[i] if i < len(aq_hourly.get("aerosol_optical_depth", [])) else None,
                "temp_2m": wx_hourly.get("temperature_2m", [None])[i] if i < len(wx_hourly.get("temperature_2m", [])) else None,
                "rh_2m": wx_hourly.get("relative_humidity_2m", [None])[i] if i < len(wx_hourly.get("relative_humidity_2m", [])) else None,
                "precip": wx_hourly.get("precipitation", [None])[i] if i < len(wx_hourly.get("precipitation", [])) else None,
                "wind_speed_10m": wx_hourly.get("wind_speed_10m", [None])[i] if i < len(wx_hourly.get("wind_speed_10m", [])) else None,
                "wind_direction_10m": wx_hourly.get("wind_direction_10m", [None])[i] if i < len(wx_hourly.get("wind_direction_10m", [])) else None,
                "fetch_time_utc": fetch_time_iso,
            }
            all_city_records.append(rec)

    df = pd.DataFrame(all_city_records)
    out_file = output_dir / f"bronze_openmeteo_{ts_str}.parquet"
    df.to_parquet(out_file, engine="pyarrow", index=False)

    manifest_path = Path(config.get("paths", {}).get("manifest", "data/manifest.csv"))
    write_manifest_row(
        file_path=out_file,
        source="Open-Meteo",
        publisher="Copernicus ECMWF & DWD (Open-Meteo)",
        url_or_doi="https://open-meteo.com/en/docs/air-quality-api",
        query=f"cities={','.join(cities.keys())}&variables=cams_pm25,cams_pm10,aod,weather",
        fetch_time_utc=fetch_time_iso,
        licence="CC BY 4.0 (Non-commercial attribution)",
        manifest_path=manifest_path,
    )

    return out_file
