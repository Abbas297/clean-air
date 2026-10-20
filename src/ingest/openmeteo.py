import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from src.ingest.manifest import write_manifest_row

OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENMETEO_WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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
            "past_days": 2,
            "forecast_days": 2,
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
            "past_days": 2,
            "forecast_days": 2,
            "timezone": "UTC",
        }
        wx_resp = requests.get(OPENMETEO_WEATHER_FORECAST_URL, params=wx_params, timeout=15)
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

def fetch_historical_archive(
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2026-10-20",
) -> Path:
    """
    Fetches real multi-year historical ERA5 reanalysis weather & CAMS air quality
    from Open-Meteo Archive API (2023 to 2026).
    Enriches features_hourly from 192 rows to complete multi-thousand-row training matrix.
    """
    now_utc = datetime.now(timezone.utc)
    fetch_time_iso = now_utc.isoformat()

    if output_dir is None:
        output_dir = Path(config.get("paths", {}).get("bronze_openmeteo", "data/bronze/openmeteo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cities = config.get("cities", {})
    all_city_records: List[Dict[str, Any]] = []

    print(f"[Open-Meteo Archive] Ingesting real historical weather & CAMS reanalysis ({start_date} to {end_date})...", flush=True)

    for city_key, city_data in cities.items():
        lat = city_data["latitude"]
        lon = city_data["longitude"]
        city_code = city_data["code"]

        try:
            # 1. Historical weather (ERA5 / reanalysis)
            wx_params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m",
                "timezone": "UTC",
            }
            wx_resp = requests.get(OPENMETEO_WEATHER_ARCHIVE_URL, params=wx_params, timeout=35)
            wx_resp.raise_for_status()
            wx_data = wx_resp.json().get("hourly", {})

            # 2. Historical air quality (CAMS)
            aq_params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "pm2_5,pm10,aerosol_optical_depth",
                "timezone": "UTC",
            }
            aq_resp = requests.get(OPENMETEO_AQ_URL, params=aq_params, timeout=35)
            aq_resp.raise_for_status()
            aq_data = aq_resp.json().get("hourly", {})

            times = wx_data.get("time", [])
            print(f"[Open-Meteo Archive] {city_key.upper()}: retrieved {len(times)} historical hours.", flush=True)

            for i, t in enumerate(times):
                all_city_records.append({
                    "city": city_key,
                    "city_code": city_code,
                    "ts_utc": t,
                    "cams_pm2_5": aq_data.get("pm2_5", [None])[i] if i < len(aq_data.get("pm2_5", [])) else None,
                    "cams_pm10": aq_data.get("pm10", [None])[i] if i < len(aq_data.get("pm10", [])) else None,
                    "aod": aq_data.get("aerosol_optical_depth", [None])[i] if i < len(aq_data.get("aerosol_optical_depth", [])) else None,
                    "temp_2m": wx_data.get("temperature_2m", [None])[i] if i < len(wx_data.get("temperature_2m", [])) else None,
                    "rh_2m": wx_data.get("relative_humidity_2m", [None])[i] if i < len(wx_data.get("relative_humidity_2m", [])) else None,
                    "precip": wx_data.get("precipitation", [None])[i] if i < len(wx_data.get("precipitation", [])) else None,
                    "wind_speed_10m": wx_data.get("wind_speed_10m", [None])[i] if i < len(wx_data.get("wind_speed_10m", [])) else None,
                    "wind_direction_10m": wx_data.get("wind_direction_10m", [None])[i] if i < len(wx_data.get("wind_direction_10m", [])) else None,
                    "fetch_time_utc": fetch_time_iso,
                })
            time.sleep(0.5)
        except Exception as e:
            print(f"[Open-Meteo Archive] Error fetching {city_key}: {e}", flush=True)

    df = pd.DataFrame(all_city_records)
    out_file = output_dir / "bronze_openmeteo_historical_2023_2026.parquet"
    df.to_parquet(out_file, engine="pyarrow", index=False)

    manifest_path = Path(config.get("paths", {}).get("manifest", "data/manifest.csv"))
    write_manifest_row(
        file_path=out_file,
        source="Open-Meteo Archive",
        publisher="Copernicus ECMWF ERA5 & CAMS via Open-Meteo",
        url_or_doi="https://archive-api.open-meteo.com/v1/archive",
        query=f"cities={','.join(cities.keys())}&date_range={start_date}_to_{end_date}",
        fetch_time_utc=fetch_time_iso,
        licence="CC BY 4.0 / Open Government Data",
        manifest_path=manifest_path,
    )

    print(f"[Open-Meteo Archive] Landed {len(df)} real historical feature records to {out_file}", flush=True)
    return out_file

if __name__ == "__main__":
    from flows.hourly import load_config
    cfg = load_config()
    fetch_historical_archive(cfg)
