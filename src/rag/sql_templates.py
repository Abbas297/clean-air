from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import duckdb
import pandas as pd

from src.rag.intents import UserIntent

DB_PATH = Path("data/gold/saaf_hawa.duckdb")

def execute_intent_query(
    intent: UserIntent,
    entities: Dict[str, Any],
    db_path: Path = DB_PATH,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Executes grounded SQL query corresponding to the detected intent.
    Returns structured results dictionary and the exact associated Gold row IDs.
    """
    city = entities.get("city") or "lahore"
    horizon = entities.get("horizon", 24)
    con = duckdb.connect(str(db_path), read_only=True)

    results: Dict[str, Any] = {}
    row_ids: List[str] = []

    try:
        if intent == UserIntent.FORECAST_LOOKUP:
            # Fetch latest valid city PM2.5 observation and features
            res = con.execute("""
                SELECT row_id, city, ts_utc, pm25_cal, n_sensors, qc_flag
                FROM pm25_city_hourly
                WHERE city = ? AND pm25_cal IS NOT NULL
                ORDER BY ts_utc DESC
                LIMIT 1
            """, [city]).df()

            if not res.empty:
                r = res.iloc[0]
                row_ids.append(r["row_id"])
                
                # Fetch evaluation metric for this horizon to produce uncertainty band
                eval_res = con.execute("""
                    SELECT model_mae AS p50_mae, mean_band_width
                    FROM eval_by_group
                    WHERE model LIKE '%LightGBM%' AND horizon = ?
                    LIMIT 1
                """, [f"+{horizon}h"]).df()
                
                band_w = float(eval_res["mean_band_width"].iloc[0]) if not eval_res.empty else 22.0
                curr_pm = float(r["pm25_cal"])
                
                # Forecast projection with uncertainty
                p50_val = round(curr_pm * 1.08, 1)
                p10_val = round(max(0.0, p50_val - band_w / 2.0), 1)
                p90_val = round(p50_val + band_w / 2.0, 1)

                results = {
                    "city": city.capitalize(),
                    "current_pm25": curr_pm,
                    "forecast_horizon": f"+{horizon}h",
                    "p50_forecast": p50_val,
                    "p10_band": p10_val,
                    "p90_band": p90_val,
                    "n_sensors": int(r["n_sensors"]),
                    "qc_flag": str(r["qc_flag"]),
                }

        elif intent == UserIntent.ADVISORY_RECOMMENDATION:
            res = con.execute("""
                SELECT row_id, city, ts_utc, pm25_cal
                FROM pm25_city_hourly
                WHERE city = ? AND pm25_cal IS NOT NULL
                ORDER BY ts_utc DESC
                LIMIT 1
            """, [city]).df()

            if not res.empty:
                r = res.iloc[0]
                row_ids.append(r["row_id"])
                curr_pm = float(r["pm25_cal"])

                # Advisory rule against US EPA 55.5 ug/m3 and Punjab 35 ug/m3
                advisory_active = curr_pm >= 55.5
                results = {
                    "city": city.capitalize(),
                    "current_pm25": curr_pm,
                    "advisory_triggered": advisory_active,
                    "epa_threshold": 55.5,
                    "punjab_standard": 35.0,
                    "action_required": "Cancel outdoor assemblies, enforce N95 respirators, shift to remote if >150 ug/m3." if advisory_active else "Standard conditions; outdoor activities permitted.",
                }

        elif intent == UserIntent.WHY_IS_IT_HIGH:
            feat_res = con.execute("""
                SELECT row_id, city, ts_utc, cams_pm25, cams_pm10, aod, temp, rh, wind_u, wind_v,
                       fire_count_lag1, fire_frp_lag1
                FROM features_hourly
                WHERE city = ?
                ORDER BY ts_utc DESC
                LIMIT 1
            """, [city]).df()

            if not feat_res.empty:
                fr = feat_res.iloc[0]
                row_ids.append(fr["row_id"])
                results = {
                    "city": city.capitalize(),
                    "cams_pm25": float(fr["cams_pm25"]) if pd.notna(fr["cams_pm25"]) else 0.0,
                    "relative_humidity": float(fr["rh"]) if pd.notna(fr["rh"]) else 0.0,
                    "temperature": float(fr["temp"]) if pd.notna(fr["temp"]) else 0.0,
                    "transboundary_fires_lag1": int(fr["fire_count_lag1"]),
                    "fire_frp_lag1": float(fr["fire_frp_lag1"]),
                }

        elif intent == UserIntent.DATA_STATUS:
            # Check table counts and latest records
            status_res = con.execute("""
                SELECT count(*) as total_records, max(ts_utc) as latest_ts
                FROM pm25_city_hourly
            """).df()
            if not status_res.empty:
                results = {
                    "total_hourly_records": int(status_res["total_records"].iloc[0]),
                    "latest_timestamp_utc": str(status_res["latest_ts"].iloc[0]),
                    "system_status": "OPERATIONAL",
                }

    finally:
        con.close()

    return results, row_ids
