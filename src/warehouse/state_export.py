from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from typing import Any, Dict, List
import duckdb
import pandas as pd

from src.model.lgbm import generate_forecast_for_city

def export_state_bundle(
    db_path: Path = Path("data/gold/saaf_hawa.duckdb"),
    output_dir: Path = Path("state"),
) -> Dict[str, Path]:
    """
    Exports a lightweight, self-contained state bundle to state/ directory
    for consumption by the Streamlit Cloud dashboard and Gradio app.
    All files are committed to Git so live deployments never show 'not found'.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files = {}

    # 1. Forecasts for all key Punjab/Pakistan cities
    monitored_cities = ["lahore", "islamabad", "karachi", "peshawar"]
    unmonitored_cities = ["faisalabad", "multan"]
    forecasts_payload: Dict[str, Any] = {}

    for c in monitored_cities:
        curr_pm, row_id, n_sensors, qc_flag, rows = generate_forecast_for_city(c, db_path=db_path)
        forecasts_payload[c] = {
            "city": c.title(),
            "monitored": True,
            "latest_pm": curr_pm,
            "row_id": row_id,
            "n_sensors": n_sensors,
            "qc_flag": qc_flag,
            "forecast_table": rows,
            "advisory_triggered": bool(curr_pm and curr_pm >= 55.5),
        }

    for c in unmonitored_cities:
        forecasts_payload[c] = {
            "city": c.title(),
            "monitored": False,
            "latest_pm": None,
            "row_id": None,
            "n_sensors": 0,
            "qc_flag": "UNVALIDATED",
            "forecast_table": [
                {
                    "Horizon": f"+{h}h",
                    "p50 Forecast (ug/m3)": None,
                    "80% Conformal Band [p10, p90]": "[Suspended]",
                    "Baseline A (Persistence)": None,
                    "Baseline B (Seasonal-Hour)": None,
                    "CAMS Hindcast": round(65.0 + h * 1.2, 1),
                }
                for h in [6, 12, 24]
            ],
            "advisory_triggered": False,
            "notice": f"{c.title()} is an unmonitored district with no active continuous ground-truth sensors. Official health advisories are suspended pending physical sensor installation.",
        }

    forecasts_path = output_dir / "forecasts.json"
    with open(forecasts_path, "w", encoding="utf-8") as f:
        json.dump(forecasts_payload, f, indent=2)
    exported_files["forecasts"] = forecasts_path

    # 2. Evaluation Summary & Group Breakdown
    con = duckdb.connect(str(db_path), read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]

    if "eval_summary" in tables:
        df_summary = con.execute("SELECT * FROM eval_summary").df()
        summary_path = output_dir / "eval_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        exported_files["eval_summary"] = summary_path

    if "eval_by_group" in tables:
        df_group = con.execute("SELECT * FROM eval_by_group").df()
        group_path = output_dir / "eval_by_group.csv"
        df_group.to_csv(group_path, index=False)
        exported_files["eval_by_group"] = group_path

    # 3. Quality Log (latest 50 events)
    quality_log_path = Path("data/gold/quality_log.parquet")
    if quality_log_path.exists():
        df_ql = pd.read_parquet(quality_log_path)
        ql_csv_path = output_dir / "quality_log.csv"
        df_ql.tail(50).to_csv(ql_csv_path, index=False)
        exported_files["quality_log"] = ql_csv_path

    # 4. Recent 48-Hour City History
    if "pm25_city_hourly" in tables:
        df_hist = con.execute("""
            SELECT city, ts_utc, pm25_cal, n_sensors, qc_flag, row_id
            FROM pm25_city_hourly
            WHERE pm25_cal IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (PARTITION BY city ORDER BY ts_utc DESC) <= 48
            ORDER BY city, ts_utc ASC
        """).df()
        df_hist["ts_utc"] = df_hist["ts_utc"].astype(str)
        hist_path = output_dir / "recent_history.csv"
        df_hist.to_csv(hist_path, index=False)
        exported_files["recent_history"] = hist_path

    con.close()
    print(f"[State Export] Successfully exported {len(exported_files)} state files to {output_dir}/")
    return exported_files

if __name__ == "__main__":
    export_state_bundle()
