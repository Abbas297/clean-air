import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
try:
    from sklearn.linear_model import Ridge
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def fit_and_apply_calibration(
    config: Dict[str, Any],
    train_end_date: str = "2024-06-30T23:59:59Z",
) -> Tuple[Path, Dict[str, Any]]:
    """
    Fits sensor-to-reference calibration strictly on the Train partition (pre-July 2024).
    Uses Ridge regression with raw PM2.5 and relative humidity interactions.
    Applies the fitted model across the entire dataset to produce pm25_cal.
    """
    silver_dir = Path(config.get("paths", {}).get("silver", "data/silver"))
    sensors_file = silver_dir / "silver_sensor_readings.parquet"
    embassy_file = silver_dir / "silver_embassy_reference.parquet"
    meteo_file = silver_dir / "silver_cams_weather.parquet"

    if not sensors_file.exists() or not embassy_file.exists():
        raise FileNotFoundError("Silver sensor readings or embassy reference not found for calibration.")

    df_sensors = pd.read_parquet(sensors_file)
    df_embassy = pd.read_parquet(embassy_file)
    df_meteo = pd.read_parquet(meteo_file) if meteo_file.exists() else pd.DataFrame()

    df_sensors["ts_utc"] = pd.to_datetime(df_sensors["ts_utc"], utc=True)
    df_embassy["ts_utc"] = pd.to_datetime(df_embassy["ts_utc"], utc=True)

    # Attach humidity from meteo if available
    if not df_meteo.empty and "rh_2m" in df_meteo.columns:
        df_meteo["ts_utc"] = pd.to_datetime(df_meteo["ts_utc"], utc=True)
        rh_lookup = df_meteo[["city", "ts_utc", "rh_2m"]].drop_duplicates(subset=["city", "ts_utc"])
        df_sensors = df_sensors.merge(rh_lookup, on=["city", "ts_utc"], how="left")
    
    if "rh_2m" not in df_sensors.columns or df_sensors["rh_2m"].isna().all():
        df_sensors["rh_2m"] = 65.0  # nominal default RH
    else:
        df_sensors["rh_2m"] = df_sensors["rh_2m"].fillna(65.0)

    # 1. Join on Train split: city and ts_utc
    train_cutoff = pd.to_datetime(train_end_date, utc=True)
    
    # Merge on city and timestamp
    df_train_pairs = df_sensors[df_sensors["ts_utc"] <= train_cutoff].merge(
        df_embassy[["city", "ts_utc", "pm25_reference"]],
        on=["city", "ts_utc"],
        how="inner",
    ).dropna(subset=["pm25_raw", "pm25_reference"])

    model_meta = {
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_end_cutoff": train_end_date,
        "n_train_samples": int(len(df_train_pairs)),
        "coefficients_by_city": {},
    }

    calibrated_dfs = []
    
    # Fit city-specific or pooled calibration
    cities = config.get("cities", {})
    for city_key in cities.keys():
        city_train = df_train_pairs[df_train_pairs["city"] == city_key]
        city_all = df_sensors[df_sensors["city"] == city_key].copy()

        if len(city_train) >= 30:
            X_train = city_train[["pm25_raw", "rh_2m"]].values
            y_train = city_train["pm25_reference"].values
            
            if HAS_SKLEARN:
                reg = Ridge(alpha=1.0)
                reg.fit(X_train, y_train)
                slope_pm = float(reg.coef_[0])
                slope_rh = float(reg.coef_[1])
                intercept = float(reg.intercept_)
            else:
                # Closed-form Ridge: (X^T X + alpha*I)^(-1) X^T y
                X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
                A = X_b.T @ X_b + 1.0 * np.eye(X_b.shape[1])
                coeffs = np.linalg.solve(A, X_b.T @ y_train)
                intercept = float(coeffs[0])
                slope_pm = float(coeffs[1])
                slope_rh = float(coeffs[2])
        else:
            # Standard physical low-cost optical sensor transfer function
            # Corrects hygroscopic growth at high humidity: ~0.85 slope + humidity offset
            slope_pm = 0.88
            slope_rh = -0.05
            intercept = 2.5

        model_meta["coefficients_by_city"][city_key] = {
            "slope_pm25": slope_pm,
            "slope_rh": slope_rh,
            "intercept": intercept,
        }

        # Apply calibration
        raw_vals = city_all["pm25_raw"].values
        rh_vals = city_all["rh_2m"].values
        
        cal_vals = intercept + slope_pm * raw_vals + slope_rh * (rh_vals - 50.0)
        # Ensure physical non-negativity
        city_all["pm25_cal"] = np.clip(cal_vals, 0.0, 1000.0)
        calibrated_dfs.append(city_all)

    df_calibrated = pd.concat(calibrated_dfs, ignore_index=True) if calibrated_dfs else df_sensors
    if "pm25_cal" not in df_calibrated.columns:
        df_calibrated["pm25_cal"] = df_calibrated["pm25_raw"]

    out_cal_parquet = silver_dir / "silver_sensor_calibrated.parquet"
    df_calibrated.to_parquet(out_cal_parquet, engine="pyarrow", index=False)

    meta_file = silver_dir / "calibration_model.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    print(f"[Calibration] Fitted on Train set (cutoff: {train_end_date}). Calibrated data saved to {out_cal_parquet}")
    return out_cal_parquet, model_meta

if __name__ == "__main__":
    from flows.hourly import load_config
    cfg = load_config()
    fit_and_apply_calibration(cfg)
