import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import pandas as pd
from src.model.lgbm import train_quantile_models, build_feature_matrix, get_feature_columns
from src.model.evaluate import HORIZONS, fit_cqr_calibration, apply_cqr_adjustment

def build_replay_dataset(output_path: Path = Path("state/replay.csv")):
    con = duckdb.connect("data/gold/saaf_hawa.duckdb")
    df_pm25 = con.execute("SELECT * FROM pm25_city_hourly").df()
    df_features = con.execute("SELECT * FROM features_hourly").df()
    con.close()

    df_pm25["ts_utc"] = pd.to_datetime(df_pm25["ts_utc"], utc=True)
    df_features["ts_utc"] = pd.to_datetime(df_features["ts_utc"], utc=True)

    df_matrix = build_feature_matrix(df_pm25, df_features)
    feature_cols = get_feature_columns(df_matrix)
    models = train_quantile_models(df_matrix)

    val_start = pd.to_datetime("2024-07-02T00:00:00Z", utc=True)
    val_end = pd.to_datetime("2025-06-30T23:59:59Z", utc=True)
    df_val = df_matrix[(df_matrix["ts_utc"] >= val_start) & (df_matrix["ts_utc"] <= val_end)]

    # Target: Lahore peak smog episode in test split (Nov 1 to Nov 15, 2025)
    ep_start = pd.to_datetime("2025-11-01T00:00:00Z", utc=True)
    ep_end = pd.to_datetime("2025-11-15T23:59:59Z", utc=True)
    df_ep = df_matrix[(df_matrix["city"] == "lahore") & (df_matrix["ts_utc"] >= ep_start) & (df_matrix["ts_utc"] <= ep_end)].copy()

    replay_df = pd.DataFrame({
        "city": "Lahore (Historical Test Episode)",
        "ts_utc": df_ep["ts_utc"].dt.strftime("%Y-%m-%d %H:%M:%SZ"),
        "ts_pkt": df_ep["ts_utc"].dt.tz_convert("Asia/Karachi").dt.strftime("%Y-%m-%d %H:%M"),
        "actual_pm25": df_ep["pm25_cal"].round(1),
    })

    for h in HORIZONS:
        target_col = f"target_h{h}"
        p10_model = models[f"lgbm_h{h}_q10"]
        p50_model = models[f"lgbm_h{h}_q50"]
        p90_model = models[f"lgbm_h{h}_q90"]

        valid_val = df_val.dropna(subset=[target_col])
        q10_val = p10_model.predict(valid_val[feature_cols])
        q90_val = p90_model.predict(valid_val[feature_cols])
        q_corr = fit_cqr_calibration(valid_val[target_col].values, q10_val, q90_val, 0.80)

        p50 = p50_model.predict(df_ep[feature_cols])
        p10 = p10_model.predict(df_ep[feature_cols])
        p90 = p90_model.predict(df_ep[feature_cols])
        p10_adj, p90_adj = apply_cqr_adjustment(p10, p90, q_corr)

        replay_df[f"forecast_p50_h{h}"] = p50.round(1)
        replay_df[f"band_p10_h{h}"] = p10_adj.round(1)
        replay_df[f"band_p90_h{h}"] = p90_adj.round(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    replay_df.to_csv(output_path, index=False)
    print(f"[Replay] Historical test episode written ({len(replay_df)} hours) -> {output_path}")

if __name__ == "__main__":
    build_replay_dataset()
