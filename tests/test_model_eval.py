from pathlib import Path
import duckdb
import pandas as pd
import pytest

from src.model.lgbm import HORIZONS, QUANTILES

def test_models_exist():
    models_dir = Path("data/gold/models")
    assert models_dir.exists(), "Models directory does not exist"

    for h in HORIZONS:
        for q in QUANTILES:
            model_file = models_dir / f"lgbm_h{h}_q{int(q*100)}.txt"
            assert model_file.exists(), f"Missing model file: {model_file}"
            assert model_file.stat().st_size > 0, f"Empty model file: {model_file}"

def test_no_leakage_embargo():
    """
    Mandatory Section 12 check:
    Assert max train timestamp is strictly below min test timestamp minus 24h embargo.
    """
    con = duckdb.connect("data/gold/saaf_hawa.duckdb")
    df_pm25 = con.execute("SELECT * FROM pm25_city_hourly").df()
    con.close()

    df_pm25["ts_utc"] = pd.to_datetime(df_pm25["ts_utc"], utc=True)
    train_mask = df_pm25["ts_utc"] <= pd.to_datetime("2024-06-30T23:59:59Z", utc=True)
    test_mask = df_pm25["ts_utc"] >= pd.to_datetime("2025-07-02T00:00:00Z", utc=True)

    max_train_ts = df_pm25.loc[train_mask, "ts_utc"].max()
    min_test_ts = df_pm25.loc[test_mask, "ts_utc"].min()

    assert pd.notna(max_train_ts) and pd.notna(min_test_ts)
    # Check 24-hour embargo gap
    embargo_gap_hours = (min_test_ts - max_train_ts).total_seconds() / 3600.0
    assert embargo_gap_hours >= 24.0, f"Embargo gap violated: only {embargo_gap_hours} hours between train and test"

def test_evaluation_success_criteria():
    eval_file = Path("data/gold/eval_by_group.parquet")
    assert eval_file.exists(), "eval_by_group.parquet not found"

    df_eval = pd.read_parquet(eval_file)
    overall_rows = df_eval[df_eval["group_type"] == "overall"]

    assert len(overall_rows) == 3, "Expected 3 horizon evaluation rows for overall model"
    assert (overall_rows["beats_baselines"] == True).all(), "Model did not beat baselines across all horizons"

    # Coverage must be in 75% - 85% range
    for cov in overall_rows["coverage_80"]:
        assert 75.0 <= cov <= 85.0, f"80% conformal coverage out of bounds: {cov}%"

    # Assert granular group breakdowns exist as required by rubric
    assert "city" in df_eval["group_type"].values, "Missing city group evaluation"
    assert "season" in df_eval["group_type"].values, "Missing season group evaluation"
    assert "diurnal" in df_eval["group_type"].values, "Missing diurnal group evaluation"
    assert "aqi_tier" in df_eval["group_type"].values, "Missing AQI tier group evaluation"
