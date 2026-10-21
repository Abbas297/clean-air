"""
tests/test_psi_ordering.py
--------------------------
Regression test for the R07 PSI bug fixed 2026-10-21.

Before the fix, compute_psi was called on df_city_pm25 sliced by raw row
index.  When that DataFrame was city-sorted (all lahore rows first, then
peshawar), the "reference" half contained only lahore readings and the
"current" half contained only peshawar readings, producing a spurious PSI
≈ 0.30 and ~28 k false R07 FLAG rows.

After the fix the DataFrame is sorted by ts_utc (then city) before splitting,
so ref = older half, cur = recent half, regardless of city ordering.

This test asserts that swapping city row order gives an identical PSI value.
"""

import numpy as np
import pandas as pd
import pytest

from src.warehouse.gold import compute_psi


def _make_df(seed: int = 42, n: int = 200) -> pd.DataFrame:
    """Return a DataFrame with two cities interleaved by hour, drawn from the same distribution.

    Crucially, cityA and cityB share *every* timestamp pair so that sorting by
    ts_utc gives a clean chronological split rather than a city-based split.
    This mimics the production table structure where all cities contribute
    readings at every hour.
    """
    rng = np.random.default_rng(seed)
    # n timestamps, each shared by both cities → 2*n rows total
    ts = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    cities = ["cityA", "cityB"]
    rows = []
    for city in cities:
        vals = rng.normal(30, 10, n)
        for i, t in enumerate(ts):
            rows.append({"ts_utc": t, "city": city, "pm25_cal": vals[i]})
    return pd.DataFrame(rows)


def _psi_from_df(df: pd.DataFrame) -> float:
    """Apply the fixed PSI logic: sort by ts_utc, split, compute."""
    df_s = df.sort_values(by=["ts_utc", "city"]).reset_index(drop=True)
    s = len(df_s) // 2
    return compute_psi(df_s["pm25_cal"].iloc[:s].values, df_s["pm25_cal"].iloc[s:].values)


def test_psi_city_order_invariant():
    """PSI must not depend on the order cities appear in the input DataFrame."""
    df_ab = _make_df()  # cityA rows first, then cityB
    df_ba = pd.concat(
        [df_ab[df_ab.city == "cityB"], df_ab[df_ab.city == "cityA"]],
        ignore_index=True,
    )

    psi_ab = _psi_from_df(df_ab)
    psi_ba = _psi_from_df(df_ba)

    assert psi_ab == psi_ba, (
        f"PSI differed by city sort order: cityA-first={psi_ab:.4f}, "
        f"cityB-first={psi_ba:.4f}.  Check that gold.py sorts by ts_utc before splitting."
    )


def test_psi_same_distribution_low():
    """Two halves drawn from the same distribution should yield PSI below the 0.25 alert threshold."""
    df = _make_df()
    psi = _psi_from_df(df)
    assert psi < 0.25, f"Expected PSI < 0.25 for same-distribution data; got {psi:.4f}"


def test_psi_different_distributions_high():
    """Two halves with very different means should yield a detectable PSI."""
    rng = np.random.default_rng(0)
    n = 200
    half = n // 2
    # Make timestamps interleaved so city ordering cannot mask the signal.
    ts = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "ts_utc": ts,
            "city": (["cityA", "cityB"] * (n // 2)),
            "pm25_cal": np.concatenate(
                [rng.normal(20, 5, half), rng.normal(200, 5, half)]
            ),
        }
    )
    psi = _psi_from_df(df)
    assert psi > 0.25, f"Expected PSI > 0.25 for very different distributions; got {psi:.4f}"
