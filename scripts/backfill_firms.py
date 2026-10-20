"""Backfill NASA FIRMS active-fire detections for the Punjab bounding box.

Why: the hourly job only downloads the last 7 days, so the fire features used for
training were zero for almost the entire history. This script lands real historical
detections into Bronze (with manifest rows) so Silver/Gold can build real lags.

Usage (needs a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/map_key/):
    set FIRMS_MAP_KEY=your_key        (Windows)   |   export FIRMS_MAP_KEY=your_key
    python scripts/backfill_firms.py --start 2023-01-01
Limits: the Area API returns at most 5 days per call; the free key allows 5,000
requests per 10 minutes. ~3.8 years is about 280 calls per sensor product.
"""
import argparse, io, os, sys, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flows.hourly import load_config
from src.ingest.manifest import write_manifest_row

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
KEEP = ["latitude", "longitude", "brightness", "frp", "acq_date", "acq_time",
        "confidence", "satellite", "instrument", "fetch_time_utc"]


def fetch_chunk(key, source, bbox, start, days):
    url = f"{API}/{key}/{source}/{bbox}/{days}/{start.isoformat()}"
    for attempt in range(3):
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and "latitude" in r.text[:300]:
            return pd.read_csv(io.StringIO(r.text))
        if r.status_code == 200:      # valid empty answer
            return pd.DataFrame()
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"FIRMS {source} {start}: HTTP {r.status_code} {r.text[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--source", default="VIIRS_SNPP_SP",
                    help="Standard-processing archive. Use VIIRS_SNPP_NRT for the last ~2 months if SP has not caught up.")
    a = ap.parse_args()
    key = os.getenv("FIRMS_MAP_KEY", "")
    if not key:
        sys.exit("Set FIRMS_MAP_KEY first (free: https://firms.modaps.eosdis.nasa.gov/api/map_key/).")

    cfg = load_config()
    b = cfg["firms_bounding_box"]
    bbox = f"{b['min_lon']},{b['min_lat']},{b['max_lon']},{b['max_lat']}"
    fetch_time = datetime.now(timezone.utc).isoformat()

    frames, d, end = [], date.fromisoformat(a.start), date.fromisoformat(a.end)
    while d <= end:
        n = min(5, (end - d).days + 1)
        df = fetch_chunk(key, a.source, bbox, d, n)
        if len(df):
            frames.append(df)
        print(f"{d} +{n}d -> {len(df)} detections", flush=True)
        d += timedelta(days=n)
        time.sleep(0.2)

    if not frames:
        sys.exit("No detections returned; nothing written.")
    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "latitude": raw["latitude"], "longitude": raw["longitude"],
        "brightness": raw.get("bright_ti4", raw.get("brightness")),
        "frp": raw["frp"], "acq_date": raw["acq_date"].astype(str),
        "acq_time": raw["acq_time"].astype(str),
        "confidence": raw.get("confidence", "nominal").astype(str),
        "satellite": raw.get("satellite", "SNPP").astype(str),
        "instrument": "VIIRS", "fetch_time_utc": fetch_time,
    })[KEEP].drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])

    out_dir = Path(cfg["paths"]["bronze_firms"]); out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"bronze_firms_backfill_{a.start.replace('-','')}_{a.end.replace('-','')}.parquet"
    out.to_parquet(f, index=False)
    write_manifest_row(
        file_path=f, source="NASA FIRMS (Area API, historical backfill)",
        publisher="NASA Earthdata / LANCE FIRMS", url_or_doi="https://firms.modaps.eosdis.nasa.gov/api/area/",
        query=f"source={a.source}&bbox={bbox}&start={a.start}&end={a.end}",
        fetch_time_utc=fetch_time, licence="NASA Open Data Policy (Public Domain)",
        manifest_path=Path(cfg["paths"]["manifest"]),
    )
    print(f"Wrote {len(out)} detections -> {f}")


if __name__ == "__main__":
    main()
