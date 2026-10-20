import json
from pathlib import Path
import pandas as pd
import pytest

from src.ingest.manifest import calculate_sha256, MANIFEST_COLUMNS

def test_bronze_files_exist():
    openaq_dir = Path("data/bronze/openaq")
    openmeteo_dir = Path("data/bronze/openmeteo")

    assert openaq_dir.exists(), "Bronze OpenAQ directory does not exist"
    assert openmeteo_dir.exists(), "Bronze Open-Meteo directory does not exist"

    openaq_files = list(openaq_dir.glob("*.parquet"))
    openmeteo_files = list(openmeteo_dir.glob("*.parquet"))

    assert len(openaq_files) >= 1, "No Bronze OpenAQ parquet files found"
    assert len(openmeteo_files) >= 1, "No Bronze Open-Meteo parquet files found"

    assert openaq_files[0].stat().st_size > 0, "Bronze OpenAQ parquet file is empty"
    assert openmeteo_files[0].stat().st_size > 0, "Bronze Open-Meteo parquet file is empty"

def test_manifest_integrity():
    manifest_path = Path("data/manifest.csv")
    assert manifest_path.exists(), "data/manifest.csv does not exist"

    df = pd.read_csv(manifest_path)
    assert list(df.columns) == MANIFEST_COLUMNS, f"Manifest columns do not match expected: {df.columns}"
    assert len(df) >= 2, f"Manifest should contain at least 2 entries, found {len(df)}"

    for _, row in df.iterrows():
        p = Path(row["file"])
        if not p.exists():
            # Legacy rows were written with an absolute path on the author's machine.
            # Map them to the repo-relative location; the SHA-256 check below still applies.
            parts = Path(str(row["file"]).replace("\\", "/")).parts
            for anchor in ("data", "src"):
                if anchor in parts:
                    p = Path(*parts[parts.index(anchor):])
                    break
        assert p.exists(), f"File declared in manifest does not exist: {p}"
        real_sha256 = calculate_sha256(p)
        assert real_sha256 == row["sha256"], f"SHA-256 mismatch for {p}: declared {row['sha256']} != actual {real_sha256}"

def test_status_json_validity():
    status_path = Path("status.json")
    assert status_path.exists(), "status.json does not exist"

    with open(status_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("run_status") == "HEALTHY", f"Expected run_status HEALTHY, got {data.get('run_status')}"
    assert "last_heartbeat_utc" in data, "Missing last_heartbeat_utc in status.json"
    assert data.get("manifest_rows_total", 0) >= 2, "Expected at least 2 manifest rows reported in status.json"
