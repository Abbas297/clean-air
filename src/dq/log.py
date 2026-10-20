from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

QUALITY_LOG_COLUMNS = [
    "ts",
    "run_id",
    "gate",
    "check",
    "rule",
    "action",
    "reason",
    "evidence_json",
    "table",
    "rows_affected",
]

def append_quality_log(
    run_id: str,
    gate: str,
    check: str,
    rule: str,
    action: str,
    reason: str,
    evidence: Dict[str, Any],
    table: str,
    rows_affected: int,
    log_path: Path = Path("data/gold/quality_log.parquet"),
) -> None:
    """
    Appends an adjudicated decision row into quality_log.parquet.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = {
        "ts": now_utc,
        "run_id": run_id,
        "gate": gate,
        "check": check,
        "rule": rule,
        "action": action,
        "reason": reason,
        "evidence_json": json.dumps(evidence),
        "table": table,
        "rows_affected": int(rows_affected),
    }

    df_new = pd.DataFrame([new_row])

    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            df_existing = pd.read_parquet(log_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_parquet(log_path, engine="pyarrow", index=False)
