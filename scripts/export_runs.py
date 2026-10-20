import asyncio
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List

PREFECT_RUNS_CSV = Path("prefect_runs.csv")

async def _fetch_prefect_runs() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    try:
        from prefect.client.orchestration import get_client
        from prefect.client.schemas.sorting import FlowRunSort

        async with get_client() as client:
            prefect_runs = await client.read_flow_runs(
                sort=FlowRunSort.EXPECTED_START_TIME_DESC,
                limit=100,
            )
            for r in prefect_runs:
                runs.append({
                    "flow_run_id": str(r.id),
                    "name": str(r.name),
                    "state": str(r.state_name),
                    "start_time": r.start_time.isoformat() if r.start_time else "",
                    "end_time": r.end_time.isoformat() if r.end_time else "",
                    "total_run_time": round(r.total_run_time.total_seconds(), 2) if r.total_run_time else 0.0,
                })
    except Exception as e:
        print(f"[Prefect Client] Query note: {e}")
    return runs

def export_prefect_runs(output_path: Path = PREFECT_RUNS_CSV) -> Path:
    """
    Exports unedited flow run history directly from the Prefect engine
    and appends new runs to prefect_runs.csv without duplicates.
    Zero synthetic fallbacks.
    """
    runs = asyncio.run(_fetch_prefect_runs())

    # Read existing runs if file exists to prevent losing previous run records
    existing_runs_by_id: Dict[str, Dict[str, Any]] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_id = row.get("flow_run_id")
                if run_id:
                    existing_runs_by_id[run_id] = row

    # Add fetched genuine Prefect runs
    for r in runs:
        run_id = r.get("flow_run_id")
        if run_id:
            existing_runs_by_id[run_id] = r

    # Sort by start_time descending
    all_runs = list(existing_runs_by_id.values())
    all_runs.sort(key=lambda x: str(x.get("start_time", "")), reverse=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["flow_run_id", "name", "state", "start_time", "end_time", "total_run_time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_runs)

    # Mirror to state/ for dashboard tracking
    state_runs_path = Path("state/prefect_runs.csv")
    state_runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_runs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_runs)

    print(f"Exported {len(all_runs)} genuine flow runs (appended without duplicates) to {output_path}")
    return output_path

if __name__ == "__main__":
    export_prefect_runs()
