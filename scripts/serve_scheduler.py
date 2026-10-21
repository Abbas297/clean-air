import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import sys

# Ensure root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Disable Prefect analytics telemetry to avoid Windows sqlite locks
os.environ["PREFECT_SERVER_ANALYTICS_ENABLED"] = "false"
os.environ["PREFECT_API_ENABLE_HTTP2"] = "false"

from flows.hourly import saaf_hawa_hourly_flow
from scripts.export_runs import export_prefect_runs

def run_loop():
    print("[Scheduler] Starting Saaf Hawa 24/7 continuous runner...", flush=True)
    print("[Scheduler] Running immediate initial cycle...", flush=True)

    while True:
        try:
            start_utc = datetime.now(timezone.utc)
            print(f"\n[Scheduler Cycle Start] {start_utc.isoformat()}", flush=True)
            
            # Execute flow
            res = saaf_hawa_hourly_flow()
            print(f"[Scheduler Cycle Success] Run Status: {res.get('run_status')} (Duration: {res.get('duration_seconds')}s)", flush=True)

            # Export genuine runs history
            try:
                export_prefect_runs()
            except Exception as e:
                print(f"[Export Error] {e}", flush=True)

            # Auto commit and push state/ to update Streamlit Cloud live
            try:
                import subprocess
                subprocess.run(["git", "add", "state/", "status.json", "prefect_runs.csv", "data/manifest.csv"], check=False)
                commit_res = subprocess.run(
                    ["git", "commit", "-m", f"chore(state): hourly heartbeat update [{start_utc.strftime('%Y%m%dT%H%MZ')}] [skip ci]"],
                    capture_output=True,
                    text=True,
                )
                if "nothing to commit" not in commit_res.stdout.lower() and "nothing to commit" not in commit_res.stderr.lower():
                    # Pull with rebase and autostash before push
                    pull_res = subprocess.run(
                        ["git", "pull", "--rebase", "--autostash", "origin", "main"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if pull_res.returncode != 0:
                        print(f"[Scheduler Git Warning] Pull --rebase stderr: {pull_res.stderr.strip()}", flush=True)

                    push_res = subprocess.run(
                        ["git", "push", "origin", "main"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if push_res.returncode == 0:
                        print("[Scheduler Git] Successfully pushed updated state to origin/main", flush=True)
                    else:
                        print(f"[Scheduler Git Error] Push failed (code {push_res.returncode}):\n{push_res.stderr.strip()}", flush=True)
                else:
                    print("[Scheduler Git] State unchanged, no commit needed.", flush=True)
            except Exception as ge:
                print(f"[Scheduler Git Notice] {ge}", flush=True)

        except Exception as e:
            print(f"[Scheduler Error] Cycle encountered exception: {e}", flush=True)

        # Sleep until the top of the next hour
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        sleep_seconds = max(60, int((next_hour - now).total_seconds()))
        print(f"[Scheduler Sleep] Next cycle scheduled at {next_hour.isoformat()} (sleeping {sleep_seconds} seconds)...", flush=True)
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    run_loop()
