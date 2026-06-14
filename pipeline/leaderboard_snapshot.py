#!/usr/bin/env python3
"""
Download the current Orbit Wars public leaderboard and save a timestamped
CSV snapshot to leaderboards/.

Usage:
    .venv/bin/python pipeline/leaderboard_snapshot.py

Designed to run on a cron (e.g. hourly) via run_pipeline.sh.
"""

import sqlite3
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "strategy" / "tracking.db"
LB_DIR = ROOT / "leaderboards"
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"
COMPETITION = "orbit-wars"


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
            filename  TEXT PRIMARY KEY,
            taken_at  TEXT
        )
    """)
    con.commit()
    return con


def main():
    LB_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    # kaggle downloads a zip; extract the CSV inside
    result = subprocess.run(
        [str(KAGGLE), "competitions", "leaderboard", COMPETITION,
         "--download", "-p", str(LB_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] kaggle leaderboard download failed:\n{result.stderr}")
        return

    # Find the downloaded zip (named orbit-wars.zip)
    zips = list(LB_DIR.glob("*.zip"))
    if not zips:
        print("[ERROR] No zip file found after download.")
        return

    zip_path = zips[0]
    dest_csv = LB_DIR / f"leaderboard_{timestamp}.csv"

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        csvs = [m for m in members if m.endswith(".csv")]
        if not csvs:
            print(f"[ERROR] No CSV inside {zip_path.name}")
            return
        # Extract and rename
        extracted = zf.extract(csvs[0], path=LB_DIR)
        Path(extracted).rename(dest_csv)

    zip_path.unlink()  # clean up the zip

    print(f"Saved: {dest_csv.name}")

    con = get_db()
    con.execute(
        "INSERT OR IGNORE INTO leaderboard_snapshots (filename, taken_at) VALUES (?, ?)",
        (dest_csv.name, now.isoformat())
    )
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
