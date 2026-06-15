#!/usr/bin/env python3
"""
Discover new episodes for tracked Kaggle submissions and record them in
strategy/tracking.db.

Usage:
    .venv/bin/python pipeline/pull_episodes.py
    .venv/bin/python pipeline/pull_episodes.py --sub 53676680

Adds new PUBLIC episodes to the DB (VALIDATION games are skipped).
Prints newly discovered episode IDs to stdout so downstream scripts can
chain on them: python pull_episodes.py | python download_replays.py --stdin
"""

import argparse
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "strategy" / "tracking.db"
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"

OUR_NAME = "Montana Schmeekler"

# Kaggle keeps only the most-recent submissions live on the ladder. We auto-detect
# the active bots from the submission list (newest first) — no manual ID editing.
ACTIVE_SLOTS = 2


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id    TEXT PRIMARY KEY,
            submission_id TEXT,
            sub_name      TEXT,
            create_time   TEXT,
            end_time      TEXT,
            episode_type  TEXT,
            discovered_at TEXT,
            downloaded    INTEGER DEFAULT 0,
            our_placement INTEGER,
            num_players   INTEGER,
            our_reward    REAL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
            filename  TEXT PRIMARY KEY,
            taken_at  TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id TEXT PRIMARY KEY,
            name          TEXT,
            status        TEXT DEFAULT 'active',
            submitted_at  TEXT,
            updated_at    TEXT,
            public_score  REAL
        )
    """)
    # Add public_score column to existing DBs that predate this schema
    try:
        con.execute("ALTER TABLE submissions ADD COLUMN public_score REAL")
    except Exception:
        pass
    con.commit()
    return con


# ---------------------------------------------------------------------------
# Submission syncing
# ---------------------------------------------------------------------------

def sync_submissions(con: sqlite3.Connection) -> None:
    """Pull all submissions from Kaggle and upsert into the submissions table."""
    result = subprocess.run(
        [str(KAGGLE), "competitions", "submissions", "orbit-wars"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [WARN] kaggle submissions failed: {result.stderr.strip()}")
        return

    now = datetime.now(timezone.utc).isoformat()
    lines = result.stdout.strip().splitlines()
    # Skip header (line 0) and separator (line 1)
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) < 3:
            continue
        sub_id   = parts[0].strip()
        # parts[1] is fileName (always main.py), parts[2] is date, parts[3] is description
        if not sub_id.isdigit():
            continue
        submitted_at = parts[2].strip() if len(parts) > 2 else ""
        name         = parts[3].strip() if len(parts) > 3 else sub_id
        public_score = None
        if len(parts) > 5:
            try:
                public_score = float(parts[5].strip())
            except (ValueError, IndexError):
                pass
        con.execute(
            """INSERT INTO submissions (submission_id, name, status, submitted_at, updated_at, public_score)
               VALUES (?, ?, 'active', ?, ?, ?)
               ON CONFLICT(submission_id) DO UPDATE SET
                   name=excluded.name,
                   public_score=excluded.public_score,
                   updated_at=excluded.updated_at""",
            (sub_id, name, submitted_at, now, public_score)
        )
    # Auto-detect active vs retired: the ACTIVE_SLOTS most-recent submissions are
    # the ones Kaggle keeps live on the ladder; everything older is retired.
    active_ids = [r[0] for r in con.execute(
        "SELECT submission_id FROM submissions ORDER BY submitted_at DESC LIMIT ?",
        (ACTIVE_SLOTS,)).fetchall()]
    if active_ids:
        ph = ",".join("?" * len(active_ids))
        con.execute(f"UPDATE submissions SET status='retired' WHERE submission_id NOT IN ({ph})", active_ids)
        con.execute(f"UPDATE submissions SET status='active'  WHERE submission_id IN ({ph})", active_ids)
    con.commit()
    print(f"  Submissions synced (active: {active_ids}).")


# ---------------------------------------------------------------------------
# Episode fetching
# ---------------------------------------------------------------------------

def fetch_episodes(submission_id: str) -> list[dict]:
    """Run kaggle CLI and parse the episode list for a submission."""
    result = subprocess.run(
        [str(KAGGLE), "competitions", "episodes", submission_id],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [WARN] kaggle episodes {submission_id} failed: {result.stderr.strip()}")
        return []

    episodes = []
    lines = result.stdout.strip().splitlines()
    # Skip header (first line) and trailing hint line
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("Use "):
            continue
        # Columns: id  createTime  endTime  state  type
        # Separated by 2+ spaces
        parts = re.split(r"\s{2,}", line)
        if len(parts) < 5:
            continue
        ep_id       = parts[0].strip()
        create_time = parts[1].strip()
        end_time    = parts[2].strip()
        ep_type     = parts[4].strip()  # e.g. EpisodeType.EPISODE_TYPE_PUBLIC
        if not ep_id.isdigit():
            continue
        episodes.append({
            "episode_id":    ep_id,
            "create_time":   create_time,
            "end_time":      end_time,
            "episode_type":  ep_type,
        })
    return episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pull new episodes from Kaggle")
    parser.add_argument("--sub", help="Only check this submission ID")
    args = parser.parse_args()

    con = get_db()
    now = datetime.now(timezone.utc).isoformat()
    new_ids: list[str] = []

    # Sync submission list first — this upserts all submissions and sets active/retired
    # status (latest ACTIVE_SLOTS = active). Then pull episodes for the ACTIVE ones.
    sync_submissions(con)

    if args.sub:
        rows = con.execute(
            "SELECT submission_id, name FROM submissions WHERE submission_id=?", (args.sub,)
        ).fetchall()
        subs = {r[0]: r[1] for r in rows} or {args.sub: args.sub}
    else:
        rows = con.execute(
            "SELECT submission_id, name FROM submissions WHERE status='active' "
            "ORDER BY submitted_at DESC"
        ).fetchall()
        subs = {r[0]: r[1] for r in rows}

    for sub_id, sub_name in subs.items():
        print(f"\n=== {sub_name} ({sub_id}) ===")
        episodes = fetch_episodes(sub_id)
        print(f"  Found {len(episodes)} episodes from Kaggle")

        for ep in episodes:
            # Skip VALIDATION / non-PUBLIC episodes
            if "PUBLIC" not in ep["episode_type"]:
                continue
            ep_id = ep["episode_id"]
            existing = con.execute(
                "SELECT episode_id FROM episodes WHERE episode_id = ?", (ep_id,)
            ).fetchone()
            if existing:
                continue
            con.execute(
                """INSERT INTO episodes
                   (episode_id, submission_id, sub_name, create_time, end_time,
                    episode_type, discovered_at, downloaded)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                (ep_id, sub_id, sub_name,
                 ep["create_time"], ep["end_time"],
                 ep["episode_type"], now)
            )
            new_ids.append(ep_id)

    con.commit()
    con.close()

    if new_ids:
        print(f"\nNew episodes discovered: {len(new_ids)}")
        for eid in new_ids:
            print(eid)
    else:
        print("\nNo new episodes.")

    return new_ids


if __name__ == "__main__":
    main()
