#!/usr/bin/env python3
"""
Download replay JSON for episodes that haven't been fetched yet.

Usage:
    .venv/bin/python pipeline/download_replays.py            # all pending
    .venv/bin/python pipeline/download_replays.py --limit 20
    .venv/bin/python pipeline/download_replays.py --episode 79926268

Replays are saved to replays/YYYY-MM-DD/<episode_id>.json and marked as
downloaded in strategy/tracking.db. Our final placement and reward are also
parsed from the JSON and stored in the DB for dashboard use.
"""

import argparse
import json
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "strategy" / "tracking.db"
REPLAYS_DIR = ROOT / "replays"
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"

OUR_NAME = "Montana Schmeekler"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def pending_episodes(con: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return con.execute(
        "SELECT * FROM episodes WHERE downloaded = 0 ORDER BY create_time DESC LIMIT ?",
        (limit,)
    ).fetchall()


# ---------------------------------------------------------------------------
# Replay download and parse
# ---------------------------------------------------------------------------

def download_replay(episode_id: str) -> Path | None:
    """Download replay JSON to today's replays/ subfolder. Returns the path."""
    date_dir = REPLAYS_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    dest = date_dir / f"{episode_id}.json"
    if dest.exists():
        return dest

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [str(KAGGLE), "competitions", "replay", episode_id, "-p", tmp],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [WARN] replay {episode_id} failed: {result.stderr.strip()}")
            return None
        # kaggle downloads as episode-<ID>-replay.json
        candidates = list(Path(tmp).glob("*.json"))
        if not candidates:
            print(f"  [WARN] no JSON found for episode {episode_id}")
            return None
        shutil.move(str(candidates[0]), str(dest))

    return dest


def parse_placement(replay_path: Path) -> dict:
    """Extract our placement, reward, and number of players from a replay JSON."""
    with open(replay_path) as f:
        data = json.load(f)

    team_names = data.get("info", {}).get("TeamNames", [])
    rewards = data.get("rewards", [])
    num_players = len(team_names)

    our_slot = next(
        (i for i, n in enumerate(team_names) if OUR_NAME in n), None
    )
    if our_slot is None or our_slot >= len(rewards):
        return {"our_placement": None, "our_reward": None, "num_players": num_players}

    our_reward = rewards[our_slot]

    # Rank by reward descending (then slot index as stable tiebreaker)
    order = sorted(range(num_players), key=lambda i: (-rewards[i], i))
    our_placement = order.index(our_slot) + 1  # 1-indexed

    return {
        "our_placement": our_placement,
        "our_reward":    our_reward,
        "num_players":   num_players,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download replay JSONs from Kaggle")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max episodes to download in one run (default 50)")
    parser.add_argument("--episode", help="Download a specific episode ID")
    args = parser.parse_args()

    con = get_db()

    if args.episode:
        episodes = [{"episode_id": args.episode}]
    else:
        episodes = pending_episodes(con, args.limit)
        print(f"Pending downloads: {len(episodes)}")

    downloaded = 0
    for ep in episodes:
        ep_id = ep["episode_id"] if isinstance(ep, sqlite3.Row) else ep["episode_id"]
        print(f"  Downloading episode {ep_id} … ", end="", flush=True)

        path = download_replay(ep_id)
        if path is None:
            print("FAILED")
            continue

        info = parse_placement(path)
        placement = info["our_placement"]
        placement_str = f"{placement}/{info['num_players']}" if placement else "?"
        print(f"saved  (placement {placement_str})")

        con.execute(
            """UPDATE episodes
               SET downloaded = 1,
                   our_placement = ?,
                   our_reward    = ?,
                   num_players   = ?
               WHERE episode_id = ?""",
            (info["our_placement"], info["our_reward"], info["num_players"], ep_id)
        )
        con.commit()
        downloaded += 1

    con.close()
    print(f"\nDownloaded {downloaded} replay(s).")


if __name__ == "__main__":
    main()
