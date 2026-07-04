# reverse-engineering-agents-replay-analysis-tooli  —  REFERENCE (AidenSong123)

## What it is
**Opponent-modeling tooling** (not an agent). Extracted to `replay_analysis.py`. Loads replay
JSON for a named agent and measures behavioral fingerprints:
- First-action turn & reaction times (how fast it responds to threats).
- Phase volumes (ship output by early/mid/late) and per-phase gains.
- Own/neutral/enemy target-size preferences; coordinated-turn detection (multi-launch coordination).
- Comet-targeting rate.

## Why it matters (directly enables strategy #1)
This is the template for **opponent modeling from replay data** — exactly what the orbit_lite
family lacks. We already have a replay ingestion pipeline (`pipeline/`, `replays/`). Run this
analysis over the top bots' replays to build per-opponent profiles (opening hold? reaction speed?
comet behavior? coordination?), then condition our bot's policy on the detected opponent type. The
do-nothing projection that every top bot uses is the biggest exploit surface in the game.
