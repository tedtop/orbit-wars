# ORCHESTRATOR STATE — snapshot for continuity (re-paste into a fresh session if context degrades)

_Last updated: 2026-06-17 ~12:00 MT (day 4). Deadline: **2026-06-23 23:59 UTC** (~6 days)._
_This is the durable memory. Canonical detail lives in `program.md` (knowledge + queue) and `LOG.md` (ledger)._

## 🔻 IMMEDIATE NEXT (incoming orchestrator: read this first)
- **Ted is about to submit `schmeekler_fmt`** (now at root `agents/schmeekler_fmt/`, baked-in 2P=1.5/4P=0.0).
  Slot rule CONFIRMED: **2 active, newest-2 auto, NO manual toggle** → submitting fmt **will bump comet_reaper**
  (oldest) → active becomes {fmt, schmeekler}, LB reads ~1075 (cosmetic until deadline). **This is intended.**
- **Strategy shift (Ted's call):** comet_reaper @1245 is NOT prize-tier (need >1500), so its slot isn't precious
  mid-week. **Use the 2 slots to live-calibrate our variants AND generate real episode replays as Track C valuenet
  training data.** Cycle bots through; **resubmit a STRONG bot (a better comet_reaper *descendant*) before the
  Jun 23 deadline.** Do NOT auto-submit — Ted runs every submission.
- **Track A is CLOSED & merged** (fmt copied to root; branch `track-a-structural-features` kept as history; its
  worktree `../orbit_wars-track-a` can be `git worktree remove`d once Ted closes that tab).
- **Track C (value function) is the live build** — its first move is LOCAL fidelity probe on ~2650 prize-zone
  episodes (gate ≥0.65 before any Jetstream2 spend).
- Poll ~hourly (tool caps 1h); WAKE TED only on: fmt converges / Track C fidelity probe lands / catastrophe.

## Live ladder (ground truth — poll `.venv/bin/kaggle competitions submissions orbit-wars`)
- **comet_reaper — 1245** — orbit_lite engine clone. **OUR BEST. Stays in slot 1. Do not touch.**
- **schmeekler — ~1075** — comet_reaper + static-planet bonus. Plateaued ~170 below CR. Gym oversold it.
- The Producer (best public) ≈ 1259. #1 ≈ 1793. Prize zone ≈ 1500.
- **Dashboard (localhost:8501, 46 live episodes):** schmeekler official **1069.5** · 2P **57% (17-13)** · 4P avg
  place **2.38** (6×1st/2×2nd/4×3rd/4×4th) · 46 games/24h. Episode graphs plot **planet-count divergence over the
  game**.
- 🔎 **DIAGNOSTIC (from episode graphs):** in several LOSSES schmeekler builds a planet lead mid-game then
  **COLLAPSES late** (over-extends, can't hold). The 1-ply scorer is blind to this; a value function on FINAL
  outcomes would penalize it → reinforces the Track C bet. (Worth a proper replay measure: peak→final planet share.)

## Active tracks (4 git worktrees; orchestrator = this session on `v5-engine-tuning`)
- **Track A** (`track-a-structural-features`) — **CLOSED & merged to root.** Best output: `schmeekler_fmt`
  (≥ schmeekler: 2P-identical, +3.72μ 4P) — now at `agents/schmeekler_fmt/`. Comet 2×2 kill-test was NOT run
  (deprioritized — replays showed comets a field-wide non-opportunity). All other bolt-ons DISCARDED.
- **Track B** (`../orbit_wars-track-b`) — COMPLETE. 2-ply search dead. Its search shell = Track C's Phase-E target.
- **Track C** (`../orbit_wars-track-c`, `track-c-value-function`) — THE moonshot, launching. Learned value function.

## Exhausted hypotheses — DO NOT REVIVE (one-line reasons)
- ❌ Optuna config tuning — 37 trials, best 0.34; base config is a tight optimum.
- ❌ BC / cloning top players — 0–16 vs engine; forum-confirmed.
- ❌ Potential-field, interdiction, phase-sizing bonuses — all DISCARD at n=150 (additive bonuses override the scorer).
- ❌ 2-ply shallow search (comet_reaper_mcts v1/v2) — n=50 parity. **Provably ≈ schmeekler — do NOT submit it.**
- ❌ Comet-aware — replays: 92% of comets never captured by ANYONE (field-wide non-opportunity). Kill-test running to confirm.

## THE mechanistic insight (shapes everything)
orbit_lite's `capture_floor`/`clears_floor` collapse each turn to **0–4 candidates (0–1 most turns)** → nothing to
re-rank → all shallow methods land at parity. **Only untested lever: EXPAND the candidate set to aggressive/
floor-blocked moves the engine refuses, judge by LEARNED outcome (value function).** That's Track C.

## Gym status (under audit)
- Engine = official `kaggle_environments` orbit_wars v1.0.9 (exact). What's NOT downloadable = the live opponent
  field + rating (server-side). Our gauntlet is a proxy.
- **gym v2** (`gauntlet_v2.py`): diverse field + mixed 2P/4P + OpenSkill placement. Clean run puts comet_reaper ≳
  schmeekler (inversion fixed); can't resolve the top near-ties (anchors within 66 live pts). Directionally OK at
  p4=0.4; understates CR's margin → trust only LARGE gym edges. 4 name-anchored opponents (lb-max-1224, lb-1000,
  heuristic-1110, lb-958) DON'T run locally (74/74 no-op) — excluded.
- **The only real gym = the live ladder (submitting).** That's why fmt gets a live test.

## Remaining bets
1. **schmeekler_fmt live test** — replace schmeekler slot (NEVER comet_reaper). Calibration + best schmeekler variant.
2. **Track C value function** — the real upside. Fidelity probe on REAL episodes (gym-independent) gates it.
3. comet 2×2 kill-test (Track A) — cheap, expected ≈0.

## Hard rules / slot discipline
- **Repo is PUBLIC — never `git push`.** Commit locally only.
- **Never edit vendored `orbit_lite/`.** All bot logic in `agents/<bot>/main.py`.
- **Ted reviews EVERY submission** — orchestrator does not auto-submit. Verify active-slot rule before any submit;
  if a 3rd sub risks retiring comet_reaper → ABORT + flag Ted.
- n≥150 for keep/discard. Re-baseline every comparison (the "fmt 66%" was an unmatched-baseline artifact).
- Tracks journal to `TRACK_{A,B,C}_NOTES.md`; orchestrator owns program.md/LOG.md/TIMELINE.md.

## Data inventory (for Track C)
- `episodes/` = **~2650 prize-zone episodes** (06-05→06-14). `replays/` = 266 (our games, incl. schmeekler/CR live).
- `strategy/tracking.db` = episode metadata (num_players, our_placement parsed). `pipeline/extract_moves.py` labels states.
- `training/*.pt` = DEAD BC clones (do not reuse as V; pipeline is reusable). schmeekler sub=53770052, CR=53707586.

## Poll cadence
~2h (tool caps a single ScheduleWakeup at 3600s/1h → self-poll hourly, wake TED only on: fmt converges / Track A
factorial done / Track C fidelity probe lands / anything catastrophic).
