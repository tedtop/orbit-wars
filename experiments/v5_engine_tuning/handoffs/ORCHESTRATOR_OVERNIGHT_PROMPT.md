# Orchestrator — overnight directive (paste into the orchestrator session)

Ted is going to sleep. Here is the consolidated plan for the next ~8 hours. Confirm by acknowledging each numbered item, then execute.

## 1. Context hygiene FIRST
You are at ~40% context with 405k uncached tokens. Before doing anything else:
- Write `experiments/v5_engine_tuning/autoresearch/ORCHESTRATOR_STATE.md` — a concise state snapshot capturing: current live scores, active tracks, exhausted hypotheses (with one-line reasons), remaining bets, hard rules, slot discipline.
- After committing it, do NOT `/clear` yet — finish this directive first. Ted will `/clear` and re-paste the snapshot into a fresh session in the morning if context degrades further.

## 2. Submit `schmeekler_fmt`
- Verify Kaggle's active-slot rule first: `kaggle competitions submissions orbit-wars`. Confirm submitting a third bot will NOT bump `comet_reaper` out of the tracked pair.
- If safe: submit `schmeekler_fmt` to slot 2, replacing `schmeekler` (live 1075). Keep `comet_reaper` (1245) in slot 1.
- If submitting risks retiring comet_reaper: **abort, flag Ted, do not submit.**
- After submission, poll the live score every ~2 hours and track the rating slope.

## 3. Re-activate Track A
- Paste the comet-factorial relay (already drafted at `experiments/v5_engine_tuning/handoffs/TRACK_A_comet_factorial.md`, or re-draft from the orchestrator's prior message) to the Track A session.
- Track A's job: build `agents/schmeekler_comet/` carrying both STATIC_BONUS and COMET_BONUS, run the 2×2 factorial on gauntlet v2, sweep COMET_BONUS at a few values, report all four cells.
- Framing: confirmatory kill-test, not hopeful build. If no clear lift at any COMET_BONUS, mark DISCARD and stop. Expected result: comet ≈ 0 or negative.

## 4. Track B is IDLE
- Do not assign Track B new work. 2-ply search is structurally dead.
- Track B's search shell is Track C's integration target in Phase E.

## 5. Launch Track C
- Ted will open a fresh Claude Code session in `/Users/Ted/src/orbit_wars-track-c` and paste the contents of `experiments/v5_engine_tuning/handoffs/TRACK_C_value_function.md` (or the revised version Ted gives him).
- After Track C session is live, your job is to poll it every ~2 hours.
- **Critical:** Track C's first move is LOCAL — build the labeler + encoder + fidelity probe on already-downloaded replays. NO Jetstream2 spend until the local fidelity probe ≥ 0.65.

## 6. Poll cadence
- Stretch from 20 minutes to **2 hours**. Track C is a multi-day build; finer polling has diminishing returns.
- Exception: if a track commits a `🚨 ORCHESTRATOR:` tagged event, respond immediately.

## 7. Scheduled wakeup conditions
Wake Ted (commit a `🚨 TED:` tagged report to `ORCHESTRATOR_STATE.md`) when any of:
- **schmeekler_fmt live score converges** — slope ≤ ±15 pts over 2 consecutive polls. Report final asymptote and whether it cleared comet_reaper's 1245.
- **Track A reports** — all 4 factorial cells at n=150 done. Report main effects + interaction + verdict.
- **Track C local fidelity probe lands** — pre-Jetstream2. Report accuracy. Below 0.65 = encoder iteration; ≥ 0.65 = clear to Jetstream2.
- **Anything catastrophic** — track crashes, prior conclusion reversed, comet_reaper bumped from tracked slots, an experiment shows a known-bad bot suddenly winning.

When you wake Ted, format the report as:
- 1-line headline
- 3-bullet detail
- Recommended next action with reasoning
- Any new entry logged in `LOG.md`

## 8. Hard "do not" rules
- Do not revive Optuna, BC, potential fields, interdiction, phase sizing, comet_reaper_mcts.
- Do not auto-submit any bot. Ted reviews every submission.
- Do not submit Track C outputs without fidelity ≥ 0.65 AND gauntlet v2 win vs comet_reaper at n=150.
- Do not poll faster than 2 hours unless a track explicitly flags.

## 9. Acknowledge
Reply with:
- A confirmation that each numbered item is understood and will be executed
- The committed sha of `ORCHESTRATOR_STATE.md`
- Your first wakeup-check ETA in clock time
- Any clarifying question that would block progress overnight

Then proceed.
