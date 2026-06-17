# Orbit Wars – Overnight Orchestrator Handoff
**Prepared:** 2026-06-17, ~11:30 (Mountain) · evening of day 4 of project
**Days to deadline:** 6 (June 23, 23:59 UTC)
**Status:** Cheap hypothesis space exhausted. Three tracks lined up for overnight work.

---

## TL;DR for the orchestrator

1. **Submit `schmeekler_fmt`** into the schmeekler slot (replaces schmeekler, keeps comet_reaper). This is the highest-fidelity gym we have access to — the live ladder itself. Verify the active-slot rule first.
2. **Re-activate Track A** to run the comet 2×2 factorial as a confirmatory kill-test (not a hopeful build).
3. **Launch Track C** for the value function build. This is the moonshot and the only remaining bet with real upside.
4. **Track B is complete** (2-ply search dead). Its search shell becomes Track C's integration target — do not assign new work to B.
5. **Stretch the poll cadence** to ~2 hours (Track C is a multi-day build, no need to poll every 20 min).
6. **Schedule a wakeup** to flag Ted when (a) schmeekler_fmt's live score settles, (b) Track A's comet factorial reports, or (c) Track C's local fidelity probe lands. Whichever happens first.

---

## State of play (what we know, with evidence)

### Confirmed live
- **comet_reaper @ 1245** – our best submitted bot. orbit_lite engine clone. Best confirmed live result. **STAYS in slot 1.**
- **schmeekler @ 1075** – static-target bonus on top of comet_reaper. Plateaued ~170 below CR. Gym oversold it. Slot 2 candidate for replacement.

### Closed with evidence (DO NOT REVISIT)
- ❌ **Optuna config tuning** – 37 trials, best 0.34 vs comet_reaper. Base config is a tight optimum. Knob-tuning alone won't help.
- ❌ **Behavioral cloning of top players** – 0–16 vs engine. Forum-confirmed independently. Engine > human moves at mechanical execution.
- ❌ **Potential fields, interdiction, phase-aware sizing** – all DISCARD at n=150.
- ❌ **2-ply shallow search (Track B v1 + v2)** – n=50 parity with schmeekler. Mechanistic reason: engine pre-filters to 0–4 candidates/turn (0–1 most turns), so there is nothing to re-rank.

### Marginal / live-untested
- ⚠️ **schmeekler_fmt** – format-aware static bonus (off in 4P). CONFIRMED at n=150: byte-identical to schmeekler in 2P, +3.72μ in 4P, 6 fewer last-places. Gym is biased AGAINST it (over-credits static play; fmt reduces static). Best schmeekler variant. **Live-test candidate.**
- ⚠️ **Comet bonus** – never built. Replay analysis shows 92% of comets never captured by anyone, field-wide. Expected result: ≈ 0 or negative. Run as confirmatory kill-test.

### The real swing
- 🎯 **Track C: learned value function** – the one remaining bet whose quality signal is **independent of the gym**. Fidelity probe runs on real downloaded episode outcomes.

---

## The mechanistic insight that shapes everything

The orbit_lite engine pre-filters candidate moves through `capture_floor`/`clears_floor` so aggressively that **64/133 turns have 0 valid candidates and 47/133 have exactly 1**. Only 22/133 turns have 2–4 choices.

This is why:
- All shallow search lands at parity (nothing to re-rank when there's 0–1 candidate)
- All re-ranking bonuses land at parity (same reason)
- The only mechanism that hasn't been tested is **expanding the candidate set to include aggressive moves the engine currently rejects** and evaluating them by learned outcome

That mechanism is the value function. It's structurally different from everything we've tried.

---

## Track assignments

### Track A — Comet 2×2 factorial (confirmatory kill-test)
**Status:** Re-activated after being declared complete.
**Worktree:** `/Users/Ted/src/orbit_wars-track-a` (branch `track-a-structural-features`)
**Goal:** Close the comet question with data, not argument.

**Work to do:**
- Build `agents/schmeekler_comet/` – a schmeekler fork carrying BOTH knobs:
  - `STATIC_BONUS` (default 1.5, same as schmeekler)
  - `COMET_BONUS` (new, scoring bonus for targeting comet planets, lookup via `obs["comet_planet_ids"]`; optionally scale by comet path-remaining)
- Run the 2×2 factorial on the seat-swapped gauntlet v2, re-baselining each cell:
  - `(static=0, comet=0)` = comet_reaper baseline
  - `(static=1.5, comet=0)` = schmeekler
  - `(static=0, comet=C)` = comet-only variant
  - `(static=1.5, comet=C)` = combined
- Sweep COMET_BONUS over a small range (e.g. 0.5, 1.0, 1.5, 2.0).
- Report all four cells in TRACK_A_NOTES (main effects + interaction).

**Framing:** This is a confirmatory kill-test. Expected result is comet ≈ 0 or negative. If COMET_BONUS shows no clear gym lift at any sweep value, mark comet DISCARD and stop. Don't over-invest.

**Cost:** Cheap, env-toggled, one bot. A few hours of gauntlet time.

---

### Track B — IDLE, do not assign new work
**Status:** Complete. 2-ply search structurally dead.
**Worktree:** `/Users/Ted/src/orbit_wars-track-b`
**Reusable artifact:** The search shell at ~14ms/turn with 800ms budget. This becomes Track C's integration target in Phase E (see below). Track C reads Track B's code as reference; does not assign Track B new work.

---

### Track C — Learned value function (the moonshot)
**Status:** Ready to launch.
**Worktree:** `/Users/Ted/src/orbit_wars-track-c` (branch `track-c-value-function`, .venv + opponents + comet_reaper symlinked in)
**Goal:** Build a learned leaf evaluator that breaks the candidate-scarcity ceiling by evaluating aggressive moves the engine currently rejects.

**Phases (from the handoff brief):**
- **Phase A — Data gen** (Jetstream2, ~150 CPU): Label real downloaded episodes with final outcomes + ε-aggressive self-play (so V sees outcomes of the floor-blocked moves we want it to value).
- **Phase B — Encoding** (LOCAL FIRST): Richer than 12 global scalars — per-planet DeepSets/attention + globals. **This is the highest-risk piece** per the advisor's flag. Iterate against the fidelity probe.
- **Phase C — Train** (A100, minutes): Tiny outcome-prediction net (~3-layer feedforward), CPU-fast inference, sigmoid output.
- **Phase D — Fidelity probe** (LOCAL, on real episodes): Does V rank real winning positions above losing ones? Threshold: ≥ 0.65 to proceed. **Gym-independent — this is what answers the "what if the gym is still bad" worry.**
- **Phase E — Integrate** as the leaf eval in Track B's search shell. The key refinement: V evaluates an **EXPANDED aggressive candidate set** (the moves the engine refuses), not the 0–4 safe ones (which is why shallow search was ≈schmeekler).

**Division of labor:** Track C session writes code + gives Ted exact commands to run heavy Phases A and C on Jetstream2. Ted runs them.

**Critical first move (LOCAL, before any Jetstream2 spend):**
Build the labeler + encoder + fidelity probe on already-downloaded replays. Get a "can V even rank real positions?" signal cheaply before committing A100 time. If the local fidelity probe fails on existing data, iterate on the encoding before generating more.

**Data check before doing anything:**
```bash
ls data/replays/ | wc -l                       # episode count
ls data/replays/**/*.json 2>/dev/null | head -5   # format check
```
If thousands of prize-zone (>1500 Elo) episodes are already downloaded → train on real data, Jetstream2 only for compute. If hundreds or fewer → Phase A self-play generation is required before Phase B.

---

## To answer Ted's specific questions

**Is ValueNet trained-and-bundled, or training-only?**
Both — same network. Trained once (offline) on Jetstream2 + A100. The trained weights file gets bundled into the submitted bot (just like The Producer ships orbit_lite alongside main.py). At game time the bot loads V into memory and calls `V(state)` as the leaf evaluator inside the MCTS tree. Inference is microseconds on CPU — fits the 1s turn budget easily.

**Optuna status:**
Done and shelved. 37 trials confirmed the base config is a tight optimum. We are not running Optuna anymore. The lesson is logged: config tuning alone doesn't move the needle.

**Track B status:**
Complete dead-end for shallow search. Reverted to depth=1. The search shell built during Track B becomes Track C's integration target in Phase E — do not assign Track B new experiments.

**Why fmt is the right submission:**
The only true "Kaggle gym" is the live ladder itself. Local gauntlets are proxies built on top of the official `kaggle_environments` package (which only ships the game engine + random/starter agents, not the opponent field or ratings). The only way to test against the real opponent distribution is to submit. fmt is the safest live test we have: byte-identical to schmeekler in 2P, +3.72μ in 4P on the gym, and the gym is provably biased against it. Slot order: keep comet_reaper, replace schmeekler with fmt. Verify the active-slot rule first.

---

## Submission discipline (unchanged)

- `comet_reaper` (1245) stays in slot 1. Do not touch.
- `schmeekler_fmt` to be submitted into slot 2, replacing `schmeekler` (live 1075).
- **Verify Kaggle's active-slot rule first.** If submitting a 3rd bot risks retiring comet_reaper, abort and re-plan. Confirm with `kaggle competitions submissions orbit-wars` before and after.
- Do NOT submit comet_reaper_mcts. It is provably ≈ schmeekler by construction (0–1 candidates per turn → identical moves). Live-testing it would just reproduce schmeekler's 1075.
- Do NOT submit anything from Track C until it clears the local fidelity probe AND beats comet_reaper on gauntlet v2.

---

## Orchestrator scheduled wakeup

**Cadence change:** Stretch the poll from 20 min to ~2 hours (Track C is a multi-day build; orchestrator overhead has diminishing returns at finer granularity).

**Scheduled wakeup events** (whichever fires first):
1. **schmeekler_fmt live score converges** – rating slope flattens to ±15 pts/hour for 2+ consecutive polls. Report final asymptote and whether it cleared comet_reaper's 1245.
2. **Track A reports comet factorial result** – all 4 cells run at n=150. Report main effects, interaction, KEEP/DISCARD verdict.
3. **Track C local fidelity probe lands** – pre-Jetstream2. Report accuracy on real-episode position ranking. Gate is ≥ 0.65 to proceed to A100.
4. **Anything catastrophic** – any track crashes, any test reverses a prior conclusion, schmeekler_fmt asymptote inverts our model.

**Context management:** The orchestrator session is currently at ~40% context with 405k uncached tokens. Before launching Track C, write a fresh `ORCHESTRATOR_STATE.md` snapshot capturing current state + active tracks + remaining bets, then `/clear` and paste that snapshot to open a fresh orchestrator session. Otherwise context will degrade and the orchestrator will start contradicting earlier decisions.

**Reporting format when waking Ted:**
- One-line headline (e.g., "schmeekler_fmt settled at 1280 — beats comet_reaper")
- 3-bullet detail
- Recommended next action with reasoning
- Any new dead-end logged

---

## Anti-goals (re-stated so the orchestrator doesn't drift)

- ❌ Do not revive Optuna config tuning
- ❌ Do not revive shallow search experiments
- ❌ Do not revive behavioral cloning of top players
- ❌ Do not submit comet_reaper_mcts (provably ≈ schmeekler)
- ❌ Do not submit Track C outputs without local fidelity ≥ 0.65 AND gauntlet v2 win vs comet_reaper
- ❌ Do not auto-submit any bot — Ted reviews every submission
- ❌ Do not poll faster than 2 hours unless a track explicitly flags an event

---

## What "winning the night" looks like by 06:00 Mountain

Best case:
- ✅ schmeekler_fmt submitted, soaking, asymptote forming
- ✅ Track A comet factorial complete, comet officially DISCARD
- ✅ Track C local fidelity probe ≥ 0.65 → A100 training kicked off

Worst tolerable case:
- ✅ schmeekler_fmt submitted, soaking
- ✅ Track A blocked on something fixable, flagged
- ⚠️ Track C local fidelity probe below 0.65 → encoder iteration needed, no A100 spend wasted

Actual disaster (wake Ted immediately):
- comet_reaper gets bumped from tracked slots (active-slot rule misread)
- A track silently crashes
- An experiment reverses comet_reaper >> everything else
