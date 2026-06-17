# HANDOFF — Track A: Structural scoring features (the proven-productive lever)

> Paste this whole file into a fresh Claude Code session opened in `/Users/Ted/src/orbit_wars`.
> You are running **Track A** on git branch **`track-a-structural-features`**. A separate session runs Track B
> (MCTS) on its own branch — stay in your lane, don't touch theirs.

## Who you are / the goal
You're improving **Montana Schmeekler's** Orbit Wars Kaggle bot. Orbit Wars = real-time-strategy: planets
orbit a sun in continuous 2D; you launch ships to capture planets; final score = ships + planets; 2P and 4P
modes; ~500 turns; ~1 s/turn compute budget. We climb from ~#144 / 1243 toward the prize zone (~1500; #1 ≈ 1793).

The whole top public field is one engine — Slawek Biel's **orbit_lite** ("The Producer"), a 1-ply greedy
flow-diff planner. Our clone is **`comet_reaper`** (it vendors `orbit_lite/`). Our **current champion** is
**`schmeekler`**: comet_reaper + a scoring bonus for **capturing static (non-rotating) planets first**
(`static_target_bonus=1.5`). It beats comet_reaper **72% 2P (seat-swapped)** and the whole public panel.

**Key established fact:** *adding structural features to the 1-ply scorer is the direction that WORKS.*
(Re-tuning the *stock* engine config → nothing, best 0.34. Simple forward-sim re-rank → no signal. BC/RL →
dead end, 0–16.) **Your job: find the next schmeekler-style feature.**

> **Note on "config tuning":** re-tuning the *base* engine knobs is the dead part. **Sweeping the knob each
> NEW feature introduces is NOT ruled out — it IS the job** (schmeekler's `static_bonus` sweep → 1.0–1.5 is the
> model). Every feature below ships its own env knob; sweep it (grid is fine; Optuna if the space is multi-dim).

## Your task — build & validate new structural scoring features
Fork **schmeekler** (the champion, not comet_reaper) into new bots under `agents/<name>/` and add ONE feature
each, env-gated so it's tunable. Ranked hypothesis queue (do them in order; each is its own bot):

1. **`agents/schmeekler_potential/`** — potential-field target value using **future** planet positions.
   Planets orbit; favor targets **rotating toward us / our cluster** (cheaper to hold) over ones drifting into
   enemy reach. Project each planet's position a few turns out (orbital angular velocity is in the engine), and
   add a scoring term ∝ (closing rate to our centroid). Env knob `POTENTIAL_WEIGHT`.
2. **`agents/schmeekler_interdict/`** — enemy-fleet **interdiction**: when an enemy fleet is en route to a
   contested neutral, add a bonus for racing the **same** neutral if our ETA ≤ theirs (deny the capture).
   In-flight fleets are in the obs tensors. Env knob `INTERDICT_BONUS`.
3. **`agents/schmeekler_phase/`** — **phase-aware ship sizing**: scale launch size by game phase
   (early ~1.2× surplus → finishing ~3.0×) and by speed bracket. Forum intel says winners do this.
   Env knobs `PHASE_EARLY_MULT`, `PHASE_LATE_MULT`.

Combine the winners afterward (stacked features) once each is individually validated.

## Hard constraints (do not violate)
- **All logic in `agents/<name>/main.py`.** **NEVER edit vendored `orbit_lite/`** — it's a shared module across
  bots; editing it corrupts every A/B comparison. Add config fields + an env override in `tensor_action` and a
  scoring term after `score_candidates`, exactly like schmeekler does (read `agents/schmeekler/main.py` first —
  it's the template).
- **Crash-guard every feature** in a `try/except` that falls back to the base scorer. A Kaggle error/timeout =
  instant loss.
- Stay within the **1 s/turn** budget (these features are cheap tensor ops, so fine — but verify).
- comet_reaper/schmeekler are **config-driven algorithms, NO weights / NO RL**. "Tuning" = sweeping the env knob.

## How to evaluate (the FIXED yardstick — don't make it easier)
Use the gauntlet. ALWAYS seat-swapped (2P has a seat-0 effect that fakes ~+14%):
```
.venv/bin/python experiments/v5_engine_tuning/autoresearch/evaluate.py <bot> 50 POTENTIAL_WEIGHT=1.0
```
It runs `<bot>` vs comet_reaper + the public panel (the-producer-v2, i-m-stronger, floor-matched, 1266-elo) and
prints win% per opponent + overall. Sweep the env knob to find the sweet spot (schmeekler's was 1.0–1.5; ≥2.0
over-committed — expect similar "too much over-commits" shapes).

**KEEP criterion — judge against the strong PUBLIC bots, not schmeekler.** schmeekler has never been on the real
Kaggle ladder, so it's only a *relative* champion; the public panel is our ground-truth proxy, and
**the-producer-v2 (the real top public lineage) + i-m-stronger are the ones that matter most.** A feature is a
KEEP only if, at n≥150 (±~5% CI): (1) its **win% vs the public panel ≥ schmeekler's, ideally higher** (especially
vs producer-v2 + i-m-stronger), and (2) it also holds its own head-to-head vs schmeekler. Always re-run schmeekler
on the same panel in the same session so the comparison is apples-to-apples (arena noise drifts run to run).

## Bookkeeping (every iteration)
- Append a row to `experiments/v5_engine_tuning/autoresearch/LOG.md` (hypothesis → gauntlet result → KEEP/DISCARD).
- Update "Accumulated knowledge" + re-rank the queue in `experiments/v5_engine_tuning/autoresearch/program.md`.
- If a feature becomes the new champion, append a milestone to root `TIMELINE.md` (append only, never rewrite).
- **Commit to branch `track-a-structural-features` only. DO NOT `git push`** — the repo is PUBLIC and we keep
  strategy secret until the competition ends. Present the commit message for approval before committing.
- **End every work session with a brief paste-able "state of play" summary** (champion, what ran, results,
  open questions) so Ted can forward it for strategic review.

## Keep yourself busy — self-paced ~20-min loop (don't idle waiting for Ted)
Work autonomously through the hypothesis queue. At the end of each work cycle, **schedule a wake-up ~20 min out
(`ScheduleWakeup`, re-passing your task) and post a brief status update** — what you just tested, the gauntlet
result, what's next — then continue. Keep looping until you either (a) hit a documented dead end on every queued
hypothesis, or (b) produce a **submission candidate** that beats the champion `schmeekler` on the gauntlet
(n≥150, outside CI, within time budget). When you hit (a) or (b), post a final state-of-play and stop scheduling.
Don't wait for Ted between cycles — he reads the status updates asynchronously and will redirect if needed.

## First moves
1. `git checkout track-a-structural-features` (confirm you're on it).
2. Read `agents/schmeekler/main.py` (the template) and `experiments/v5_engine_tuning/autoresearch/program.md`.
3. Build `agents/schmeekler_potential/`, evaluate, sweep, log. Then interdiction, then phase-sizing.
