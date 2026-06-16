# Orbit Wars — Project Timeline & Provenance

A running, dated record of what we tried and why — the *change of events* across
experiments. Kept so the provenance of the final solution is defensible: if anyone
asks "how did you arrive at this?", the answer is here, in order, with the dead ends
left visible rather than erased.

> **Maintenance:** append a new dated phase whenever a milestone lands (a submission,
> a strategy pivot, an experiment that succeeds or fails). Don't rewrite history —
> add. Mark anything not yet verified with ⚠️.

---

## Phase 0 — 23 bots from across the sciences (Jun 13)

Started broad: built **23 homegrown bots**, each implementing a different theory drawn
from a different scientific field — control theory, graph neural nets, epidemiology
(SIR), reaction-diffusion (Turing patterns), stigmergy/pheromones, minimax, Bayesian
methods, kinematics, macroeconomics, portfolio theory, and more.

- 20 "algorithm-strategy" agents + 3 earlier homegrown bots (renamed by method).
- **Best performer: `markowitz_portfolio_optimization`** — an *economics* approach
  (Markowitz mean-variance portfolio optimization applied to fleet/planet allocation).
- These now live in `archive/agents/` (21 of them) plus the two that were submitted,
  still at the top level of `agents/`.
- Git: `4366f1d` "add 20 algorithm-strategy agents and roster write-up",
  `6f8be2c` "rename the 3 homegrown bots by method", `324bd2e` "add arena workflow".

## Phase 1 — Evaluation infrastructure (Jun 13–14)

Built the harness to tell the 23 bots apart objectively.

- OpenSkill-rated **arena** (round-robin, seat-aware), a **pipeline**, and a
  **Streamlit dashboard** with episode cards, an HTML leaderboard, submission DB sync,
  and a countdown.
- Archived 21 bots into `archive/agents/`.
- Git: `8cb91e5` "add OpenSkill arena, pipeline, dashboard, and archive 21 bots",
  `1ea297c`, `c672ce3`.

## Phase 2 — First submissions (Jun 14) ⚠️ verify IDs/placement

Submitted the **best two** bots:

- `markowitz_portfolio_optimization`
- `coordinated_strike_interceptor` v1 — ⚠️ ~36 episodes (confirm exact count)

**Result: placed in the ~500s** on the public leaderboard. ⚠️ exact submission IDs and
scores to be filled in (memory has live submissions in the ~535–546 range). Screenshots
kept separately by Ted.

> *"Then we slept."* — a deliberate pause before the next direction.

## Phase 3 — Reverse-engineer the top of the leaderboard (Jun 14–15)

Studied the public leaderboard and found the best / most interesting bots were **all
the same lineage**: built on **The Producer**, which runs on the **`orbit_lite`
engine** — a self-contained planning package ("speed-first flow-diff producer";
zero-sum competitive scorer that prefers enemy captures).

- Built our own clone of that engine → **`comet_reaper`** (vendors `orbit_lite/` next
  to `main.py`). This became our strongest bot.
- Distinction worth keeping straight: **The Producer** = the top public *bot*;
  **orbit_lite** = the *engine/library* it's built on; **comet_reaper** = our clone.

## Phase 4 — Prize-zone data + neural cloning (Jun 15) — DEAD END

In parallel with Phase 3, pulled **public episodes from "density-max" / prize-zone
games** (matches where teams rated **>1500** participated) and tried to learn from the
top humans:

1. **Behavior-clone the top public players** (`rl/bc_train.py`): label their recorded
   moves and train a neural `PlanetPolicy` to imitate them (`--team` clones one team →
   `training/clone_*.pt`).
2. **Self-play against the clones** (`rl/selfplay.py --clones`): use those clones as
   fixed PPO league opponents.
3. **Result — failed.** The cloned-from-humans net hit a **"BC ceiling"** and lost
   **0–16 to `comet_reaper`**. The engine is better than anything a net imitating human
   players can reach. Also recovered a genuine *insight* (top teams target enemy ≫
   neutral, near-dead, nearby, high-production planets), but the engine already encodes it.

## Phase 5 — comet_reaper fork bake-off (Jun 15) — no champion

Forked `comet_reaper` five times, each adding one bolt-on heuristic (precog, kingmaker,
maestro, helmsman, oracle), and ran them against the engine on **seat-swapped** arena.
**All ≈parity-or-worse.** The orbit_lite engine is a tight, well-tuned local optimum.

- **Conclusion: submit `comet_reaper` unchanged** — it ties the real Producer (~14–14)
  and beats the rest of the public field (~67%).
- The whole experiment is filed under `archive/experiments/comet_reaper_forks/`
  (see its `README.md`). The neural BC/self-play track (Phase 4) is kept in `rl/` but
  carries `DEAD END` banners.

---

## Open questions / next directions

- Multi-knob config search (CMA-ES / Optuna) over the engine's ~20 knobs vs a fixed
  gauntlet — the most promising untried lever.
- Proper seat-rotated 4P meta-play.
- Add the engine itself as a self-play league opponent (selfplay currently only loads
  neural opponents).
