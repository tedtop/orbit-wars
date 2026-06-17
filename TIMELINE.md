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

## Phase 6 — Config tuning + forum intel (Jun 16–17)

**Milestone: `comet_reaper` hit a new best — #144 / 1243.8** (up from #183 / 1235.9; first time inside
the top ~150).

Finally read the Kaggle competition notebooks (pulled via `kaggle kernels pull` →
`experiments/v5_engine_tuning/intel_kernels/`):

- `souldrive/why-cloning-the-1-bot-loses-to-greedy` **independently confirms our Phase-4 dead end:** a BC
  clone of #1 (LB **1793**) wins only **~17%** vs the greedy Producer (~1240); PPO made it worse (4%).
  *"If a fast search exists, run the search — don't clone."* And **inference is not the bottleneck** —
  ~1 s/turn budget vs ~25 ms used → **~30× headroom for deeper search.**
- `improved-agent-v2`, `improved-heuristic-agent` (target ~1500): 1-ply heuristics with better
  **scoring/sizing** — NPV/snowball scoring with ownership multipliers (enemy 2.05× / neutral 1.4× /
  contested 0.7× / friendly 0.3×), phase-aware ship sizing (1.2×→3.0×), speed-bracket sizing, enemy-fleet
  interdiction, 4P weakest-enemy + elimination bonus.
- **Landscape: comet_reaper ≈ Producer ≈ 1240; #1 ≈ 1793.** Two levers — better scoring/sizing
  (1240→~1500), and **deeper multi-ply search** (~1500→1793, the real prize lever; we have the budget).

Plan (`experiments/v5_engine_tuning/`): low-noise seat-rotated **gauntlet** → bench comet_reaper vs the
improved field → port what wins → **Optuna** over the ~22 config knobs → **deeper search**, with an
**autoresearch** ratchet as orchestrator.

**🏆 Result — `schmeekler` (Ted's own idea) BEATS comet_reaper.** A comet_reaper fork that adds a tunable
scoring bonus for capturing **static (non-rotating) planets** first (the safe periphery land-grab — static
planets can't drift into enemy reach). Swept `static_target_bonus`: **1.0–1.5 is the sweet spot** (beats
comet_reaper ~71–72% 2P, seat-swapped), ≥2.0 over-commits and tanks. Validated:
- 2P vs comet_reaper: **72%** (36–14, n=50). vs the public panel: producer-v2 **77%**, floor-matched 77%,
  i-m-stronger 70%, 1266-elo 60% — beats the whole field (comet_reaper only *tied* producer-v2).
- 4P (focal vs 3× comet_reaper): **best in the pod** — 35% first-places vs each comet_reaper's ~22%.
- **Submitted 2026-06-17** as our new bot. First thing all project to genuinely beat the engine.

Also this phase: `comet_reaper_tuned` (knob-exposed for Optuna; config sweep found nothing, best 0.34 —
base config is a tight optimum) and `comet_reaper_search` (Numba forward-sim lookahead, ~80k rollouts/turn;
the single-move signal washes out over the rollout — being fixed). ⚠️ search in progress.

---

## 2026-06-17 — Track A comet 2×2 factorial kill-test: DISCARD

Tested `comet_target_bonus=1.5` (additive score bonus for targeting ephemeral comet-planets) across
a 2×2 factorial: with/without the static bonus (schmeekler vs comet_reaper base) × with/without the
comet bonus. Implementation: `obs_tensors["comet_planet_ids"]` → `torch.isin()` mask → additive bonus.

Results (n=50/opp, 5-opp panel, seat-swapped):
- **schmeekler_comet** (static=1.5, comet=1.5): OVERALL 74% = schmeekler baseline 74% → **+0pp**
- **comet_reaper_comet** (static=0, comet=1.5): OVERALL ~61% ≈ comet_reaper baseline → **+0pp**

Per-opp breakdown: schmeekler_comet 72%/74%/78%/78%/66%; comet_reaper_comet 50%/52%/67%/71%/66%.

The flow scorer already handles ephemeral target valuation implicitly. A flat additive bonus is noise —
comets are correctly de-prioritized by the engine because they are structurally low-value (prod=1.0,
short time window). COMET bonus sweep (0.5/1.0/2.0) not run; verdict: DISCARD.

---

## Open questions / next directions

- Multi-knob config search (CMA-ES / Optuna) over the engine's ~20 knobs vs a fixed
  gauntlet — the most promising untried lever.
- Proper seat-rotated 4P meta-play.
- Add the engine itself as a self-play league opponent (selfplay currently only loads
  neural opponents).
