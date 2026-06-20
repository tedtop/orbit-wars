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

## 2026-06-17 — Track C value function: NULL RESULT (submittable), V model preserved

Built `comet_reaper_vf`: bolt-on value function trained on 746K (obs, outcome) pairs from 2937 prize-zone episodes.

**What worked:**
- V model (MLP 2×128, DeepSets encoding): AUC=0.9835, Pearson=0.917 — genuinely predictive of game outcomes
- Integration gate (15ms/turn) well within the 800ms budget

**What didn't:**
- Phase E arena 2P: 17-17-6 **parity** vs comet_reaper — both bots use the same engine, so base moves are identical; VF aggressive expansions add noise, not signal
- Phase E arena 4P: 12 vs 23 firsts — **HURT** — aggressive moves from secondary sources expose the bot to 3-opponent counter-attacks simultaneously

**Insight:** Bolt-on VF (add extra candidates if V improves) is the wrong integration strategy. The right path: replace `score_candidates` inside orbit_lite with V-based scoring (policy-level, not post-planning). That requires a deep refactor with ~6 days to deadline — not feasible.

**Preserved:** `track_c/data/value_probe.pt` (trained V model) — genuinely accurate, useful if we revisit next season.

**Only remaining offensive bet: `comet_reaper_stochastic` (Boltzmann opponent model, τ sweep running).**

---

## 2026-06-17 — Track B Boltzmann stochastic search: NULL RESULT

Built `comet_reaper_stochastic`: Boltzmann 2-ply opponent search on top of comet_reaper base.

**Design:** P(R1_j) ∝ exp(orbit_lite_score_j / τ); EV(W1) = Σ_j P_j × competitive_score(W1+R1_j).
Z-score correction: `score = 1ply_score + EV_BONUS * (ev - ev_mean) / ev_std` (EV_BONUS=2.0).

**5 correctness bugs fixed during development** (summary in TRACK_B_NOTES.md).

**Results (n=20/opp, seat-swapped, panel=comet_reaper+4 opponents):**

τ=2.0 (EV_BONUS=2.0, N_CAND=192, TOP_K_OPP=5):
```
vs comet_reaper      10-10  (50%)
vs the-producer-v2   11-9   (55%)
vs i-m-stronger      12-8   (60%)
vs floor-matched     12-8   (60%)
vs 1266-elo          16-4   (80%)
OVERALL: 61/100 = 61%
```

schmeekler baseline (same panel, same n):
```
OVERALL: 78/100 = 78%
```

**VERDICT: DISCARD** — stochastic 61% < schmeekler 78%; did not clear +3pp threshold.

**Key insight:** Stochastic search excels vs hard opponents (80% vs 1266-elo, +20pp over schmeekler)
but fails vs comet_reaper (50%). Root cause: base is plain comet_reaper which already loses 80-20
to schmeekler. EV corrections can't overcome base strength gap vs aggressive opponents.

**Next idea:** Apply Boltzmann search on top of schmeekler base (with static bonus). Unlikely to pursue
before deadline — time better spent on submission strategy.

## 2026-06-17 — schmeekler_elim: DISCARD (bug found + fixed, still below baseline)

Tested elimination mode bolt-on to schmeekler_fmt: when opponent has ≤3 planets + we own more, add +8.0 enemy planet score bonus and lower ROI threshold to 1.0.

**Bug found:** Original threshold=3 fires at game START (opponent always starts with 1-2 planets → immediate overaggression). Fixed with dominance guard (`n_my_planets > n_enemy_planets`).

**Full panel results (n=150, post-fix):**

| Opponent | schmeekler_elim | schmeekler_fmt baseline |
|---|---|---|
| comet_reaper | 56.7% | ~80% |
| the-producer-v2 | 57.3% | ~85% |
| i-m-stronger | 70.0% | ~80% |
| floor-matched | 69.3% | ~85% |
| 1266-elo | **72.0%** | ~60% ← only win |
| **OVERALL** | **65.1%** | **~78%** |

Direct matchup vs schmeekler_fmt (n=20): 9-11 (45%) — ELIM LOSES TO ITS OWN BASE.

**Key insight:** Both elim (+12pp) and stochastic (+20pp, prior track) add edge specifically vs the 1266-elo opponent, but hurt vs the rest. orbit_lite greedy 1-ply is suboptimal in close/late-game vs skilled opponents. The fix (elim or EV) over-corrects and destroys medium-game performance.

**Preserved:** `agents/schmeekler_elim/main.py` for reference. schmeekler_fmt remains best gym bot.

---

## 2026-06-17 — Track D multi-fleet coordination: DISCARD

Tested coordinated 2-source attacks on targets neither fleet can take alone (4 implementations: crude ETA → tensor intercept_angle with accurate ETAs via orbit_lite's `intercept_angle`).

Economic analysis of 438 combo decisions: average value = -75.6 ships. 95% of combos negative EV. With strict filter (prod≥3, combined_ships≤40, turns≥10): only 0.8 profitable combos per game.

Root cause: staggered arrival (first fleet softens, second finishes) destroys the first fleet as attrition. The action-space gap is real (orbit_lite's `clears_floor` gate blocks multi-source attacks), but exploiting it requires simultaneous arrival (same-turn), not sequential. Simultaneous arrival needs L=2 LaunchSet in `plan_lite_waves` — invasive, didn't pursue.

**VERDICT: DISCARD.** `comet_reaper_combo` preserved with COMBO_ENABLED=0 (identical to comet_reaper).

---

## 2026-06-18 — v5 CLOSED — Full RL/PPO pivot for final 5 days

**v5 Engine Tuning is complete.** All tracks exhausted. Crystallized finding:

> orbit_lite's `capture_floor`/`clears_floor` collapses each turn to 0–4 candidates (0–1 most turns). Every bolt-on intervention (potential fields, EV scoring, elim bonus, 2-ply search, VF re-rank) lands at parity or hurts because there is nothing meaningful to re-rank at depth=1. The fix requires either (a) replacing the decision architecture entirely or (b) training a learned policy.

**Summary of 20+ experiments across 4 tracks:**
- Track A (structural features): all DISCARD — additive bonuses override flow scorer
- Track B (search/lookahead): all DISCARD — 2-ply exact search ≈ parity; stochastic EV incompatible with static-grab
- Track C (value function bolt-on): AUC=0.9835 pass, Phase E HURT in 4P. V model preserved.
- Track D (multi-fleet coordination): DISCARD — staggered combos negative EV
- Config tuning (Optuna, 37 trials): DISCARD — base config is a tight optimum

**Live situation at pivot:** comet_reaper=1234.7 (best, inactive), schmeekler_fmt~1125 (active, converging), schmeekler~1098 (active). Gap to prize zone (~1500): 265 pts. Heuristic ceiling confirmed at ~1240.

**Forum intel gathered (2026-06-18):** Lin Myat Ko (#1, 1793 elo) = JAX env + PPO self-play 600M steps (~$150). Radek (68th) = 1303 LB in $15 on GH200 in 6.5hrs. Mendrika (48th) = 1420 LB pure BC per-planet fire heads. Abhyuday (31st) = RL beat heuristic in 1 day. `orbit-wars-torch` (MIT) = complete PyTorch GPU batched env + PPO. Fanghhhh's notebook = complete working PPO loop.

**v6 plan:** PPO self-play, per-planet action heads, kaggle_environments (8K SPS target), train on Jetstream2/GH200, 2P first then optionally 4P. New branch: `v6-rl-selfplay`.

---

## 2026-06-18 — v6 PPO GAE Bug Found and Fixed (session 2)

**v6 RL training is now running correctly on 8 Jetstream2 instances.**

### Experiment log

| Exp | Config | Result | Verdict |
|-----|--------|--------|---------|
| v6-baseline | reward_scale=0.001, pure self-play | CF≈0, eval=25% at U=200 | ❌ DISCARD |
| v6-H1 | reward_scale=0.01, pure self-play | CF=0.052 at U=91 (EP spike), eval=25% at U=200 | ❌ DISCARD |
| v6-H1-remote | reward_scale=0.01, 32 remote jobs | CF=0.04-0.06 at U=10 (promising) | ❌ KILLED (GAE bug) |

### Root cause diagnosis (GAE structural bug)

`compute_gae()` grouped all per-planet decisions by `env_i` and treated them as sequential timesteps. With N owned planets per turn, planets 0..N-2 got `nv = V(same_state)` → delta ≈ r_t (no bootstrapping). Only the last planet per turn got a real TD advantage. ~70-80% of buffer entries had near-zero advantages regardless of reward_scale. Direct explanation for CF≈0 and entropy stuck at 5.0 for 1.2M steps.

**Smoking gun:** entropy never decayed across 300 updates and two separate runs — a learning policy always shows entropy decay.

### Fix implemented

- Tagged each buffer entry with `step_i` (rollout step index, 0..rollout_steps-1)
- Rewrote `compute_gae` to group by `(env_i, step_i)` → true timestep sequence
- Broadcast one advantage per timestep to all planet decisions at that step
- Validation: CF=0.17/0.08/0.16 at U=1-3 with EP=0, entropy decaying (5.00→4.92)

### Other changes this session

- **Comet_reaper as P1 cold-start opponent**: loads via importlib, fires from step 1, provides combat signal before self-play pool has checkpoints
- **Fleet monitoring**: tmux_fleet.sh (8-pane, per-instance tables), monitor.sh (refreshing CF/EV/Ent/EP/SPS display, color-coded)
- **VecEnv callable P1**: step() accepts callable agent(obs)->launches alongside per-planet tuple lists

### Fleet status (as of 2026-06-18 19:15 MT)

8 Jetstream2 instances relaunched with fixed train.py:
- 5× m3.2xl (64 CPU, 250 GB): 149.165.174.18/133/171.142/170.73/171.248
- 3× m3.xl (32 CPU, 125 GB): 149.165.175.105/170.84/175.177
- 32 parallel runs total (4 per machine)

### Next gate

With GAE fix, expect CF=0.05-0.3 sustained from U=1 (not just at terminal spikes). Gate passes when: CF in-band + EV rising + entropy decaying — all three, not just CF.

---

## 2026-06-19 — v6 RL CLOSED: FAIL at pre-committed gate

### Pre-committed failure condition (set ~20:00 MT 2026-06-18)
At U=500: best seed < 25% vs greedy AND 0% vs comet_reaper → FAIL. Close v6 RL track.

### Result
- v6_cr4 (diverse cold-start: greedy + comet_reaper alternating) — final experiment
- Greedy eval: 20%, 20%, 20%, 20%, 22% at U=100 through U=500. **Completely flat.**
- vs comet_reaper: 0% at every checkpoint, confirmed by direct eval (not instrument artifact)
- Three cold-start regimes tested: comet (v6_cr2), greedy, diverse (v6_cr4) — identical ~20% ceiling

### Diagnosis
Entropy decayed cleanly 4.84 → 2.1 across training. Policy committed — to a local optimum scoring 20% vs greedy. This is the worst-case pattern: not "still exploring," but "converged to something bad." Feature design or encoding changes won't un-commit a settled policy; the ceiling is structural given achievable depth (~11M steps by deadline, vs Radek's JAX pipeline at 15× throughput).

### Bugs fixed during v6 (all resolved, no competitive value extracted)
8 critical bugs found and fixed: GAE grouping, ent_coef (50× game signal), eval null-agent, eval n=20 noise, no checkpoint resume, shaped-reward drift, env reset logic, GAE rollout boundary bootstrap. First green gate: eval_fix_test U=100 → 40% vs greedy. But subsequent depth showed 40% was noise: plateau appears immediately at U=200 with all configs.

### Infrastructure built (retains value)
- eval_checkpoints.py: seat-balanced RL vs RL eval, symmetry gate, JSON output
- sync_checkpoints.sh: fleet champion promotion loop
- Fleet: 9 Jetstream2 instances, 32 × 64-env + 4 × 8-env jobs, ~1,312 SPS
- ORCHESTRATOR_STATE.md: structured session log

### Decision
**Pivot to Plan B: defend and harden comet_reaper (current live submission ~1235 Elo).**
- Verify submission slot eligibility before 2026-06-23 deadline
- Investigate #1047 trig inversion / [Y,X] coordinate transpose in orbit_lite engine
- Lock comet_reaper as the active defended submission


## 2026-06-19 — SECOND PROMOTION (v3 champion, U=400)

**Champion v2 → v3: ppo-1-of-3_job1 U=400 (WR=36.7% vs greedy)**

- 15:35 MT: v2 promoted (1000-game eval WR=0.753, pool_size=2)
- 15:56 MT: v3 promoted same model, duplicate eval (WR=0.752 — bookkeeping quirk, concurrent sync)
- Net: champion.pt is ppo-1-of-3_job1 U=400 WR=36.7% vs greedy
- vs comet_reaper: 0.000 (still zero at U=400)
- Champion pushed to all 9 fleet hosts ✓

Trajectory of ppo-1-of-3_job1 inline evals (n=30 greedy):
  U=100: 20%  U=200: 16%  U=300: 33% [v1]  U=400: 37% [v3]

Upcoming: Phase 3 gate at U=500 (~16:30 MT) — greedy WR > 25% likely; comet WR > 0% is the stretch goal

## 2026-06-19 21:07 MT — Phase 3 Gate: PASS (condition 1 only)

**Trigger:** U=500 eval on 4 deep seeds (ppo-1-of-3_j1/j3, ppo-3-of-3_j3/j4)

| Seed | greedy WR | comet WR | Pass? |
|------|-----------|----------|-------|
| j1 (1-of-3) | **36.7%** | 0.0% | ✓ cond 1 |
| j3 (1-of-3) | 23.3% | 0.0% | ✗ collapsed |
| j3 (3-of-3) | 30.0% | 0.0% | ✓ cond 1 |
| j4 (3-of-3) | 26.7% | 0.0% | ✓ cond 1 |

**Verdict:** Gate PASS (cond 1: greedy > 25%). Continue fleet.
**Comet ceiling confirmed:** vs_comet_reaper = 0.000% across ALL seeds at ALL depths through U=500.
**Strategic implication:** RL is learning (beats greedy at 37%) but shows no signal vs comet_reaper. June 22 re-submit trigger still in effect.
**Fresh seeds ETA to eligibility:** ~21:45 MT (U=450 threshold) from ppo-2-of-3 (170.84) and ppo-1-of-3_j2/j4 (175.105).

## 2026-06-19 22:19 MT — RL Ceiling Confirmed at U=500

**All 4 deep seeds evaluated at U=500. No challengers. Track effectively closed.**

| Seed | U=500 greedy WR | comet WR | best_model status |
|------|----------------|----------|------------------|
| ppo-1-of-3_j1 (champion) | 36.7% | 0.0% | stuck @ U=400 (no improvement) |
| ppo-1-of-3_j3 (collapsed) | 23.3% | 0.0% | stuck @ U=300 |
| ppo-3-of-3_j3 | 30.0% | 0.0% | stuck @ U=400 |
| ppo-3-of-3_j4 | 27.0% | 0.0% | stuck @ U=300 |

**Structural finding:** 32 seeds, 28 have inverted-U WR trend (peak U=100-200, decline after). Only j1 improved monotonically. n_games=30 variance causes spurious early saves that block later eligibility.

**One remaining watch:** ppo-2-of-6_j2 (174.133) shows genuine improvement (28%→27%→33%→33%→37%) — the only fresh seed with an upward trend at U=400. At U=420, 33 SPS → ETA U=500 in ~2.5h (~01:00 MT June 20). Needs ≥40% at U=500 to save a new best_model and challenge.

**comet ceiling:** 0.000% across all seeds, all depths — zero signal that RL can beat comet_reaper.

**June 22 trigger still active.** Fleet continues but prognosis is bearish.
