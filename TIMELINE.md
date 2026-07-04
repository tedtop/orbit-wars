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

## Phase 2 — First submissions (Jun 14)

Submitted the **best two** bots:

- `markowitz_portfolio_optimization` v1 — 36 episodes, Official score: 578.2
- `coordinated_strike_interceptor` v1 — 35 episodes, Official score: 531.7

**Result:** Placed in the ~500s on the public leaderboard. The scores confirm the live submissions reached 531.7 and 578.2. Screenshots preserved in `strategy/screenshots/`.

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
the single-move signal washes out over the rollout — being fixed). (Search later concluded in Track B: DISCARD).

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

---

## 2026-06-20 — v8 comet_reaper hardening audit: CLEAN, no changes made

With 3 days to deadline and comet_reaper resubmitted as the protected floor (sub 53871873,
~1235 Elo), ran a full correctness + latency audit before locking it.

### Task 1: Latency audit

Instrumented `run_turn` (the main planning call) across 5 seeds (2P, 500 steps each) and
one 4P run:

| Config | Median | p99 | Max (local) | Max × 5× (tournament est.) |
|--------|--------|-----|-------------|----------------------------|
| 2P (5 seeds) | 6–7 ms | ~8 ms | 85 ms† | ~425 ms |
| 4P (seed 42) | 5.1 ms | 6.8 ms | 7.6 ms | ~38 ms |

†The 85 ms max is a one-time PyTorch lazy kernel compilation spike (fires at most once per
game, not every turn — confirmed by repeating the same game in the same process: zero spikes
on runs 2 and 3). All other turns are under 10 ms.

**Verdict: no latency issue.** Zero turns exceed 100 ms locally. At 5× tournament slowdown
every turn is well under the 1 s budget. No fix needed.

### Task 2: Coordinate-order check

Checked `orbit_wars.py` (the live environment source) — line 8 declares:

```
# Planets and fleets share a common [id, owner, x, y, ...] prefix.
Planet = ["id", "owner", "x", "y", "radius", "ships", "production"]
```

Index 2 = X, index 3 = Y — not Y-before-X. Traced the full stack:

- `adapter.py`: unpacks `pid, owner, x, y, r, ships, prod = p[:7]` — correct
- `obs.py`: `x = planets[..., 2]`, `y = planets[..., 3]` — correct
- `intercept_aim.py`: `atan2(t0y − CENTER, t0x − CENTER)` — standard `atan2(Δy, Δx)` convention — correct

**Verdict: no coordinate bug.**

### Overall finding

Bot is clean. No latency overage, no coordinate swap, no correctness bug found.
Sub 53871873 locked as-is — do not touch before the 2026-06-23 deadline.

---

## 2026-06-20 — v7 Critic A/B: CLOSED FAIL

**Hypothesis:** Dense per-step reward (reward_scale=0.01) adds noise to GAE returns,
suppressing critic EV vs pure sparse ±1 terminal (reward_scale=0.0). Expected: var_a EV
climbs to ≥0.90 while ctrl stays ~0.80.

**Setup:** 8 jobs across ppo-1 + ppo-2 (4 ctrl, 4 var_a), 64 envs each, comet_reaper
cold-start opponent, per-planet PPO. Branch: v7-critic-ab.

**Result — both arms identical:**

| U | ctrl greedy | var_a greedy | comet_reaper_WR | ctrl EV | var_a EV |
|---|------------|-------------|-----------------|---------|---------|
| 100 | 30–47% | 30–40% | 0% | 0.97 | 0.97 |
| 200 | 33–43% | 27–47% | 0% | 0.93 | 0.93 |
| 300 | 30–37% | 27–43% | 0% | 0.87 | 0.85 |
| 400 | — | — | 0% | 0.65–0.97 | — |

Entropy collapsed to 1.18–3.13 by U=390–400 (started ~4.9). Fleet killed at U=390–400.

**Why hypothesis failed:** The "fast filter" (EV ≥ 0.90) was trivially met by both arms
from U=10 onward — short 64-step rollouts mean the buffer is dominated by near-zero dense
rewards, making the critic's job easy regardless of reward_scale. EV was not the real bottleneck.

**Diagnosis:** RL from scratch at CPU scale (~67 SPS, ~1.6M steps) cannot close the
comet_reaper gap. Policy peaks at U=100 (~40% greedy) then entropy-collapses. Same
structural failure as v6 with a different config. comet_reaper_WR = 0% at every checkpoint
across all 8 seeds.

**Fleet:** ppo-1 (149.165.175.228) + ppo-2 (149.165.175.188) killed 2026-06-20 ~14:30 MT.

**Next:** v8 — per-planet behavior cloning from top-ranked game replays (Mendrika's approach,
1420 Elo with pure BC per-planet fire heads).

---

## 2026-06-20 — v8 Per-Planet BC: OPENED

**Branch:** v8-planet-bc | **Spec:** experiments/v8_planet_bc/SPEC.md

**Hypothesis:** Mendrika (48th, ~1420 Elo) achieved their rating with pure behavior cloning
using per-planet fire heads on top-ranked replays. Our current `ActorCritic` in train.py has
exactly this architecture. Prior BC failure (Phase 4, lost 0–16) used a global `PlanetPolicy`
(rl/policy.py), not per-planet heads. This combination has never been tested.

**Plan:**
1. Download prize-zone episodes: `pipeline/pull_topbot_episodes.py --require-rating 1400`
2. Extract labeled moves: `pipeline/extract_moves.py --min-rating 1400 --out training/moves_v8.jsonl.gz`
3. Train: `agents/rl_ppo/bc_train.py` (new script) — CE loss on fire/target/frac per owned planet
4. Gate: `eval_checkpoints.py --vs-comet --n-games 200` — comet_reaper_WR > 0% → submit

**Status:** CLOSED FAIL 2026-06-21.

**Results:** 20 epochs on 982k per-planet examples from 1400+ Elo teams (2,650 episodes).
fire=91%, tgt=19% (vs 5% chance), frac=79%. comet_reaper_WR=0/200, avg game 88/500 steps —
crushed before midgame. BC ceiling confirmed: imitation captures *what* top players do,
not *why*. orbit_lite exploits the gap immediately.

**Final answer: comet_reaper (sub 53707586, ~1235 Elo).** Competition ends 2026-06-23.

---

## 2026-06-22 — v10-jax: Pure-JAX Engine (commit 105ab1e)

**Milestone:** JAX game engine + two-phase validation gate committed to `v10-jax` branch.

**orbit_jax.py** — fully-vectorised `step()` with no Python loops in hot path:
- Fleet ring-buffer write slots via prefix-cumsum (no loop)
- Fleet-planet swept-pair CCD via `(F,P)` broadcasting — no nested `vmap`
- Comet spawning/expiry, orbit rotation, production all vectorised
- Compile time: <10 s on CPU (was 3-5 min with nested vmap)

**Bugs fixed vs Python interpreter (4 parity bugs):**
1. Comet spawn trigger: `cur_step == 50` (not 49)
2. Spawn position: `path[0]` immediately; `c_path_idx = 0` (not -99,-99 then advance)
3. No double-advance on spawn tick (`already_adv = active_c & ~should_spawn`)
4. Terminal condition: `slimit = cur_step >= EPISODE_STEPS-1` (not -2)

**Validation result:**
- Phase 1 (no-op exact): 20/20 (100%) — zero step/winner errors
- Phase 2 (bot winner agreement): 15/20 (75%) — threshold 65%

**train_jax.py** — ActorCriticET Transformer PPO, smoke test exit 0 (768 steps, 4 envs).
**launch_gpu.sh** — validates engine (N=20) then runs training on A100.

**Next:** deploy to Jetstream2 A100, measure SPS target (≥1000 SPS / 10k parallel envs).

---

## 2026-06-24 — v10-jax A100 runs: CLOSED FAIL

**Status:** SHUT DOWN (all Jetstream2 instances shelved by prof; CPU fleet also unreachable).

**What ran:** 2× g3.xl A100 instances (GPU1 seed=1, GPU2 seed=2), each ~20 hours, ~107M steps (U≈1640).

**Results:** `eval_vs_greedy` locked at **0.2333 (7/30)** from U=100 through U=1600 on both seeds — never moved. `comet_reaper_WR=null` entire run.

**Root causes diagnosed:**
1. **Broken eval signal** — `make_rl_agent` uses argmax (deterministic), eval uses fixed seeds 0–29. Once the policy stabilises on those 30 games (U=100), the result never changes. Fix implemented: random seed offset per eval call.
2. **comet_reaper eval null** — `comet_reaper/main.py` imports `torch`; jax_env had no torch. Fix implemented: CPU-only torch installed.
3. **No opponent quality** — snapshot pool seeded from random weights. CPU `train.py` used comet_reaper as P1 cold-start → 40% vs greedy at U=100. JAX had no cold-start → policy learned to beat random play only.

**Fixes committed but not yet proven:**
- `evaluate_vs_greedy` now uses random `seed_offset` per call
- `torch` (CPU-only) installed in jax_env on both GPU instances
- `SnapshotPool` class with 50% current / 50% historical P1 mixing
- `--seed` arg for JAX PRNGKey diversity

**Next when GPU instances are reactivated:**
- Implement comet_reaper bootstrap phase (~60 lines): run Python games vs comet_reaper for first 50 updates to warm-start snapshot pool with a competent policy before switching to JAX self-play.
- The JAX engine + Tensorboard + sync_runs.sh infrastructure is all in place.

**Final submission:** comet_reaper (sub 53707586, ~1235 Elo). Competition ended 2026-06-23.

---

## 2026-06-25 — v9 CPU fleet: CLOSED FAIL + metrics archived

**Final result:** `comet_reaper_WR = 0.0` across all 24 seeds (6 of 9 instances reachable), all 700+ updates, ~69 hours of training each. Gate was cr_WR > 0% at U=500 — never triggered.

**Final training state (ppo-1 sample):**
- U=710–723, elapsed ~69h, entropy ~0.08 (collapsed), explained_variance ~0.82–0.90
- eval_vs_greedy stuck at 0.2333 (7/30) — same 23.3% plateau as JAX runs
- terminal_win_pct = 0.0 — training games never terminated with a winner

**Metrics archived** to `agents/rl_ppo_cpu/runs_archive/` (24 × metrics.jsonl, no checkpoints — best_model.pt files were all saved at cr_WR=0%, not worth keeping).

**All Jetstream2 instances shelved/deleted.** No RL experiment ever beat comet_reaper.

**Lesson confirmed:** Both CPU (v9 ET-PPO, 24 seeds, 69h) and JAX (v10, 2 seeds, 20h) converged to the same 23% greedy ceiling and 0% vs comet_reaper. Root cause: self-play without a strong opponent bootstrap → policy learns to beat random play only, not transferable to orbit_lite opponents.

## 2026-06-28 to 2026-06-30 — v10 JAX GPU w/ comet_reaper training signal: CLOSED FAIL

**Config:** 2× A100 (seeds 1 & 2), pool_size=64, cr_games_per_update=4, 1024 JAX envs
**Results:**
- GPU1 (seed 1): ran to U=567, comet_reaper_WR=0.0 at every eval (U=100–500)
- GPU2 (seed 2): ran to U=1100+, comet_reaper_WR=0.0 at every eval (U=100–1100)
- eval_vs_greedy: flat 3–17%, trending toward 0 by U=700+

**Root cause:** 4 Python CR games = ~256 transitions vs 65,536 JAX self-play transitions per update. CR signal is 0.4% of the batch — completely drowned out by self-play.

**Bugs fixed this run:** fire_l[0,0] indexing, pool_size 512→64 (was 5hr startup), checkpoint resume logic added.

**Status:** GPU instances shelved. RL self-play approach closed.

---

## 2026-07-04 — RL campaign closed: final post-mortem

**All RL experiments across every track (v6, v9 CPU, v10 JAX): `comet_reaper_WR = 0%`. Campaign closed.**

**Why the top-LB players' PPO worked and ours didn't — the real answer:**

The top of the leaderboard runs the orbit_lite engine. Their PPO policies almost certainly operate over *orbit_lite macro-action parameters* (aggression level, target priority, fleet-split ratio — a handful of floats), not raw ship commands. Orbit_lite executes the actual micro. This collapses the decision horizon from 498 raw steps to ~20–40 macro-decisions, turning γ^498 ≈ 0.007 into γ^30 ≈ 0.74. Credit assignment becomes tractable. Dense reward flows naturally from orbit_lite's internal state (planet delta, ship production lead per step). Action space is tiny.

We were doing PPO over raw actions in a 498-step sparse-reward environment. They were selecting strategy knobs and letting a deterministic engine handle the rest.

**Why behavioral cloning also failed (earlier experiment):** BC on top-player replays clones the move sequence, but the move only makes sense in the context of the macro strategy orbit_lite was already pursuing. You get the cursor without the hand — syntactically correct moves with no strategic coherence.

**What would have had to change to make RL work:**
1. Wrap a strong handcrafted engine (orbit_lite or comet_reaper) as the macro-action executor
2. Train PPO over strategy parameters only (small discrete/continuous space)
3. Dense reward from engine internals every step, not terminal-only
4. Fixed diverse opponent pool for training, not pure self-play

**Final submission:** comet_reaper (sub 53707586, ~1235 Elo). This remains the best bot produced in the project.

---

## 2026-07-04 — Project presentation website v2 (`website_fable/`)

Built a fresh single-page presentation site from the whole repo's mission data
(new build, independent of the earlier `website/` draft): Next.js 16 + Tailwind v4,
fully static, Vercel-ready, zero chart/animation libraries.

- **Data:** `scripts/build_data.py` regenerates everything from raw sources —
  712 leaderboard CSVs → Elo race series + final standings (4,752 teams, us #415);
  6 curated episode replays compacted ~15 MB → ~500 KB each; curated RL metrics.
- **Sections:** cinematic orbital hero → game briefing → 23-bot arena → engine
  lineage → leaderboard climb chart → 19-experiment KEEP/DISCARD ledger +
  zero-choice exhibit → RL moonshot (0% hero number, plateau + entropy-collapse
  charts, macro-action post-mortem) → canvas replay theater of real ranked games →
  seven lessons → final standings + 24-phase mission log.
- **QA:** `next build` + eslint clean; headless-Chrome screenshot pass at desktop
  and mobile widths; zero console errors.

## 2026-07-04 — Presentation site deployed to Vercel

`website_fable/` deployed to production: **https://orbit-wars-smoky.vercel.app**
(project `orbit-wars`; the plain `orbit-wars.vercel.app` subdomain was already
taken by another account). Final content: mission-log timeline with education
post-mortem + next-season playbook, lab field notes (MCTS, Boltzmann search),
replay theater, GitHub repo link.

**Update:** production domain finalized — **https://kaggle-orbit-wars.vercel.app**
(added as a project domain, auto-assigned to every future production deploy).
