# Orbit Wars — Experiments Context & Supplementary Detail

Compiled from raw conversation pastes and session notes. Intended as a reference for *why* each experiment was tried and what sub-findings aren't captured in TIMELINE.md. Cross-references TIMELINE.md phase/section names where applicable.

---

## 1. The orbit_lite Engine — What It Actually Is

Across many sessions there was confusion about what exactly comet_reaper is. Settled understanding:

- **comet_reaper is not a neural network.** No weights file. It is a deterministic algorithm: the `orbit_lite` planner (`Slawek Biel's "The Producer"`) steered by a config of ~22 numbers.
- **"Tuning" means searching for better numbers** — no gradients, no training data, no `.pt` files.
- The planner is **pure 1-ply greedy**: it builds an 18-turn forward projection (assuming all opponents do nothing), scores all `(source, target)` wave candidates, and fires the best. No search over future decisions.
- `comet_reaper`'s **no-op rate is 78%** (launches on only 22% of turns), but dispatches ~5,700 ships/game. This is *selective aggression*, not pathological passivity — hold unless there's a +ROI wave, then commit hard. This is distinct from the RL passivity trap.
- **Per-turn time: ~25ms used out of ~1s budget** → 30× compute headroom. This headroom was the central motivation for the deeper search experiments.
- Engine source is **byte-identical** across all public orbit_lite family bots — only `main.py` config presets differ. This means all public bots can be run in-process in arena safely.

### The Full Knob Set (~22 knobs)

| Group | Knobs |
|---|---|
| Aggression/selection | `roi_threshold`, `min_ships_to_launch`, `max_waves_per_turn`, `max_offensive_targets`, `max_defensive_targets`, `max_sources_per_lane` |
| Lookahead/reinforce | `horizon`, `reinforce_size_beta`, `reinforce_eta_free`, `reinforce_eta_scale` |
| Regroup | `enable_regroup`, `max_regroup_time`, `regroup_pressure_delta_min`, `max_regroup_sources_per_lane`, `max_regroup_targets_per_source`, `regroup_time_penalty_weight` |
| Endgame surge | `terminal_phase_turns`, `terminal_roi_threshold`, `terminal_max_waves_per_turn`, `terminal_enable_regroup` |
| Misc/4P | `comet_evac_steps`, `ffa_leader_attack_bonus`, `ffa_target_prod_bonus` (4P only; comet_reaper sets both to 0) |

**Important:** `min_ships_to_launch` controls the minimum garrison a planet needs before launching. Higher = more conservative. Ted's initial intuition was to raise it for aggression — this is backwards. Lower = more aggressive. Optuna confirmed: best early trial had `min_ships=3.24` (down from 4.0).

---

## 2. The Low-Noise Gauntlet — Why It Matters

Early experiments (comet_reaper_forks bolt-on bots) produced "61% win" results that evaporated when properly measured. Root cause: **n=28, seat-biased evaluations had ±18% confidence intervals**. The "wins" were noise.

**The fix:** a seat-rotated gauntlet at n≥150–200 games per candidate gives ~±5% CI.

This is more important than it sounds. The autoresearch/ratchet loop (inspired by Karpathy's "autoresearch" pattern) only works if the yardstick is trustworthy. With a noisy evaluator, the ratchet keeps noise improvements and rejects real ones.

**Gauntlet composition used for v5 experiments:**
- `comet_reaper` (control)
- `the-producer-v2`
- `i-m-stronger`
- `floor-matched`
- `1266-elo`

Seat-rotated 2P + 4P. Bots run in-process via `arena.py`.

---

## 3. Schmeekler — The One Structural Win (TIMELINE Phase 6)

**Origin:** Ted's human intuition while playing the game in the browser: "I send ships to the closest static planets first — they can't drift into enemy reach." The orbit_lite engine already weights proximity + competitive value but does not distinguish static vs rotating planets.

**Implementation:** A single additive bonus (`STATIC_BONUS`) for targeting non-rotating planets. Format-aware: `STATIC_BONUS_2P=1.5`, `STATIC_BONUS_4P=0.0` (4P dynamics are different; static bonus hurts in FFA).

**Why it works:** Static planets stay in geometrically safe space and form a reliable production base. The orbit_lite scorer already uses lead-aim (accounts for future planet position at arrival time), but doesn't weight "this planet will stay in my reach" vs "this planet will drift away." Schmeekler adds exactly that weight.

**Activity rate:** schmeekler launches on 62% of turns vs comet_reaper's 22%. The static bonus genuinely changes behavior — it's not a cosmetic change.

**Sweep results:**

| STATIC_BONUS | vs comet_reaper (2P, n=50) | Notes |
|---|---|---|
| 0.0 | baseline | comet_reaper base |
| 1.0 | ~71% | sweet spot low end |
| 1.5 | **72%** | submitted value |
| 2.0+ | declining | over-commits, tanks |

**Cross-format validation (n=50 each):**

| Opponent | schmeekler win% |
|---|---|
| comet_reaper | 72% |
| the-producer-v2 | 77% |
| floor-matched | 77% |
| i-m-stronger | 70% |
| 1266-elo | 60% |
| **OVERALL** | **~72%** |

4P (focal vs 3× comet_reaper): 35% first-places vs ~22% per comet_reaper.

**Why this class of idea works when search doesn't:** The orbit_lite `capture_floor`/`clears_floor` filter reduces candidates to 0–4 per turn (0–1 on most turns). Re-ranking 0–1 candidates is structurally useless — search just reproduces the engine's own choice. But changing what the engine *targets* (via a scoring modification) changes which planets get attacked, which is a structurally different lever.

---

## 4. The MCTS / Deeper Search Dead End — Anatomy

(TIMELINE Phase 6, various Track B entries)

### Why 30× headroom didn't help

The Numba forward-model benchmark showed the hardware is fast enough:

| Rollout depth | Speed | Rollouts in 800ms |
|---|---|---|
| 20 turns | 1.4 µs each | ~567,000 |
| 60 turns | 2.4 µs each | ~328,000 |

(Compile is a one-time 2.7s on turn 0, pre-triggered at import.)

This definitively proved that MCTS rollouts are *computationally* feasible. The failure was not speed.

### The structural problem

Profiling of 133 comet_reaper games:
- **64 out of 133 turns: zero valid candidates** (engine's `clears_floor` gate blocks all moves)
- **47 out of 133 turns: exactly one candidate**
- Only 0–4 candidates on any given turn

When search has 0–1 candidates to re-rank, it makes the same move as the 1-ply engine by construction. This is a math fact, not a gym-trust question.

### Track B v1 and v2

- **v1** (flat 2-ply with exact flow scorer as leaf): 75% vs schmeekler's 74% at n=50. Statistically indistinguishable.
- **v2** (true depth-2 + state-advancement opponent model): also 75% at n=50. Same structural problem.
- Both showed **false promise at n=20** (80% at small sample sizes) that evaporated at proper n.

### The stochastic opponent model idea (Track B revival)

Track B's argmax 2-ply was killed partly by the "oracle opponent" problem: assuming the opponent always plays their single best counter to our exact move. Against a perfect oracle, nothing wins.

The proposed fix: replace argmax with a **Boltzmann/softmax distribution over opponent candidates**:

```
P(opp attacks planet j | state) ∝ exp(orbit_lite_score_j / τ)
EV(W1) = Σ_j P_j × competitive_score(W1 + R1_j)
```

This is a proper Bellman equation for a 2-player MDP. Temperature τ → 0 is the oracle; τ → ∞ is random; optimal τ matches real opponent behavior.

**Stochastic 2-ply also resolves the candidate-pool problem:** with softmax over all ~192 `(src, tgt)` pairs (not just the 0–4 floor-cleared ones), the expectation naturally penalizes losing attacks near zero. The hard pre-filter is no longer needed.

### Track B stochastic result (TIMELINE "Track B Boltzmann stochastic search: NULL RESULT")

Despite the theoretical fix, stochastic search (τ=2.0, EV_BONUS=2.0, N_CAND=192, TOP_K_OPP=5) produced:
- vs comet_reaper: 50%
- OVERALL: 61% vs schmeekler baseline of 78%

Root cause: the stochastic base was plain comet_reaper (not schmeekler). Schmeekler already wins 80–20 vs comet_reaper, so a stochastic wrapper on comet_reaper was starting from a disadvantaged base.

**Key finding:** Stochastic search added +20pp specifically vs the `1266-elo` opponent but hurt vs all others. The pattern — search adds edge vs skilled opponents but destroys medium-game performance — also appeared in `schmeekler_elim`. This suggests the engine is already a tight optimum for the general case, and perturbations that help in close endgames hurt in the more common mid-game decisions.

### The "rollout policy reproduces the candidate move" finding

A separate forward-sim re-ranking experiment (comet_reaper_search with aggressive rollout policy) found:

```
H= 6: near-noop=+0.0   near-far=+0.0
H=40: near-noop=-3.0   near-far=-3.0
```

The rollout policy ("greedily attack the best target") reproduces the candidate move anyway — so whether we apply the candidate explicitly or let the policy do it, the rollout converges to the same outcome. The explicit move adds no information the policy didn't already encode. This is why the horizon parameter λ made no difference.

**Summary: Simple search (rollout / 2-ply) is fundamentally redundant with the engine's 1-ply scorer.** Real gains from search would require a learned value function that can evaluate board states the engine refuses to enter — a structurally different problem (Track C).

---

## 5. Config Tuning / Optuna (TIMELINE Phase 6)

**37 trials, result: best 0.34 — base config is a tight optimum.**

Context from early Optuna runs (14 trials, before final verdict):
- Best-so-far trial used aggressive settings: `roi_threshold=1.03`, `min_ships=3.24`, `reinforce_beta=1.70`
- Yet still only won 20% vs comet_reaper
- Key insight: "the answer isn't *more aggression*, it's *calibrated aggression* — which is exactly what search buys you (lookahead tells you which aggressive moves don't blow up)"

The 4P surface was the highest-EV tuning target because all public bots share the same 2P config, but differ in CONFIG_4P. `i-m-stronger` (stronger than comet_reaper) uses: `roi_threshold=1.55`, `min_ships_to_launch=5.0`, `max_offensive_targets=7`, plus `ffa_leader_attack_bonus=0.035`, `ffa_target_prod_bonus=0.08`. comet_reaper sets the ffa bonuses to 0.

Despite testing i-m-stronger's 4P config and running 37+ Optuna trials, no improvement was found. The 4P config surface is also saturated.

---

## 6. Track A: Comet Factorial Kill-Test (TIMELINE "Track A comet 2×2 factorial")

**Why it was tried:** Comet planets are transient (appear/disappear, ~53-step lifespan, production=1.0). The hypothesis was that capturing them early (before they drift into enemy reach) could be a bonus, analogous to the static-planet idea.

**Prior evidence against:** Replay analysis showed 92% of comets are never captured by anyone. The engine implicitly values them low.

**2×2 factorial design:**

| Cell | Static bonus | Comet bonus |
|---|---|---|
| (0,0) | 0 | 0 | = comet_reaper baseline |
| (1.5,0) | 1.5 | 0 | = schmeekler |
| (0,C) | 0 | 1.5 | = comet_reaper_comet |
| (1.5,C) | 1.5 | 1.5 | = schmeekler_comet |

**Results:** schmeekler_comet added 0pp over schmeekler. comet_reaper_comet added ~0pp over comet_reaper. Closed with DISCARD.

**Why:** The flow scorer already handles ephemeral target valuation implicitly. Comets are structurally low-value (prod=1.0, short time window, gone before terminal 5× bonus applies). A flat additive bonus is noise.

---

## 7. Track C: Value Function (TIMELINE "Track C value function: NULL RESULT")

**Design rationale:** Trained on 746K (obs, outcome) pairs from 2,937 prize-zone episodes. MLP 2×128 with DeepSets encoding. The pitch was that it could expand the candidate set to aggressive moves the engine refuses (floor-blocked) and evaluate them by learned outcome — bypassing the 0–4 candidate ceiling.

**What worked:**
- V model quality: AUC=0.9835, Pearson=0.917 — genuinely predictive of game outcomes
- Integration gate: 15ms/turn, well within the 800ms budget

**What failed:**
- 2P Phase E arena: 17-17-6 parity vs comet_reaper. Root cause: both bots use the same engine, so base moves are identical. VF "aggressive expansions" add noise, not signal.
- 4P Phase E arena: 12 vs 23 firsts — actually *hurt*. Aggressive floor-blocked moves expose the bot to simultaneous counter-attacks from 3 opponents.

**Key architectural lesson:** Bolt-on VF (add extra candidates if V improves) is the wrong integration strategy. The right path: replace `score_candidates` inside orbit_lite with V-based scoring at the policy level, not post-planning. That requires a deep refactor of orbit_lite. Estimated ~6 days, not feasible with ~2 days remaining.

**Preserved:** `track_c/data/value_probe.pt` — the V model itself is accurate and useful if revisited.

---

## 8. Track D: Multi-Fleet Coordination (TIMELINE "Track D multi-fleet coordination")

**Hypothesis:** Some target planets can't be captured by a single fleet but could be taken with coordinated 2-source attacks. orbit_lite's `clears_floor` gate explicitly blocks these.

**Economic analysis of 438 combo decisions:**
- Average value: **-75.6 ships**
- 95% of combos: negative EV
- With strict filter (prod≥3, combined_ships≤40, turns≥10): only **0.8 profitable combos per game**

**Root cause:** Staggered arrival (first fleet softens → second fleet finishes) destroys the first fleet as attrition. Simultaneous arrival (same turn) would work, but requires `L=2 LaunchSet` in `plan_lite_waves` — invasive engine change. Not pursued.

---

## 9. Track B Boltzmann: 5 Correctness Bugs Fixed

During development of the stochastic search, 5 correctness bugs were found and fixed (summarized in `experiments/v5_engine_tuning/autoresearch/TRACK_B_NOTES.md`). These are worth knowing because they illustrate the difficulty of building correct 2-ply search on top of orbit_lite:

1. Z-score de-meaning applied incorrectly (before vs after baseline)
2. EV_BONUS scaling error
3. Opponent distribution computed on wrong game state
4. N_CAND count off-by-one
5. Seat-swap not applied to stochastic panel

The bugs were fixed, the algorithm ran correctly, and still didn't beat schmeekler. This closes the "maybe the stochastic search had bugs" hypothesis.

---

## 10. schmeekler_elim: Elimination Mode Bolt-On

**Hypothesis:** When an opponent is nearly dead (≤3 planets), add an elimination bonus (+8.0 enemy planet score, lower ROI threshold to 1.0) to close them out decisively.

**Bug found:** The original threshold=3 fires at game start (opponents always start with 1–2 planets → immediate overaggression from turn 1). Fixed with a dominance guard: only fire when `n_my_planets > n_enemy_planets`.

**Results (n=150, post-fix):**

| Opponent | schmeekler_elim | Notes |
|---|---|---|
| comet_reaper | 56.7% | vs schmeekler base ~80% |
| the-producer-v2 | 57.3% | vs schmeekler base ~85% |
| i-m-stronger | 70.0% | |
| floor-matched | 69.3% | |
| 1266-elo | **72.0%** | only case where elim helps |
| **OVERALL** | **65.1%** | vs schmeekler ~78% |

Direct matchup vs schmeekler_fmt: 9-11 (45%) — elim loses to its own base.

**Pattern:** Both elim and stochastic search add edge specifically vs `1266-elo` (+12pp and +20pp respectively) but hurt vs the rest. This suggests the `1266-elo` bot has exploitable late-game passivity. Neither intervention is worth submitting.

---

## 11. GYM vs LIVE Calibration Notes

The gym (local `kaggle-environments`) is an imperfect proxy for the live leaderboard. Confirmed calibration issues:

- `schmeekler` won 72% vs comet_reaper in gym but only rated ~1075 live vs comet_reaper's ~1235
- `schmeekler_fmt` beat schmeekler in gym — never live-tested, slot not used
- The gym over-credits the static bonus for 2P specifically (likely because live games have more diverse opponents)

**The GYM ≠ LIVE gap is not fully understood.** Key finding: large gym differences (e.g., schmeekler 72% vs comet_reaper) track the correct direction (schmeekler is a real improvement), but the magnitude is overstated and absolute ratings don't translate directly.

**One live datum on timing:** The "No Lag" bots (~1300+) on the leaderboard optimize specifically for tournament CPU speed (reported to be 3–5× slower than local hardware). The hypothesis is that `comet_reaper` may occasionally exceed the 1s/turn budget on tournament hardware, costing games silently. This was not investigated before the RL pivot.

---

## 12. Forum Intel Summary (TIMELINE Phase 6)

Gathered ~2026-06-16–18. Key findings:

| Source | Finding |
|---|---|
| souldrive | BC clone of #1 (1793 Elo) → 17% vs greedy Producer; PPO made it *worse* (4%). Independent confirmation of Phase 4 dead end. |
| improved-agent-v2, improved-heuristic-agent (~1500) | 1-ply with ownership multipliers: enemy 2.05×, neutral 1.4×, contested 0.7×, friendly 0.3×; phase-aware sizing 1.2×→3.0×; speed-bracket sizing; enemy-fleet interdiction; 4P weakest-enemy + elimination bonus |
| Lin Myat Ko (#1, 1793) | JAX env + PPO self-play 600M steps (~$150) |
| Radek (68th, ~1303) | $15 on GH200 in 6.5hrs |
| Mendrika (48th, ~1420) | Pure BC per-planet fire heads |
| Abhyuday (31st) | RL beat heuristic in 1 day |
| Jonathan Roy / Bovard (staff) | Tournament CPU runs at 1/3–1/5 of local speed |

**Leaderboard ladder (as understood ~Jun 16):**
```
~1240  comet_reaper / Producer    ← 1-ply greedy, moderate aggression
~1500  improved heuristics        ← 1-ply, better scoring/sizing
~1793  Lin Myat Ko                ← deeper search + calibrated aggression
```

---

## 13. v6 PPO RL: GAE Bug and Its Signature

(TIMELINE "v6 PPO GAE Bug Found and Fixed")

The most important debugging insight of the RL track: the GAE structural bug had a **smoking gun diagnostic that appeared from the very first run** but wasn't recognized:

> **Entropy never decayed across 300 updates and two separate runs.** A learning policy always shows entropy decay. Entropy stuck at 5.0 = the policy was not learning, regardless of other metrics.

The bug: `compute_gae()` grouped per-planet decisions by `env_i` and treated them as sequential timesteps. With N planets owned per turn, planets 0..N-2 got `nv = V(same_state)` → delta ≈ r_t (no bootstrapping). Only the last planet per turn got a real TD advantage. ~70–80% of buffer entries had near-zero advantages.

**Fix:** Tag each buffer entry with `step_i` (rollout step index, 0..rollout_steps-1). Group by `(env_i, step_i)` for true timestep sequence. Broadcast one advantage per timestep to all planet decisions at that step.

**Validation:** CF=0.17/0.08/0.16 at U=1–3 with EP=0, entropy decaying (5.00→4.92). Bug was real; fix confirmed.

### The 8 critical bugs fixed during v6 (all resolved, no competitive value extracted)

| # | Bug |
|---|---|
| 1 | GAE grouping (the structural one — described above) |
| 2 | ent_coef 50× game signal (entropy coefficient too large, masked learning) |
| 3 | eval null-agent (evaluator crashed silently) |
| 4 | eval n=20 noise (too few games for meaningful eval) |
| 5 | no checkpoint resume (restarts wasted all progress) |
| 6 | shaped-reward drift (reward normalization drifted over time) |
| 7 | env reset logic (environment state not properly reset between episodes) |
| 8 | GAE rollout boundary bootstrap (boundary condition in multi-rollout GAE) |

First clean gate: eval_fix_test U=100 → 40% vs greedy. But subsequent depth showed 40% was noise; plateau appeared immediately at U=200 with all configs.

---

## 14. v6 RL: Why It Failed Structurally

(TIMELINE "v6 RL CLOSED: FAIL at pre-committed gate")

Entropy decayed cleanly 4.84 → 2.1. The policy *committed* — to a local optimum scoring ~20% vs greedy. This is the worst-case pattern: not "still exploring," but "converged to something bad."

**The structural ceiling:** ~11M steps achievable by deadline vs Radek's JAX pipeline at 15× throughput. The RL failure was not a bug problem (all 8 bugs fixed) — it was a depth-of-training problem plus possibly a feature-design ceiling.

Three cold-start regimes tested:
- `v6_cr2`: comet_reaper cold-start only
- Greedy cold-start only  
- `v6_cr4`: diverse (greedy + comet_reaper alternating)

All produced ~20% greedy ceiling, 0% vs comet_reaper at every checkpoint.

**The inverted-U pattern:** 32 seeds total, 28 showed peak win rate at U=100–200 then decline. Only `ppo-1-of-3_j1` improved monotonically (to 37% at U=400). The n=30 variance at early checkpoints creates spurious saves that block later eligibility.

---

## 15. Ideas Generated But Not Executed

These emerged during brainstorming sessions and were either deprioritized or came too late for the deadline:

### Periodic Orbit Timing ("WHEN to attack, not just WHETHER")
From Track B closure note. For each target planet, scan the next 150 steps to find the minimum capture floor (planets orbit with period ~40–80 steps; distance from nearest source varies → floor varies). If minimum floor occurs T_future steps from now and current floor is 2× that minimum, suppress the attack and accumulate garrison, then attack at the optimal orbital window. Compute cost: negligible (floor already computed for K_eta steps). Never implemented; closed with v5.

### Potential Field with Future Planet Positions
orbit_lite uses lead-aim (accounts for where a planet will be at arrival) but doesn't weight "this planet is rotating toward my territory" vs "this planet is drifting away." A potential-field overlay scoring planets by future geometry would be orthogonal to schmeekler's static-planet bonus. Never implemented.

### Objective Recovery / Preference Ranking (Track A variant)
From the "Orbit Wars — four engine-layer bots" planning doc. For each prize-zone replay state, regenerate orbit_lite's candidate waves, compute per-candidate features, and fit scoring weights so the top team's actual move ranks top (pairwise logistic/ranking loss). Plug recovered weights into a parameterized `competitive_score`. This would answer "what are the top teams optimizing for?" without having their code. Never implemented (value function was prioritized instead).

### Stochastic Search on Schmeekler Base
The natural extension of Track B stochastic: apply Boltzmann search on top of schmeekler rather than comet_reaper. Stochastic search on comet_reaper was hampered by the base strength gap vs schmeekler. Never tested.

### "No Lag" Latency Optimization
Competition CPU runs at 1/3–1/5 local speed. Forum bots named "No Lag" reached ~1300+ by optimizing to stay under the per-turn budget on slow tournament hardware. Numba on orbit_lite's hot path + capping planner horizon/candidate counts in expensive branches is the mechanism. Never verified whether comet_reaper times out on tournament hardware, and never implemented.

---

## 16. Autoresearch Pattern — What It Was

Not a package or plugin. The "Karpathy autoresearch" reference is a 3-file pattern:
- **Fixed yardstick** (`prepare.py` analog) = the low-noise gauntlet evaluator
- **Editable code** (`train.py` analog) = comet_reaper config / planner
- **Goals** (`program.md` analog) = "raise win%, don't bust time budget, don't regress 2P"
- **Ratchet** = Claude Code proposes a change → gauntlet scores it → keep only if it beats baseline

The ratchet was run manually (not as a formal framework). schmeekler was the first and only "keep." Optuna served as the numeric inner loop for config sweeps; Claude Code was the structural outer loop proposing ideas.

The key difference from Karpathy's version: Karpathy had a **deterministic metric** (val_bpb). Orbit Wars has **stochastic games** — the gauntlet is only as trustworthy as the number of seat-rotated games run. This is why the low-noise evaluator (n≥150) was the prerequisite for any reliable autoresearch.

---

## 17. Screenshot Archive — Visual Evidence Record

152 screenshots spanning Jun 13–20. All Jun 13 (4 files), Jun 15 (10 files), and Jun 17 overnight AM (42 files, 1:49–8:40 AM) are blank/unrenderable — likely the screenshots were taken of a dark or locked screen. All substantive content came from: named dashboard files, Jun 17 PM session, Jun 18 evening, Jun 19, and Jun 20.

### Phase 0–2: Early Bot Scores (Jun 13–15, named files)

**Local arena leaderboard** (`my_1v1_leaderboard.png`, ~Jun 13–14):
This is the Phase 0 evaluation — 80,480 games, 23 bots + 2 baselines. Full ranking:

| Rank | Bot | Win% |
|---|---|---|
| 1 | markowitz_portfolio_optimization | 66% |
| 2 | value_function_collapse | 64% |
| 3 | stategraphic_pheromone_routing | 63% |
| 4 | coordinated_strike_interceptor | 63% |
| 5 | deep_q_network_macro_strategist | 62% |
| 6 | lstm_fleet_trajectory_forecaster | 62% |
| 7 | comet_riding_ephemeris_exploitation | 61% |
| 8 | predictive_kinematic_interceptor | 59% |
| … | distributed_pid, lyapunov (55%), frontline (51%) | 55–51% |
| 15 | path_aware_lead_interceptor | 48% |
| 16 | greedy_lead_interceptor | 42% |
| 17 | artificial_potential_fields | 41% |
| 23 | cascading_classifier_regressor | 17% |
| 24 | random (baseline) | 6% |

Top 8 marked "ELITE TIER — SUBMIT CANDIDATES." This is why markowitz and coordinated_strike were the first submissions.

**First Kaggle submissions** (named screenshot files, ~Jun 14):
- `markowitz_portfolio_optimization v1` · 36 episodes: official score **578.2**, 2P win 46% (13W/15L)
- `coordinated_strike_interceptor v1` · 35 episodes: official score **531.7**, 2P win 44% (11W/14L)

**Early dashboard** (`dashboard_1.png`, ~Jun 14): Score **553.7**, Rank **#3117** (+13 places). Top of leaderboard visible: #1 Ialiah @ Tufts Labs **1767.5**, #2 Jake WIS **1743.8**, #3 H.K. **1624.4**. The gap between position 3117 and position 1 at this point was enormous.

---

### Phase 6 / v5 Engine Tuning: Jun 17 PM Session

This was the main orchestrator session. Multiple live ladder snapshots visible from the Streamlit dashboard and orchestrator terminal.

**Live ladder progression (Jun 17 PM, from screenshots):**

| Time (MT) | comet_reaper | schmeekler | schmeekler_fmt | Notes |
|---|---|---|---|---|
| ~12:11 PM | 1234.7 (inactive) | — | — | Starting state |
| ~1:59 PM | 1234.7 (inactive) | 1091.3 (52 eps) | 1120.8 (28 eps) | fmt just submitted |
| ~2:01 PM | 1234.7 (inactive) | 1098.7 (55 eps) | 1125.8 (51 eps) | fmt reversed after peak |
| ~2:37 PM | 1234.7 (inactive) | 1098.2 (51 eps) | 1125.8 (51 eps) | oscillating |
| ~4:14 PM | — | — | **1141.1** at **#653** | Dashboard shows −174 pts since last snapshot |

**schmeekler_fmt trajectory** (Jun 17, from orchestrator logs): 1048 → 1126 → 1144 → 1167 → 1126 — peaked at 1167 then reversed. Still converging with only 28–51 episodes.

**Track C fidelity probe landing** (Jun 17 ~12:11 PM, screenshot `12.11.19 PM`): This was the biggest single moment of the day. Full metrics from the screenshot:

| Metric | Value | Threshold |
|---|---|---|
| AUC (overall) | **0.9905** | ≥ 0.65 |
| Pearson correlation | 0.9145 | — |
| Val MSE | 0.033 | — |
| AUC (contested 2P states) | 0.9532 | — |
| AUC (early game, step<150) | 0.9613 | — |
| Phase E timing | 19 ms/turn | < 800 ms |

Comment from session: "This isn't a marginal pass — it's a dominant signal." Yet as documented in TIMELINE.md, the gym arena then showed parity 2P (17-17-6) and hurt in 4P (12 vs 23 firsts).

**Autoresearch dashboard tab** (`2.15.45 PM` screenshot): 15 experiments done, KEEP rate **2/15 (13%)**, 4 running overnight, Track C AUC 0.99 ✅, Deadline: 6 days. Shows bot lineage chart (tree from orbit_lite_engine base through comet_reaper/schmeekler variants) and experiment scatter plot.

**Track B closure** (Jun 17, `12.57.02 PM` screenshot): Terminal showing profiling data — orbit_lite's `clears_floor` filter leaves 8–4 valid candidates per turn, **44 turns with 0 candidates, 47 with 1**, out of 133 profiled. This is the quantitative backing for the candidate-scarcity finding in the raw pastes.

**Submit.py bot list** (Jun 17 ~11–12 AM, multiple screenshots): 8 bots visible in `agents/` at time of v5 peak:
1. comet_reaper, 2. comet_reaper_mcts, 3. comet_reaper_search, 4. comet_reaper_tuned, 5. coordinated_strike_interceptor, 6. markowitz_portfolio_optimization, 7. schmeekler, 8. schmeekler_potential

**Kaggle submissions table** (Jun 17 `12.55.56 PM` screenshot):
| Sub ID | File | Date | Score |
|---|---|---|---|
| 53785483 | schmeekler_fmt.tar.gz | 2026-06-17 18:55 | PENDING |
| 53779065 | schmeekler_fmt.tar.gz | 2026-06-17 08:45 | 1074.4 |
| 53770080 | main.py (coordinated_strike) | 2026-06-14 22:34 | 523.5 |

---

### v6 RL PPO: Jun 18–19 Sessions

**RL Dashboard tab** (`6.34.00 PM` Jun 18 screenshot): First view of the RL Training tab. State at start of fleet deployment:
- Active runs: 3, Fleet SPS: 1,612, Total env-steps: 3.80M
- ETA to 100M steps: 16.6h
- **Gate: RED** — CF=0, reward signal too weak
- Best run: h1_test · U=148 · 606,208 steps (CF=0.0015)

**Bug triage table** (`8.23.01 PM` Jun 18 screenshot): This matches and confirms the TIMELINE.md list. Screenshot shows exact 3-bug table for the first round:

| # | Bug | Effect | Fix |
|---|---|---|---|
| 1 | GAE grouping by env_i only | 70–80% of advantages = 0, policy couldn't learn | Fixed 2 hours prior |
| 2 | ent_coef=0.05 | Entropy bonus 50× game signal, policy stayed random | Fixed, redeploying |
| 3 | evaluate_vs_greedy(20) | Eval tested null agent, always returned 25% | Fixed, redeploying |

**Anvil HPC access discovered** (`8.18.38 PM` Jun 18): Screenshots reveal Ted had access to **Anvil HPC cluster** (Purdue) with partition data showing 21/21 GPU slots idle and 93 idle 128-CPU wholenode nodes. This was considered but the main training ran on Jetstream2. The CPU vs GPU analysis showed: Anvil CPU node (128 CPU) → ~138 SPS vs Anvil GPU node (128 CPU + A100) → ~70–80 SPS. Since the bottleneck is game simulation (CPU), not neural net updates, more CPU cores beat GPU.

**Fleet deployment fix confirmation** (`8.36.18 PM` Jun 18): After fixes redeployed, CF went from ~0.000–0.034 (broken ent_coef) to 0.012–0.079 across all remote instances. SPS jumped from 46–47 to 63–68. This is the moment the fleet started training correctly.

**7-bug table** (`8.54.32 PM` Jun 18): The full bug count grew to 7 by end of session (same as TIMELINE.md's 8-bug list, with one more found later):

| # | Bug | Impact |
|---|---|---|
| 1 | GAE grouping per-planet as sequential timesteps | Advantages = 0 |
| 2 | ent_coef=0.05 | Entropy 50× game signal |
| 3 | evaluate_vs_greedy(20) passed int not policy | Always 25% null baseline |
| 4 | prev_adv baseline never updated per-step | Shaped rewards = cumulative drift |
| 5 | reset_env called once per planet on done | Garbage ep_count, wasted resets |
| 6 | GAE rollout boundary used nv=0.8 for live episodes | Undervalued last-step return |
| 7 | ship_advantage re-evaluated after env reset mid-loop | Wrong delta for 2nd+ planets on terminal step |

**Fleet health at U=10 post-fix** (`7.52.55 PM` Jun 18 and `3.31.06 AM` Jun 19): Full per-job CF/EV/ENT/SPS table from screenshots. At U=10 after relaunch:
- CF: 0.012–0.079 (positive, in-band)
- EV: 0.341–0.713 (varying, positive)
- ENT: 4.956–4.991 (healthy, not collapsed)
- SPS: 46–68 (64–68 after scaling)
- EP: 0 across all (no terminal events yet)

**Entropy collapse evidence** (`9.15.37 PM` Jun 19): Multi-pane monitor showing some jobs with ENT=0.00 (complete entropy collapse) while others still at 4.35. Mixed fleet health at higher update counts. This is the "converged to local optimum" finding from TIMELINE.md, visible in the monitor data.

**Champion v3 promotion** (`9.18.03 PM` and `9.19.35 PM` Jun 19): Screenshots show champion promoted. At U=400:
- ppo-1-of-3_j1: 80–75.29% vs greedy (confirmed 9/9 hosts)
- vs comet_reaper: **0.000% at every checkpoint tested through U=400**

**Final RL state** (`11.47.37 PM` Jun 19): Last screenshot of the RL track. Question: "Can ppo-v3 score 12/50 wins (24%) vs comet at U=500?" Next check scheduled for 08:13 MT. Still running, prognosis bearish.

**Jun 20 v7 critic A/B** (`12.54.49 AM` Jun 20): Despite v6 being closed per TIMELINE.md, this screenshot shows a `v7_critic_ab` experiment scaffold being set up — new `SPEC.md`, `launch_control.sh`, `launch_variant.sh`. `reward_scale=0` hypothesis being tested. 18 files changed, 343 insertions. This appears to be a brief continuation after the v6 "close" decision, possibly exploring whether a different reward structure could break the 20% ceiling.

---

### Notable Findings from Screenshots Not in Other Sources

1. **The leaderboard top** was much more spread than expected. Jun 14 dashboard showed #1 at 1767.5, #2 at 1743.8, #3 at 1624.4, #10 at 1508 — a real spread across the top 10, not a single dominant cluster.

2. **coordinated_strike_interceptor in local arena**: 4th place at 63% win rate in Phase 0 arena. Was submitted as the second bot. Live score 531.7 vs markowitz's 578.2 — consistent with local ranking.

3. **schmeekler_fmt's live trajectory was noisy**: 1048 → 1126 → 1144 → 1167 → 1126 within a single afternoon (~28–51 episodes). The reversal from 1167 back to 1126 is visible in the screenshots. Its final settled score (~1074–1126) confirmed it was below comet_reaper's 1234.7.

4. **The "no-op but active" pattern**: Schmeekler ran at 62% action rate vs comet_reaper's 22%. This was visible behaviorally in game replay charts — schmeekler dispatched ships far more frequently. This higher activity rate is the mechanism behind the static-bonus effect.

5. **Anvil HPC was available and idle but unused**: 93 wholenode CPU nodes idle, 21/21 GPU slots idle. The decision to stay on Jetstream2 was based on deployment convenience (already set up), not resource availability.

6. **The "No Lag" bot intel** (in champion v3 screenshot): "Orbit Wars Agent (with No Lag) 1300+" appeared in the leaderboard data visible in one screenshot. The "(with No Lag)" label indicates latency-optimized orbit_lite variants were reaching 1300+ by Jun 19. This was the final credible lead that went uninvestigated before the RL track was closed.

---

## Cross-Reference Index

| TIMELINE section | Supplementary detail above |
|---|---|
| Phase 0 (23 bots) | §17 (local arena full ranking table, Phase 0) |
| Phase 2 (first submissions) | §17 (exact Kaggle scores: markowitz 578.2, CSI 531.7) |
| Phase 3 (comet_reaper) | §1 (engine anatomy, no-op rate, compute budget) |
| Phase 4 (BC dead end) | §12 (forum confirms), §4 (why cloning ≠ planning depth) |
| Phase 5 (fork bake-off) | §2 (why n=28 was too noisy) |
| Phase 6 (schmeekler) | §3 (sweep data, why it worked), §2 (gauntlet design), §17 (live trajectory) |
| Phase 6 (Optuna tuning) | §5 (37 trials, knob directions, 4P surface) |
| Track A comet factorial | §6 (2×2 design, 92% never-captured stat) |
| Track B Boltzmann stochastic | §4 (stochastic model theory, oracle problem, 5 bugs) |
| Track B MCTS/search | §4 (Numba benchmark, 0–4 candidate ceiling, anatomy), §17 (44 zero-candidate turns) |
| Track C value function | §7 (AUC result, why bolt-on fails), §17 (full fidelity probe metrics) |
| Track D multi-fleet | §8 (economic analysis, −75.6 avg, staggered-arrival problem) |
| schmeekler_elim | §10 (bug, per-opponent breakdown, 1266-elo pattern) |
| v6 GAE bug | §13 (entropy signature, fix mechanism, 8-bug list), §17 (bug triage table) |
| v6 RL failure | §14 (structural ceiling, inverted-U pattern, 3 cold-start regimes), §17 (fleet metrics, entropy collapse) |
