# The orbit_lite Family — Deep Dive (the engine behind the top of the leaderboard)

**The single most important finding from this whole study:** the top ~7 public bots are not
7 different agents. They are **one codebase** — Slawek Biel's `orbit_lite` planner ("The
Producer") — with different config presets and small bolt-on phases. Beating the leaderboard
is therefore a *policy-layer* problem on top of a known, shared engine, not a "build a better
heuristic from scratch" problem.

## The engine (`producer-orbit-wars-utils/orbit_lite/`, ~4,400 LOC, torch + stdlib)

A batched (designed for `B` games × `P` planets on GPU) tensor RTS planner, ported to
single-game for Kaggle. Calibrated on 535 real Kaggle replays. Per-turn pipeline (`run_turn`):

1. **`parse_obs` + `PlanetMovement`** — builds a forward model of the board over a horizon `H`:
   every planet's position, owner, and garrison **projected at each future turn 0..H** under a
   do-nothing assumption (`garrison_status`). This is the core advantage over reactive bots.
2. **Shortlists** (`build_target_shortlist`): up to `max_offensive_targets` enemy/neutral
   targets by proximity ∪ `max_defensive_targets` of my own planets the projection shows
   **flipping** within `H` (ranked by urgency = ships I'd lose). Sources = my planets with
   ≥ `min_ships_to_launch`, ranked by garrison.
3. **`safe_drain`** (closed form): the max ships a source can shed while the do-nothing
   projection keeps it **held** over `H`. This is the only fleet size it ever sends — one size
   per source, "send the safe surplus or nothing."
4. **`capture_floor`** with **ETA-aware reinforcement risk**: ships needed to take a target at
   arrival turn `k`, inflated by `reinforce_size_beta · ρ(k) · reachable-enemy-mass`. ρ ramps
   with flight time, so it **declines captures the enemy has time to reinforce mid-flight** —
   a heuristic patch for the fact that the flow scorer projects opponents as passive.
5. **`reachable_mask` + `intercept_angle`**: provably-correct swept-segment physics — accounts
   for the target's orbital motion, sun avoidance, surface offsets, and fleet speed scaling
   with fleet size. Never false-prunes a viable shot.
6. **Scoring** (`score_candidates` → `competitive_score`): each candidate wave is scored by the
   **exact combat flow projection** as `Δnet_me − Σ_opp Δnet_opp` (net ships = produced −
   lost). Zero-sum-aware: a launch is worth my gain minus opponents' gains.
7. **`_greedy_select`**: fire the best wave each iteration (one per target, source-budget
   aware, role-mutex so a reinforced planet can't also be a source), up to `max_waves_per_turn`,
   only if score > `roi_threshold`.
8. **`_plan_regroup`** (pressure-gradient): marshal leftover ships from safe planets toward
   stressed frontier planets, capped by `safe_drain`, only if the destination is still mine on
   arrival.

## Structural weaknesses (where to attack)
These are shared by the ENTIRE family — exploit any of them and you beat all of them at once:

- **Opponents modeled as do-nothing.** The only reactivity is the `reinforce_beta` floor patch
  and the defensive-flip shortlist. No real opponent move prediction, no anticipation of THEIR
  attacks beyond "my planet will flip." A bot that feints/baits can mislead the projection.
- **4P treats opponents as an equal-weight sum** (`Σ_opp`). Only i-m-stronger/floor-matched add
  a mild `ffa_leader_attack_bonus` (0.035). No coalition modeling, no "let A and B bleed each
  other" meta-play. **4P is where the prize money is won** and the family plays it shallow.
- **One fleet size per source** (`safe_drain`). No partial commits, no splitting a garrison
  across two targets in one turn except via the multi-contributor wave on a *single* target.
  Suboptimal ship allocation under contention.
- **Static horizon** (18/20). Captures requiring >H turns of orbital setup are invisible. Long
  rotating-planet snipes and slow multi-turn sieges are out of its planning window.
- **Greedy, not search.** `_greedy_select` is myopic per-turn; no lookahead over its OWN future
  turns, no MCTS. The lb-1110 and search-value bots *do* search and still score lower — but a
  search bot tuned against the orbit_lite family specifically could exploit its predictable greedy responses.
- **`roi_threshold` is a blunt global gate.** v2-gru's whole premise was to modulate it
  dynamically with a GRU — and they **shipped it disabled**. That's free signal left on the table.

## Evolutionary tree (config deltas)
| Bot | Score | What it changed |
|---|---|---|
| the-producer-v2 | ~1248 | Baseline. 2P: H=18, roi=1.5, β=2.2, 6 waves. 4P: H=13. |
| i-m-stronger | ~1226 | 4P **anti-leader bias** `ffa_leader_attack_bonus=0.035`; tighter 4P (offensive 7, roi 1.55, min_ships 5). |
| v2-gru | — | i-m-stronger + a `TinyGRU` meta-controller that would modulate roi/waves/ffa from a 12-step feature history — **but `GRU_WEIGHTS_AVAILABLE=False`, so it runs identically to i-m-stronger.** The neural net is dormant. |
| exp50 | ~1175 | **Terminal phase**: in the endgame, roi→1.0, 9 waves, regroup off (dump everything). |
| floor-matched | ~1175 | **Comet evacuation** (flee own comet planets ~4 turns before expiry) + target veto on soon-to-expire comets + anti-leader bias. |
| agent-lyonel | ~1200 | Conservative tune: H=15, 5 waves, roi=1.4, with config-validation asserts. |
| v44 (1266 elo) | ~1266 | Most aggressive: H=20, roi=1.2 (fires more), 7 waves, terminal phase. Ships its own orbit_lite copy. |

**Takeaway:** the gap from #1 (producer-v2 ~1248) to v44 (~1266) is *pure config tuning* of one
engine. The biggest unexploited levers in the family's own design are (a) a working learned
roi/wave controller (v2-gru's dead GRU), (b) real 4P coalition play, and (c) opponent modeling
to replace the do-nothing projection. See `strategy/v4_candidates.md`.
