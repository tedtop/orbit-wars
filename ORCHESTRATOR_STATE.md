# ORCHESTRATOR STATE — v6 RL/PPO Campaign
*Branch: v6-rl-selfplay | Updated: 2026-06-18 ~19:46 MT | Deadline: 2026-06-23*

---

## CURRENT STATUS: 🟡→🟢 EVAL BROKE 25% FLOOR — FLEET RUNNING STALE CODE

**Four fixes confirmed in code, eval methodology also fixed:**
1. ✅ GAE step_i fix — CF from 0.003 → 0.06+ 
2. ✅ comet_reaper cold-start opponent — combat signal from step 1
3. ✅ ent_coef=0.001 in PPO loss — prevents premature convergence
4. ✅ **eval_fix: n_games=200 + both seats (P0 AND P1)** — was n_games=20 P0-only (biased)

**eval_fix_test (200 updates, 204K steps, 16 envs) — BREAKTHROUGH:**
- **40% win rate vs greedy at U=100** (was stuck at 25%) ✅ FIRST IMPROVEMENT
- **35% at U=200** — slight dip but well above 25% floor
- Entropy: 5.013 → 2.664 (aggressive decay — policy committed to strategy)
- CF: 0.087 mean, 85%+ in-band throughout

**full_eval_test (at U=100, eval imminent as of 20:38):**
- CF mean=0.076, entropy=4.342 — healthy
- Will show first eval result shortly — confirmation pending

**Remote fleet (DEPLOYED BUT STALE):**
- ALL 32 jobs running across 8 instances ✅
- BUT: **deployed before ent_coef fix** — CF=0.028 (stale code behavior)
- SPS per job: ~65 (Jetstream2 Xeon is 30× slower than local Apple M3 for Python steps)
- MUST REDEPLOY with current code: `bash agents/rl_ppo/deploy_all.sh`

---

## Fleet (8 instances total)

### m3.2xl — 64 CPU, 250 GB RAM (original)
| Instance | IP | Status |
|---|---|---|
| orbit_wars_ppo 1 of 6 | 149.165.174.18 | Ready — launch pending |
| orbit_wars_ppo 2 of 6 | 149.165.174.133 | Ready — launch pending |
| orbit_wars_ppo 3 of 6 | 149.165.171.142 | Ready — launch pending |
| orbit_wars_ppo 4 of 6 | 149.165.170.73 | Ready — launch pending |
| orbit_wars_ppo 5 of 6 | 149.165.171.248 | Ready — launch pending |

### m3.xl — 32 CPU, 125 GB RAM (bonus fleet, added 2026-06-18)
| Instance | IP | Status |
|---|---|---|
| orbit_wars_ppo 1 of 3 | 149.165.175.105 | Ready — deploying now |
| orbit_wars_ppo 2 of 3 | 149.165.170.84 | Ready — deploying now |
| orbit_wars_ppo 3 of 3 | 149.165.175.177 | Ready — deploying now |

**Fleet target: 32 parallel runs (8 machines × 4 jobs) at ~17K SPS**
- 5 × m3.2xl × 4 jobs × ~540 SPS = ~10,800 SPS
- 3 × m3.xl  × 4 jobs × ~540 SPS = ~6,480 SPS

---

## Protected Floor
| Bot | Live Score | Status |
|-----|-----------|--------|
| comet_reaper | 1234.7 | Inactive (protected) |
| schmeekler_fmt | ~1125 | Active, converging |

**Target:** Beat comet_reaper at n≥150 in both 2P and 4P regimes.
**Prize cutoff:** ~1534 Elo.

---

## Correctness Gate Checklist

| Check | Target | Observed | Status |
|-------|--------|----------|--------|
| CF (with all 3 fixes)   | 0.05–0.3 | **0.096 mean, 85% in-band** | ✅ PASSING |
| Eval vs greedy          | >50%     | 25% at U=100, 200 — scale blocker | 🟡 SCALE — need 100M steps |
| Entropy | decaying gradually | -1.17 nat in 200 updates | ✅ HEALTHY |
| Explained variance | rising | oscillating ~0.06 mean (low-n artifact) | 🟡 BORDERLINE |
| P0/P1 balance | ~50% | not checked | 🔴 PENDING |

**Gate result: 🟡 YELLOW — all 3 code fixes confirmed; eval stuck at 25% due to insufficient steps (204K vs 100M target). Fleet deployment is the unlock.**

---

## Root Cause Analysis (REVISED — confirmed by H1 DISCARD)

**The bug is in `compute_gae`, NOT in reward scale.**

`compute_gae` groups buffer entries by `env_i` and treats ALL per-planet decisions from all
timesteps as one flat sequential array. For a game where the agent owns N planets per turn:

- Planet decisions 0 through N-2 at timestep t: `nv = vals[next_entry] = V(same_state_t)`
  → `delta ≈ r_t + γ×V(s_t) − V(s_t) ≈ r_t` (bootstrapping collapses — same state!)
- Planet decision N-1 (last) at timestep t: `nv = V(s_{t+1})` → proper TD delta ✅

**Result:** Only 1/N planet decisions per timestep get a real bootstrapped advantage. The rest
get `adv ≈ r_t` (tiny). After advantage normalization, those all become near-zero. CF≈0.

This explains why 10× reward scale made zero difference: normalization erases the signal regardless
of scale when N-1 of N entries have the same small magnitude.

**Entropy stuck at 5.0 for 300 updates confirms:** policy weights are essentially unchanged.

**The fix (one function rewrite):**
Tag each buffer entry with `"step_idx"` (which rollout step, 0..rollout_steps-1).
In `compute_gae`, group by `(env_i, step_idx)` → compute one V(s_t) per timestep → compute
one GAE advantage per timestep → broadcast that advantage to all planet decisions at that step.

```python
# Pseudocode for fixed compute_gae
for env_i in envs:
    # Get unique timesteps in order
    timesteps = sorted(set(e["step_idx"] for e in env_entries[env_i]))
    for t in timesteps:
        # All planets at this step share the same obs → same V(s_t)
        val_t  = entries_at[env_i][t][0]["val"]   # same for all planets
        rew_t  = entries_at[env_i][t][0]["rew"]   # same for all planets
        done_t = entries_at[env_i][t][0]["done"]  # same for all planets
        val_t1 = 0.0 if done_t else val_at[env_i][t+1]
        delta_t = rew_t + gamma * val_t1 - val_t
        # Broadcast to ALL planets at this timestep
        for entry in entries_at[env_i][t]:
            entry["adv"] = gae_t  # same for all
```

---

## Open Hypotheses (priority order)

| # | Hypothesis | Metric | Status |
|---|-----------|--------|--------|
| **BUG** | GAE treats per-planet decisions as sequential timesteps → ~1/N get real gradient | CF enters 0.05–0.3 after fix | **CRITICAL — fix before anything else** |
| H2 | P0 systematic advantage (feature asymmetry #1047) | P0 win% in symmetric self-play = 50% | Pending (post-GAE-fix) |
| H3 | Per-planet action heads cause vanishing gradient in early training | gradient norms per head | Pending (may be solved by GAE fix) |
| H4 | Self-play alone insufficient — BC warmup needed | EV rate vs random init | Pending |

---

## Experiment Log (KEEP/DISCARD)

| # | Change | Result | Verdict |
|---|--------|--------|---------|
| 0 | Initial scaffold (reward_scale=0.001, terminal_bonus=1.0) | CF~0, eval=25%, no learning | ❌ RED GATE |
| 1 | H1: reward_scale=0.001→0.01 (300 updates) | CF still ~0.003, eval=**25%** — zero change | ❌ DISCARD — not the root cause |
| 2 | **GAE fix: step_i tagging + (env_i,step_i) grouping + comet_reaper cold-start** | CF mean 0.062, 29/50 in-band | ✅ **KEEP** |
| 3 | gae_full (no ent_coef): CF crashed 0.057→0.014 by U=300, 4× evals all 25% | Policy convergence without entropy reg | ❌ DISCARD — confirmed ent_coef is required |
| 4 | **ent_fix_test (ent_coef=0.001)**: CF 0.096, 85% in-band, entropy -1.17 nat, eval=25% @ U=200 | Old eval was biased (20 games, P0 only) | ✅ **KEEP** ent_coef |
| 5 | **eval_fix: n_games=200 + both seats**: eval_fix_test → **40% at U=100**, 35% at U=200 | First improvement above 25% floor | ✅ **KEEP** — gate breaking GREEN |
| 6 | **full_eval_test**: CF=0.076, entropy=4.342 at U=100 — eval imminent | confirming 40% result | 🔜 PENDING |
| 7 | Fleet redeploy with current code (ent_coef + eval_fix) | expect CF=0.08+ on remotes | 🔜 **DEPLOY NOW** |

---

## League Composition
- **Current:** Pure self-play (no opponent pool — pool requires ≥1 checkpoint at update 1000)
- **Next addition (U=1000):** comet_reaper + orbit_lite as fixed opponents

---

## Dashboard
- Run: `.venv/bin/streamlit run dashboard/app.py` → RL Training tab
- Shows: fleet table (all 32 runs), CF/EV/entropy/loss charts, eval chart, fleet SPS + ETAs
- Sync metrics from remotes: `bash agents/rl_ppo/sync_checkpoints.sh`

---

## Eval Matrix (last run)
| Run | U | Eval vs greedy | Notes |
|-----|---|----------------|-------|
| train_local (rs=0.001) | 200 | 25% | Fire head collapsed — GAE bug |
| h1_test (rs=0.01) | 200 | **25%** | Identical — confirms GAE bug, not reward scale |
| gae_fix_test (smoke) | 50 | not reached | CF passing, 73 eps — KEEP. Stopped early. |
| gae_full (pending) | 200 | **TBD** | Must launch now |

---

## Submission-Ready Checkpoints
None yet. `best_model.pt` requires eval >25% (current best).

---

## Next Actions (ordered)

### 🚨 IMMEDIATE (do before anything else):
1. ~~Fix `compute_gae`~~ — **DONE** ✅
2. ~~Entropy fix (ent_coef=0.001)~~ — **DONE** ✅
3. ~~Eval fix (n_games=200, both seats)~~ — **DONE** ✅ → 40% at U=100!
4. **REDEPLOY FLEET NOW**: `bash agents/rl_ppo/deploy_all.sh` — fleet is running stale code (pre-ent_coef)
   - Kill existing stale jobs first on each host, then redeploy
   - Expected SPS: ~65/job (Jetstream2 Xeon, much slower than Apple M3 for Python)
   - At 65 × 32 jobs = 2,080 total SPS → 100M steps in ~13.4 hours (not 1.6h as estimated)
5. **Wait for full_eval_test eval result** (happening right now) — confirms 40% or higher
6. **Sync metrics every cycle**: `bash agents/rl_ppo/sync_checkpoints.sh`

### Then:
7. **At U=1000:** Add orbit_lite + comet_reaper to opponent pool
8. **Vs comet_reaper eval** as soon as any run hits U=200 with proper code
9. **Wake Ted:** When RL policy beats comet_reaper at n≥150 across both 2P + 4P

---

## PPO Health Targets (reference)
- `clip_frac`: 0.05–0.30 (near 0 = not learning; >0.5 = exploding updates)
- `explained_variance`: start near 0, should rise steadily toward 0.8+
- `entropy`: should start high (~5.0 for 3 combined heads) and decay to ~1.0 over training
- `kl_approx`: typically 0.005–0.02 per update
- `value_loss`: should converge from ~0.5 to ~0.05 as value head fits returns

## AgileRL Note (from dashboard)
Candidate for Population-Based Training (PBT) hyperparameter search. Requires wrapping
orbit_wars as a PettingZoo env + custom policy extractor for per-planet heads (~1 day work).
Not relevant until gate is green and we want to sweep lr/clip_eps/ent_coef at scale.
Pocket it for after U=1000 if current hyperparams aren't converging fast enough.
