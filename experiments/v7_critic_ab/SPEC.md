# v7 — Critic A/B Experiment

**Opened:** 2026-06-20  
**Deadline:** 2026-06-21 18:00 MT  
**Branch:** v7-critic-ab  
**Fleet:** 9× m3.xl (ppo-1 through ppo-9, homogeneous for clean comparison)

---

## Hypothesis

v6 showed EV 0.75–0.84 vs Billy's (7th place) 0.90–0.96. The critic is underfitting.

**Primary cause:** The dense per-step reward (`delta * reward_scale=0.01`) adds
high-frequency noise to GAE returns. Ship advantage oscillates throughout a game, making
the cumulative discounted return harder to predict than a pure sparse signal. With
`reward_scale=0`, returns simplify to `gamma^(T-t) * ±1` — much cleaner for the critic.

**Confirmed facts (before deploying):**
- Terminal reward IS already ±1 (orbit_wars.py lines 712-715: winner gets 1, loser -1)
- Win condition: highest total ship count (planets + in-flight fleets) at termination
- Scale-mismatch hypothesis (large raw score) was WRONG — terminal is already binary
- #1047 trig: atan2(dy,dx) consistent. Coordinate order: unverified, low-risk.

---

## Arms

### CONTROL — reward_scale=0.01 (current config)
- Instances: ppo-1, ppo-2 (2 control jobs each, alongside variant on same box)
- Run name prefix: `ctrl`
- Launch: `bash experiments/v7_critic_ab/launch_control.sh IP`

### VARIANT-A — reward_scale=0.0 (pure sparse terminal ±1)
- Instances: ppo-1, ppo-2 (2 variant jobs each, same box as control for isolation)
- Run name prefix: `var_a`
- Launch: `bash experiments/v7_critic_ab/launch_variant_a.sh IP`

### SPARE (ppo-5 through ppo-9)
- Reserved for VARIANT-B (separate critic head) if VARIANT-A EV filter fails
- Do not deploy until VARIANT-A fast filter result is known

---

## Decision Gates

### Fast filter (~150 updates, ~1–2h)
Watch `explained_variance` in training log every 10 updates.

| Result | Action |
|--------|--------|
| VARIANT-A EV climbs toward ≥0.90, CONTROL stays ~0.80 | Advance to slow gate |
| VARIANT-A EV flat / stays near CONTROL | Hypothesis A failed → deploy VARIANT-B (separate critic head) |
| Both flat | Critic is not the bottleneck → close RL, comet_reaper is final |

### Slow gate (n=1000, NOT n=30)
Run `eval_checkpoints.py` manually for any arm that passes EV ≥ 0.90.

| Result | Action |
|--------|--------|
| vs_comet_reaper_WR > 0% | Signal found — press fleet on that arm |
| vs_comet_reaper_WR = 0% | Critic fixed but didn't close comet gap → RL closed |

### Hard kill: Jun 21 18:00 MT
Regardless of state: no arm passes slow gate → RL bet is closed.
comet_reaper (sub 53871873) is the final submission.

---

## Results Log

*(append here as cycle data comes in)*

| Time | Arm | U | EV | vs_greedy_WR | vs_comet_reaper_WR | Notes |
|------|-----|---|----|--------------|--------------------|-------|
