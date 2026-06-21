# v7 — Critic A/B Experiment

**Opened:** 2026-06-20  
**Closed:** 2026-06-20 — FAIL  
**Deadline:** 2026-06-21 18:00 MT  
**Branch:** v7-critic-ab  
**Fleet:** 2 active (ppo-1, ppo-2); ppo-3 through ppo-9 shelved to save SUs — unshelve if VARIANT-B needed

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

### SPARE (ppo-3 through ppo-9 — shelved)
- Unshelve and deploy if VARIANT-B (separate critic head) is needed
- Do not unshelve until VARIANT-A fast filter result is known

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

| Time | Arm | U | EV | greedy_WR | comet_reaper_WR | Notes |
|------|-----|---|----|-----------|--------------------|-------|
| 2026-06-20 07:xx | ctrl | 100 | 0.97 | 30–47% | 0% | ppo-1+2, 4 seeds |
| 2026-06-20 09:xx | ctrl | 200 | 0.93 | 33–43% | 0% | regression from U=100 |
| 2026-06-20 xx:xx | ctrl | 300 | 0.87 | 30–37% | 0% | continued regression |
| 2026-06-20 07:xx | var_a | 100 | 0.97 | 30–40% | 0% | ppo-1+2, 4 seeds |
| 2026-06-20 09:xx | var_a | 200 | 0.93 | 27–47% | 0% | mixed; one seed improved |
| 2026-06-20 xx:xx | var_a | 300 | 0.85 | 27–43% | 0% | entropy collapsing |
| 2026-06-20 14:xx | both | 390–400 | 0.65–0.97 | — | 0% | entropy 1.18–3.13; fleet killed |

## Verdict: CLOSED FAIL

**Fast filter:** Both arms hit EV ≥ 0.90 from U=10 onward — but this was trivially due to
short rollouts (64 steps / 499-step game). Most buffer steps have near-zero dense reward,
so the critic trivially predicts zero and achieves high EV. The fast filter was not a useful signal.

**Slow gate:** comet_reaper_WR = 0% at every checkpoint (n=100 games) across all 8 seeds.
No signal at U=100, 200, or 300.

**Hypothesis verdict:** reward_scale made no difference. Both ctrl (0.01) and var_a (0.0)
showed identical behavior: greedy WR peaks at U=100 (~40%), decays by U=300, entropy collapses
by U=400. Same pattern as v6 with different config.

**Root cause:** RL from scratch cannot close the gap to comet_reaper at this compute scale
(~1.6M steps, ~67 SPS on CPU). The comet_reaper cold-start opponent provides no learning
signal because the policy collapses to entropy=1 before accumulating enough experience.

**Next:** v8 — per-planet behavior cloning from top-ranked game replays.
