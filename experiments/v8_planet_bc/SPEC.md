# v8 — Per-Planet Behavior Cloning

**Opened:** 2026-06-20
**Branch:** v8-planet-bc
**Deadline:** 2026-06-23 (competition end)

---

## Hypothesis

Mendrika (48th, ~1420 Elo) achieved their score with pure behavior cloning using per-planet
fire heads trained on top-ranked replay data. We now have the same architecture:
`ActorCritic` in `agents/rl_ppo/train.py` has exactly per-planet heads (fire, target, frac).
The v4/Phase 4 BC failure used a **global policy** (`rl/policy.py:PlanetPolicy`), not the
current per-planet design. This is the one untested combination.

**If BC with per-planet heads lands > 1235 Elo → surpasses comet_reaper.**

Prior BC failure (`rl/bc_train.py`) was explicitly noted as a dead end due to the global
architecture, not a fundamental ceiling on BC itself.

---

## Data Pipeline

### Step 1: Download prize-zone episodes
```bash
python pipeline/pull_topbot_episodes.py \
    --days-back 5 --max-per-day 300 --require-rating 1400
```
Downloads to `episodes/<date>/<episode_id>.json`. Target: ~1000-1500 prize-zone episodes.

### Step 2: Extract labeled moves
```bash
python pipeline/extract_moves.py \
    --src episodes --min-rating 1400 \
    --out training/moves_v8.jsonl.gz
```
Produces (obs, action) pairs stamped with team rating.

---

## Architecture

Reuse `ActorCritic` from `agents/rl_ppo/train.py` verbatim.

Per-planet decisions at each step:
- `fire`: Bernoulli — did this planet launch? (CE on binary label)
- `target`: Categorical over MAX_PLANETS — which planet was targeted? (CE on index)
- `frac`: Categorical over N_FRACS=[0.25, 0.5, 0.75, 1.0] — ship fraction bucket (CE on bucket)

Only owned planets with ships >= MIN_SHIPS generate training examples.

### Action parsing
Each recorded action: `[from_planet_id, angle_rad, num_ships]`

- Build `launched_from = {from_planet_id: (angle, ships) for each launch}`
- For each owned planet p at index i in planets list:
  - `fire_label = 1 if p[0] in launched_from else 0`
  - If fire=1:
    - `target_idx = argmin_j |atan2(py_j - py_src, px_j - px_src) - angle|` (across all planets)
    - `frac_label = argmin_k |FRAC_BINS[k] - ships/p_ships|`
  - If fire=0: skip target/frac (don't penalize — we only train on launched planets for target/frac)

### Loss
```
L = CE(fire_logit, fire_label)                              # all owned planets
  + CE(tgt_logit[launched], tgt_label[launched])            # only launched
  + CE(frac_logit[launched], frac_label[launched])          # only launched
```

---

## Training Script

`agents/rl_ppo/bc_train.py` — new file, reuses `encode_obs` / `ActorCritic` from train.py.

```
python agents/rl_ppo/bc_train.py \
    --data training/moves_v8.jsonl.gz \
    --epochs 10 \
    --lr 1e-3 \
    --batch-size 512 \
    --out agents/rl_ppo/checkpoints/bc_best.pt
```

---

## Decision Gates

### Gate 1: Data sanity (~30 min)
- Extract moves → inspect summary.json
- Need: ≥ 500 episodes with ≥ 1 launch step each, from teams rated ≥ 1400
- If data is sparse: lower --require-rating to 1200 and retry

### Gate 2: BC training loss (~1-2h local)
- fire accuracy > 70% on held-out steps → network is learning who fires
- target accuracy > 30% → network is learning where to fire (chance = 1/n_planets)
- If neither metric moves after 3 epochs → architecture mismatch, check obs encoding

### Gate 3: comet_reaper_WR (the real test)
```bash
python agents/rl_ppo/eval_checkpoints.py \
    --checkpoint-a agents/rl_ppo/checkpoints/bc_best.pt \
    --vs-comet --n-games 200
```
- comet_reaper_WR > 0% → submit bc_best.pt via make_submission.py
- comet_reaper_WR = 0% → BC ceiling confirmed; close v8, keep comet_reaper as final

---

## Results Log

| Date | Gate | Metric | Value | Notes |
|------|------|--------|-------|-------|
| 2026-06-21 | Training | fire acc | 91% | 20 epochs, 982k examples |
| 2026-06-21 | Training | tgt acc | 19% (chance=5%) | Still improving at E20 |
| 2026-06-21 | Training | frac acc | 79% | |
| 2026-06-21 | Gate 3 | comet_reaper_WR | 0% (0/200) | Mean episode length 88 steps — crushed early |

## Verdict: CLOSED FAIL

**BC ceiling confirmed.** 0/200 vs comet_reaper bot, avg game length 88/500 steps —
the BC model is destroyed before the midgame. Same result as Phase 4 PlanetPolicy BC
(which lost 0-16). Training metrics looked healthy (fire=91%, tgt=19% vs 5% chance)
but imitation of *what* top players do doesn't capture *why*, and orbit_lite
exploits the gap immediately.

**comet_reaper (sub 53707586, ~1235 Elo) is the final submission.**
