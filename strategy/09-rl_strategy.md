# Orbit Wars — RL Strategy (BC → self-play PPO)

The path to top-10 is a policy that **exceeds** hand-tuned play. Pure from-scratch RL on a
500-step, huge-action-space RTS is slow and fragile. The proven recipe (AlphaStar / TLOL) is
**behavior-clone a strong teacher, then fine-tune with self-play PPO**. We have every ingredient:
orbit_lite as an unlimited scripted teacher, prize-zone replays as elite demos, the RL tutorial's
`PlanetPolicy`/PPO as a starting codebase, and our local gym as the evaluation harness.

---

## 1. Action space — the single most important design choice
Raw actions (per owned planet: continuous angle + integer ships) are intractable. **Factor it the
way the strong bots think:** aiming is solved physics; only *target selection* and *sizing* are
strategic.

**Per-source factored policy** (extends the RL tutorial's `PlanetPolicy`):
- For each owned planet *s*, generate **K candidate targets** (e.g. K=8 nearest enemy/neutral
  planets + highest-production reachable). The policy outputs, per source:
  - a **target head**: softmax over {no-op, cand₁…cand_K} (with masking of illegal/unreachable),
  - a **ship-bucket head**: softmax over B buckets (fractions of safe-drain garrison, e.g. {0, ¼,
    ½, ¾, all}).
- **Angle is computed deterministically** by orbit_lite's `intercept_angle` lead-aim solver — the
  policy never learns geometry, only "which planet, how many ships." This collapses a continuous
  joint action into a small discrete per-source choice and reuses the provably-correct physics.
- All sources decided in one forward pass (shared encoder + per-source heads) → a factored joint
  action. This is exactly the tutorial's structure; we keep it.

**Action masking** (critical): mask candidates that are unreachable, sun-blocked, or that the
forward-projection says we can't capture; mask ship-buckets exceeding `safe_drain`. Masking is the
difference between RL that learns and RL that flails.

---

## 2. Observations / features
Reuse + extend the tutorial's `features.py` (self dim 11, candidate dim 14, global dim 8):
- **Self (per source)**: garrison, production, orbital radius/phase, dist-to-sun, # incoming
  threats, dist to nearest enemy, safe-drain.
- **Candidate (per source×target)**: target ownership, garrison, production, ETA (travel turns),
  **defenders-at-arrival** (forward-projected — the orbit_lite advantage), is-comet + comet life,
  distance, sun-blocked flag, **owner strength** (4P).
- **Global**: my rank, #players alive, board-control share, **per-opponent score margins**, phase
  (early/mid/late/terminal), step/500.
- **ADD (our edge): 4P opponent-asymmetry features** — each opponent's strength, who-is-attacking-
  whom (from the arrival ledger), is-an-opponent-near-elimination. This is the signal the whole
  orbit_lite family ignores; giving it to a learned policy is where we beat them.

---

## 3. Reward structure  ⭐ (the part you asked me to ultrathink)

The native reward is **sparse**: win/loss (2P) or final placement (4P), only at game end. Sparse +
500 steps = very slow learning. We add **dense shaping that provably can't be gamed.**

### 3a. Terminal reward (the true objective)
- **2P:** +1 win / −1 loss.
- **4P:** placement-shaped, NOT linear — the rating system punishes last place hardest, so:
  `1st=+1.0, 2nd=+0.3, 3rd=−0.3, 4th=−1.0`. This directly encodes the OpenSkill variance-management
  insight (avoid catastrophic 4th) into the objective. (Tune the middle values; they set risk
  appetite.)
- Optional tiny margin term (final production share) to break ties among wins — keep small to avoid
  rewarding reckless over-extension.

### 3b. Potential-based shaping (dense, provably safe)
Use **potential-based reward shaping** (Ng, Harada, Russell 1999): `r' = r + γΦ(s') − Φ(s)`. Because
it telescopes over the episode, it **cannot change the optimal policy** — it only guides exploration.
This is the key to dense rewards without reward-hacking.

Define the potential as a normalized dominance score (production-weighted, mirroring orbit_lite's
own 535-replay early-termination calibration that weighs production 5:1 over ships in 2P):

```
Φ(s) = w_p · (my_prod − mean_opp_prod)        / (total_prod   + ε)
     + w_s · (my_ships − best_opp_ships)       / (total_ships  + ε)
     + w_c · (my_planets − mean_opp_planets)   / (total_planets+ ε)
   with w_p ≈ 0.6, w_s ≈ 0.2, w_c ≈ 0.2   (production leads; it compounds)
```

Every captured planet / lost garrison nudges Φ, giving a gradient every turn, while the telescoping
guarantees the agent still ultimately optimizes win/placement. **This is the core of the reward
design.**

### 3c. Curriculum shaping (non-potential, annealed to 0)
Early in training only, to break symmetry / bootstrap, then linearly annealed out so it can't
distort the final policy:
- small penalty for passing every turn early (encourage acting),
- **elimination bonus** in 4P for removing an opponent (seeds the asymmetry behavior),
- penalty for losing the home planet (survival).
Anneal all of these to 0 over the first ~30–40% of training.

### 3d. Discounting / advantage
- γ ≈ **0.997–0.999** (long horizon; production compounds over all 500 steps).
- **GAE(λ≈0.95)** to tame the variance that high γ + long episodes create.
- Normalize advantages per batch; normalize/clip the value target (returns span a wide range).

---

## 4. Algorithm & framework — what to install
- **Algorithm: PPO** (clipped, on-policy) — stable, the standard for self-play, and **already
  implemented in the RL tutorial** (`agents/opponents/.../src/ppo.py`) purpose-built for this exact
  candidate/ship-bucket action space with masking.
- **Framework recommendation:** start by **reusing the tutorial's PPO + PlanetPolicy** — zero new
  deps, and it already matches our action space (SB3 fits this structured action poorly; CleanRL is
  a fine upgrade later if we want its tooling).
- **What you'll need to install (minimal → recommended):**
  - Already have: `torch 2.12`, `numpy`, `kaggle-environments`.
  - **Minimal:** nothing — reuse the tutorial's PPO.
  - **Recommended:** `pip install tensorboard` (training curves) and `pip install gymnasium`
    (clean env API if we wrap). Optional `wandb` for run tracking.
  - GPU optional: the policy net is tiny (CPU-fine). A GPU only matters if we adopt orbit_lite's
    **batched** simulator for throughput (see §6) — then it's a big win.

---

## 5. Training curriculum (BC → PPO → league)
1. **BC pretrain.** Clone (a) orbit_lite/producer-v2 rollouts (unlimited, free) and (b) prize-zone
   replays (`extract_moves --min-rating 1500`). Supervised cross-entropy on (target head, ship
   head). Output: a fast NN that ~matches SOTA without the 1s torch-sim per turn. Low-risk insurance
   that already lands near the public ceiling.
2. **PPO vs the scripted field.** Fine-tune against the 14 downloaded bots (our gym = the training
   env). Shaped reward from §3. Learn to beat the known field.
3. **4P self-play league** (AlphaStar-style). Maintain a pool = past policy snapshots + scripted
   bots; sample opponents with **PFSP** (prioritized fictitious self-play — play more against those
   that beat us). Add periodic **exploiter** agents that train solely to beat our main agent, then
   fold their lessons back. This prevents self-play cycles/collapse and forces robust strategy.
4. **Train primarily in 4P** (3 league opponents) — the prize format. Coalition/anti-leader/
   elimination behavior (strategies #2/#6) **emerge** from 4P self-play rather than being hand-coded.

---

## 6. Throughput — train in orbit_lite's batched sim  ⭐ big lever
`kaggle_environments` runs ONE python game at a time (slow). **orbit_lite is a batched torch
simulator** (`movement.py`, the B-axis) the Producer team built precisely for GPU self-play —
hundreds/thousands of games in parallel on one GPU. If we train inside orbit_lite's sim:
- opponents that ARE orbit_lite are free (run in the same batch),
- throughput jumps ~100–1000×,
- it's the same engine the top bots use, so the learned policy is grounded in the real dynamics.
Plan: prototype the loop in `kaggle_environments` + our gym (simple, correct), then port the hot
loop to orbit_lite's batched step for scale. Always **validate in `kaggle_environments` + the gym**
to catch sim-to-real drift.

---

## 7. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Reward hacking | potential-based shaping (§3b) — provably preserves the optimum |
| Self-play collapse / cycles | league + PFSP + exploiter agents (§5.3) |
| Sparse/slow learning | dense Φ shaping + curriculum + BC warm-start |
| High-variance returns (γ→1, 500 steps) | GAE(λ), advantage & value normalization |
| Illegal actions | logit masking on target + ship heads |
| Sim-to-real gap (orbit_lite vs kaggle engine) | periodic gym/kaggle validation each iteration |
| Compute | tiny net (CPU ok); orbit_lite batched sim for scale; GPU optional |

---

## 8. Concrete next steps (what I'll build / what you do)
1. **You (when up):** approve the framework choice; if we want tooling, `pip install tensorboard
   gymnasium`. (Nothing required to start — tutorial PPO is self-contained.)
2. **Data:** finish the prize-zone pull (rate-limit cooldown) → `extract_moves --min-rating 1500`
   → BC dataset. Also generate orbit_lite teacher rollouts (unlimited).
3. **Code (I can scaffold next):** `rl/features.py` (extend tutorial + asymmetry feats), `rl/policy.py`
   (PlanetPolicy + masking), `rl/reward.py` (the §3 shaped reward), `rl/bc_train.py`, `rl/selfplay.py`
   (PPO loop + PFSP league), `rl/rl_agent.py` (inference → Kaggle agent).
4. **Validate** every checkpoint in the gym (2P + 4P) before it ever touches a submission slot.

**Bottom line:** BC gets us ~SOTA cheaply and low-risk; the 4P self-play league with production-
weighted potential shaping is what can push past ~1533. The reward design (§3) — placement-shaped
terminal + potential-based production-led dense shaping + annealed curriculum — is the heart of it.
