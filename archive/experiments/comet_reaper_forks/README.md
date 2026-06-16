# Archived experiment — comet_reaper forks

**Date:** 2026-06-15 · **Verdict:** no actionable champion. Filed away.

## What this was

A batch run that forked `comet_reaper` (the orbit_lite engine bot) five times, each
adding **one** bolt-on heuristic, and pitted them against `comet_reaper` to see if any
beat it. They didn't. Seat-swapped win% vs `comet_reaper` (50% = parity):

| fork | idea (one knob) | clean win% | verdict |
|------|-----------------|------------|---------|
| maestro   | standing-aware roi_threshold      | 50–52% | parity |
| precog    | opponent-response modeling        | ~50%   | parity |
| kingmaker | 4P leader-weighting               | ~parity | parity |
| helmsman  | deeper projection horizon         | parity (worse if >1.0) | idea fails |
| oracle    | recovered-objective target bias   | ~17%   | regresses |

**Lesson:** the orbit_lite engine is a tight, well-tuned local optimum that *already*
encodes the recovered target preferences (its zero-sum score prefers enemy captures),
so bolting more on top is parity-or-worse. **Submit `comet_reaper` unchanged** — it
ties the real Producer (14–14) and beats the rest of the public field (~67%).

The "overnight" name was an artifact of a fresh, context-light session; the real
content is this fork bake-off.

## Contents

- `RESULTS.md` — the full run readout (the morning summary + the appended driver log).
- `bots/{precog,kingmaker,maestro,helmsman,oracle}/` — the five fork bots (each a
  folder bot: `main.py` + a vendored `orbit_lite/`). Knobs are env-var driven; see
  each `main.py` header.
- `pipeline/{overnight.py,_eval_worker.py}` — the self-looping knob-tuning + self-play
  driver and its eval worker.
- `rl/objective_recovery.py` — Track A: learn what top teams target (the one real
  *insight*: enemy ≫ neutral, near-dead enemies, nearby, higher-production).
- `rl/train_clones.sh` — drove the behavior-cloning of top public players (see below).
- `logs/`, `notes/` — raw run logs and intermediate eval text dumps.

## The BC / self-play track was also a dead end

Related, and abandoned for the same reason. The full chain was:

1. **Behavior-clone the top public players** (`rl/bc_train.py`). Take the top teams'
   moves from the public episode replays, label each `(obs, action)` (recover which
   candidate planet a source launched at + a ship bucket), and train a neural
   `PlanetPolicy` (supervised cross-entropy) to *imitate* them. `--team "Jake Will"`
   clones one specific team; output is `training/clone_*.pt`.
2. **Self-play against the clones** (`rl/selfplay.py --clones`). Load those `clone_*.pt`
   as **fixed** PPO league opponents and run clipped-PPO self-play against them.
3. **Result — dead end.** The cloned-from-humans policy hit a **"BC ceiling"** and lost
   **0–16 to `comet_reaper`**. The orbit_lite engine is simply better than anything a
   net imitating human players can reach, and self-play against weak (cloned) opponents
   can't push past that ceiling.

Both files stayed in `rl/` as reusable infra, but the clone/BC paths are marked with
`DEAD END` banners in place (in `selfplay.py` the clone-league code is fenced with
`--- BC-clone league ---` markers). Read those before reviving any of it.
