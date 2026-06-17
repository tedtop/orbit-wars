# Orbit Wars — Competitor Intel (distilled)

_Sources: Kaggle notebooks (pulled to `intel_kernels/`), forum discussions + Discord (pasted by Ted).
I can pull `/code/` notebooks via `kaggle kernels pull`; discussions/Discord must be pasted (no API)._

## The landscape
- **comet_reaper ≈ "the Producer" 1-ply greedy search ≈ LB ~1240** (= us, #144/1243). **#1 ≈ 1793.**
- comet_reaper **beats the whole public field** in our bench (see RESULTS.md). The gap to the prize zone is
  the **private** top teams.

## What loses (confirmed dead ends)
- **Behavior-cloning / imitation** (souldrive `why-cloning-the-1-bot-loses-to-greedy`): a 2.4M-param clone
  of #1 wins only ~17% vs the greedy Producer; **PPO made it worse (4%)**. Cloning a search agent distills
  its moves but not its lookahead. *"If a fast search exists, run the search — don't clone."* (Matches our
  Phase-4 0–16 result.)
- **The public "improved" heuristic agents** (improved-agent-v2, improved-heuristic-agent): claim ~1500 but
  are **passive** (no-op-heavy) and lose to comet_reaper (50–0 in our bench). Their *ideas* may still help;
  their *execution* doesn't beat us.

## What likely wins (the edge)
- **Deeper search.** #1 is a search agent; comet_reaper is only **1-ply**. **Inference is not the
  bottleneck** — ~1 s/turn budget vs ~25 ms used = **~30× headroom**. This is the prize-zone lever.
- **Aggression.** Universal theme (forums + Discord): winners attack *early*; clones learn passivity from
  no-op-heavy data. Bias toward early aggression (lower early roi, dynamic sizing).

## Concrete techniques seen (test, don't assume)
- **NPV / production-snowball scoring** with ownership multipliers: enemy **2.05×**, neutral **1.4×**,
  contested **0.7×**, friendly **0.3×**. (Quantifies our objective-recovery finding: enemy ≫ neutral.)
- **Phase-aware ship sizing:** early 1.2× → mid 1.5× → late 2.0× → finishing 3.0× (dump everything).
- **Speed-bracket sizing:** `fleet_speed` is log-concave → discrete ship-count thresholds that drop travel
  time a full turn; size sends to cross them.
- **Enemy-fleet interdiction:** race to neutrals enemy fleets target, or arrive after to pick up the pieces.
- **4P:** weakest-enemy targeting + elimination bonus (improved-agent) vs leader-attack bonus (i-m-stronger).
- **Deception:** feints / diversionary attacks (improved-heuristic-agent).
- **Infra tip (Discord):** orbit_lite is slow PyTorch — **numba/numpy would speed it up** → more games/sec.

## To request from Ted (no API access)
Discussion 707869 (vkhydras's hinted edge), 708209, 705475; any Discord strategy talk → paste here.
