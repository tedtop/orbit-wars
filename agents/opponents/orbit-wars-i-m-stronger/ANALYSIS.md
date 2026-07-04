> Shared engine: see `agents/opponents/ORBIT_LITE_FAMILY.md` for the full orbit_lite deep-dive.
> This file covers only what THIS fork changes and how to beat it.

# orbit-wars-i-m-stronger  —  ~1226, #2 (Roman Tamrazov)

## Delta vs producer-v2
Adds `ffa_leader_attack_bonus=0.035`: in 4P, inflates the score of attacking the **strongest**
opponent's planets by `0.035 · (their_strength − my_strength)`. Tightens 4P (offensive targets
7, roi 1.55, min_ships 5). Otherwise identical engine. Note: deliberate **opening hold** — emits
no launches until ~turn 40 while it accumulates and waits for orbital geometry.

## How to beat it
- The anti-leader bias is tiny (0.035) and only targets the *current* leader — it can be
  baited: stay quietly 2nd, let it hammer the leader, then surge late.
- Opening hold means it cedes early neutral grabs — a fast-expansion bot can bank production
  before i-m-stronger acts, then defend the lead.

## Ideas to steal
The anti-leader FFA term is the right idea but under-tuned; a learned/larger coalition weight is
a top-10 lever (see strategy #3).
