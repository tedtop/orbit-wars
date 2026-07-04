> Shared engine: see `agents/opponents/ORBIT_LITE_FAMILY.md` for the full orbit_lite deep-dive.
> This file covers only what THIS fork changes and how to beat it.

# 1266-elo v44 agent  —  ~1266, highest of the family (Omkar Kadam)

## Delta vs producer-v2
The most aggressive config + ships its OWN copy of orbit_lite (44 iterations of tuning): H=20
(longest horizon), roi=1.2 (lowest gate → fires the most), max_waves=7, plus a terminal phase
(roi→1.0, 8 waves, regroup off). "100% self-contained" = writes the whole engine inline, no
external dataset dependency.

## Why it tops the family
Lower roi + more waves + longer horizon = it simply acts more and plans slightly further, and
the exact-flow scorer keeps those extra actions positive-EV. Diminishing returns: 44 iterations
bought ~18 points over baseline.

## How to beat it
- Aggression cuts both ways: roi=1.2 and 7 waves means it commits more ships forward → thinner
  rear garrisons. A counter-attacking bot that punishes over-extension exploits this directly.
- Still the do-nothing opponent model and equal-weight 4P sum.

## Ideas to steal
Use v44's config as the *starting point* for our own self-play tuning (it's the best-known
point), then push past it with a learned controller / opponent model rather than more hand-tuning.
