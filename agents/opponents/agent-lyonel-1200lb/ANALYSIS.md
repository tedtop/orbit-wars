> Shared engine: see `agents/opponents/ORBIT_LITE_FAMILY.md` for the full orbit_lite deep-dive.
> This file covers only what THIS fork changes and how to beat it.

# agent-lyonel-1200lb  —  ~1200 (Marwan Ashref)

## Delta vs producer-v2
A conservative re-tune of the same engine with input validation: H=15, max_waves=5, roi=1.4
(slightly more eager but fewer waves), plus dataclass asserts guarding the knobs. No new phase.

## How to beat it
- Shorter horizon (15) → even more blind to long setups than baseline.
- Fewer waves/turn (5) → under heavy multi-front pressure it can't respond to everything; open
  two simultaneous threats and it must concede one.

## Ideas to steal
Mostly a data point that conservative tuning ≈ baseline. Confirms config tuning has plateaued
around ~1200–1266; the next gains need a new mechanism, not new constants.
