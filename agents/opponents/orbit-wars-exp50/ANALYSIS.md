> Shared engine: see `agents/opponents/ORBIT_LITE_FAMILY.md` for the full orbit_lite deep-dive.
> This file covers only what THIS fork changes and how to beat it.

# orbit-wars-exp50  —  ~1175 (ShumingFang)

## Delta vs producer-v2
Adds a **terminal phase**: when the game is near its end, switches to `terminal_roi_threshold=1.0`
(fire on almost any positive-value wave), `terminal_max_waves_per_turn=9`, and disables regroup —
i.e. stop hoarding and dump every safe ship into captures for the final score.

## How to beat it
- Before the terminal phase it is just producer-v2; all baseline exploits apply.
- The endgame dump is predictable: it will throw everything forward on a known turn. Hold a
  defensive reserve sized for that wave and counter-punch the planets it strips to attack.

## Ideas to steal
A terminal "spend it all" phase is clearly worth ~points and is trivial to add. Include a tuned
endgame phase in any heuristic we ship (strategy #5).
