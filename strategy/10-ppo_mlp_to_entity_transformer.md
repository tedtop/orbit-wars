What we've been building, in plain English

---
The game and the goal

Orbit Wars is a real-time strategy game where you and an opponent race to conquer planets. Each turn every planet you own produces ships, and you can launch fleets to
attack enemy planets or reinforce your own. The winner is whoever owns all the planets (or has the most ships when time runs out).

Our goal: train an AI that can beat comet_reaper, our hand-coded bot sitting at ~1235 Elo on the live leaderboard.

---
What PPO is

PPO (Proximal Policy Optimization) is a method for training an AI to play a game by having it play thousands of games against itself, learning from wins and losses.
Think of it like a chess student playing blitz games all day — each game teaches it something, and gradually it gets better.

PPO has two main parts that learn simultaneously:

The Policy — the part that actually decides what to do each turn. "I own planets 1, 3, and 7. Enemy owns 2, 4, 5. I should launch from planet 3 toward planet 4." This is
the player's brain.

The Critic — a separate part that watches the game and tries to predict "how good is this situation for me right now?" It doesn't make moves itself — it's more like a
coach watching from the sideline saying "you're up 60% right now" or "this looks bad." The critic's predictions are used to guide policy updates: moves that led to
better-than-expected outcomes get reinforced, moves that led to worse-than-expected outcomes get penalized.

EV (Explained Variance) is how accurate the critic's predictions are. EV=0.90 means the critic's "how good is this?" score explains 90% of what actually happened. Bad EV
= bad coaching = policy learns slowly or not at all.

---
MLP — what we tried before

MLP (Multi-Layer Perceptron) is the simplest neural network architecture. To describe the game state to the AI, we took every planet's stats (owner, ship count,
production rate, position, etc.) and flattened them into one long list of ~300 numbers. The MLP reads that list and outputs a decision.

The problem: flattening destroys relationships. Planet 3 is close to planet 7 and they're both yours — the MLP has to figure that out from scratch every time by
memorizing patterns in the flat list. It's like describing a chess board by reading off all 64 squares left-to-right and expecting someone to understand piece
relationships from that.

After hundreds of hours of training, our MLP bots hit a ceiling of about 20-37% win rate against the greedy heuristic (a bot that just always attacks the nearest enemy).
They never beat comet_reaper at all. Not once in thousands of games.

---
Entity Transformer — what we're doing now

The Entity Transformer (ET) treats each planet as its own object ("entity") that can look at all the other planets and ask "how does this one relate to me?" This is
called attention — each planet attends to every other planet when deciding what matters.

Instead of one big flat list, the ET sees: "I'm Planet 3. Planet 7 is mine, close, low ships. Planet 4 is the enemy, close, high ships. Planet 1 is mine but far away."
The relational structure is explicit in the architecture.

It's the difference between handing someone a list of 300 numbers versus handing them an actual map.

The results so far are striking: EV hit 0.93-0.99 across 36 seeds at U=40, well past the 0.90 threshold that signals a working critic. The MLP peaked around 0.84 and
stalled. A healthy critic means the policy can actually learn.

---
Why didn't we try ET first?

Honestly — we underestimated how much the architecture mattered. The early bets were:

1. The reward signal was the problem, not the model. We spent weeks debugging reward shaping, terminal bonuses, critic normalization. Turned out the MLP just couldn't
represent the game well enough to make any reward signal work.
2. Complexity fear. Transformers are more code, more hyperparameters, more things to debug. When you're racing a competition deadline you tend to start simple and
escalate. We started simple and it didn't work.
3. The EV diagnostic came late. We only started tracking EV carefully in the last week. If we'd known from day one that MLP EV was stuck at 0.84 (while ET hits 0.97),
we'd have pivoted sooner.

In hindsight: the game is fundamentally relational (planets relate to each other, fleets relate to planets). The architecture needed to match that structure from the
start.

---
The JAX rewrite — what and why

Right now each training game runs in Python, one step at a time. The bottleneck isn't the neural network — it's the game engine itself, a Python simulator that processes
one game step, hands off to our AI, gets an action, processes the next step, etc. 36 parallel games on 9 servers gives us ~27 game-steps per second.

JAX is a framework that lets you write math that runs on a GPU (graphics card) instead of a CPU. But more importantly for us — if you rewrite the game rules themselves
in JAX, you can run thousands of games simultaneously on a single GPU, because GPUs are designed to do millions of simple operations in parallel.

The JAX rewrite means: instead of 36 games at 27 steps/second, potentially 10,000+ games at 1,000+ steps/second. That's a 10,000x throughput increase, roughly.

The catch: rewriting the game engine in JAX requires it to behave identically to the Python original. That's why the new Claude Code instance is building a validation
harness first — it must prove the JAX engine produces identical outcomes before we trust any training results from it.

---
Realistic odds in the time remaining

Honest assessment — ~24-36 hours left:

┌────────────────────────────────────┬────────────┬─────────────────────────────────────────────────────────┐
│                Path                │ Likelihood │                 What needs to go right                  │
├────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
│ ET CPU fleet beats greedy          │ ~85%       │ Already happening — EV looks strong, first evals in ~2h │
├────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
│ ET CPU fleet beats comet_reaper    │ ~20-30%    │ Would require greedy_WR >70% AND cr_WR signal appearing │
├────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
│ JAX engine validated + training    │ ~30%       │ JAX rewrite takes time; validation may find bugs        │
├────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
│ JAX-trained bot beats comet_reaper │ ~10-15%    │ Even if JAX works, needs to train long enough           │
└────────────────────────────────────┴────────────┴─────────────────────────────────────────────────────────┘

The most realistic win scenario: the CPU ET fleet finds a seed that beats the greedy bot at 70%+ in the next few hours. That's our signal that the architecture is
genuinely learning the game. Whether that translates to beating comet_reaper in the time remaining is uncertain — comet_reaper is a much harder opponent than the greedy
heuristic.

The protected floor is comet_reaper itself at ~1235 Elo, already submitted. Any RL bot we train only replaces it if it actually wins a 1000-game eval against it. We
don't risk our existing submission — we can only improve on it.

Bottom line: we're in a better position than we've been at any point in this competition. The architecture is finally working. Whether it's enough in 24 hours is
genuinely uncertain, but the signal we're waiting for — greedy_WR on the dashboard — arrives in ~2 hours and will tell us a lot.