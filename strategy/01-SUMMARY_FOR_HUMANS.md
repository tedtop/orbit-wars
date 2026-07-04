Here's a plain-English writeup:

---

## How We've Been Teaching a Computer to Conquer the Solar System

A few days ago we entered an AI competition called **Orbit Wars** — a space strategy game where you write a bot (an automated player) that conquers planets by sending fleets of ships across a 100×100 map. There's a sun in the middle that destroys any fleet that flies through it, planets orbit around it, and comets fly through periodically. You win by having the most ships at the end of 500 turns. Up to 4,400 teams are competing globally and the top 10 each win $5,000.

You don't play the game yourself. You write a program that plays it for you, submit it to Kaggle (a competition platform), and it automatically gets matched against other teams' bots 24 hours a day. The better your bot plays, the higher its Elo rating climbs — same system chess uses.

---

### Phase 1 — Throw Everything at the Wall

We started by building **23 different bots**, each one inspired by a different scientific field. One used portfolio theory from finance (treat attacking planets like making investments — diversify risk, maximize return). One used epidemic modeling from biology (treat the enemy like a virus spreading through sectors). One used ant colony behavior from nature (leave virtual pheromone trails that guide ships toward good targets). One used chaos theory. One used traffic flow mathematics. And so on.

We ran them all against each other in a 10-hour tournament — 80,000 games total. The winner was the **finance bot** (Markowitz portfolio optimization), which beat everyone at 66%. We submitted it and our battle-hardened "champion" bot to Kaggle and Montana Schmeekler entered the competition.

---

### Phase 2 — Discovering We Were Reinventing the Wheel

While our bots started playing real games, we looked at what the top public competitors had already built. What we found was humbling: the #1 public bot — called **The Producer** — was built on a sophisticated planning library called `orbit_lite`. Instead of hand-coding rules like "attack the nearest planet," orbit_lite does something much smarter: it simulates 18 turns into the future, figures out how many ships will be on every planet at every future moment accounting for all fleets already in flight, and scores every possible attack by how much it would improve your position versus your opponent's. It's not just asking "is this a good target?" — it's asking "is this a good target *given everything that's already happening on the board for the next 18 turns?*"

We downloaded orbit_lite, built our own bot on top of it called **comet_reaper**, and it immediately tied The Producer head-to-head. We had gone from hand-coded heuristics to matching the best public bot in about a day.

---

### Phase 3 — Turning the Knobs

Every sophisticated system has configuration settings — numbers that control its behavior. orbit_lite has about 22 of them. Things like "how risk-averse should the bot be?", "how many targets should it attack at once?", "how many ships must a planet have before it's allowed to launch?"

We used a tool called **Optuna** — an automated knob-tuner — to systematically try thousands of combinations of these settings and find the ones that produce the highest win rate. Think of it like a robot that spends all night adjusting every dial on a mixing board, playing a song after each adjustment, and zeroing in on the combination that sounds best.

Result: after 37 attempts, Optuna declared the settings were already nearly optimal. Turning the knobs didn't meaningfully improve things. The bot was already at a local peak.

---

### Phase 4 — Trying to Learn From the Best Players (and Failing)

We had an idea: what if we could watch the top teams' actual games and teach our bot to imitate how they play? This is called **behavioral cloning** — you record thousands of (situation → action) pairs from an expert player and train a neural network to copy their decisions.

We downloaded games played by teams rated above 1500 Elo, extracted every move they made, and trained a neural network on that data. Then we tested it against comet_reaper.

It lost 0–16. Completely crushed.

The reason turned out to be elegant and a bit humbling: **the engine is already better than the humans at the mechanical execution**. The orbit_lite planner calculates exact fleet arrival times, accounts for orbital motion, sizes fleets precisely, and plans 18 turns out — better than any human can do mentally. What the top human teams have figured out isn't better mechanics, it's better *strategy* — when to be aggressive, which targets matter most, how to handle four-player politics. Imitating their moves without imitating their underlying judgment just gives you mechanical precision in service of the wrong goals. The internet forum for this competition independently confirmed the same finding: behavioral cloning of top players consistently underperforms the hand-coded engine.

---

### Phase 5 — The Structural Features Experiment (and What It Taught Us)

Between the knob-tuning dead end and the behavioral cloning dead end, we tried a series of structural ideas — adding new scoring factors to the bot's decision-making:

**The one that worked:** we noticed the engine doesn't distinguish between planets that orbit the sun and planets that stay fixed. Fixed (static) planets are more strategically valuable — they're predictable, they stay in safe space, enemies can't drift them into your territory. We added a small bonus for capturing static planets first, called this bot **schmeekler**, and it won 72% of games against comet_reaper in our local testing.

**The ones that failed:** we tried adding awareness of enemy fleets in flight (don't attack a planet if the enemy is already sending reinforcements), potential field planning (treat the board like a gravitational field and flow ships toward weak points), and phase-aware sizing (send bigger fleets early, smaller fleets late). All failed at scale.

The reason everything kept failing led to the most important discovery of the whole project:

---

### The Key Discovery — The Engine Is Nearly Deterministic

When we profiled what comet_reaper actually does each turn, we found something striking: **64 out of every 133 turns, there are literally zero valid moves to make**. 47 more turns have exactly one valid move. Only 22 turns per game have 2–4 choices.

The engine's filters are so precise — they reject any fleet that would arrive with too few ships to guarantee capture, any path that crosses the sun, any move that wouldn't pay back its cost before the game ends — that most turns the decision is already made for you. There's nothing to improve because there's nothing to choose.

This explains why every clever adjustment we tried landed at parity: **you can't optimize a decision that isn't being made**. It's like trying to improve a chess player's endgame when they're winning by checkmate on move 3.

---

### Phase 6 — Two Live Work Tracks

This discovery reframed the whole project. There are now two serious paths forward, running simultaneously:

**Track A — Structural Features**
The one kind of improvement that can work: adding information the engine doesn't currently have that changes which moves it considers valid in the first place. The comet blind spot is the current hypothesis — the engine doesn't explicitly track when comets enter the board on a fixed schedule (turns 50, 150, 250, 350, 450) and time attacks accordingly. Comets are temporary high-production planets — grabbing one at the right moment and launching a massive fleet from it just before it exits is a genuine edge the engine doesn't exploit. We're testing whether adding comet timing awareness creates enough new valid moves to improve performance.

**Track B — MCTS with a Learned Value Function**
This is the moonshot. MCTS stands for Monte Carlo Tree Search — the same family of algorithms that powers the world's best chess and Go AIs. The idea: instead of just evaluating the current board and picking the best move, you simulate many possible futures. "If I do X, my opponent might do A, B, or C. If they do A, I then have options 1, 2, 3..." You explore a tree of possibilities and pick the branch that leads to the best outcomes.

The reason simple search failed earlier is that the engine only generates 0–4 candidates per turn — you can't build a meaningful tree with one branch. So Track B is doing something more ambitious: **expand the candidate set to include moves the engine currently rejects** — marginal attacks, speculative interceptions, moves that don't guarantee capture but might be worth taking strategically — and use a neural network trained on thousands of real game outcomes to evaluate them.

This neural network (called a **value function**) learns to look at a board position and predict "how likely is this position to result in a win?" from having watched hundreds of thousands of real games play out. Once trained, it gives the search tree an independent judgment that the hand-coded engine fundamentally can't provide. We're planning to generate the training data and train this network on a supercomputer (Jetstream2, a national research computing cluster).

---

### The Autoresearch Wrapper — Making the Bot Improve Itself

The most architecturally interesting piece of the whole project is how we're managing the improvement process. Inspired by a framework built by AI researcher Andrej Karpathy, we built an **autoresearch loop**:

Imagine a research lab with an **orchestrator** (the manager) and two **worker tracks** (Track A and Track B). The orchestrator wakes up every 20 minutes, checks what both workers are doing, reads the latest results, polls the live Kaggle leaderboard to see how our submitted bots are actually performing, and decides what each track should work on next.

The key insight is the **ratchet**: there's a fixed evaluator (our testing gauntlet) that doesn't change mid-experiment, and a rule that only improvements get kept. The orchestrator proposes a hypothesis ("what if we add a comet bonus?"), the worker builds and tests it, and the result is one of three things: KEEP (it's better, this is now the new champion), DISCARD (it's worse, record why and move on), or INVESTIGATE (ambiguous, needs more data).

Everything — every hypothesis tried, every result, every dead end, every insight — gets logged in a timeline document so the history is never lost. When a new Claude Code session starts, it reads the timeline and picks up exactly where the last one left off without repeating failed experiments.

The orchestrator is itself a Claude AI instance running in a loop, reading the state of play and deciding what the workers should do next. It's an AI directing two other AIs to improve a bot that's competing against thousands of other bots. Turtles all the way down.

---

### Where We Are Now

Our best submitted bot (**comet_reaper**) sits at Elo ~1245. The prize zone starts at ~1534. The leader is at ~1748.

Every cheap improvement has been tried and either discarded or shown to land near parity. The two remaining serious bets are the comet timing feature and the value function. We have 6 days until the submission deadline. The supercomputer is ready. The pipeline is built.

Montana Schmeekler is still in the game. 🚀