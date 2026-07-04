After I showed Claude.ai our round robin ranking system here is what we came up with. It would be worth re-thinking about our ranking system and using something more like what we think Kaggle is using for scoring. Doesn't sound to me like Wilson CI is it!

Thanks for the detailed breakdown — that's a clean, well-designed system. Let me give you an honest assessment of its strengths and the one significant blind spot.
What's Working Well
Wilson CI for adaptive stopping is the right call. The naive normal approximation breaks down near 0% and 100% win rates, which is exactly where your dominant bots will be. Wilson handles those gracefully.
Side-swapping on the same seed is elegant — you're controlling for both positional bias and map variance simultaneously. That's rigorous.
The adaptive game count behavior you observed (dominant bots settle fast, close matchups hit ~300+) is exactly what the math predicts. That's a good sign your system is working as intended.

The Blind Spot: Win Rate ≠ Ranking Quality
Your current ranking is aggregate win rate across all opponents, which has a meaningful flaw:

Strength of Schedule is ignored.

If bot A went 90% against a field of weak bots, and bot B went 85% against stronger bots, bot A ranks higher — but bot B might actually be better. In a full round-robin this is partially mitigated (everyone plays everyone), but it's not fully solved because win rate is not transitive.
Classic example:

A beats B 70%
B beats C 70%
C beats A 70% (non-transitive / rock-paper-scissors dynamic)

With aggregate win rates, all three look identical at ~70%, but they have very different matchup profiles. In a space conquest game with diverse strategies, this is actually likely — aggressive rushers beat turtlers, turtlers beat economic builders, builders beat rushers.

What This Means for Finals Submission
This is why your 2 finals submissions matter strategically, not just statistically.
Rather than just submitting your top 2 by win rate, look at the pairwise matrix:

Which bots have the most consistent win rates across diverse opponent styles? That's your "safe" submission — generalist dominance.
Which bot specifically beats the bots you expect to see in the final tournament field? If you've scouted other competitors, that's your "meta" submission.

Practically, from your arena data you can extract:
python# Win rate of bot A specifically against top-5 opponents
# vs win rate against bottom-5 opponents
# A big gap = specialist. Consistent = generalist.
The bots that hit 300+ games against each other are your closest rivals — their head-to-head record is the most statistically reliable signal you have for finals prediction.

One Upgrade Worth Considering
If you want better ranking before finals, Bradley-Terry is a drop-in improvement over raw win rate that accounts for opponent strength. It's essentially logistic regression on your pairwise win/loss counts — not much harder to compute, but gives you a proper strength-of-schedule-adjusted score. Python has choix for this.
But honestly, for selecting your 2 finals submissions, the pairwise matrix analysis matters more than the ranking method.


Key Things That Change Our Approach
The rating system is already TrueSkill-flavored
The competition uses Gaussian N(μ, σ²) — that's exactly the TrueSkill model we talked about earlier. So the competition itself is handling strength-of-schedule. Your local arena's raw win rate is a proxy, but the competition's μ is the real ranking signal once your bots play enough episodes.
It's Kaggle — 4-player FFA only
Looking at the game spec:

2-player OR 4-player depending on episode
The competition pools all submissions and matches similar ratings
You don't control who you play against

This confirms the 4v4 blind spot is real and urgent.


What the Numbers Actually Mean
In Kaggle's system starting at 600:

600 — brand new submission, no history
1200 — solidly beating average competition
1259 — where The Producer sits
1539+ — prize territory (top 10)
1745 — current leader

The gap between 1259 and 1539 sounds like 280 points but in practice means the top bots are winning roughly 80%+ of their games against The Producer level.
Why Kaggle Uses a Variant (TrueSkill/Gaussian)
Pure Elo has one weakness — it only handles two players at a time and assumes binary win/loss. Kaggle's system uses a Gaussian N(μ, σ²) model where:

μ (mu) = your estimated skill, what shows on the leaderboard
σ (sigma) = uncertainty in that estimate, shrinks as you play more games

This handles 4-player games naturally and means a brand new submission can climb fast (high σ = big updates per game) but eventually converges as σ shrinks and the system becomes confident in your true rating.
That's why submitting tonight matters — the sooner your bot starts playing, the sooner σ drops and your rating stabilizes before the June 23 deadline.


Ranking from Mixed 2v2 and 4-player Matches
For 1v1 matches, your adaptive round-robin is probably already using something like Elo or TrueSkill. The adaptive part (more games for close matchups) is a hallmark of TrueSkill-style systems — it keeps playing until the uncertainty (sigma) collapses below a threshold.
For 4-player matches, you need a system that can extract pairwise signal from a multi-player result. The best options:

TrueSkill (Microsoft) — designed exactly for this. It handles N-player matches natively by treating the finish order as a series of pairwise comparisons (1st beat 2nd, 2nd beat 3rd, etc.). It's the gold standard for this use case.
Glicko-2 — extends Elo with rating deviation and volatility, but is fundamentally pairwise, so you'd need to decompose 4-player results into pairwise outcomes manually.
OpenSkill — an open-source TrueSkill alternative worth looking at if you want something easier to implement.

The key insight: a 4-player match where you finish 1st gives you signal against 3 opponents simultaneously, which is why 4-player formats are efficient for ranking.
Combining Both Match Types
Run a unified rating system (TrueSkill or OpenSkill) that ingests both 1v1 and 4-player results into the same rating pool. Don't maintain separate rankings — the same mu/sigma per agent should absorb all match results. Your adaptive system is likely already doing this if it's well-designed.
Submission Strategy (5/day open, 2 for finals)
A few thoughts:
For daily open competition:

Your top ~5 by rating are obvious candidates, but rating uncertainty matters too. An agent with high sigma (fewer games, less certain) might actually be better than its mu suggests.
Submit agents that are stylistically diverse — if two of your top agents have nearly identical strategies, you're wasting a slot.

For the 2 finals submissions, you want:

Your highest-confidence best agent — highest mu, lowest sigma
A strategically different second agent — not just your #2 by rating, but one that plays differently, in case the meta of the final tournament favors a different style

The real question: do you know which agents are converging vs. still improving? The ones hitting 300+ games in your adaptive system are the interesting ones — they're closely matched and the system hasn't resolved them yet. Those are worth extra scrutiny before finals.
What ranking system is your adaptive round-robin actually using under the hood? That'll help nail down whether your current ratings are trustworthy enough for finals selection.