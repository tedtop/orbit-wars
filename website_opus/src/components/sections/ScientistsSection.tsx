"use client";
import scientists from "../../../public/data/scientists.json";

const CLUSTER_COLORS: Record<string, string> = {
  physics:   "#3b82f6",
  life:      "#10b981",
  math_cs:   "#a855f7",
  economics: "#f59e0b",
  other:     "#64748b",
};

const CLUSTER_LABELS: Record<string, string> = {
  physics:   "Physics & Control",
  life:      "Life Sciences",
  math_cs:   "Math & ML",
  economics: "Economics",
  other:     "Strategy",
};

// Approximate ratings from the overnight arena run (from TIMELINE.md Phase 0)
// markowitz_portfolio_optimization won the arena
const ARENA_WINNER = "markowitz_portfolio_optimization";

export function ScientistsSection() {
  const byCluster = scientists.reduce<Record<string, typeof scientists>>((acc, s) => {
    acc[s.cluster] = acc[s.cluster] ?? [];
    acc[s.cluster].push(s);
    return acc;
  }, {});

  return (
    <section id="scientists" className="section">
      {/* Header */}
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-purple-500/30 bg-purple-500/10 text-purple-400 text-xs font-medium uppercase tracking-wide mb-4">
          Night One
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          The 23 Scientists
        </h2>
        <p className="text-slate-400 max-w-xl mx-auto leading-relaxed">
          Before touching a leaderboard, we invented 23 bots from scratch — each applying
          a different scientific discipline to an orbital-mechanics strategy game.
          They battled each other for&nbsp;<strong className="text-white">10 hours and 10 minutes</strong>.
        </p>
      </div>

      {/* Stat callout */}
      <div className="flex flex-wrap justify-center gap-4 mb-12">
        {[
          { v: "23",      l: "Bots",             c: "#a855f7" },
          { v: "10h 10m", l: "Arena Run",         c: "#3b82f6" },
          { v: "5",       l: "Scientific Fields", c: "#10b981" },
          { v: "#~500",   l: "Where They'd Rank", c: "#ef4444" },
        ].map(({ v, l, c }) => (
          <div key={l} className="glass px-6 py-4 text-center min-w-[110px]">
            <div className="text-2xl font-bold" style={{ color: c, fontFamily: "var(--font-space)" }}>{v}</div>
            <div className="text-xs text-slate-500 mt-1 uppercase tracking-wide">{l}</div>
          </div>
        ))}
      </div>

      {/* Bot grid by cluster */}
      <div className="space-y-8">
        {Object.entries(byCluster).map(([cluster, bots]) => (
          <div key={cluster}>
            <div className="flex items-center gap-3 mb-4">
              <span
                className="w-2 h-2 rounded-full"
                style={{ background: CLUSTER_COLORS[cluster] }}
              />
              <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
                {CLUSTER_LABELS[cluster]}
              </h3>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {bots.map((bot) => {
                const isWinner = bot.name === ARENA_WINNER;
                return (
                  <div
                    key={bot.name}
                    className="glass p-4 relative group"
                    style={isWinner ? { borderColor: CLUSTER_COLORS[bot.cluster], borderWidth: 1 } : {}}
                  >
                    {isWinner && (
                      <div className="absolute -top-2.5 left-4 px-2 py-0.5 rounded-full text-xs font-bold bg-amber-500 text-black">
                        🏆 Arena Winner
                      </div>
                    )}
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <code
                        className="text-xs font-mono break-all leading-tight"
                        style={{ color: CLUSTER_COLORS[bot.cluster] }}
                      >
                        {bot.name}
                      </code>
                    </div>
                    <div className="text-xs text-slate-500 font-medium mb-2 uppercase tracking-wide">
                      {bot.field}
                    </div>
                    <p className="text-sm text-slate-400 leading-relaxed">{bot.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Transition */}
      <div className="mt-16 p-6 rounded-xl border border-rose-500/20 bg-rose-500/5 text-center">
        <p className="text-slate-300 text-lg">
          Then we looked at the actual leaderboard.
        </p>
        <p className="text-slate-500 mt-1">
          None of our 23 scientists would have cracked the top 400.
          The best submission placed <strong className="text-rose-400">#536</strong>.
        </p>
        <p className="text-slate-500 mt-3 text-sm">
          Time to understand what the top bots were actually doing. ↓
        </p>
      </div>
    </section>
  );
}
