import screenshots from "../../../public/data/screenshots_index.json";

type Shot = { filename: string; caption: string; category: string; key_numbers: string; ocr_summary: string };

const CATEGORY_COLORS: Record<string, string> = {
  leaderboard:           "#f59e0b",
  submission:            "#10b981",
  autoresearch:          "#a855f7",
  terminal_agent:        "#3b82f6",
  dashboard:             "#06b6d4",
  fleet_monitor:         "#f97316",
  game_board:            "#ef4444",
};

export function AutonomousSection() {
  const shots = screenshots as Shot[];

  return (
    <section id="autonomous" className="section">
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-amber-500/30 bg-amber-500/10 text-amber-400 text-xs font-medium uppercase tracking-wide mb-4">
          Behind the Scenes
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          Autonomous AI Development
        </h2>
        <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed">
          Most of this project was built by{" "}
          <strong className="text-white">Claude Code running autonomously overnight</strong> — designing
          experiments, writing code, running evaluations, promoting champions, filing audit reports,
          and deciding when to kill failing tracks. I&apos;d wake up to a full experiment log and a
          state-of-play summary ready for strategic review.
        </p>
      </div>

      {/* Meta-narrative stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-10">
        {[
          { v: "8",   l: "Critical bugs found autonomously", c: "#10b981" },
          { v: "6",   l: "Research hypotheses tested",       c: "#3b82f6" },
          { v: "11%", l: "Experiment hit rate",              c: "#f59e0b" },
          { v: "3",   l: "Autonomous overnight sessions",    c: "#a855f7" },
        ].map(({ v, l, c }) => (
          <div key={l} className="glass p-4 text-center">
            <div className="text-3xl font-bold" style={{ color: c, fontFamily: "var(--font-space)" }}>{v}</div>
            <div className="text-xs text-slate-500 mt-1 leading-tight">{l}</div>
          </div>
        ))}
      </div>

      {/* Interesting quotes from AUDITOR_LOG */}
      <div className="space-y-4 mb-10">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">From the Independent Auditor Log</h3>
        {[
          {
            quote: "v6_cr4 LOCAL: ⛔ DEAD — ESCALATE. CF=0/50 for last 50 updates. Entropy=0.086: policy fully committed, zero exploration. Not learning, collapsing. This run should be killed.",
            context: "2026-06-19 05:25 MT — Claude Code's independent auditor catching a collapsed training run before it wasted compute"
          },
          {
            quote: "DISAGREE: 'Campaign FAIL — RL bet closed.' Best fleet seed is 32.5% vs greedy at U=100 with CF in-band and entropy still decaying. The gate should not have been declared failed based on local seeds alone when the fleet has a seed showing >25%.",
            context: "The auditor subagent disagreeing with the orchestrator's failure call, providing a second opinion"
          },
          {
            quote: "Structural finding: 32 seeds, 28 have inverted-U WR trend (peak U=100-200, decline after). Only j1 improved monotonically. The pre-committed gate was designed assuming near-zero draws — at U=200 with weak policies, stalemates are common.",
            context: "Pattern analysis across the entire fleet of 32 parallel training runs"
          },
        ].map(({ quote, context }, i) => (
          <div key={i} className="glass p-5 border-l-4 border-slate-700">
            <blockquote className="text-sm text-slate-300 italic leading-relaxed mb-2">&ldquo;{quote}&rdquo;</blockquote>
            <p className="text-xs text-slate-600">{context}</p>
          </div>
        ))}
      </div>

      {/* What the autonomous loop did */}
      <div className="grid md:grid-cols-3 gap-4 mb-10">
        {[
          {
            icon: "🔬",
            title: "Hypothesis Generation",
            body: "Proposed 6 research hypotheses based on forum intel and game analysis. Each framed as a falsifiable experiment with a pre-committed threshold.",
          },
          {
            icon: "🏃",
            title: "Experiment Execution",
            body: "Wrote bot code, ran seat-balanced evaluation panels (n=50–150), collected results, and updated the experiment registry — all without prompting.",
          },
          {
            icon: "📋",
            title: "State of Play Reports",
            body: "Ended each session with a paste-able strategic summary for human review, including what passed the gate, what was killed, and what remained.",
          },
        ].map(({ icon, title, body }) => (
          <div key={title} className="glass p-5">
            <div className="text-2xl mb-3">{icon}</div>
            <h3 className="font-semibold text-white mb-2">{title}</h3>
            <p className="text-sm text-slate-500 leading-relaxed">{body}</p>
          </div>
        ))}
      </div>

      {/* Screenshot gallery */}
      {shots.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500 mb-4">
            Progress Screenshots <span className="text-slate-700 normal-case font-normal">— {shots.length} moments</span>
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {shots.map(s => {
              const color = CATEGORY_COLORS[s.category] ?? "#64748b";
              return (
                <a
                  key={s.filename}
                  href={`/screenshots/${s.filename}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="glass overflow-hidden rounded-lg hover:border-white/20 transition-colors flex flex-col group"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/screenshots/${s.filename}`}
                    alt={s.caption}
                    className="w-full h-36 object-cover object-top group-hover:opacity-90 transition-opacity"
                    loading="lazy"
                  />
                  <div className="px-2.5 pt-2 pb-2.5 flex flex-col gap-1.5 flex-1">
                    <span
                      className="text-[10px] font-medium px-1.5 py-0.5 rounded-full self-start"
                      style={{ color, background: `${color}18`, border: `1px solid ${color}30` }}
                    >
                      {s.category.replace("_", " ")}
                    </span>
                    <p className="text-xs text-slate-400 leading-snug line-clamp-2">{s.caption}</p>
                  </div>
                </a>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
