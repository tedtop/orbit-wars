import scoreHistory from "../../../public/data/score_history.json";
import submissions from "../../../public/data/submissions.json";

export function HeroSection() {
  const latest = scoreHistory.at(-1);
  const totalGames = 523;
  const finalRank = latest?.rank ?? "—";
  const finalScore = latest?.score?.toFixed(0) ?? "—";

  // Count active submission details
  const activeSub = (submissions as Array<{name:string;status:string;public_score:number|null}>)
    .find(s => s.status === "active" && s.public_score != null);

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center text-center overflow-hidden px-6">
      {/* Orbital rings decoration */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none" aria-hidden>
        <div className="orbit-ring" style={{ width: 500, height: 500, top: "50%", left: "50%", transform: "translate(-50%,-50%)" }} />
        <div className="orbit-ring" style={{ width: 700, height: 700, top: "50%", left: "50%", transform: "translate(-50%,-50%)", opacity: 0.5 }} />
        <div className="orbit-ring" style={{ width: 900, height: 900, top: "50%", left: "50%", transform: "translate(-50%,-50%)", opacity: 0.25 }} />
      </div>

      <div className="relative z-10 max-w-3xl">
        {/* Eyebrow */}
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400 text-xs font-medium tracking-wide uppercase mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
          Kaggle Competition Journal · June 2026
        </div>

        {/* Title */}
        <h1
          className="text-6xl md:text-8xl font-bold mb-4 leading-tight"
          style={{ fontFamily: "var(--font-space)" }}
        >
          <span className="gradient-text">Orbit Wars</span>
        </h1>

        <p className="text-lg md:text-xl text-slate-400 max-w-xl mx-auto mb-10 leading-relaxed">
          7 days building an AI to compete in a real-time orbital-strategy game.
          23 scientific theories. 19 experiments. 491 leaderboard snapshots.
          One very stubborn engine.
        </p>

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 max-w-2xl mx-auto mb-12">
          {[
            { label: "Final Rank",    value: `#${finalRank}`,  accent: "#3b82f6" },
            { label: "Peak Elo",      value: "1328",           accent: "#f59e0b" },
            { label: "Experiments",   value: "19",             accent: "#a855f7" },
            { label: "Games Played",  value: totalGames.toLocaleString(), accent: "#10b981" },
          ].map(({ label, value, accent }) => (
            <div key={label} className="glass p-4 flex flex-col items-center gap-1">
              <span className="text-2xl md:text-3xl font-bold" style={{ color: accent, fontFamily: "var(--font-space)" }}>{value}</span>
              <span className="text-xs text-slate-500 uppercase tracking-wide">{label}</span>
            </div>
          ))}
        </div>

        {/* CTA */}
        <a
          href="#journey"
          className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold transition-colors"
        >
          See the journey
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M8 3v10M3 8l5 5 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </a>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 text-slate-600 text-xs animate-bounce">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <path d="M10 4v12M4 10l6 6 6-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      </div>
    </section>
  );
}
