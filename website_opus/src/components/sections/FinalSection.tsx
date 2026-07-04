import scoreHistory from "../../../public/data/score_history.json";
import type { ScorePoint } from "@/data/types";

export function FinalSection() {
  const data = scoreHistory as ScorePoint[];
  const peak = Math.max(...data.map(d => d.score));
  const peakRank = Math.min(...data.map(d => d.rank));

  return (
    <section id="final" className="section">
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-slate-500/30 bg-slate-500/10 text-slate-400 text-xs font-medium uppercase tracking-wide mb-4">
          Wrap-Up
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          Final Standings
        </h2>
      </div>

      {/* Final stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-12">
        {[
          { label: "Peak Elo",           value: peak.toFixed(0),   accent: "#f59e0b", sub: "comet_reaper_1235" },
          { label: "Peak Rank",          value: `#${peakRank}`,    accent: "#3b82f6", sub: "all-time" },
          { label: "Experiments Run",    value: "19",              accent: "#a855f7", sub: "across 6 tracks" },
          { label: "Successful (KEEP)",  value: "2",               accent: "#10b981", sub: "11% hit rate" },
          { label: "Games Played",       value: "523",             accent: "#3b82f6", sub: "2P + 4P" },
          { label: "Days Competing",     value: "7",               accent: "#f97316", sub: "Jun 14–20" },
        ].map(({ label, value, accent, sub }) => (
          <div key={label} className="glass p-5">
            <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">{label}</div>
            <div className="text-3xl font-bold" style={{ color: accent, fontFamily: "var(--font-space)" }}>{value}</div>
            <div className="text-xs text-slate-600 mt-1">{sub}</div>
          </div>
        ))}
      </div>

      {/* Links */}
      <div className="flex flex-wrap gap-4 justify-center mb-16">
        <a
          href="https://www.kaggle.com/competitions/llm-20-questions/leaderboard"
          target="_blank"
          rel="noopener noreferrer"
          className="glass px-6 py-3 text-sm font-medium text-slate-300 hover:text-white hover:border-blue-500/40 transition-colors flex items-center gap-2"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
          </svg>
          Kaggle Competition
        </a>
        <a
          href="https://github.com/tedtop/orbit-wars"
          target="_blank"
          rel="noopener noreferrer"
          className="glass px-6 py-3 text-sm font-medium text-slate-300 hover:text-white hover:border-blue-500/40 transition-colors flex items-center gap-2"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
          </svg>
          GitHub (post-competition)
        </a>
      </div>

      {/* Footer */}
      <div className="border-t border-slate-800 pt-8 text-center">
        <p className="text-slate-600 text-sm">
          Built with{" "}
          <span className="text-slate-500">Next.js 16 · TypeScript · Recharts · D3</span>
          {" "}·{" "}
          Competition data collected June 14–20, 2026
        </p>
        <p className="text-slate-700 text-xs mt-2">
          Ted Toporkov · Orbit Wars Kaggle Competition
        </p>
      </div>
    </section>
  );
}
