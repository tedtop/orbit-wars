import scoreHistory from "../../../public/data/score_history.json";
import submissions from "../../../public/data/submissions.json";
import { ScoreChart } from "@/components/charts/ScoreChart";
import type { ScorePoint } from "@/data/types";

export function ScoreProgressionSection() {
  const data = scoreHistory as ScorePoint[];
  const subs = (submissions as Array<{name:string;submitted_at:string;public_score:number|null;status:string}>)
    .filter(s => s.submitted_at)
    .map(s => ({ time: s.submitted_at, name: s.name }))
    .sort((a,b) => a.time.localeCompare(b.time));

  const peakScore = Math.max(...data.map(d => d.score));
  const startScore = data[0]?.score ?? 0;
  const peakRank = Math.min(...data.map(d => d.rank));

  return (
    <section id="score" className="section">
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-amber-500/30 bg-amber-500/10 text-amber-400 text-xs font-medium uppercase tracking-wide mb-4">
          Live Leaderboard
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          Elo Over 6 Days
        </h2>
        <p className="text-slate-400 max-w-xl mx-auto">
          {data.length} snapshots, captured every 15 minutes around the clock.
          Each color is a different bot submission.
        </p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        {[
          { label: "Starting Elo",  value: startScore.toFixed(0),  sub: "First submission", accent: "#64748b" },
          { label: "Peak Elo",      value: peakScore.toFixed(0),   sub: "comet_reaper_1235", accent: "#f97316" },
          { label: "Peak Rank",     value: `#${peakRank}`,         sub: "All-time best",     accent: "#10b981" },
          { label: "Total Snapshots", value: data.length.toString(), sub: "~15-min cadence", accent: "#3b82f6" },
        ].map(({ label, value, sub, accent }) => (
          <div key={label} className="glass p-4">
            <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">{label}</div>
            <div className="text-2xl font-bold" style={{ color: accent, fontFamily: "var(--font-space)" }}>{value}</div>
            <div className="text-xs text-slate-600 mt-0.5">{sub}</div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="glass p-6">
        <ScoreChart data={data} submissionTimes={subs} />
      </div>

      {/* Submission events legend */}
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
        {subs.map(s => {
          const sub = (submissions as Array<{name:string;submitted_at:string;public_score:number|null}>)
            .find(x => x.name === s.name);
          return (
            <div key={s.name} className="glass px-4 py-3 flex items-center gap-3">
              <div className="w-2 h-2 rounded-full flex-shrink-0 bg-blue-500" />
              <div>
                <div className="text-xs font-mono text-slate-300">{s.name}</div>
                <div className="text-xs text-slate-600">{new Date(s.time).toLocaleDateString("en-US", { month: "short", day: "numeric" })} · {sub?.public_score?.toFixed(0) ?? "—"} Elo</div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
