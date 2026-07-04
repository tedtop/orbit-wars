"use client";
import { useState } from "react";
import replaysIndex from "../../../public/data/replays_index.json";
import type { ReplayMeta } from "@/data/types";
import { OrbitalCanvas } from "@/components/charts/OrbitalCanvas";

const OUR_NAME = "Montana Schmeekler";

// Pick 8 highlight replays: mix of wins and losses, varied lengths
function getHighlights(all: typeof replaysIndex): typeof replaysIndex {
  const wins  = all.filter(r => r.our_result === "win").sort((a,b) => b.steps - a.steps).slice(0, 5);
  const losses = all.filter(r => r.our_result === "loss").sort((a,b) => b.steps - a.steps).slice(0, 3);
  return [...wins, ...losses];
}

export function ReplayGallerySection() {
  const all = replaysIndex as typeof replaysIndex;
  const highlights = getHighlights(all);
  const [selected, setSelected] = useState<(typeof replaysIndex)[0] | null>(null);

  return (
    <section id="replays" className="section">
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400 text-xs font-medium uppercase tracking-wide mb-4">
          Game Replays
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          Watch the Bot Play
        </h2>
        <p className="text-slate-400 max-w-xl mx-auto">
          {all.length} games played. Click any card to replay the full match with orbital mechanics.
        </p>
      </div>

      {/* Stat row */}
      <div className="flex flex-wrap gap-3 justify-center mb-10">
        {[
          { v: all.filter(r => r.our_result === "win").length.toString(),  l: "Wins",   c: "#10b981" },
          { v: all.filter(r => r.our_result === "loss").length.toString(), l: "Losses", c: "#ef4444" },
          { v: Math.max(...all.map(r => r.steps)).toString(),              l: "Longest Game (steps)", c: "#3b82f6" },
          { v: Math.round(all.reduce((s,r) => s + r.steps, 0) / all.length).toString(), l: "Avg Steps", c: "#a855f7" },
        ].map(({ v, l, c }) => (
          <div key={l} className="glass px-5 py-3 text-center">
            <div className="text-xl font-bold" style={{ color: c, fontFamily: "var(--font-space)" }}>{v}</div>
            <div className="text-xs text-slate-500 mt-0.5">{l}</div>
          </div>
        ))}
      </div>

      {/* Highlight grid */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {highlights.map(replay => {
          const isWin = replay.our_result === "win";
          const opp = replay.teams.find(t => !t.toLowerCase().includes(OUR_NAME.toLowerCase().split(" ")[1])) ?? replay.teams[0];
          return (
            <button
              key={replay.id}
              onClick={() => setSelected(selected?.id === replay.id ? null : replay)}
              className={`glass p-4 text-left transition-all hover:border-blue-500/50 ${selected?.id === replay.id ? "border-blue-500/60 bg-blue-500/5" : ""}`}
            >
              <div className="flex items-center justify-between mb-3">
                <span
                  className="text-xs font-bold px-2 py-0.5 rounded-full"
                  style={isWin
                    ? { background: "#10b98130", color: "#10b981" }
                    : { background: "#ef444430", color: "#ef4444" }}
                >
                  {isWin ? "WIN" : "LOSS"}
                </span>
                <span className="text-xs text-slate-600 font-mono">{replay.steps} steps</span>
              </div>
              <div className="text-xs text-slate-400 truncate mb-1">vs {opp?.slice(0, 20) ?? "—"}</div>
              <div className="text-xs text-slate-600 font-mono">{replay.date}</div>
              <div className="text-xs text-slate-700 mt-2">{replay.num_players}P match</div>
            </button>
          );
        })}
      </div>

      {/* Orbital canvas replay */}
      {selected && (
        <div className="glass p-4 md:p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <span className="text-sm font-semibold text-white">
                {selected.our_result === "win" ? "🟢 Win" : "🔴 Loss"} — {selected.steps} steps
              </span>
              <span className="text-xs text-slate-500 ml-3 font-mono">#{selected.id}</span>
            </div>
            <button onClick={() => setSelected(null)} className="text-slate-500 hover:text-slate-300 text-sm">✕ Close</button>
          </div>
          <OrbitalCanvas episodeId={selected.id} teams={selected.teams} />
        </div>
      )}

      {/* All games summary table */}
      <details className="mt-8 glass p-4">
        <summary className="text-sm text-slate-400 cursor-pointer hover:text-slate-200 select-none">
          View all {all.length} games →
        </summary>
        <div className="mt-4 overflow-x-auto max-h-80 overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-slate-900">
              <tr>
                {["Result","Steps","Opponent","Date","Players"].map(h => (
                  <th key={h} className="text-left py-2 px-3 text-slate-500 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {all.slice().sort((a,b) => (b.steps - a.steps)).map(r => {
                const opp = r.teams.find(t => !t.toLowerCase().includes(OUR_NAME.toLowerCase().split(" ")[1])) ?? r.teams[0];
                return (
                  <tr key={r.id} className="border-b border-slate-800/40 hover:bg-white/2">
                    <td className="py-1.5 px-3">
                      <span style={{ color: r.our_result === "win" ? "#10b981" : r.our_result === "loss" ? "#ef4444" : "#64748b" }}>
                        {r.our_result === "win" ? "W" : r.our_result === "loss" ? "L" : "?"}
                      </span>
                    </td>
                    <td className="py-1.5 px-3 font-mono text-slate-400">{r.steps}</td>
                    <td className="py-1.5 px-3 text-slate-400 max-w-[160px] truncate">{opp?.slice(0,25) ?? "—"}</td>
                    <td className="py-1.5 px-3 text-slate-600 font-mono">{r.date}</td>
                    <td className="py-1.5 px-3 text-slate-600">{r.num_players}P</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}
