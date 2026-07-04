"use client";
import experiments from "../../../public/data/experiments.json";
import { useState } from "react";
import type { Experiment } from "@/data/types";

const VERDICT_COLOR = {
  "KEEP":        "#10b981",
  "DISCARD":     "#ef4444",
  "IN PROGRESS": "#f59e0b",
};

const TRACK_COLORS: Record<string, string> = {
  "A":       "#3b82f6",
  "B":       "#a855f7",
  "C":       "#10b981",
  "Config":  "#f59e0b",
  "Search":  "#f97316",
};

// Bot lineage data (hardcoded from known structure)
const LINEAGE = [
  { id: "orbit_lite",    label: "orbit_lite engine",    parent: null,         verdict: "external",   x: 0,   y: 0   },
  { id: "comet_reaper",  label: "comet_reaper",          parent: "orbit_lite", verdict: "KEEP",       x: 0,   y: 80  },
  { id: "precog",        label: "precog",                parent: "comet_reaper", verdict: "DISCARD",  x: -240, y: 160 },
  { id: "kingmaker",     label: "kingmaker",             parent: "comet_reaper", verdict: "DISCARD",  x: -160, y: 160 },
  { id: "maestro",       label: "maestro",               parent: "comet_reaper", verdict: "DISCARD",  x: -80,  y: 160 },
  { id: "helmsman",      label: "helmsman",              parent: "comet_reaper", verdict: "DISCARD",  x: 0,    y: 160 },
  { id: "oracle",        label: "oracle",                parent: "comet_reaper", verdict: "DISCARD",  x: 80,   y: 160 },
  { id: "cr_tuned",      label: "comet_reaper_tuned",    parent: "comet_reaper", verdict: "DISCARD",  x: 160,  y: 160 },
  { id: "schmeekler",    label: "schmeekler ✨",         parent: "comet_reaper", verdict: "KEEP",     x: 240,  y: 160 },
  { id: "sch_potential", label: "schmeekler_potential",  parent: "schmeekler", verdict: "DISCARD",    x: 100,  y: 260 },
  { id: "sch_interdict", label: "schmeekler_interdict",  parent: "schmeekler", verdict: "DISCARD",    x: 180,  y: 260 },
  { id: "sch_phase",     label: "schmeekler_phase",      parent: "schmeekler", verdict: "DISCARD",    x: 260,  y: 260 },
  { id: "sch_fmt",       label: "schmeekler_fmt ✅",     parent: "schmeekler", verdict: "KEEP",       x: 340,  y: 260 },
];

const NODE_COLOR: Record<string, string> = {
  "KEEP":     "#10b981",
  "DISCARD":  "#ef4444",
  "external": "#64748b",
};

export function ExperimentLabSection() {
  const exps = experiments as Experiment[];
  const [selected, setSelected] = useState<Experiment | null>(null);

  const keep    = exps.filter(e => e.verdict === "KEEP").length;
  const done    = exps.filter(e => e.verdict !== "IN PROGRESS").length;
  const hitRate = Math.round((keep / done) * 100);

  // Scatter: epoch × delta, colored by verdict
  const SCATTER_W = 560;
  const SCATTER_H = 260;
  const PAD = { l: 50, r: 20, t: 20, b: 30 };
  const plotW = SCATTER_W - PAD.l - PAD.r;
  const plotH = SCATTER_H - PAD.t - PAD.b;
  const minD = -55, maxD = 15;
  const epochMin = 1, epochMax = 19;

  const toX = (epoch: number) => PAD.l + ((epoch - epochMin) / (epochMax - epochMin)) * plotW;
  const toY = (delta: number) => PAD.t + ((maxD - delta) / (maxD - minD)) * plotH;

  return (
    <section id="experiments" className="section">
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-purple-500/30 bg-purple-500/10 text-purple-400 text-xs font-medium uppercase tracking-wide mb-4">
          Experiment Lab
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          19 Experiments
        </h2>
        <p className="text-slate-400 max-w-xl mx-auto">
          Every fork, every bolt-on heuristic, every failed clever idea —
          measured against a held-out panel of strong opponents.
        </p>
      </div>

      {/* KPIs */}
      <div className="flex flex-wrap gap-4 justify-center mb-10">
        {[
          { v: `${done}`,       l: "Experiments run",  c: "#3b82f6" },
          { v: `${keep}`,       l: "Kept (KEEP)",       c: "#10b981" },
          { v: `${hitRate}%`,   l: "Hit rate",          c: "#f59e0b" },
          { v: "6",             l: "Tracks A–D + more", c: "#a855f7" },
        ].map(({ v, l, c }) => (
          <div key={l} className="glass px-6 py-4 text-center min-w-[110px]">
            <div className="text-3xl font-bold" style={{ color: c, fontFamily: "var(--font-space)" }}>{v}</div>
            <div className="text-xs text-slate-500 mt-1 uppercase tracking-wide">{l}</div>
          </div>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Bot lineage SVG */}
        <div className="glass p-5">
          <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500 mb-4">Bot Lineage</h3>
          <div className="overflow-x-auto">
            <svg width="460" height="300" viewBox="-40 -20 480 320" className="min-w-[380px]">
              {/* Edges */}
              {LINEAGE.filter(n => n.parent).map(node => {
                const par = LINEAGE.find(n => n.id === node.parent);
                if (!par) return null;
                return (
                  <line
                    key={node.id}
                    x1={par.x + 200} y1={par.y + 14}
                    x2={node.x + 200} y2={node.y}
                    stroke="#334155"
                    strokeWidth="1"
                  />
                );
              })}
              {/* Nodes */}
              {LINEAGE.map(node => {
                const c = NODE_COLOR[node.verdict] ?? "#64748b";
                const isKeep = node.verdict === "KEEP";
                return (
                  <g key={node.id} transform={`translate(${node.x + 200}, ${node.y})`}>
                    <rect
                      x={-Math.min(node.label.length * 4.2, 72)}
                      y={-10}
                      width={Math.min(node.label.length * 8.4, 144)}
                      height={26}
                      rx={4}
                      fill={isKeep ? `${c}20` : "#0f172a"}
                      stroke={c}
                      strokeWidth={isKeep ? 1.5 : 0.8}
                    />
                    <text
                      textAnchor="middle"
                      dominantBaseline="middle"
                      y={3}
                      fontSize={9}
                      fill={c}
                      fontFamily="var(--font-mono)"
                    >
                      {node.label.length > 16 ? node.label.slice(0, 15) + "…" : node.label}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
          <div className="flex gap-4 mt-3 text-xs text-slate-500">
            {[["#10b981","KEEP"],["#ef4444","DISCARD"],["#64748b","External"]].map(([c,l]) => (
              <span key={l} className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm inline-block" style={{ background: c }} />
                {l}
              </span>
            ))}
          </div>
        </div>

        {/* Scatter chart */}
        <div className="glass p-5">
          <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500 mb-1">
            Δ Win% vs Schmeekler Baseline
          </h3>
          <p className="text-xs text-slate-600 mb-4">Click a point to see details</p>
          <div className="overflow-x-auto">
            <svg
              width={SCATTER_W}
              height={SCATTER_H}
              className="cursor-pointer min-w-[340px]"
              viewBox={`0 0 ${SCATTER_W} ${SCATTER_H}`}
            >
              {/* Grid */}
              {[-40,-20,0,10].map(d => (
                <g key={d}>
                  <line x1={PAD.l} y1={toY(d)} x2={SCATTER_W - PAD.r} y2={toY(d)} stroke="#1e293b" />
                  <text x={PAD.l - 6} y={toY(d) + 4} textAnchor="end" fontSize={9} fill="#475569">{d}</text>
                </g>
              ))}
              {/* Zero line */}
              <line x1={PAD.l} y1={toY(0)} x2={SCATTER_W - PAD.r} y2={toY(0)} stroke="#334155" strokeWidth={1.5} />
              {/* Epoch ticks */}
              {[1,5,10,15,19].map(e => (
                <text key={e} x={toX(e)} y={SCATTER_H - 10} textAnchor="middle" fontSize={9} fill="#475569">{e}</text>
              ))}
              <text x={SCATTER_W / 2} y={SCATTER_H - 2} textAnchor="middle" fontSize={9} fill="#475569">Experiment #</text>
              {/* Points */}
              {exps.map(exp => {
                const x = toX(exp.epoch);
                const y = toY(exp.delta ?? 0);
                const c = VERDICT_COLOR[exp.verdict] ?? "#888";
                const isSelected = selected?.name === exp.name;
                return (
                  <g key={exp.name} onClick={() => setSelected(selected?.name === exp.name ? null : exp)}>
                    <circle
                      cx={x} cy={y} r={isSelected ? 9 : 6}
                      fill={c}
                      opacity={exp.delta === null ? 0.4 : 0.85}
                      stroke={isSelected ? "white" : "none"}
                      strokeWidth={1.5}
                    />
                    {exp.verdict === "KEEP" && (
                      <text x={x} y={y - 12} textAnchor="middle" fontSize={8} fill={c} fontWeight="bold">
                        {exp.name.length > 12 ? exp.name.slice(0,12) : exp.name}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
          {/* Legend */}
          <div className="flex gap-4 mt-2 text-xs">
            {Object.entries(VERDICT_COLOR).map(([v,c]) => (
              <span key={v} className="flex items-center gap-1 text-slate-500">
                <span className="w-2 h-2 rounded-full inline-block" style={{ background: c }} />
                {v}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Selected experiment detail */}
      {selected && (
        <div className="mt-6 glass p-5 border-l-4" style={{ borderColor: VERDICT_COLOR[selected.verdict] }}>
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2 flex-wrap">
                <code className="text-sm font-mono" style={{ color: VERDICT_COLOR[selected.verdict] }}>{selected.name}</code>
                <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: `${VERDICT_COLOR[selected.verdict]}20`, color: VERDICT_COLOR[selected.verdict] }}>
                  {selected.verdict}
                </span>
                <span className="text-xs text-slate-500">Track {selected.track} · {selected.category}</span>
              </div>
              <p className="text-sm text-slate-400 leading-relaxed">{selected.note}</p>
              {selected.win_pct !== null && (
                <div className="flex gap-6 mt-3">
                  <div>
                    <div className="text-xs text-slate-600">Win %</div>
                    <div className="font-mono font-bold" style={{ color: VERDICT_COLOR[selected.verdict] }}>{selected.win_pct}%</div>
                  </div>
                  {selected.delta !== null && (
                    <div>
                      <div className="text-xs text-slate-600">Δ vs baseline</div>
                      <div className="font-mono font-bold" style={{ color: selected.delta >= 0 ? "#10b981" : "#ef4444" }}>
                        {selected.delta > 0 ? "+" : ""}{selected.delta}pp
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            <button onClick={() => setSelected(null)} className="text-slate-600 hover:text-slate-400 flex-shrink-0">✕</button>
          </div>
        </div>
      )}

      {/* Experiment table */}
      <div className="mt-8 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-800">
              {["#","Bot","Track","Category","Win%","Δ","Verdict"].map(h => (
                <th key={h} className="text-left py-2 px-3 text-slate-500 font-medium uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {exps.map(exp => {
              const c = VERDICT_COLOR[exp.verdict];
              return (
                <tr
                  key={exp.name}
                  className="border-b border-slate-800/50 hover:bg-white/3 cursor-pointer transition-colors"
                  onClick={() => setSelected(selected?.name === exp.name ? null : exp)}
                >
                  <td className="py-2 px-3 text-slate-600 font-mono">{exp.epoch}</td>
                  <td className="py-2 px-3 font-mono text-slate-300">{exp.name}</td>
                  <td className="py-2 px-3">
                    <span className="px-1.5 py-0.5 rounded text-xs" style={{ background: `${TRACK_COLORS[exp.track] ?? "#888"}20`, color: TRACK_COLORS[exp.track] ?? "#888" }}>
                      {exp.track}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-slate-500">{exp.category}</td>
                  <td className="py-2 px-3 font-mono">{exp.win_pct != null ? `${exp.win_pct}%` : "—"}</td>
                  <td className="py-2 px-3 font-mono" style={{ color: exp.delta != null ? (exp.delta >= 0 ? "#10b981" : "#ef4444") : "#64748b" }}>
                    {exp.delta != null ? `${exp.delta > 0 ? "+" : ""}${exp.delta}pp` : "—"}
                  </td>
                  <td className="py-2 px-3">
                    <span className="font-medium" style={{ color: c }}>{exp.verdict}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
