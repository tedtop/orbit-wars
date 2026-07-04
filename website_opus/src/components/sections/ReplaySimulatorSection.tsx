"use client";
import { useState, useMemo } from "react";
import replaysIndex from "../../../public/data/replays_index.json";
import { OrbitalCanvas } from "@/components/charts/OrbitalCanvas";

interface ReplayMeta {
  id: string;
  teams: string[];
  our_result: string;
  steps: number;
  date: string;
  placement: number | null;
  num_players: number;
  available: boolean;
}

const OUR_NAME = "Montana Schmeekler";

function opponentName(teams: string[]): string {
  const others = teams.filter(t => !t.toLowerCase().includes("schmeekler") && !t.toLowerCase().includes("montana"));
  if (others.length === 0) return teams[0] ?? "Unknown";
  if (others.length === 1) return others[0];
  return `${others[0]} +${others.length - 1}`;
}

const RESULT_COLOR: Record<string, string> = { win: "#10b981", loss: "#ef4444", unknown: "#64748b" };

export function ReplaySimulatorSection() {
  const available = useMemo(
    () => (replaysIndex as ReplayMeta[]).filter(r => r.available),
    []
  );

  const [filter, setFilter]   = useState<"all" | "win" | "loss">("all");
  const [search, setSearch]   = useState("");
  const [sort, setSort]       = useState<"date" | "steps">("steps");
  const [selected, setSelected] = useState<ReplayMeta>(available[0]);

  const filtered = useMemo(() => {
    let list = available;
    if (filter !== "all") list = list.filter(r => r.our_result === filter);
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(r => r.teams.some(t => t.toLowerCase().includes(q)));
    }
    return [...list].sort((a, b) =>
      sort === "steps" ? b.steps - a.steps : b.date.localeCompare(a.date)
    );
  }, [available, filter, search, sort]);

  // Overall stats
  const wins   = available.filter(r => r.our_result === "win").length;
  const losses = available.filter(r => r.our_result === "loss").length;
  const maxSteps = Math.max(...available.map(r => r.steps));

  return (
    <section id="replays" className="section">
      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400 text-xs font-medium uppercase tracking-wide mb-4">
          Game Replays
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          Replay Simulator
        </h2>
        <p className="text-slate-400 max-w-2xl mx-auto text-sm leading-relaxed">
          {available.length} games available for instant playback — select any match to watch the orbital battle unfold step by step.
        </p>
      </div>

      {/* Fleet summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        {[
          { v: String(wins),    l: "Wins",              c: "#10b981" },
          { v: String(losses),  l: "Losses",            c: "#ef4444" },
          { v: String(maxSteps), l: "Longest (steps)",  c: "#3b82f6" },
          { v: `${available.filter(r => r.num_players === 4).length}`, l: "4-Player Matches", c: "#a855f7" },
        ].map(({ v, l, c }) => (
          <div key={l} className="glass p-4 text-center">
            <div className="text-2xl font-bold" style={{ color: c, fontFamily: "var(--font-space)" }}>{v}</div>
            <div className="text-xs text-slate-500 mt-1">{l}</div>
          </div>
        ))}
      </div>

      {/* Main two-column layout */}
      <div className="flex flex-col lg:flex-row gap-4" style={{ minHeight: 600 }}>
        {/* Sidebar */}
        <div className="lg:w-72 xl:w-80 flex flex-col gap-3 flex-shrink-0">
          {/* Controls */}
          <div className="glass p-3 flex flex-col gap-2">
            <input
              type="text"
              placeholder="Search opponent…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full bg-transparent text-sm text-slate-300 placeholder-slate-600 border border-slate-800 rounded-md px-3 py-1.5 focus:outline-none focus:border-slate-600"
            />
            <div className="flex gap-1">
              {(["all","win","loss"] as const).map(f => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className="flex-1 py-1 rounded-md text-xs capitalize font-medium transition-colors"
                  style={{
                    background: filter === f ? (f === "win" ? "#10b98130" : f === "loss" ? "#ef444430" : "#3b82f630") : "transparent",
                    color: filter === f ? (f === "win" ? "#10b981" : f === "loss" ? "#ef4444" : "#3b82f6") : "#475569",
                    border: `1px solid ${filter === f ? (f === "win" ? "#10b98140" : f === "loss" ? "#ef444440" : "#3b82f640") : "#1e293b"}`,
                  }}
                >
                  {f}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-600">
              <span>Sort:</span>
              {(["steps","date"] as const).map(s => (
                <button
                  key={s}
                  onClick={() => setSort(s)}
                  className={`capitalize transition-colors ${sort === s ? "text-slate-300" : "hover:text-slate-400"}`}
                >
                  {s}
                </button>
              ))}
            </div>
            <div className="text-xs text-slate-700">{filtered.length} matches</div>
          </div>

          {/* Replay list */}
          <div className="glass overflow-y-auto flex-1" style={{ maxHeight: 500 }}>
            {filtered.length === 0 ? (
              <div className="p-4 text-xs text-slate-600 text-center">No matches found</div>
            ) : filtered.map(r => {
              const isSelected = r.id === selected?.id;
              const opp = opponentName(r.teams);
              const color = RESULT_COLOR[r.our_result] ?? "#64748b";
              return (
                <button
                  key={r.id}
                  onClick={() => setSelected(r)}
                  className="w-full text-left px-3 py-2.5 border-b border-slate-900 transition-colors last:border-0"
                  style={{
                    background: isSelected ? "rgba(59,130,246,0.08)" : "transparent",
                    borderLeft: isSelected ? "2px solid #3b82f6" : "2px solid transparent",
                  }}
                >
                  <div className="flex items-center justify-between gap-2 mb-0.5">
                    <span
                      className="text-[10px] font-bold uppercase tracking-wide px-1.5 py-0.5 rounded"
                      style={{ color, background: `${color}15` }}
                    >
                      {r.our_result}
                    </span>
                    <span className="text-[10px] text-slate-600 font-mono">{r.steps}s</span>
                  </div>
                  <div className="text-xs text-slate-300 truncate leading-tight">vs {opp}</div>
                  <div className="text-[10px] text-slate-600 mt-0.5">{r.date} · {r.num_players}P</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Main canvas area */}
        <div className="flex-1 glass p-4 flex flex-col gap-4">
          {selected ? (
            <>
              {/* Game header */}
              <div className="flex items-start justify-between gap-3 flex-wrap">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className="px-2 py-0.5 rounded text-xs font-bold uppercase"
                      style={{
                        color: RESULT_COLOR[selected.our_result],
                        background: `${RESULT_COLOR[selected.our_result]}15`,
                      }}
                    >
                      {selected.our_result}
                    </span>
                    <span className="text-sm font-semibold text-white" style={{ fontFamily: "var(--font-space)" }}>
                      {selected.steps} steps
                    </span>
                    <span className="text-xs text-slate-500">#{selected.id}</span>
                  </div>
                  <div className="text-xs text-slate-500">
                    {selected.num_players}P · {selected.date}
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {selected.teams.map((t, i) => (
                    <span
                      key={t}
                      className="text-xs px-2 py-0.5 rounded-full border flex items-center gap-1.5"
                      style={{
                        borderColor: i === 0 ? "#3b82f640" : i === 1 ? "#ef444440" : "#a855f740",
                        color: i === 0 ? "#3b82f6" : i === 1 ? "#ef4444" : "#a855f7",
                        background: i === 0 ? "#3b82f610" : i === 1 ? "#ef444410" : "#a855f710",
                      }}
                    >
                      <span
                        className="w-2 h-2 rounded-full inline-block flex-shrink-0"
                        style={{ background: i === 0 ? "#3b82f6" : i === 1 ? "#ef4444" : "#a855f7" }}
                      />
                      {t.slice(0, 20)}{t.length > 20 ? "…" : ""}
                    </span>
                  ))}
                </div>
              </div>

              {/* Canvas */}
              <OrbitalCanvas episodeId={selected.id} teams={selected.teams} />
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
              Select a match from the sidebar to begin playback
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
