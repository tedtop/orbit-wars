"use client";
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import { useState } from "react";
import type { ScorePoint } from "@/data/types";

const BOT_COLORS: Record<string, string> = {
  "markowitz_portfolio_optimization v1": "#64748b",
  "coordinated_strike_interceptor v1":   "#64748b",
  "comet_reaper v1":                     "#f59e0b",
  "schmeekler@1.5":                      "#3b82f6",
  "schmeekler_fmt":                      "#10b981",
  "comet_reaper_1235":                   "#f97316",
};

const BOT_SHORT: Record<string, string> = {
  "markowitz_portfolio_optimization v1": "markowitz",
  "coordinated_strike_interceptor v1":   "CSI",
  "comet_reaper v1":                     "comet_reaper",
  "schmeekler@1.5":                      "schmeekler",
  "schmeekler_fmt":                      "schmeekler_fmt",
  "comet_reaper_1235":                   "comet_reaper_1235",
};

const RANGE_OPTS = [
  { label: "All",  hours: null },
  { label: "3d",   hours: 72  },
  { label: "24h",  hours: 24  },
  { label: "6h",   hours: 6   },
];

const CustomTooltip = ({ active, payload, label }: {active?: boolean; payload?: Array<{value: number; payload: ScorePoint}>; label?: string}) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="glass px-3 py-2 text-xs">
      <div className="text-slate-400 mb-1 font-mono">{new Date(d.time).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</div>
      <div className="font-semibold text-white">Score: {d.score.toFixed(1)}</div>
      <div className="text-slate-400">Rank: #{d.rank}</div>
      <div className="text-slate-500 mt-0.5">{BOT_SHORT[d.bot_name] ?? d.bot_name}</div>
    </div>
  );
};

interface Props {
  data: ScorePoint[];
  submissionTimes?: Array<{ time: string; name: string }>;
}

export function ScoreChart({ data, submissionTimes = [] }: Props) {
  const [range, setRange] = useState<number | null>(null);
  const [view, setView] = useState<"score" | "rank">("score");

  const filtered = (() => {
    if (!range) return data;
    const cutoff = new Date(data.at(-1)!.time).getTime() - range * 3600 * 1000;
    return data.filter(d => new Date(d.time).getTime() >= cutoff);
  })();

  // Find submission events within visible range
  const visibleSubs = submissionTimes.filter(s => {
    const t = new Date(s.time).getTime();
    const lo = new Date(filtered[0]?.time ?? 0).getTime();
    const hi = new Date(filtered.at(-1)?.time ?? 0).getTime();
    return t >= lo && t <= hi;
  });

  const scores = filtered.map(d => d.score);
  const yMin = Math.min(...scores) - 30;
  const yMax = Math.max(...scores) + 80;

  const ticks = filtered.reduce<string[]>((acc, d, i) => {
    if (i % Math.ceil(filtered.length / 6) === 0) acc.push(d.time);
    return acc;
  }, []);

  const fmtTime = (iso: string) =>
    new Date(iso).toLocaleString("en-US", { month: "numeric", day: "numeric", hour: "2-digit" });

  return (
    <div>
      {/* Controls */}
      <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
        <div className="flex gap-1">
          {(["score", "rank"] as const).map(v => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${view === v ? "bg-blue-600 text-white" : "text-slate-500 hover:text-slate-300"}`}
            >
              {v === "score" ? "Elo Score" : "Rank"}
            </button>
          ))}
        </div>
        <div className="flex gap-1">
          {RANGE_OPTS.map(({ label, hours }) => (
            <button
              key={label}
              onClick={() => setRange(hours)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${range === hours ? "bg-slate-700 text-white" : "text-slate-500 hover:text-slate-300"}`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Score chart */}
      {view === "score" && (
        <ResponsiveContainer width="100%" height={320}>
          <AreaChart data={filtered} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              {Object.entries(BOT_COLORS).map(([bot, color]) => (
                <linearGradient key={bot} id={`grad-${bot.replace(/\s/g, "_")}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={color} stopOpacity={0.02} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis
              dataKey="time"
              ticks={ticks}
              tickFormatter={fmtTime}
              tick={{ fontSize: 10, fill: "#475569" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[yMin, yMax]}
              tick={{ fontSize: 10, fill: "#475569" }}
              axisLine={false}
              tickLine={false}
              width={45}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={1500} stroke="#a855f7" strokeDasharray="5 4" strokeWidth={1.5} label={{ value: "Prize ~1500", fill: "#a855f7", fontSize: 10, position: "insideTopRight" }} />
            <ReferenceLine y={1328} stroke="#f97316" strokeDasharray="4 3" strokeWidth={1} label={{ value: "Best 1328", fill: "#f97316", fontSize: 9, position: "insideTopRight" }} />
            {visibleSubs.map(s => (
              <ReferenceLine
                key={s.time}
                x={s.time}
                stroke={BOT_COLORS[s.name] ?? "#888"}
                strokeDasharray="6 4"
                strokeWidth={1.5}
                label={{ value: BOT_SHORT[s.name] ?? s.name, fill: BOT_COLORS[s.name] ?? "#888", fontSize: 9, position: "insideTopLeft", angle: -45 }}
              />
            ))}
            <Area
              type="monotone"
              dataKey="score"
              stroke="#3b82f6"
              strokeWidth={2}
              fill="url(#grad-schmeekler@1.5)"
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}

      {/* Rank chart */}
      {view === "rank" && (
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={filtered} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis
              dataKey="time"
              ticks={ticks}
              tickFormatter={fmtTime}
              tick={{ fontSize: 10, fill: "#475569" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              reversed
              tick={{ fontSize: 10, fill: "#475569" }}
              axisLine={false}
              tickLine={false}
              width={45}
              label={{ value: "Rank ↓ better", angle: -90, fill: "#475569", fontSize: 9 }}
            />
            <Tooltip content={<CustomTooltip />} />
            {visibleSubs.map(s => (
              <ReferenceLine
                key={s.time}
                x={s.time}
                stroke={BOT_COLORS[s.name] ?? "#888"}
                strokeDasharray="6 4"
                strokeWidth={1.5}
              />
            ))}
            <Line
              type="monotone"
              dataKey="rank"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}

      {/* Bot legend */}
      <div className="flex flex-wrap gap-3 mt-4 justify-center">
        {Object.entries(BOT_SHORT).map(([full, short]) => (
          <div key={full} className="flex items-center gap-1.5 text-xs text-slate-500">
            <span className="w-3 h-0.5 rounded-full inline-block" style={{ background: BOT_COLORS[full] }} />
            {short}
          </div>
        ))}
      </div>
    </div>
  );
}
