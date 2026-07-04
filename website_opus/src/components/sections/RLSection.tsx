"use client";
import { useState, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import rlRuns from "../../../public/data/rl_runs.json";

// Pick interesting seeds to display — best performers
const FEATURED_RUNS = [
  "ppo-1-of-3_job1",
  "ppo-3-of-3_job3",
  "ppo-3-of-3_job4",
];
const RUN_COLORS = ["#3b82f6", "#a855f7", "#10b981", "#f59e0b"];

export function RLSection() {
  const [metric, setMetric] = useState<"entropy" | "clip_frac" | "eval_wr">("entropy");

  const allRuns = rlRuns as Record<string, Array<{
    update: number; clip_frac: number; entropy: number | null; eval_wr?: number | null;
  }>>;

  // Find featured runs or fall back to first 3
  const runKeys = FEATURED_RUNS.filter(k => allRuns[k]).length > 0
    ? FEATURED_RUNS.filter(k => allRuns[k])
    : Object.keys(allRuns).slice(0, 3);

  // Build a merged timeline: for each update, one entry per run
  const merged = useMemo(() => {
    const updates = new Set<number>();
    runKeys.forEach(k => allRuns[k]?.forEach(r => updates.add(r.update)));
    return Array.from(updates).sort((a,b) => a-b).map(u => {
      const entry: Record<string, number | null> = { update: u };
      runKeys.forEach((k, i) => {
        const row = allRuns[k]?.find(r => r.update === u);
        if (row) {
          entry[`run_${i}`] = metric === "entropy" ? (row.entropy ?? null)
            : metric === "clip_frac" ? row.clip_frac
            : (row.eval_wr ?? null);
        }
      });
      return entry;
    });
  }, [metric, runKeys, allRuns]);

  const METRICS = [
    { key: "entropy",   label: "Entropy",   desc: "Policy randomness — should decay as policy commits" },
    { key: "clip_frac", label: "Clip Frac",  desc: "PPO update clipping — healthy range 0.05–0.30" },
    { key: "eval_wr",   label: "Win Rate vs Greedy", desc: "% wins in evaluation games (goal: >25%)" },
  ];

  return (
    <section id="rl" className="section">
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-rose-500/30 bg-rose-500/10 text-rose-400 text-xs font-medium uppercase tracking-wide mb-4">
          v6 · RL Self-Play
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          The Neural Net Attempt
        </h2>
        <p className="text-slate-400 max-w-xl mx-auto">
          With 4 days left, we pivoted to PPO self-play on a 9-machine HPC cluster.
          The plan: replace the heuristic engine entirely with a learned policy.
        </p>
      </div>

      {/* Fleet stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-10">
        {[
          { label: "Jetstream2 Instances", value: "9",       c: "#3b82f6" },
          { label: "Parallel Seeds",       value: "32",      c: "#a855f7" },
          { label: "Peak Throughput",      value: "1,312 SPS", c: "#10b981" },
          { label: "Best vs Greedy",       value: "37%",     c: "#f59e0b" },
        ].map(({ label, value, c }) => (
          <div key={label} className="glass p-4 text-center">
            <div className="text-2xl font-bold" style={{ color: c, fontFamily: "var(--font-space)" }}>{value}</div>
            <div className="text-xs text-slate-500 mt-1 uppercase tracking-wide">{label}</div>
          </div>
        ))}
      </div>

      {/* Training curve */}
      <div className="glass p-6 mb-8">
        <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
          <h3 className="text-sm font-semibold text-slate-300">Training Dynamics — Top Seeds</h3>
          <div className="flex gap-1">
            {METRICS.map(m => (
              <button
                key={m.key}
                onClick={() => setMetric(m.key as typeof metric)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${metric === m.key ? "bg-slate-700 text-white" : "text-slate-500 hover:text-slate-300"}`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
        <p className="text-xs text-slate-600 mb-4">
          {METRICS.find(m => m.key === metric)?.desc}
        </p>

        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={merged} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis dataKey="update" tick={{ fontSize: 10, fill: "#475569" }} axisLine={false} tickLine={false} label={{ value: "PPO Update", position: "insideBottom", offset: -4, fill: "#475569", fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10, fill: "#475569" }} axisLine={false} tickLine={false} width={40} />
            <Tooltip
              contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 }}
              labelStyle={{ color: "#94a3b8" }}
            />
            {metric === "clip_frac" && (
              <>
                <ReferenceLine y={0.05} stroke="#10b981" strokeDasharray="4 3" strokeWidth={1} label={{ value: "min 0.05", fill: "#10b981", fontSize: 9 }} />
                <ReferenceLine y={0.30} stroke="#10b981" strokeDasharray="4 3" strokeWidth={1} label={{ value: "max 0.30", fill: "#10b981", fontSize: 9 }} />
              </>
            )}
            {metric === "eval_wr" && (
              <ReferenceLine y={0.25} stroke="#f59e0b" strokeDasharray="4 3" strokeWidth={1.5} label={{ value: "Gate 25%", fill: "#f59e0b", fontSize: 10 }} />
            )}
            {runKeys.map((k, i) => (
              <Line
                key={k}
                dataKey={`run_${i}`}
                name={k.length > 20 ? k.slice(0,20) + "…" : k}
                stroke={RUN_COLORS[i % RUN_COLORS.length]}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>

        <div className="flex flex-wrap gap-4 mt-3">
          {runKeys.map((k, i) => (
            <div key={k} className="flex items-center gap-1.5 text-xs text-slate-500">
              <span className="w-4 h-0.5 rounded-full inline-block" style={{ background: RUN_COLORS[i % RUN_COLORS.length] }} />
              {k}
            </div>
          ))}
        </div>
      </div>

      {/* Diagnosis */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="glass p-5 border-l-4 border-rose-500/50">
          <h3 className="font-semibold text-white mb-3">What Happened</h3>
          <ul className="space-y-2 text-sm text-slate-400">
            <li className="flex gap-2"><span className="text-rose-400 mt-0.5">✕</span>Policy converged to a <strong className="text-white">local optimum at ~20% vs greedy</strong> — not still exploring, but committed to something bad</li>
            <li className="flex gap-2"><span className="text-rose-400 mt-0.5">✕</span>Zero signal vs comet_reaper at <em>every</em> depth, <em>every</em> seed: <strong className="text-white">0.000% across 32 seeds</strong></li>
            <li className="flex gap-2"><span className="text-rose-400 mt-0.5">✕</span>32 of 32 seeds showed an inverted-U trajectory: peak at U=100–200, decline after</li>
            <li className="flex gap-2"><span className="text-amber-400 mt-0.5">!</span>8 critical bugs found and fixed during the run (GAE grouping, entropy coefficient 50× too high, eval null-agent, etc.)</li>
          </ul>
        </div>

        <div className="glass p-5 border-l-4 border-blue-500/50">
          <h3 className="font-semibold text-white mb-3">Why It Failed</h3>
          <p className="text-sm text-slate-400 leading-relaxed mb-3">
            The heuristic engine collapses each turn to <strong className="text-white">0–4 candidates</strong> via its capture-floor filter.
            A neural policy needs to learn to beat a greedy opponent that already plays near-optimally in this constrained action space.
          </p>
          <p className="text-sm text-slate-400 leading-relaxed">
            The forum leader (#1 at 1793 Elo, Lin Myat Ko) trained for <strong className="text-white">600M steps with JAX</strong>.
            We had ~11M steps available by deadline. The throughput gap (~15×) was unbridgeable in 4 days.
          </p>
        </div>

        <div className="glass p-5 border-l-4 border-emerald-500/50 md:col-span-2">
          <h3 className="font-semibold text-white mb-3">Infrastructure That Remains Useful</h3>
          <div className="grid sm:grid-cols-3 gap-3 text-sm text-slate-400">
            <div><strong className="text-emerald-400">eval_checkpoints.py</strong> — seat-balanced RL vs RL evaluation with JSON output</div>
            <div><strong className="text-emerald-400">sync_checkpoints.sh</strong> — fleet champion promotion loop across 9 hosts</div>
            <div><strong className="text-emerald-400">ORCHESTRATOR_STATE.md</strong> — structured autonomous session log format</div>
          </div>
        </div>
      </div>
    </section>
  );
}
