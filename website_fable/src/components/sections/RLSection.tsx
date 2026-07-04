"use client";

import { useEffect, useState } from "react";
import Section from "../ui/Section";
import Reveal from "../ui/Reveal";
import LineChart, { type Series } from "../charts/LineChart";

type RlRuns = Record<
  string,
  { label: string; points: { u: number; ent: number | null; ev: number | null }[] }
>;

const SEED_COLORS = ["var(--s-blue)", "var(--s-violet)", "var(--s-green)", "var(--s-amber)"];
const ENTROPY_KEYS = ["m3quad-v2_job1", "m3quad-v2_job2", "m3quad-v2_job3", "m3quad-v2_job4"];

/* Eval trajectories reconstructed from the experiment log (TIMELINE.md):
   inline evals were n=30 games vs the scripted greedy baseline. */
const EVAL_SERIES: Series[] = [
  {
    id: "j1",
    label: "best seed vs greedy",
    color: "var(--s-blue)",
    points: [
      { x: 100, y: 20 }, { x: 200, y: 16 }, { x: 300, y: 33 },
      { x: 400, y: 37 }, { x: 500, y: 36.7 },
    ],
  },
  {
    id: "v9",
    label: "v9 CPU · 24 seeds",
    color: "var(--s-violet)",
    points: [100, 200, 300, 400, 500, 600, 700].map((u) => ({ x: u, y: 23.3 })),
  },
  {
    id: "v10",
    label: "v10 JAX · A100",
    color: "var(--s-amber)",
    points: [100, 400, 800, 1200, 1600].map((u) => ({ x: u, y: 23.3 })),
  },
  {
    id: "cr",
    label: "ALL runs vs comet_reaper",
    color: "var(--s-red)",
    points: [100, 400, 800, 1200, 1600].map((u) => ({ x: u, y: 0 })),
  },
];

const CAMPAIGNS = [
  ["v6", "per-planet PPO, CPU fleet", "8 bugs found incl. a structural GAE grouping error; best seed 37% vs greedy", "FAIL"],
  ["v7", "sparse-vs-dense reward A/B", "both arms identical; entropy collapsed by U≈400", "FAIL"],
  ["v8", "behavior cloning, 982k moves from 1400+ Elo replays", "91% move accuracy — crushed in 88 turns, 0/200 vs comet_reaper", "FAIL"],
  ["v9", "entity-transformer PPO, 24 seeds × 69 hours", "eval frozen at 23.3% the entire run", "FAIL"],
  ["v10", "pure-JAX engine rebuild, 2× A100, 107M steps/run", "same 23.3% ceiling; comet signal 0.4% of batch — drowned out", "FAIL"],
] as const;

export default function RLSection() {
  const [entropySeries, setEntropySeries] = useState<Series[] | null>(null);

  useEffect(() => {
    let alive = true;
    fetch("/data/rl_runs.json")
      .then((r) => r.json())
      .then((runs: RlRuns) => {
        if (!alive) return;
        setEntropySeries(
          ENTROPY_KEYS.map((k, i) => ({
            id: k,
            label: `seed ${i + 1}`,
            color: SEED_COLORS[i],
            points: runs[k].points
              .filter((p) => p.ent !== null)
              .map((p) => ({ x: p.u, y: p.ent as number })),
          })),
        );
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  return (
    <Section
      id="rl"
      kicker="phases v6–v10 · the moonshot"
      title="teaching a neural net to fly"
      lede={
        <>
          The heuristic ceiling was ~1240 Elo and the leader — a JAX + PPO
          self-play pipeline — sat at 1793. So we went all-in on reinforcement
          learning: a nine-machine Jetstream2 fleet, 32 parallel seeds, then a
          from-scratch pure-JAX game engine on A100s. Five campaigns.
          Every single one closed{" "}
          <span className="font-mono text-[#e66c6c]">FAIL</span> — and the
          post-mortem is the most valuable artifact of the whole project.
        </>
      }
      wide
    >
      {/* The hero number */}
      <Reveal>
        <div className="card flex flex-col items-center gap-2 px-6 py-10 text-center">
          <span className="font-mono text-xs uppercase tracking-[0.3em] text-ink-3">
            win rate vs our own heuristic · every seed · every architecture · ~400M steps
          </span>
          <span className="font-display text-8xl font-bold leading-none text-s-red sm:text-9xl">
            0%
          </span>
          <span className="max-w-xl text-sm text-ink-2">
            comet_reaper_WR = 0.000 at every checkpoint of every run — CPU and
            GPU, self-play and cold-start, MLP and transformer, PyTorch and JAX.
          </span>
        </div>
      </Reveal>

      <div className="mt-8 grid gap-8 lg:grid-cols-2">
        <Reveal>
          <div className="card p-5 sm:p-6">
            <h3 className="font-display text-lg font-bold text-ink">
              Exhibit A — the plateau<span className="text-s-blue">.</span>
            </h3>
            <p className="mb-4 mt-1 text-xs leading-relaxed text-ink-3">
              Win rate (%) vs the scripted greedy baseline, by training update.
              One seed learned something (37%). The fleets flatlined at 23.3% —
              and nobody ever touched the engine.
            </p>
            <LineChart
              series={EVAL_SERIES}
              formatX={(u) => `U${Math.round(u)}`}
              formatY={(v) => `${Math.round(v)}%`}
              yDomain={[0, 50]}
              height={300}
              yLabel="Eval win rate vs greedy baseline by training update"
            />
          </div>
        </Reveal>

        <Reveal delay={80}>
          <div className="card p-5 sm:p-6">
            <h3 className="font-display text-lg font-bold text-ink">
              Exhibit B — entropy collapse<span className="text-s-violet">.</span>
            </h3>
            <p className="mb-4 mt-1 text-xs leading-relaxed text-ink-3">
              Policy entropy across four v9 seeds. Clean decay from ~4.9 to
              near-zero: the policy <em>committed</em> — to a local optimum that
              beats random play and nothing else. The worst possible outcome:
              not confusion, but confident mediocrity.
            </p>
            {entropySeries ? (
              <LineChart
                series={entropySeries}
                formatX={(u) => `U${Math.round(u)}`}
                formatY={(v) => v.toFixed(1)}
                yDomain={[0, 5]}
                height={300}
                yLabel="Policy entropy by training update, four seeds"
              />
            ) : (
              <div className="flex h-[300px] items-center justify-center font-mono text-xs text-ink-3">
                loading training curves…
              </div>
            )}
          </div>
        </Reveal>
      </div>

      {/* Campaign ledger */}
      <Reveal className="mt-8">
        <div className="card overflow-x-auto">
          <table className="w-full min-w-[640px] text-left">
            <thead>
              <tr className="border-b border-white/[0.07] font-mono text-[10px] uppercase tracking-widest text-ink-3">
                <th className="px-5 py-3 font-medium">campaign</th>
                <th className="px-3 py-3 font-medium">approach</th>
                <th className="px-3 py-3 font-medium">what happened</th>
                <th className="px-5 py-3 font-medium">verdict</th>
              </tr>
            </thead>
            <tbody>
              {CAMPAIGNS.map(([v, approach, result, verdict]) => (
                <tr key={v} className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.03]">
                  <td className="px-5 py-3 font-mono text-xs font-bold text-ink">{v}</td>
                  <td className="px-3 py-3 text-xs text-ink-2">{approach}</td>
                  <td className="px-3 py-3 text-xs text-ink-3">{result}</td>
                  <td className="px-5 py-3">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-critical/15 px-2.5 py-0.5 font-mono text-[10px] font-bold tracking-wider text-[#e66c6c]">
                      <span aria-hidden>✕</span>
                      {verdict}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Reveal>

      {/* Post-mortem */}
      <Reveal className="mt-10">
        <div className="card border-l-2 border-l-s-blue px-6 py-6 sm:px-8">
          <p className="kicker mb-2">the post-mortem — why theirs worked</p>
          <p className="max-w-4xl text-sm leading-relaxed text-ink-2 sm:text-base">
            The winners weren&apos;t training PPO to fly ships. They were training
            PPO to <strong className="text-ink">turn the knobs of a strong
            engine</strong> — aggression, target priority, fleet-split ratio — and
            letting orbit_lite execute the micro. That collapses the horizon from
            498 raw steps to ~30 macro-decisions:
          </p>
          <div className="mt-5 grid max-w-2xl gap-3 sm:grid-cols-2">
            <div className="rounded-xl bg-critical/10 px-5 py-4">
              <span className="block font-mono text-xs text-[#e66c6c]">raw actions · our approach</span>
              <span className="mt-1 block font-mono text-2xl font-bold text-ink">γ<sup>498</sup> ≈ 0.007</span>
              <span className="mt-1 block text-xs text-ink-3">the win signal never reaches move one</span>
            </div>
            <div className="rounded-xl bg-good/10 px-5 py-4">
              <span className="block font-mono text-xs text-[#2fc32f]">strategy knobs · theirs</span>
              <span className="mt-1 block font-mono text-2xl font-bold text-ink">γ<sup>30</sup> ≈ 0.74</span>
              <span className="mt-1 block text-xs text-ink-3">credit assignment becomes tractable</span>
            </div>
          </div>
          <p className="mt-5 max-w-4xl text-sm leading-relaxed text-ink-2 sm:text-base">
            It also explains the behavior-cloning failure: imitating a top
            player&apos;s moves without the macro strategy behind them gets you{" "}
            <em className="text-ink">the cursor without the hand</em> —
            syntactically perfect moves with no strategic coherence, crushed in 88
            turns.
          </p>
        </div>
      </Reveal>
    </Section>
  );
}
