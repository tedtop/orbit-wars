import type { ReactNode } from "react";
import Section from "../ui/Section";
import Reveal from "../ui/Reveal";

/* Curated from TIMELINE.md — strictly chronological, one entry per beat of
   the story. Post-deadline autopsies are narrated as such, so v9/v10 closing
   after Jun 23 reads as intentional rather than out of order. */

type EntryType = "build" | "ship" | "breakthrough" | "deadend" | "pivot" | "insight";

const TYPE_META: Record<EntryType, { label: string; color: string }> = {
  build: { label: "build", color: "var(--s-blue)" },
  ship: { label: "submitted", color: "var(--s-violet)" },
  breakthrough: { label: "breakthrough", color: "var(--s-green)" },
  deadend: { label: "dead end", color: "var(--s-red)" },
  pivot: { label: "pivot", color: "var(--s-amber)" },
  insight: { label: "insight", color: "var(--s-amber)" },
};

const LOG: { date: string; type: EntryType; title: string; body: ReactNode }[] = [
  {
    date: "jun 13",
    type: "build",
    title: "23 bots from across the sciences",
    body: "Day one: breadth over depth. One bot per discipline — SIR epidemiology, Turing patterns, ant-colony stigmergy, minimax, Markowitz portfolio theory, 18 more. The finance bot wins the internal arena at 66%.",
  },
  {
    date: "jun 13–14",
    type: "build",
    title: "The harness",
    body: "OpenSkill round-robin arena, a leaderboard pipeline polling Kaggle every 15 minutes, and a live dashboard. 80,000 games in 10 hours to tell 23 theories apart objectively.",
  },
  {
    date: "jun 14",
    type: "ship",
    title: "First blood",
    body: "markowitz_portfolio_optimization (578.2) and coordinated_strike_interceptor (531.7) go live — mid-pack of 4,400+ teams. “Then we slept.”",
  },
  {
    date: "jun 14–15",
    type: "breakthrough",
    title: "The engine discovery",
    body: "Leaderboard archaeology: the entire top of the ladder is one lineage — The Producer, running the orbit_lite planning engine. We clone the approach, name ours comet_reaper, and tie the best public bot within a day.",
  },
  {
    date: "jun 15",
    type: "deadend",
    title: "Clone the humans",
    body: "Behavior-clone the teams rated >1500 and self-play against the clones. The clone loses 0–16 to our own engine. Lesson one, paid early: mechanics beat imitation.",
  },
  {
    date: "jun 15",
    type: "deadend",
    title: "The fork bake-off",
    body: "Five bolt-on heuristics — precog, kingmaker, maestro, helmsman, oracle — all land at parity against base comet_reaper. First hint that the engine is a tight local optimum.",
  },
  {
    date: "jun 16–17",
    type: "breakthrough",
    title: "Top 150",
    body: "comet_reaper reaches #144 at Elo 1243.8. Forum intel names the two levers above us: better scoring/sizing (→ ~1500) and deeper search (→ 1793, the leader). We have 30× compute headroom per turn.",
  },
  {
    date: "jun 17",
    type: "breakthrough",
    title: "schmeekler beats the engine",
    body: "Our one genuinely novel strategy: capture the static planets first — they don't rotate, so they hold the safe periphery forever. 72% vs comet_reaper seat-swapped, best-in-pod in 4P. Submitted the same day.",
  },
  {
    date: "jun 17",
    type: "deadend",
    title: "Five kill-tests in 24 hours",
    body: "Comet-targeting bonus, a learned value function (AUC 0.983 — genuinely predictive, still didn't help), Boltzmann opponent search, elimination mode, multi-fleet coordination. Verdict, five times over: DISCARD.",
  },
  {
    date: "jun 18",
    type: "pivot",
    title: "The zero-choice discovery closes v5",
    body: "Profiling crystallizes why nothing sticks: the engine's filters leave 0–4 candidate moves per turn — most turns zero or one. There is nothing to re-rank at depth 1. With five days left, we go all-in on reinforcement learning.",
  },
  {
    date: "jun 18–19",
    type: "deadend",
    title: "v6 — PPO self-play at CPU scale",
    body: "A structural GAE bug found and fixed, nine Jetstream2 machines, 32 parallel seeds. Best seed: 37% vs a scripted greedy baseline, 0% vs comet_reaper. Fails its pre-committed gate and is closed on schedule.",
  },
  {
    date: "jun 20",
    type: "build",
    title: "Harden the floor",
    body: "Full latency and coordinate-order audit of the live bot before locking it: median 6 ms/turn against a 1 s budget, no bugs found. comet_reaper locked as the defended submission.",
  },
  {
    date: "jun 20–21",
    type: "deadend",
    title: "v7 and v8 fall in 48 hours",
    body: "The sparse-vs-dense reward A/B: both arms identical, entropy collapse by U≈400. Then per-planet behavior cloning on 982k expert moves: 91% move accuracy, crushed in 88 turns, 0/200 vs the engine.",
  },
  {
    date: "jun 22",
    type: "build",
    title: "v10 — rewrite the game in JAX",
    body: "The whole game engine vectorised for GPUs in a day: 20/20 exact-parity validation against the reference implementation, four subtle physics bugs fixed, deployed to two A100s the day before deadline.",
  },
  {
    date: "jun 23",
    type: "ship",
    title: "Deadline",
    body: "Final answer: comet_reaper — the day-three engine clone, unmodified, because nothing in 19 experiments ever beat it. #415 of 4,752 at the deadline, before the post-deadline ladder (suddenly 4P-heavy, our weakest format) began grinding everyone's ratings toward their final resting places.",
  },
  {
    date: "jun 24–30",
    type: "deadend",
    title: "The autopsies",
    body: "Training kept running past the deadline to finish the science: v9's 24 CPU seeds (69 hours each) and v10's A100 runs (107M+ steps, then a retry with the engine injected into the training signal) all converge to the same 23.3% ceiling — and 0% vs comet_reaper.",
  },
  {
    date: "jul 4",
    type: "insight",
    title: "The post-mortem",
    body: (
      <>
        We didn&apos;t place top 10 — a day-three engine clone, #415 of 4,752 at
        the deadline, was our ceiling this season. What we took away instead: a crash course in PPO
        self-play, behavior cloning, value functions, MCTS, and Boltzmann search; a
        pure-JAX game engine and a nine-machine CPU + A100{" "}
        <a
          href="https://jetstream-cloud.org/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-s-blue underline decoration-s-blue/40 underline-offset-4 hover:decoration-s-blue"
        >
          Jetstream2
        </a>{" "}
        training fleet; Optuna sweeps, OpenSkill gauntlets, and a live ops dashboard
        — plus seven hard-earned lessons about search, imitation, evaluation, and
        problem framing.
      </>
    ),
  },
  {
    date: "next szn",
    type: "pivot",
    title: "The playbook for a top-10 run",
    body: "Everything above points one direction: keep a strong engine in the loop. Train PPO over the engine's strategy knobs — aggression, target priority, fleet sizing — instead of raw ship commands. Feed it dense reward from the engine's own board evaluation every turn, and train against a fixed, diverse pool of strong opponents instead of pure self-play. Spend the 30× compute headroom on deeper multi-ply search — the lever that separates 1500 from 1793. And start all of it on day one, not day six.",
  },
];

export default function MissionLogSection() {
  return (
    <Section
      id="log"
      kicker="jun 13 → jul 4 · the provenance trail"
      title="the mission log"
      lede={
        <>
          The repository keeps a dated, append-only timeline of every milestone —
          dead ends left visible, never rewritten. This is that document,
          retold in order: three weeks from 23 hand-rolled bots to a closed RL
          campaign and one honest answer.
        </>
      }
    >
      <div className="mb-8 flex flex-wrap gap-x-5 gap-y-2">
        {Object.values(TYPE_META).map((m) => (
          <span key={m.label} className="flex items-center gap-2 font-mono text-[11px] text-ink-3">
            <span aria-hidden className="h-2.5 w-2.5 rounded-full" style={{ background: m.color }} />
            {m.label}
          </span>
        ))}
      </div>

      <ol className="relative ml-2 border-l border-white/[0.09] sm:ml-24">
        {LOG.map((e, i) => {
          const meta = TYPE_META[e.type];
          return (
            <Reveal as="li" key={e.title} delay={(i % 2) * 60} className="relative pb-10 pl-8 last:pb-0">
              {/* node */}
              <span
                aria-hidden
                className="absolute -left-[7px] top-1.5 h-3.5 w-3.5 rounded-full border-2 border-void"
                style={{ background: meta.color, boxShadow: `0 0 12px -2px ${meta.color}` }}
              />
              {/* date — gutter on desktop, inline on mobile */}
              <span className="mb-1 block font-mono text-[11px] tracking-widest text-ink-3 sm:absolute sm:-left-24 sm:top-1 sm:mb-0 sm:w-20 sm:text-right">
                {e.date}
              </span>
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <h3 className="font-display text-lg font-bold leading-snug text-ink">
                  {e.title}
                </h3>
                <span
                  className="rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest"
                  style={{ color: meta.color, background: `color-mix(in srgb, ${meta.color} 14%, transparent)` }}
                >
                  {meta.label}
                </span>
              </div>
              <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-ink-2">{e.body}</p>
            </Reveal>
          );
        })}
      </ol>
    </Section>
  );
}
