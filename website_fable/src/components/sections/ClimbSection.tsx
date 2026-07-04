"use client";

import Section from "../ui/Section";
import Reveal from "../ui/Reveal";
import LineChart, { type Series } from "../charts/LineChart";
import race from "@/data/race.json";

type RacePoint = {
  t: string;
  top1: number | null;
  prize: number | null;
  us: number | null;
  rank: number | null;
  teams: number;
};

const pts = (race as RacePoint[]).map((r) => ({ ...r, ms: Date.parse(r.t) }));

function seriesOf(key: "us" | "prize" | "top1", label: string, color: string, dash?: string): Series {
  return {
    id: key,
    label,
    color,
    dash,
    points: pts.filter((p) => p[key] !== null).map((p) => ({ x: p.ms, y: p[key] as number })),
  };
}

const SERIES: Series[] = [
  seriesOf("top1", "leader (#1)", "var(--s-violet)"),
  seriesOf("prize", "prize line (#10)", "var(--s-amber)", "5 4"),
  seriesOf("us", "montana schmeekler", "var(--s-blue)"),
];

const ANNOTATIONS = [
  { x: Date.parse("2026-06-15T10:52:00Z"), label: "comet_reaper" },
  { x: Date.parse("2026-06-17T08:53:00Z"), label: "schmeekler" },
  { x: Date.parse("2026-06-20T06:28:00Z"), label: "resubmit" },
];

const fmtDay = (ms: number) =>
  new Date(ms).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

export default function ClimbSection() {
  return (
    <Section
      id="climb"
      kicker="the ladder · jun 14–22"
      title="the climb"
      lede={
        <>
          A pipeline polled the live leaderboard every 15 minutes for nine days —
          712 snapshots of a 4,752-team field. Here is the whole campaign in one
          picture: our ladder score, the prize line we were chasing, and the
          leader receding above both. Elo starts near 600 for a fresh submission
          and converges as games accumulate — every fresh bot pays the climb tax
          from the bottom.
        </>
      }
      wide
    >
      <Reveal>
        <div className="card p-5 sm:p-7">
          <LineChart
            series={SERIES}
            annotations={ANNOTATIONS}
            formatX={fmtDay}
            formatY={(v) => String(Math.round(v))}
            yLabel="Leaderboard Elo over time"
            height={380}
          />
        </div>
      </Reveal>

      <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {(
          [
            ["1,243.8", "peak Elo — rank #144, first time in the top 150", "text-s-blue"],
            ["#415", "final rank of 4,752 — top 9%, two submissions", "text-s-blue"],
            ["~1,530", "where the money started: the rank-10 prize line", "text-s-amber"],
            ["1,694", "the leader — JAX + PPO self-play, 600M steps, ~$150 of GPU", "text-s-violet"],
          ] as const
        ).map(([num, label, color], i) => (
          <Reveal key={label} delay={i * 60}>
            <div className="card card-hover px-5 py-4">
              <span className={`block font-display text-2xl font-bold ${color}`}>{num}</span>
              <span className="mt-1 block text-xs leading-relaxed text-ink-3">{label}</span>
            </div>
          </Reveal>
        ))}
      </div>

      <Reveal className="mt-6">
        <p className="font-mono text-xs text-ink-3">
          full standings live on{" "}
          <a
            href="https://www.kaggle.com/competitions/orbit-wars/leaderboard"
            target="_blank"
            rel="noopener noreferrer"
            className="text-s-blue underline decoration-s-blue/40 underline-offset-4 hover:decoration-s-blue"
          >
            the kaggle leaderboard ↗
          </a>
        </p>
      </Reveal>
    </Section>
  );
}
