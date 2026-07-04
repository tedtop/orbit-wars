import Section from "../ui/Section";
import Reveal from "../ui/Reveal";
import experiments from "@/data/experiments.json";

type Experiment = {
  epoch: number;
  name: string;
  base: string;
  track: string;
  verdict: "KEEP" | "DISCARD";
  win_pct: number | null;
  delta: number | null;
  category: string;
  note: string;
};

function VerdictChip({ verdict }: { verdict: Experiment["verdict"] }) {
  const keep = verdict === "KEEP";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 font-mono text-[10px] font-bold tracking-wider ${
        keep ? "bg-good/15 text-[#2fc32f]" : "bg-critical/15 text-[#e66c6c]"
      }`}
    >
      <span aria-hidden>{keep ? "✓" : "✕"}</span>
      {verdict}
    </span>
  );
}

function WinBar({ pct }: { pct: number | null }) {
  if (pct === null)
    return <span className="font-mono text-[11px] text-ink-3">n/a</span>;
  return (
    <div className="flex items-center gap-2">
      <div
        className="relative h-2 w-full max-w-[140px] overflow-hidden rounded-full bg-[rgba(255,255,255,0.06)]"
        role="img"
        aria-label={`${pct}% win rate on the evaluation panel`}
      >
        <div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{
            width: `${pct}%`,
            background: pct > 55 ? "var(--s-green)" : pct < 45 ? "var(--s-red)" : "var(--baseline)",
          }}
        />
        {/* 50% parity tick */}
        <div className="absolute inset-y-0 left-1/2 w-px bg-[rgba(242,244,251,0.35)]" />
      </div>
      <span className="font-mono text-[11px] tabular-nums text-ink-2">{pct}%</span>
    </div>
  );
}

/* Plain-words explainers for the most interesting weapons in the ledger.
   `bots` are footnote links to the ledger rows above (tr ids = exp-<name>). */
const FIELD_NOTES = [
  {
    bots: ["CR_mcts_v1", "CR_mcts_v2"],
    accent: "var(--s-violet)",
    title: "MCTS — Monte Carlo Tree Search",
    body: (
      <>
        The algorithm family behind AlphaGo and modern chess engines. Instead of
        scoring only your immediate options, you play out imagined futures —
        &ldquo;if I do X, they answer A or B; if A, I could do Y or Z…&rdquo; —
        building a tree of possibilities, running quick simulated games down the
        promising branches, and picking the move whose branch wins most often. We
        built two variants on top of comet_reaper. Both landed at 75% on the panel
        — one point over baseline — and were discarded. The reason is the exhibit
        below: the engine&apos;s filters left 0–4 candidate moves per turn, and you
        can&apos;t grow a search tree when most turns have one branch.
      </>
    ),
  },
  {
    bots: ["CR_stochastic"],
    accent: "var(--s-amber)",
    title: "Boltzmann search — playing the odds on your opponent",
    body: (
      <>
        Standard search assumes the opponent plays their single best move.
        Boltzmann search assumes they play <em>probabilistically</em>: each
        plausible reply gets a probability weighted by how good it is — P ∝
        exp(score/τ), the softmax distribution borrowed from statistical physics,
        where the temperature τ says how sharp the opponent is (low τ = near-perfect,
        high τ = sloppy). Your move is chosen by expected value over that whole
        distribution instead of betting on one predicted response. It scored 61%
        overall against schmeekler&apos;s 78% → DISCARD. One fascinating quirk: it
        was our <em>best</em> bot against the weakest opponent (80% vs 1266-elo) —
        modeling sloppy opponents probabilistically is exactly right; it just gave
        too much away against strong ones.
      </>
    ),
  },
] as const;

/* The decision-funnel exhibit: most turns there is nothing to decide. */
function ZeroChoiceExhibit() {
  const segs = [
    { n: 64, label: "0 valid moves", color: "#104281" },
    { n: 47, label: "exactly 1", color: "#1c5cab" },
    { n: 22, label: "2–4 choices", color: "#3987e5" },
  ];
  const total = 133;
  return (
    <div className="card px-6 py-6">
      <p className="kicker mb-1 !text-s-amber">the key discovery</p>
      <h3 className="font-display text-2xl font-bold text-ink">
        Most turns, there is nothing to decide<span className="text-s-amber">.</span>
      </h3>
      <p className="mt-3 max-w-3xl text-sm leading-relaxed text-ink-2">
        Profiling comet_reaper&apos;s planner over a full game: of 133 turns where a
        decision could happen, the engine&apos;s capture-guarantee filters left{" "}
        <strong className="text-ink">zero</strong> candidate moves on 64 of them and
        exactly one on another 47. Only 22 turns per game offered a real choice.
      </p>
      <div className="mt-5 flex h-9 w-full gap-0.5 overflow-hidden rounded-lg" role="img" aria-label="Of 133 decision turns: 64 have zero valid moves, 47 exactly one, 22 have two to four choices">
        {segs.map((s) => (
          <div
            key={s.label}
            className="flex items-center justify-center"
            style={{ width: `${(s.n / total) * 100}%`, background: s.color }}
          >
            <span className="font-mono text-[11px] font-bold text-white/90">{s.n}</span>
          </div>
        ))}
      </div>
      <div className="mt-2 flex gap-0.5">
        {segs.map((s) => (
          <span
            key={s.label}
            className="font-mono text-[10px] text-ink-3"
            style={{ width: `${(s.n / total) * 100}%` }}
          >
            {s.label}
          </span>
        ))}
      </div>
      <p className="mt-5 border-l-2 border-l-s-amber pl-4 text-sm italic leading-relaxed text-ink-2">
        You can&apos;t optimize a decision that isn&apos;t being made. Every bolt-on —
        potential fields, 2-ply search, value-function re-ranking, elimination
        bonuses — landed at parity because there was nothing to re-rank at depth 1.
        The engine is a tight local optimum with the decisions already compiled away.
      </p>
    </div>
  );
}

export default function LabSection() {
  const rows = experiments as Experiment[];
  const discards = rows.filter((r) => r.verdict === "DISCARD").length;
  return (
    <Section
      id="lab"
      kicker="phases 5–6 · the ratchet"
      title="the experiment ledger"
      lede={
        <>
          Every idea got the same treatment: a hypothesis, a seat-swapped
          gauntlet against a fixed opponent panel, and a pre-committed verdict —{" "}
          <span className="font-mono text-[#2fc32f]">KEEP</span> only if it beat
          the champion by a threshold, otherwise{" "}
          <span className="font-mono text-[#e66c6c]">DISCARD</span> with the
          reason logged.
          <span className="mt-6 block font-display text-3xl font-bold text-ink sm:text-4xl">
            {rows.length} experiments entered.{" "}
            <span className="text-s-green">{rows.length - discards} survived.</span>
          </span>
          <span className="mt-3 block">
            That&apos;s not failure — that&apos;s what a ratchet looks like: the
            champion can only get stronger.
          </span>
        </>
      }
      wide
    >
      <Reveal>
        <div className="card overflow-x-auto">
          <table className="w-full min-w-[640px] text-left">
            <thead>
              <tr className="border-b border-white/[0.07] font-mono text-[10px] uppercase tracking-widest text-ink-3">
                <th className="px-5 py-3 font-medium">#</th>
                <th className="px-3 py-3 font-medium">experiment</th>
                <th className="px-3 py-3 font-medium">idea</th>
                <th className="px-3 py-3 font-medium">panel win rate · 50% = parity</th>
                <th className="px-5 py-3 font-medium">verdict</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.epoch}
                  id={`exp-${r.name}`}
                  className="border-b border-white/[0.04] transition-colors last:border-0 hover:bg-white/[0.03]"
                >
                  <td className="px-5 py-2.5 font-mono text-[11px] text-ink-3">{r.epoch}</td>
                  <td className="px-3 py-2.5 font-mono text-xs text-ink">{r.name}</td>
                  <td className="px-3 py-2.5 text-xs text-ink-3">{r.category}</td>
                  <td className="px-3 py-2.5"><WinBar pct={r.win_pct} /></td>
                  <td className="px-5 py-2.5"><VerdictChip verdict={r.verdict} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Reveal>

      <div className="mt-10 grid gap-4 lg:grid-cols-2">
        {FIELD_NOTES.map((n, i) => (
          <Reveal key={n.title} delay={i * 80}>
            <div
              className="card card-hover flex h-full flex-col px-6 py-5 sm:px-7 sm:py-6"
              style={{ borderLeft: `2px solid ${n.accent}` }}
            >
              <p
                className="font-mono text-[10px] uppercase tracking-[0.25em]"
                style={{ color: n.accent }}
              >
                field note
              </p>
              <h3 className="mt-1.5 font-display text-lg font-bold text-ink">{n.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-2">{n.body}</p>
              <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-white/[0.06] pt-4">
                <span className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
                  tested as
                </span>
                {n.bots.map((b) => (
                  <a
                    key={b}
                    href={`#exp-${b}`}
                    className="rounded-full bg-s-blue/10 px-2.5 py-0.5 font-mono text-[11px] text-s-blue transition-colors hover:bg-s-blue/20"
                  >
                    {b} ↑
                  </a>
                ))}
              </div>
            </div>
          </Reveal>
        ))}
      </div>

      <Reveal className="mt-10">
        <ZeroChoiceExhibit />
      </Reveal>
    </Section>
  );
}
