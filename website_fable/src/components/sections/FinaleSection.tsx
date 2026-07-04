import Section from "../ui/Section";
import Reveal from "../ui/Reveal";
import finalLb from "@/data/final_leaderboard.json";
import timeline from "@/data/timeline.json";

type LbTeam = { rank: number; name: string; score: number; subs: number };
type Phase = { phase: string; date: string; title: string; body_md: string };

function LbRow({ t, highlight = false }: { t: LbTeam; highlight?: boolean }) {
  return (
    <tr
      className={`border-b border-white/[0.04] last:border-0 ${
        highlight ? "bg-s-blue/10" : "hover:bg-white/[0.03]"
      }`}
    >
      <td className="px-5 py-2.5 font-mono text-xs tabular-nums text-ink-3">#{t.rank}</td>
      <td className={`px-3 py-2.5 text-sm ${highlight ? "font-bold text-s-blue" : "text-ink-2"}`}>
        {t.name}
        {highlight && <span className="ml-2 font-mono text-[10px] text-ink-3">← us</span>}
      </td>
      <td className="px-5 py-2.5 text-right font-mono text-xs tabular-nums text-ink">
        {t.score.toFixed(1)}
      </td>
    </tr>
  );
}

export default function FinaleSection() {
  const lb = finalLb as { n_teams: number; top: LbTeam[]; us: LbTeam };
  const phases = timeline as Phase[];

  return (
    <Section
      id="finale"
      kicker="2026-06-23 · deadline"
      title="final standings"
      lede={
        <>
          When the music stopped, the answer was the one we&apos;d had since day
          three: <code className="font-mono text-s-blue">comet_reaper</code>, the
          engine clone, unmodified — because nothing in 19 experiments and five RL
          campaigns ever beat it. Top 9% of {lb.n_teams.toLocaleString()} teams,
          and a complete, honest lab notebook of how we got there.
        </>
      }
    >
      <div className="grid gap-8 lg:grid-cols-2">
        <Reveal>
          <div className="card overflow-hidden">
            <div className="border-b border-white/[0.06] px-5 py-3">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
                final snapshot · {lb.n_teams.toLocaleString()} teams ·{" "}
                <a
                  href="https://www.kaggle.com/competitions/orbit-wars/leaderboard"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-s-blue hover:underline"
                >
                  live leaderboard ↗
                </a>
              </span>
            </div>
            <table className="w-full text-left">
              <tbody>
                {lb.top.map((t) => (
                  <LbRow key={t.rank} t={t} />
                ))}
                <tr aria-hidden>
                  <td colSpan={3} className="px-5 py-1.5 text-center font-mono text-xs text-ink-3">
                    ⋯ 404 teams ⋯
                  </td>
                </tr>
                <LbRow t={lb.us} highlight />
              </tbody>
            </table>
          </div>
        </Reveal>

        <Reveal delay={80}>
          <div className="card max-h-[480px] overflow-y-auto px-5 py-4">
            <span className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
              the full mission log · {phases.length} phases
            </span>
            <div className="mt-3 flex flex-col">
              {phases.map((p, i) => (
                <details key={i} className="group border-b border-white/[0.05] py-2.5 last:border-0">
                  <summary className="cursor-pointer list-none font-mono text-xs text-ink-2 transition-colors hover:text-ink [&::-webkit-details-marker]:hidden">
                    <span aria-hidden className="mr-2 inline-block text-s-blue transition-transform group-open:rotate-90">
                      ▸
                    </span>
                    {p.title}
                  </summary>
                  <p className="mt-2 whitespace-pre-line pl-5 text-xs leading-relaxed text-ink-3">
                    {p.body_md.replace(/[*`#>]/g, "").slice(0, 700)}
                    {p.body_md.length > 700 ? "…" : ""}
                  </p>
                </details>
              ))}
            </div>
          </div>
        </Reveal>
      </div>

      <Reveal className="mt-16">
        <div className="flex flex-col items-center gap-4 text-center">
          <p className="font-display text-2xl font-bold text-ink sm:text-3xl">
            Montana Schmeekler, signing off<span className="text-s-blue">.</span>
          </p>
          <p className="max-w-lg text-sm leading-relaxed text-ink-3">
            Every experiment, dead end, and 3 a.m. GPU restart in this report is
            real and dated in the repository&apos;s provenance timeline.
          </p>
          <div className="mt-2 flex flex-wrap justify-center gap-3 font-mono text-xs">
            <a
              href="https://www.kaggle.com/competitions/orbit-wars/overview"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full bg-s-blue/15 px-4 py-2 text-s-blue transition-colors hover:bg-s-blue/25"
            >
              the competition ↗
            </a>
            <a
              href="https://www.kaggle.com/competitions/orbit-wars/leaderboard"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full bg-white/[0.05] px-4 py-2 text-ink-2 transition-colors hover:bg-white/[0.09]"
            >
              live leaderboard ↗
            </a>
          </div>
        </div>
      </Reveal>

      <footer className="mt-20 border-t border-white/[0.05] pt-6 text-center font-mono text-[11px] leading-relaxed text-ink-3">
        712 leaderboard snapshots · 511 episode replays · 19 experiments · 5 RL
        campaigns · built from the raw mission data
      </footer>
    </Section>
  );
}
