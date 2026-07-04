import Section from "../ui/Section";
import Reveal from "../ui/Reveal";
import scientists from "@/data/scientists.json";

const CLUSTER_META: Record<string, { label: string; color: string }> = {
  physics: { label: "physics & control", color: "var(--s-violet)" },
  life: { label: "life sciences", color: "var(--s-green)" },
  math_cs: { label: "math, games & ML", color: "var(--s-blue)" },
  economics: { label: "economics", color: "var(--s-amber)" },
  other: { label: "military & opportunism", color: "var(--s-red)" },
};

/** Break long snake_case names at underscore boundaries, never mid-word. */
function BotName({ name }: { name: string }) {
  const parts = name.split("_");
  return (
    <>
      {parts.map((p, i) => (
        <span key={i}>
          {p}
          {i < parts.length - 1 && (
            <>
              _<wbr />
            </>
          )}
        </span>
      ))}
    </>
  );
}

export default function ScientistsSection() {
  return (
    <Section
      id="scientists"
      kicker="phase 0 · jun 13"
      title="23 theories walk into an arena"
      lede={
        <>
          Day one strategy: breadth. We built 23 bots, each a serious
          implementation of a theory from a different scientific field —
          epidemiology, control theory, ant colonies, chaos, macroeconomics —
          then ran a 10-hour, 80,000-game round-robin arena with OpenSkill
          ratings to find out which science plays the best space war.{" "}
          <strong className="text-ink">
            The winner: finance.
          </strong>{" "}
          Markowitz portfolio optimization — treat every attack as an
          investment, diversify the risk — beat the field at 66%.
        </>
      }
      wide
    >
      <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {(scientists as Array<{ name: string; field: string; cluster: string; description: string }>).map(
          (s, i) => {
            const meta = CLUSTER_META[s.cluster] ?? CLUSTER_META.other;
            const champion = s.name === "markowitz_portfolio_optimization";
            return (
              <Reveal as="li" key={s.name} delay={(i % 4) * 55}>
                <div
                  className={`card card-hover relative h-full px-4 py-4 ${
                    champion ? "outline outline-1 outline-s-amber/60" : ""
                  }`}
                >
                  {champion && (
                    <span className="absolute -top-3 right-4 inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-[#f2c14e]/50 bg-gradient-to-r from-[#3a2a08] to-[#59400f] px-3 py-1 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-[#f2c14e] shadow-[0_0_18px_-4px_rgba(242,193,78,0.55)]">
                      <span aria-hidden>🏆</span>
                      arena champion
                    </span>
                  )}
                  <div className="flex items-center gap-2">
                    <span
                      aria-hidden
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ background: meta.color }}
                    />
                    <span className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
                      {s.field}
                    </span>
                  </div>
                  <h3 className="mt-2 font-mono text-[13px] font-semibold text-ink">
                    <BotName name={s.name} />
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-ink-2">
                    {champion
                      ? s.description.replace(/\s*🏆 Won the arena\.?/, "")
                      : s.description}
                  </p>
                </div>
              </Reveal>
            );
          },
        )}
      </ul>

      <Reveal className="mt-8">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          {Object.values(CLUSTER_META).map((m) => (
            <span key={m.label} className="flex items-center gap-2 font-mono text-[11px] text-ink-3">
              <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: m.color }} />
              {m.label}
            </span>
          ))}
        </div>
      </Reveal>

      <Reveal className="mt-10">
        <div className="card border-l-2 border-l-s-amber px-6 py-5">
          <p className="text-sm leading-relaxed text-ink-2">
            <span className="font-display font-bold text-ink">Result:</span>{" "}
            we submitted the two best — the finance bot and a coordinated-strike
            interceptor — and landed at Elo ~578 and ~532, mid-pack among 4,400+
            teams. Respectable for hand-rolled science. Then we looked at what the
            top of the leaderboard was running, and everything changed.
          </p>
        </div>
      </Reveal>
    </Section>
  );
}
