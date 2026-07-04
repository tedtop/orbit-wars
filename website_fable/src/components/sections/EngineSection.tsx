import Section from "../ui/Section";
import Reveal from "../ui/Reveal";

function LineageDiagram() {
  const node =
    "card px-4 py-3 text-center font-mono text-xs text-ink transition-colors";
  return (
    <div className="flex flex-col items-center gap-2">
      <div className={node}>
        <span className="block text-[10px] uppercase tracking-widest text-ink-3">
          #1 public bot
        </span>
        the-producer
      </div>
      <span aria-hidden className="font-mono text-ink-3">│ runs on</span>
      <div className={`${node} outline outline-1 outline-s-violet/50`}>
        <span className="block text-[10px] uppercase tracking-widest text-s-violet">
          the engine
        </span>
        orbit_lite
      </div>
      <span aria-hidden className="font-mono text-ink-3">│ we cloned it</span>
      <div className={`${node} outline outline-1 outline-s-blue/60`}>
        <span className="block text-[10px] uppercase tracking-widest text-s-blue">
          our champion
        </span>
        comet_reaper
      </div>
      <span aria-hidden className="font-mono text-ink-3">│ 19 forks later…</span>
      <div className="flex flex-wrap justify-center gap-2">
        {["precog", "kingmaker", "maestro", "helmsman", "oracle", "schmeekler ★"].map((f) => (
          <div
            key={f}
            className={`card px-3 py-1.5 font-mono text-[11px] ${
              f.startsWith("schmeekler") ? "text-s-green outline outline-1 outline-s-green/50" : "text-ink-3"
            }`}
          >
            {f}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function EngineSection() {
  return (
    <Section
      id="engine"
      kicker="phase 3 · jun 14–15"
      title="reverse-engineering the summit"
      lede={
        <>
          Leaderboard archaeology revealed something humbling: the best public
          bots weren&apos;t 23 different ideas — they were{" "}
          <strong className="text-ink">one lineage</strong>. The #1 public bot,
          The Producer, runs on <code className="font-mono text-s-violet">orbit_lite</code>:
          a planning engine that simulates 18 turns of the future — every fleet
          in flight, every planet&apos;s garrison at every future tick — and
          scores each candidate attack by how much it shifts the whole board in
          your favor. We cloned the approach, named ours{" "}
          <code className="font-mono text-s-blue">comet_reaper</code>, and tied
          the best public bot within a day.
        </>
      }
    >
      <div className="grid items-start gap-10 md:grid-cols-2">
        <Reveal>
          <LineageDiagram />
        </Reveal>
        <div className="grid gap-3">
          {(
            [
              ["14–14", "comet_reaper vs The Producer, head-to-head — a dead tie with the best public bot"],
              ["~67%", "win rate against the rest of the public field"],
              ["#144", "peak leaderboard rank at Elo 1243.8 — first time inside the top 150"],
              ["0.34", "best Optuna score across 37 config-tuning trials: the base config is a tight local optimum. The knobs were already set."],
            ] as const
          ).map(([num, text], i) => (
            <Reveal key={num} delay={i * 60}>
              <div className="card card-hover flex items-baseline gap-4 px-5 py-4">
                <span className="w-20 shrink-0 text-right font-mono text-sm font-bold text-s-violet">
                  {num}
                </span>
                <span className="text-sm leading-relaxed text-ink-2">{text}</span>
              </div>
            </Reveal>
          ))}
          <Reveal delay={260}>
            <div className="card border-l-2 border-l-s-green px-5 py-4">
              <p className="text-sm leading-relaxed text-ink-2">
                <span className="font-display font-bold text-s-green">
                  The one fork that ever beat it —{" "}
                  <code className="font-mono">schmeekler</code>.
                </span>{" "}
                Our one genuinely novel strategy, not borrowed from anyone&apos;s
                leaderboard bot: capture the <em>static</em> planets first.
                They don&apos;t rotate, so they hold the safe periphery forever and
                can never drift into enemy reach — a land-grab the engine
                systematically undervalued. It beat comet_reaper 72% seat-swapped
                and swept the whole public panel in the gym. The live ladder had
                other opinions — that story comes later.
              </p>
            </div>
          </Reveal>
        </div>
      </div>
    </Section>
  );
}
