import Section from "../ui/Section";
import Reveal from "../ui/Reveal";

const RULES = [
  ["100×100", "continuous map with a lethal sun at the center — fleets that fly through it are destroyed"],
  ["500", "turns per game; whoever holds the most ships at the end wins"],
  ["2P & 4P", "matches — half your games are four-player pods with three rivals ganging and grinding"],
  ["6 u/turn", "fleet speed. You launch [from_planet, angle, ships] — then physics takes over. No steering."],
  ["t=50…450", "comets streak through on a fixed schedule: temporary high-value planets on rails"],
  ["1 s/turn", "compute budget per move — roughly 30× more than the top bots actually use"],
] as const;

/** Static-with-motion SVG schematic of the board. */
function BoardDiagram() {
  return (
    <svg
      viewBox="0 0 300 300"
      role="img"
      aria-label="Schematic of the Orbit Wars board: planets orbit a central sun, fleets travel between them"
      className="w-full max-w-md"
    >
      {/* orbit rings */}
      {[52, 86, 120].map((r) => (
        <circle key={r} cx="150" cy="150" r={r} fill="none" stroke="var(--grid)" strokeWidth="1" />
      ))}
      {/* sun */}
      <circle cx="150" cy="150" r="17" fill="url(#sunGrad)" />
      <circle cx="150" cy="150" r="26" fill="none" stroke="var(--sun-glow)" strokeOpacity="0.25" strokeDasharray="2 4" />
      <defs>
        <radialGradient id="sunGrad">
          <stop offset="0%" stopColor="#fff3da" />
          <stop offset="60%" stopColor="#ffd9a0" />
          <stop offset="100%" stopColor="#ff9d3c" />
        </radialGradient>
      </defs>
      {/* orbiting planets — CSS animation groups */}
      <g className="orbit-slow" style={{ transformOrigin: "150px 150px" }}>
        <circle cx="150" cy="64" r="7" fill="var(--s-blue)" />
        <text x="150" y="48" textAnchor="middle" className="tick-label">ours</text>
      </g>
      <g className="orbit-med" style={{ transformOrigin: "150px 150px" }}>
        <circle cx="236" cy="150" r="6" fill="var(--s-red)" />
        <text x="236" y="133" textAnchor="middle" className="tick-label">enemy</text>
      </g>
      <g className="orbit-fast" style={{ transformOrigin: "150px 150px" }}>
        <circle cx="150" cy="270" r="5" fill="var(--ink-3)" />
        <text x="150" y="288" textAnchor="middle" className="tick-label">neutral</text>
      </g>
      {/* static planet */}
      <circle cx="34" cy="44" r="6" fill="var(--s-green)" />
      <text x="34" y="28" textAnchor="middle" className="tick-label">static</text>
      {/* comet path */}
      <path d="M 292 30 Q 210 96 150 150" fill="none" stroke="var(--s-amber)" strokeOpacity="0.55" strokeWidth="1.4" strokeDasharray="3 5" />
      <circle cx="255" cy="57" r="4" fill="var(--s-amber)" />
      <text x="262" y="44" className="tick-label">comet</text>
      {/* fleet in transit */}
      <path d="M 60 220 L 118 178" fill="none" stroke="var(--s-blue)" strokeWidth="1.6" markerEnd="url(#arrow)" />
      <defs>
        <marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill="var(--s-blue)" />
        </marker>
      </defs>
      <text x="66" y="240" className="tick-label">fleet · no steering</text>
    </svg>
  );
}

export default function GameSection() {
  return (
    <Section
      id="game"
      kicker="mission briefing · 01"
      title="the game"
      lede={
        <>
          <a
            href="https://www.kaggle.com/competitions/orbit-wars/overview"
            target="_blank"
            rel="noopener noreferrer"
            className="text-s-blue underline decoration-s-blue/40 underline-offset-4 hover:decoration-s-blue"
          >
            Orbit Wars
          </a>{" "}
          is a Kaggle simulation competition: you don&apos;t play — you write a bot
          that plays for you, around the clock, against thousands of other teams&apos;
          bots on a live Elo ladder. Top 10 teams split $50,000.
        </>
      }
    >
      <div className="grid items-center gap-10 md:grid-cols-2">
        <Reveal className="flex justify-center">
          <div className="card p-6">
            <BoardDiagram />
          </div>
        </Reveal>
        <div className="grid gap-3">
          {RULES.map(([num, text], i) => (
            <Reveal key={num} delay={i * 60}>
              <div className="card card-hover flex items-baseline gap-4 px-5 py-4">
                <span className="w-20 shrink-0 text-right font-mono text-sm font-bold text-s-blue">
                  {num}
                </span>
                <span className="text-sm leading-relaxed text-ink-2">{text}</span>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </Section>
  );
}
