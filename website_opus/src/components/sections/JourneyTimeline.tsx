import timeline from "../../../public/data/timeline.json";

const STATUS_CONFIG = {
  submitted:      { color: "#3b82f6",  icon: "🚀", label: "Submitted"      },
  dead_end:       { color: "#ef4444",  icon: "💀", label: "Dead End"       },
  breakthrough:   { color: "#10b981",  icon: "✨", label: "Breakthrough"   },
  infrastructure: { color: "#64748b",  icon: "🔧", label: "Infrastructure" },
  closed:         { color: "#a855f7",  icon: "🔒", label: "Closed"        },
  ongoing:        { color: "#f59e0b",  icon: "⚡", label: "Ongoing"        },
};

// Only show the "top-level" phases, not every sub-entry
const MAIN_PHASES = [
  "Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4",
  "Phase 5", "Phase 6", "2026-06-18", "2026-06-19",
];

export function JourneyTimeline() {
  const phases = (timeline as Array<{
    phase: string; date: string; title: string;
    headline: string; body_md: string; status: string;
  }>).filter(p =>
    MAIN_PHASES.some(m => p.phase.startsWith(m) || p.title.startsWith(m) || p.phase === m)
  ).slice(0, 12);

  return (
    <section id="journey" className="section">
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-xs font-medium uppercase tracking-wide mb-4">
          The Journey
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          7 Days, Step by Step
        </h2>
        <p className="text-slate-400 max-w-xl mx-auto">
          Every pivot, every dead end, every late-night breakthrough — documented in order.
        </p>
      </div>

      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-6 md:left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-blue-500/50 via-slate-700/50 to-transparent" />

        <div className="space-y-8">
          {phases.map((phase, i) => {
            const cfg = STATUS_CONFIG[phase.status as keyof typeof STATUS_CONFIG] ?? STATUS_CONFIG.infrastructure;
            const isRight = i % 2 === 1;

            return (
              <div
                key={i}
                className={`relative flex gap-4 md:gap-0 ${isRight ? "md:flex-row-reverse" : "md:flex-row"}`}
              >
                {/* Mobile: left dot */}
                <div
                  className="relative z-10 flex-shrink-0 flex md:hidden w-12 h-12 items-center justify-center rounded-full border-2 text-lg"
                  style={{ borderColor: cfg.color, background: `${cfg.color}15` }}
                >
                  {cfg.icon}
                </div>

                {/* Desktop: center dot */}
                <div className="hidden md:flex absolute left-1/2 -translate-x-1/2 z-10 w-10 h-10 items-center justify-center rounded-full border-2 text-base top-3"
                  style={{ borderColor: cfg.color, background: "#08080f" }}>
                  {cfg.icon}
                </div>

                {/* Card */}
                <div className={`glass p-5 flex-1 md:w-5/12 md:max-w-[45%] ${isRight ? "md:mr-[55%]" : "md:ml-[55%]"}`}>
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <span
                      className="text-xs font-bold px-2 py-0.5 rounded-full"
                      style={{ color: cfg.color, background: `${cfg.color}20` }}
                    >
                      {cfg.label}
                    </span>
                    {phase.date && (
                      <span className="text-xs text-slate-600 font-mono">{phase.date}</span>
                    )}
                  </div>
                  <h3 className="font-semibold text-white mb-1 text-sm leading-snug">
                    {phase.title.length > 80 ? phase.title.slice(0, 80) + "…" : phase.title}
                  </h3>
                  {phase.headline && (
                    <p className="text-xs text-slate-500 leading-relaxed">{phase.headline}</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
