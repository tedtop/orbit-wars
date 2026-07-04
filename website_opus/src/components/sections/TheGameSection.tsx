export function TheGameSection() {
  return (
    <section id="game" className="section">
      <div className="grid md:grid-cols-2 gap-12 items-center">
        {/* Left: explanation */}
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400 text-xs font-medium uppercase tracking-wide mb-4">
            The Game
          </div>
          <h2 className="text-4xl font-bold mb-6" style={{ fontFamily: "var(--font-space)" }}>
            What is Orbit Wars?
          </h2>
          <p className="text-slate-400 leading-relaxed mb-6">
            A real-time strategy game where planets orbit a central sun. You control a fleet of ships
            spread across your planets. Each turn, you can send ships from any planet to attack or
            reinforce others. Planets you own produce ships each turn — capture more to build a
            bigger fleet. Last team standing wins.
          </p>
          <p className="text-slate-400 leading-relaxed mb-6">
            The twist: because planets <em>orbit</em>, timing matters enormously. A planet that&apos;s
            capturable now might drift out of reach in 5 turns. Comets appear periodically,
            offering bonus production for whoever captures them first. Games run 2-player or 4-player.
          </p>

          <div className="grid grid-cols-2 gap-3">
            {[
              { icon: "🪐", title: "Orbiting planets",    body: "Every planet moves. Target prediction is essential." },
              { icon: "🚀", title: "Fleet dispatch",       body: "Send ships from any owned planet to any target." },
              { icon: "⚡",  title: "Production economy",  body: "More planets → more ships/turn → compounding advantage." },
              { icon: "☄️", title: "Comet events",         body: "Temporary high-production planets that orbit past." },
            ].map(({ icon, title, body }) => (
              <div key={title} className="glass p-4">
                <div className="text-xl mb-2">{icon}</div>
                <div className="text-sm font-semibold text-white mb-1">{title}</div>
                <div className="text-xs text-slate-500 leading-relaxed">{body}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: links + competition info */}
        <div className="space-y-4">
          <div className="glass p-6">
            <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500 mb-4">Competition</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-500">Platform</span>
                <span className="text-white font-medium">Kaggle Simulations</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Duration</span>
                <span className="text-white font-medium">May – July 2026</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Participants</span>
                <span className="text-white font-medium">~500+ teams</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Game type</span>
                <span className="text-white font-medium">2P &amp; 4P, Elo-rated</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Languages</span>
                <span className="text-white font-medium">Any (Python)</span>
              </div>
            </div>
          </div>

          {/* Orbit Wars mini-diagram */}
          <div className="glass p-6 flex flex-col items-center gap-4">
            <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500 self-start">Game State (Schematic)</h3>
            <svg width="220" height="220" viewBox="-110 -110 220 220" className="text-slate-700">
              {/* Sun */}
              <circle cx="0" cy="0" r="10" fill="#fbbf24" opacity="0.9" />
              <circle cx="0" cy="0" r="14" fill="none" stroke="#fbbf24" strokeWidth="0.5" opacity="0.4" />
              {/* Orbits */}
              <circle cx="0" cy="0" r="45" fill="none" stroke="currentColor" strokeWidth="0.5" strokeDasharray="3 3" />
              <circle cx="0" cy="0" r="70" fill="none" stroke="currentColor" strokeWidth="0.5" strokeDasharray="3 3" />
              <circle cx="0" cy="0" r="95" fill="none" stroke="currentColor" strokeWidth="0.5" strokeDasharray="3 3" />
              {/* Player 1 planets (blue) */}
              <circle cx="0" cy="-45" r="8" fill="#3b82f6" opacity="0.9" />
              <text x="0" y="-45" textAnchor="middle" dominantBaseline="middle" fill="white" fontSize="7" fontWeight="bold">6</text>
              <circle cx="45" cy="0" r="6" fill="#3b82f6" opacity="0.7" />
              <text x="45" y="0" textAnchor="middle" dominantBaseline="middle" fill="white" fontSize="6">3</text>
              {/* Player 2 planets (red) */}
              <circle cx="0" cy="70" r="9" fill="#ef4444" opacity="0.9" />
              <text x="0" y="70" textAnchor="middle" dominantBaseline="middle" fill="white" fontSize="7" fontWeight="bold">8</text>
              <circle cx="-60" cy="-35" r="7" fill="#ef4444" opacity="0.7" />
              <text x="-60" y="-35" textAnchor="middle" dominantBaseline="middle" fill="white" fontSize="6">4</text>
              {/* Neutral */}
              <circle cx="85" cy="40" r="6" fill="#64748b" opacity="0.8" />
              <text x="85" y="40" textAnchor="middle" dominantBaseline="middle" fill="white" fontSize="5">2</text>
              <circle cx="-90" cy="15" r="5" fill="#64748b" opacity="0.6" />
              {/* Fleet in transit */}
              <line x1="0" y1="-45" x2="85" y2="40" stroke="#3b82f6" strokeWidth="0.8" strokeDasharray="4 3" opacity="0.6" />
              <polygon points="78,35 88,33 80,45" fill="#3b82f6" opacity="0.8" />
            </svg>
            <p className="text-xs text-slate-600 text-center">Numbers = garrison strength. Dashed line = fleet in transit.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
