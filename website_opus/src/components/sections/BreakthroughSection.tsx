export function BreakthroughSection() {
  const sweepData = [
    { bonus: 0.0,  win: 66, label: "baseline" },
    { bonus: 0.5,  win: 68, label: "" },
    { bonus: 1.0,  win: 72, label: "sweet spot" },
    { bonus: 1.5,  win: 74, label: "✨ submitted" },
    { bonus: 2.0,  win: 61, label: "over-commits" },
    { bonus: 3.0,  win: 52, label: "tanks" },
  ];

  const W = 460, H = 200;
  const PAD = { l: 40, r: 20, t: 20, b: 36 };
  const pw = W - PAD.l - PAD.r;
  const ph = H - PAD.t - PAD.b;
  const xStep = pw / (sweepData.length - 1);
  const yMin = 48, yMax = 80;
  const toX = (i: number) => PAD.l + i * xStep;
  const toY = (v: number) => PAD.t + ph - ((v - yMin) / (yMax - yMin)) * ph;

  const path = sweepData.map((d, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(d.win)}`).join(" ");
  const area = `${path} L${toX(sweepData.length-1)},${H - PAD.b} L${PAD.l},${H - PAD.b} Z`;

  return (
    <section id="breakthrough" className="section">
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-xs font-medium uppercase tracking-wide mb-4">
          The Breakthrough
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          Finding the Engine&apos;s Blind Spot
        </h2>
      </div>

      <div className="grid md:grid-cols-2 gap-10 items-start">
        <div className="space-y-6">
          <p className="text-slate-400 leading-relaxed">
            After reverse-engineering the #1 bot&apos;s engine, we cloned it exactly. Our{" "}
            <code className="text-amber-400">comet_reaper</code> tied the original Producer
            bot 14–14 and beat ~67% of the public field. But we were stuck at ~1240 Elo — the engine
            was a tight optimum that 19 experiments couldn&apos;t crack.
          </p>
          <p className="text-slate-400 leading-relaxed">
            The insight came from watching games in-browser: <strong className="text-white">static
            planets</strong> (planets that don&apos;t orbit, sitting at fixed coordinates) are the
            safest early-game targets. They can&apos;t drift into enemy reach. The engine treated them
            identically to rotating planets — it didn&apos;t know they were safer.
          </p>
          <p className="text-slate-400 leading-relaxed">
            One parameter. A flat additive bonus for targeting static planets.
            Swept values from 0→3. At <code className="text-emerald-400">static_target_bonus=1.5</code>:
          </p>

          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "vs comet_reaper",   value: "72%",  c: "#10b981" },
              { label: "vs Producer",        value: "77%",  c: "#10b981" },
              { label: "Peak rank",          value: "#144", c: "#3b82f6" },
            ].map(({ label, value, c }) => (
              <div key={label} className="glass p-3 text-center">
                <div className="text-2xl font-bold" style={{ color: c, fontFamily: "var(--font-space)" }}>{value}</div>
                <div className="text-xs text-slate-500 mt-1">{label}</div>
              </div>
            ))}
          </div>

          {/* Code snippet */}
          <div className="glass rounded-lg overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-800 bg-slate-900/50">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
              <span className="text-xs text-slate-500 ml-2 font-mono">schmeekler/main.py</span>
            </div>
            <pre className="p-4 text-xs overflow-x-auto" style={{ fontFamily: "var(--font-mono)" }}>
{`# The one change that beat everything
STATIC_TARGET_BONUS = 1.5  # additive score bonus

def score_planet(planet, obs):
    base_score = orbit_lite.score(planet, obs)

    # Static planets don't orbit — safe early grabs
    if planet.angular_velocity == 0:
        base_score += STATIC_TARGET_BONUS

    return base_score`}
            </pre>
          </div>
        </div>

        {/* Win rate sweep chart */}
        <div className="glass p-5">
          <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500 mb-4">
            Win Rate Sweep — static_target_bonus
          </h3>
          <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
            {/* Grid lines */}
            {[50,60,70,80].map(v => (
              <g key={v}>
                <line x1={PAD.l} y1={toY(v)} x2={W - PAD.r} y2={toY(v)} stroke="#1e293b" />
                <text x={PAD.l - 6} y={toY(v) + 4} textAnchor="end" fontSize={9} fill="#475569">{v}%</text>
              </g>
            ))}
            {/* Area */}
            <path d={area} fill="#10b981" opacity={0.08} />
            {/* Line */}
            <path d={path} fill="none" stroke="#10b981" strokeWidth={2} />
            {/* Points */}
            {sweepData.map((d, i) => {
              const isSubmitted = d.bonus === 1.5;
              return (
                <g key={d.bonus}>
                  <circle
                    cx={toX(i)} cy={toY(d.win)} r={isSubmitted ? 7 : 4}
                    fill={isSubmitted ? "#10b981" : "#0f172a"}
                    stroke={isSubmitted ? "white" : "#10b981"}
                    strokeWidth={1.5}
                  />
                  <text x={toX(i)} y={H - PAD.b + 14} textAnchor="middle" fontSize={9} fill="#475569">
                    {d.bonus}
                  </text>
                  {d.label && (
                    <text x={toX(i)} y={toY(d.win) - 11} textAnchor="middle" fontSize={8}
                      fill={isSubmitted ? "#10b981" : "#64748b"} fontWeight={isSubmitted ? "bold" : "normal"}>
                      {d.label}
                    </text>
                  )}
                </g>
              );
            })}
            <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="#475569">
              static_target_bonus value
            </text>
            <text x={18} y={H / 2} textAnchor="middle" fontSize={9} fill="#475569" transform={`rotate(-90, 14, ${H/2})`}>
              2P win %
            </text>
          </svg>

          <div className="mt-4 space-y-2 text-xs text-slate-500">
            <div className="flex items-start gap-2">
              <span className="text-emerald-400 font-bold mt-0.5">1.0–1.5</span>
              <span>Sweet spot — captures safe territory early without over-committing</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-rose-400 font-bold mt-0.5">≥ 2.0</span>
              <span>Over-commits — bot chases static planets even when enemy is attacking</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
