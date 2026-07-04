export function LessonsSection() {
  const lessons = [
    {
      n: "01",
      tag: "Strategy",
      tagColor: "#3b82f6",
      title: "Clone the winner before inventing your own",
      body: "23 original scientific bots placed ~500th. A clone of the #1 engine placed #144. Reverse engineering beats invention when the environment is already well-optimized.",
      callout: "23 bots → #500. 1 clone → #144.",
    },
    {
      n: "02",
      tag: "Eval Methodology",
      tagColor: "#f59e0b",
      title: "The gym vs. live gap is real — and direction-reversing",
      body: "schmeekler won 72% of local evals but underperformed initially on the live leaderboard. Local win rate against a narrow panel of 4–5 bots ≠ Elo vs 500 diverse agents. Your gym opponent panel is your biggest confound.",
      callout: "72% local → underperformed live (first week)",
    },
    {
      n: "03",
      tag: "Experiment Design",
      tagColor: "#10b981",
      title: "Pre-commit failure conditions before running",
      body: "Setting 'if greedy win rate < 25% at U=500, shut down' before the RL run made the closure decision clean and unchallengeable. No sunk-cost rationalization. No 'let's run just 1000 more steps.'",
      callout: "Pre-committed gates → clean kills",
    },
    {
      n: "04",
      tag: "Meta-AI",
      tagColor: "#a855f7",
      title: "An independent auditor catches what the orchestrator rationalizes",
      body: "The auditor subagent caught a completely collapsed training run (CF=0/50) that the orchestrator had missed while focused on the fleet average. Two autonomous agents with different briefs and briefs see different things.",
      callout: "2 agents > 1 agent",
    },
    {
      n: "05",
      tag: "Architecture",
      tagColor: "#ef4444",
      title: "Candidate scarcity is the real constraint",
      body: "orbit_lite produces 0–4 action candidates per turn via its capture-floor filter. No re-ranking algorithm — 2-ply search, value function, EV correction — can improve what it can't see. Fix the architecture, not the scorer.",
      callout: "0–4 candidates/turn = ceiling, not scorer",
    },
    {
      n: "06",
      tag: "Infrastructure",
      tagColor: "#f97316",
      title: "The bottleneck was throughput, not algorithm",
      body: "The #1 bot trained 600M PPO steps with JAX on TPUs. We had ~11M steps available in 4 days on 9 CPU instances. The only path to RL competitive play was faster infrastructure, not smarter training code.",
      callout: "600M JAX vs 11M CPU — gap was infrastructure",
    },
    {
      n: "07",
      tag: "Tooling",
      tagColor: "#06b6d4",
      title: "Decide your observability stack on day one — never let the LLM improvise it",
      body: "We ended up with three parallel visualization systems: tmux monitoring boards, a custom Streamlit dashboard that grew into a mess, and ad-hoc tables generated inline each session. The root cause: the LLM is stateless between sessions and reaches for whatever visualization feels natural in the moment. Tensorboard + CleanRL's validated PPO implementation would have given real-time loss curves, multi-run seed comparison, and standardized logging in ~3 lines — for free. Instead we reimplemented PPO from scratch (reintroducing bugs CleanRL already fixed), rebuilt dashboards repeatedly, and manually diffed seed metrics that Tensorboard would have overlaid automatically. The fix: lock the visualization tool before writing a single training line, and explicitly tell the LLM 'all metrics go to Tensorboard — never generate inline tables.'",
      callout: "3 viz systems rebuilt vs tensorboard --logdir runs/",
    },
    {
      n: "08",
      tag: "Architecture",
      tagColor: "#8b5cf6",
      title: "Validate with standard tools first — go custom only when you've proven they're the bottleneck",
      body: "We built a custom JAX game engine early to get 1024-env GPU throughput — and that made CleanRL and Tensorboard incompatible, because Gym's one-env-at-a-time interface and JAX's compile-everything-into-one-kernel design are fundamentally opposed. The engine was fast, but we built it before knowing throughput was actually the constraint. The right sequence: wrap kaggle-environments in a Gym interface first, use CleanRL's validated PPO, prove the architecture works at CPU speed. If and only if CPU throughput is provably the ceiling — then build the custom JAX engine, knowing you're intentionally trading away standard tooling for speed. Custom infrastructure is justified by evidence, not by anticipation.",
      callout: "Prove standard tools are the bottleneck before abandoning them",
    },
  ];

  return (
    <section id="lessons" className="section">
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-xs font-medium uppercase tracking-wide mb-4">
          What We Learned
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ fontFamily: "var(--font-space)" }}>
          Lessons Learned
        </h2>
        <p className="text-slate-400 max-w-xl mx-auto text-sm leading-relaxed">
          Eight things that were genuinely surprising — not obvious from the competition page, not obvious from the code.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-5">
        {lessons.map(({ n, tag, tagColor, title, body, callout }) => (
          <div key={n} className="glass p-6 flex flex-col gap-4 group hover:border-white/15 transition-colors">
            <div className="flex items-start justify-between gap-3">
              <span
                className="text-5xl font-bold leading-none select-none"
                style={{ color: "#0f172a", fontFamily: "var(--font-space)" }}
              >
                {n}
              </span>
              <span
                className="px-2.5 py-0.5 rounded-full text-xs font-medium border shrink-0 mt-1"
                style={{ color: tagColor, borderColor: `${tagColor}40`, background: `${tagColor}10` }}
              >
                {tag}
              </span>
            </div>
            <div>
              <h3 className="font-semibold text-white text-base mb-2 leading-snug">{title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{body}</p>
            </div>
            <div
              className="mt-auto rounded-lg px-3 py-2 text-xs font-mono"
              style={{ color: tagColor, background: `${tagColor}10`, border: `1px solid ${tagColor}20` }}
            >
              {callout}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
