import Section from "../ui/Section";
import Reveal from "../ui/Reveal";

const LESSONS = [
  {
    title: "If a fast search exists, run the search — don't clone it.",
    body: "Behavior cloning the #1 player produced a bot that lost 0–16 to our own engine. The engine already out-executes every human mechanically; imitating their moves without their macro strategy gives you the cursor without the hand. The competition forum independently confirmed the same result.",
    tag: "imitation",
  },
  {
    title: "The gym is not the ladder.",
    body: "schmeekler beat our champion 72% in seat-swapped local evals and swept the public panel — then underperformed it live. Pairwise win rate against a narrow, strong panel is not a rating against a 4,700-team diverse field. Always seat-swap; always check the live ladder before crowning a champion.",
    tag: "evaluation",
  },
  {
    title: "You can't optimize a decision that isn't being made.",
    body: "Profiling showed the engine's safety filters left zero or one candidate move on 111 of 133 decision turns. Every clever re-ranking bolt-on landed at parity because there was nothing to re-rank. Measure where the decisions actually are before optimizing them.",
    tag: "profiling",
  },
  {
    title: "In RL, the action space is the strategy.",
    body: "PPO over raw ship commands faces a 498-step sparse-reward horizon (γ⁴⁹⁸ ≈ 0.007). PPO over engine strategy-knobs faces ~30 macro-decisions (γ³⁰ ≈ 0.74). The winners chose the second. We spent five campaigns and two A100s learning why that choice matters more than compute.",
    tag: "reinforcement learning",
  },
  {
    title: "Self-play without a strong opponent teaches you to beat nobody.",
    body: "Every self-play fleet converged to the same 23.3% ceiling versus a greedy baseline and 0% versus the engine. A snapshot pool seeded from random weights only ever learns to beat random play. Strong opponents must be in the training signal from step one — and at more than 0.4% of the batch.",
    tag: "training signal",
  },
  {
    title: "Pre-commit the kill criteria, log the dead ends.",
    body: "Every experiment declared its failure condition before launch, met a fixed evaluator, and got a KEEP or DISCARD on the record — 17 discards, 2 keeps, all still visible in the timeline. The ratchet means the champion only ever gets stronger, and no failed idea gets re-run by a forgetful future session.",
    tag: "methodology",
  },
  {
    title: "An AI can run the lab.",
    body: "The improvement loop itself was operated by Claude agent sessions: an orchestrator reading the live leaderboard and experiment state, worker tracks running the gauntlets, every milestone appended to a provenance timeline that the next session picks up cold. Turtles all the way down — and the audit trail survived all of them.",
    tag: "autoresearch",
  },
] as const;

export default function LessonsSection() {
  return (
    <Section
      id="lessons"
      kicker="the debrief"
      title="seven lessons, paid in full"
      lede={
        <>
          Ten days, ~$150 of compute, 80,000+ local games, and five closed RL
          campaigns bought these. They&apos;re written the expensive way — each
          one is a thing we believed that the data refused.
        </>
      }
    >
      <ol className="grid gap-4">
        {LESSONS.map((l, i) => (
          <Reveal as="li" key={l.title} delay={(i % 3) * 60}>
            <div className="card card-hover flex gap-5 px-6 py-5 sm:gap-7 sm:px-8">
              <span
                aria-hidden
                className="font-display text-4xl font-bold leading-none text-s-blue/35 sm:text-5xl"
              >
                {String(i + 1).padStart(2, "0")}
              </span>
              <div>
                <div className="flex flex-wrap items-center gap-3">
                  <h3 className="font-display text-lg font-bold leading-snug text-ink">
                    {l.title}
                  </h3>
                  <span className="rounded-full bg-white/[0.05] px-2.5 py-0.5 font-mono text-[10px] tracking-widest text-ink-3">
                    {l.tag}
                  </span>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-ink-2">{l.body}</p>
              </div>
            </div>
          </Reveal>
        ))}
      </ol>
    </Section>
  );
}
