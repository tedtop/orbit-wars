import { StarfieldBg }               from "@/components/ui/StarfieldBg";
import { HeroSection }              from "@/components/sections/HeroSection";
import { ScientistsSection }        from "@/components/sections/ScientistsSection";
import { TheGameSection }           from "@/components/sections/TheGameSection";
import { JourneyTimeline }          from "@/components/sections/JourneyTimeline";
import { ScoreProgressionSection }  from "@/components/sections/ScoreProgressionSection";
import { ExperimentLabSection }     from "@/components/sections/ExperimentLabSection";
import { BreakthroughSection }      from "@/components/sections/BreakthroughSection";
import { RLSection }                from "@/components/sections/RLSection";
import { ReplaySimulatorSection }   from "@/components/sections/ReplaySimulatorSection";
import { AutonomousSection }        from "@/components/sections/AutonomousSection";
import { LessonsSection }           from "@/components/sections/LessonsSection";
import { FinalSection }             from "@/components/sections/FinalSection";

export default function Home() {
  return (
    <main>
      <StarfieldBg />

      {/* Sticky nav */}
      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 bg-[#08080f]/80 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <a href="#" className="text-sm font-bold text-white flex items-center gap-2" style={{ fontFamily: "var(--font-space)" }}>
            <span className="text-amber-400">⬡</span> Orbit Wars
          </a>
          <div className="hidden md:flex items-center gap-6 text-xs text-slate-500">
            <a href="#scientists"  className="hover:text-slate-200 transition-colors">23 Scientists</a>
            <a href="#score"       className="hover:text-slate-200 transition-colors">Score Chart</a>
            <a href="#experiments" className="hover:text-slate-200 transition-colors">Experiments</a>
            <a href="#replays"     className="hover:text-slate-200 transition-colors">Replays</a>
            <a href="#lessons"     className="hover:text-slate-200 transition-colors">Lessons</a>
            <a href="#final"       className="hover:text-slate-200 transition-colors">Final</a>
          </div>
        </div>
      </nav>

      <div className="pt-14">
        <HeroSection />

        <div className="max-w-[1200px] mx-auto">
          <ScientistsSection />
          <Divider />
          <TheGameSection />
          <Divider />
          <JourneyTimeline />
          <Divider />
          <ScoreProgressionSection />
          <Divider />
          <ExperimentLabSection />
          <Divider />
          <BreakthroughSection />
          <Divider />
          <RLSection />
          <Divider />
          <ReplaySimulatorSection />
          <Divider />
          <AutonomousSection />
          <Divider />
          <LessonsSection />
          <Divider />
          <FinalSection />
        </div>
      </div>
    </main>
  );
}

function Divider() {
  return (
    <div className="px-6">
      <div className="h-px bg-gradient-to-r from-transparent via-slate-800 to-transparent" />
    </div>
  );
}
