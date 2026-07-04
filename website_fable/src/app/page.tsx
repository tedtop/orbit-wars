import Nav from "@/components/ui/Nav";
import Starfield from "@/components/ui/Starfield";
import HeroSection from "@/components/sections/HeroSection";
import GameSection from "@/components/sections/GameSection";
import ScientistsSection from "@/components/sections/ScientistsSection";
import EngineSection from "@/components/sections/EngineSection";
import ClimbSection from "@/components/sections/ClimbSection";
import LabSection from "@/components/sections/LabSection";
import RLSection from "@/components/sections/RLSection";
import TheaterSection from "@/components/sections/TheaterSection";
import LessonsSection from "@/components/sections/LessonsSection";
import FinaleSection from "@/components/sections/FinaleSection";

export default function Home() {
  return (
    <main>
      <Starfield />
      <Nav />
      <HeroSection />
      <GameSection />
      <ScientistsSection />
      <EngineSection />
      <ClimbSection />
      <LabSection />
      <RLSection />
      <TheaterSection />
      <LessonsSection />
      <FinaleSection />
    </main>
  );
}
