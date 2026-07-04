import type { ReactNode } from "react";
import Reveal from "./Reveal";

export default function Section({
  id,
  kicker,
  title,
  accent = ".",
  lede,
  children,
  wide = false,
}: {
  id: string;
  kicker: string;
  title: string;
  accent?: string;
  lede?: ReactNode;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <section id={id} className="relative scroll-mt-20 px-5 py-24 sm:px-8 md:py-32">
      <div className={`mx-auto ${wide ? "max-w-7xl" : "max-w-5xl"}`}>
        <Reveal>
          <p className="kicker mb-4">{kicker}</p>
          <h2 className="font-display text-4xl font-bold tracking-tight text-ink sm:text-5xl">
            {title}
            <span className="text-s-blue">{accent}</span>
          </h2>
          {lede && (
            <div className="mt-6 max-w-3xl text-base leading-relaxed text-ink-2 sm:text-lg">
              {lede}
            </div>
          )}
        </Reveal>
        <div className="mt-12">{children}</div>
      </div>
    </section>
  );
}
