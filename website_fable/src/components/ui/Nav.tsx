"use client";

import { useEffect, useState } from "react";

const LINKS = [
  ["game", "the game"],
  ["scientists", "23 bots"],
  ["engine", "the engine"],
  ["climb", "the climb"],
  ["lab", "the lab"],
  ["rl", "the moonshot"],
  ["theater", "replays"],
  ["lessons", "lessons"],
  ["log", "mission log"],
  ["finale", "finale"],
] as const;

export default function Nav() {
  const [active, setActive] = useState("");
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > window.innerHeight * 0.6);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) setActive(e.target.id);
        }
      },
      { rootMargin: "-30% 0px -60% 0px" },
    );
    for (const [id] of LINKS) {
      const el = document.getElementById(id);
      if (el) io.observe(el);
    }
    return () => {
      window.removeEventListener("scroll", onScroll);
      io.disconnect();
    };
  }, []);

  return (
    <nav
      aria-label="Sections"
      className={`fixed inset-x-0 top-0 z-50 transition-all duration-500 ${
        scrolled
          ? "translate-y-0 opacity-100"
          : "-translate-y-full opacity-0"
      }`}
    >
      <div className="border-b border-white/[0.06] bg-[rgba(5,6,13,0.82)] backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center gap-1 overflow-x-auto px-4 py-2.5 [scrollbar-width:none]">
          <a
            href="#top"
            className="mr-3 shrink-0 font-display text-sm font-bold tracking-tight text-ink"
          >
            orbit wars<span className="text-s-blue">.</span>
          </a>
          {LINKS.map(([id, label]) => (
            <a
              key={id}
              href={`#${id}`}
              className={`shrink-0 rounded-full px-3 py-1 font-mono text-[11px] tracking-wide transition-colors ${
                active === id
                  ? "bg-s-blue/15 text-s-blue"
                  : "text-ink-3 hover:text-ink-2"
              }`}
            >
              {label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
}
