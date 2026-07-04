"use client";

import { useEffect, useRef } from "react";

/* A cinematic orbital scene: glowing sun, tilted orbit trails, planets,
   and tiny fleets in transit — the game itself, painted large. */

type Orbit = { a: number; b: number; tilt: number; speed: number; phase: number; pr: number; hue: string };
type Ship = { orbit: number; t: number; speed: number; len: number };

const PLANET_HUES = ["#3987e5", "#9085e9", "#b7bfd6", "#199e70", "#e66767"];

export default function HeroSection() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let running = true;
    let W = 0;
    let H = 0;
    let sunX = 0;
    let sunY = 0;
    let sunR = 0;
    let orbits: Orbit[] = [];
    let ships: Ship[] = [];

    const setup = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = wrap.clientWidth;
      H = wrap.clientHeight;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const compact = W < 768;
      sunX = compact ? W * 0.5 : W * 0.78;
      sunY = compact ? H * 0.13 : H * 0.24;
      sunR = Math.min(W, H) * (compact ? 0.17 : 0.2);
      orbits = Array.from({ length: 5 }, (_, i) => {
        const a = sunR * (1.7 + i * 0.62) * (0.9 + Math.random() * 0.2);
        return {
          a,
          b: a * (0.32 + Math.random() * 0.1),
          tilt: -0.38 + Math.random() * 0.12,
          speed: (0.05 + Math.random() * 0.05) / (1 + i * 0.35),
          phase: Math.random() * Math.PI * 2,
          pr: 2.2 + Math.random() * 2.6,
          hue: PLANET_HUES[i % PLANET_HUES.length],
        };
      });
      ships = Array.from({ length: 7 }, () => ({
        orbit: Math.floor(Math.random() * orbits.length),
        t: Math.random() * Math.PI * 2,
        speed: 0.12 + Math.random() * 0.1,
        len: 10 + Math.random() * 16,
      }));
    };

    const orbitPoint = (o: Orbit, angle: number) => {
      const x0 = Math.cos(angle) * o.a;
      const y0 = Math.sin(angle) * o.b;
      return {
        x: sunX + x0 * Math.cos(o.tilt) - y0 * Math.sin(o.tilt),
        y: sunY + x0 * Math.sin(o.tilt) + y0 * Math.cos(o.tilt),
      };
    };

    const drawSun = (t: number) => {
      const pulse = 1 + (reduced ? 0 : Math.sin(t * 0.0004) * 0.012);
      const r = sunR * pulse;
      // far glow
      let g = ctx.createRadialGradient(sunX, sunY, r * 0.2, sunX, sunY, r * 3.2);
      g.addColorStop(0, "rgba(255,157,60,0.28)");
      g.addColorStop(0.4, "rgba(255,120,40,0.10)");
      g.addColorStop(1, "rgba(255,120,40,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);
      // body
      g = ctx.createRadialGradient(
        sunX - r * 0.18,
        sunY - r * 0.18,
        r * 0.1,
        sunX,
        sunY,
        r,
      );
      g.addColorStop(0, "#fff3da");
      g.addColorStop(0.35, "#ffd9a0");
      g.addColorStop(0.75, "#ff9d3c");
      g.addColorStop(1, "#e5661a");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(sunX, sunY, r, 0, Math.PI * 2);
      ctx.fill();
      // corona rim
      ctx.strokeStyle = "rgba(255,200,120,0.5)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(sunX, sunY, r + 1, 0, Math.PI * 2);
      ctx.stroke();
    };

    const draw = (t: number) => {
      ctx.clearRect(0, 0, W, H);

      // orbit trails (behind sun half fades naturally via low alpha)
      for (const o of orbits) {
        ctx.strokeStyle = "rgba(255,170,90,0.13)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let a = 0; a <= Math.PI * 2 + 0.02; a += 0.05) {
          const p = orbitPoint(o, a);
          if (a === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
      }

      drawSun(t);

      // planets
      orbits.forEach((o) => {
        const ang = o.phase + t * 0.0002 * o.speed * 60;
        const p = orbitPoint(o, ang);
        const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, o.pr * 3.5);
        glow.addColorStop(0, `${o.hue}55`);
        glow.addColorStop(1, `${o.hue}00`);
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(p.x, p.y, o.pr * 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = o.hue;
        ctx.beginPath();
        ctx.arc(p.x, p.y, o.pr, 0, Math.PI * 2);
        ctx.fill();
      });

      // fleets: short bright streaks gliding along orbits between planets
      for (const s of ships) {
        const o = orbits[s.orbit];
        s.t += reduced ? 0 : 0.0016 * s.speed * 60 * 0.016;
        const head = orbitPoint(o, s.t);
        const tail = orbitPoint(o, s.t - s.len / o.a);
        const grad = ctx.createLinearGradient(tail.x, tail.y, head.x, head.y);
        grad.addColorStop(0, "rgba(242,244,251,0)");
        grad.addColorStop(1, "rgba(242,244,251,0.75)");
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(tail.x, tail.y);
        ctx.lineTo(head.x, head.y);
        ctx.stroke();
      }

      // bottom fade into the page — also the scrim the headline sits on
      const fade = ctx.createLinearGradient(0, H * 0.34, 0, H);
      fade.addColorStop(0, "rgba(5,6,13,0)");
      fade.addColorStop(0.55, "rgba(5,6,13,0.72)");
      fade.addColorStop(1, "rgba(5,6,13,1)");
      ctx.fillStyle = fade;
      ctx.fillRect(0, 0, W, H);

      if (running && !reduced) raf = requestAnimationFrame(draw);
    };

    setup();
    draw(performance.now());

    const onResize = () => {
      setup();
      if (reduced) draw(performance.now());
    };
    window.addEventListener("resize", onResize);

    const io = new IntersectionObserver(([e]) => {
      const shouldRun = e.isIntersecting;
      if (shouldRun && !running) {
        running = true;
        if (!reduced) raf = requestAnimationFrame(draw);
      } else if (!shouldRun) {
        running = false;
        cancelAnimationFrame(raf);
      }
    });
    io.observe(wrap);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      io.disconnect();
    };
  }, []);

  return (
    <div ref={wrapRef} id="top" className="relative flex min-h-svh flex-col overflow-hidden">
      <canvas ref={canvasRef} aria-hidden className="absolute inset-0 h-full w-full" />

      <div className="relative z-10 mx-auto flex w-full max-w-6xl flex-1 flex-col items-center justify-end px-5 pb-20 pt-32 text-center sm:pb-24">
        <p className="kicker mb-6">team montana schmeekler · kaggle campaign</p>
        <h1 className="font-display text-6xl font-bold leading-none tracking-tight text-ink sm:text-8xl md:text-9xl">
          orbit wars<span className="text-s-blue">.</span>
        </h1>
        <p className="mt-6 max-w-xl text-base leading-relaxed text-ink-2 sm:text-lg">
          Ten days. 23 bots. 19 experiments. One reverse-engineered engine, a
          nine-machine supercomputer fleet, and a $150 lesson in why
          reinforcement learning is hard.
        </p>

        <dl className="mt-10 grid grid-cols-2 gap-3 font-mono text-xs sm:grid-cols-4">
          {[
            ["4,752", "teams in the field"],
            ["#415", "final rank · top 9%"],
            ["1,243.8", "peak elo · rank 144"],
            ["80,000+", "games simulated"],
          ].map(([num, label]) => (
            <div key={label} className="card px-4 py-3">
              <dt className="sr-only">{label}</dt>
              <dd>
                <span className="block font-display text-xl font-bold text-ink">{num}</span>
                <span className="mt-1 block tracking-wide text-ink-3">{label}</span>
              </dd>
            </div>
          ))}
        </dl>

        <a
          href="#game"
          className="kicker mt-14 inline-block text-ink-3 transition-colors hover:text-s-blue"
        >
          scroll for mission log ↓
        </a>
      </div>
    </div>
  );
}
