"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Section from "../ui/Section";
import Reveal from "../ui/Reveal";
import manifest from "@/data/replays_manifest.json";

/* Compact replay format (scripts/build_data.py):
   steps[i].p = planets  [id, owner, x, y, radius, ships, production]
   steps[i].f = fleets   [owner, x, y, heading, ships]
   steps[i].c = comet planet ids */
type Replay = {
  id: string;
  label: string;
  teams: string[];
  our_seat: number;
  n_steps: number;
  steps: { p: number[][]; f: number[][]; c: number[] }[];
};

type ManifestEntry = {
  id: string;
  label: string;
  teams: string[];
  our_seat: number;
  result: string;
  placement: number | null;
  n_steps: number;
  players: number;
};

const NEUTRAL = "#4a5370";
const OPPONENT_COLORS = ["#e66767", "#c98500", "#9085e9"];

function seatColors(nPlayers: number, ourSeat: number): string[] {
  const colors: string[] = [];
  let oi = 0;
  for (let s = 0; s < nPlayers; s++) {
    colors.push(s === ourSeat ? "#3987e5" : OPPONENT_COLORS[oi++ % OPPONENT_COLORS.length]);
  }
  return colors;
}

export default function TheaterSection() {
  const entries = manifest as ManifestEntry[];
  const [selected, setSelected] = useState(entries[0].id);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [playing, setPlaying] = useState(false);
  const loading = !replay || replay.id !== selected;
  const [speed, setSpeed] = useState(2);
  const [stepIdx, setStepIdx] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stepRef = useRef(0);
  const playingRef = useRef(false);
  const speedRef = useRef(2);
  const replayRef = useRef<Replay | null>(null);

  /* load replay on selection */
  useEffect(() => {
    let alive = true;
    fetch(`/data/replays/${selected}.json`)
      .then((r) => r.json())
      .then((rep: Replay) => {
        if (!alive) return;
        replayRef.current = rep;
        stepRef.current = 0;
        setReplay(rep);
        setStepIdx(0);
        setPlaying(true);
        playingRef.current = true;
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [selected]);

  const selectReplay = (id: string) => {
    playingRef.current = false;
    setPlaying(false);
    setSelected(id);
  };

  const draw = useCallback((rep: Replay, idx: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = canvas.clientWidth;
    if (canvas.width !== size * dpr) {
      canvas.width = size * dpr;
      canvas.height = size * dpr;
    }
    ctx.setTransform((size * dpr) / 100, 0, 0, (size * dpr) / 100, 0, 0);

    const step = rep.steps[Math.min(idx, rep.steps.length - 1)];
    const colors = seatColors(rep.teams.length, rep.our_seat);
    const comets = new Set(step.c);

    // board
    ctx.fillStyle = "#05060d";
    ctx.fillRect(0, 0, 100, 100);

    // sun
    const sun = ctx.createRadialGradient(50, 50, 0.5, 50, 50, 7);
    sun.addColorStop(0, "#fff3da");
    sun.addColorStop(0.4, "#ffb45e");
    sun.addColorStop(1, "rgba(255,120,40,0)");
    ctx.fillStyle = sun;
    ctx.beginPath();
    ctx.arc(50, 50, 7, 0, Math.PI * 2);
    ctx.fill();

    // fleets — triangles pointing along heading, size ~ sqrt(ships)
    for (const f of step.f) {
      const [owner, x, y, heading, ships] = f;
      const r = Math.min(1.6, 0.35 + Math.sqrt(ships) * 0.16);
      ctx.fillStyle = colors[owner] ?? NEUTRAL;
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(heading);
      ctx.beginPath();
      ctx.moveTo(r * 1.6, 0);
      ctx.lineTo(-r, r * 0.8);
      ctx.lineTo(-r, -r * 0.8);
      ctx.closePath();
      ctx.globalAlpha = 0.95;
      ctx.fill();
      ctx.restore();
    }
    ctx.globalAlpha = 1;

    // planets
    ctx.textAlign = "center";
    for (const p of step.p) {
      const [pid, owner, x, y, radius, ships] = p;
      const col = owner === -1 ? NEUTRAL : (colors[owner] ?? NEUTRAL);
      const r = Math.max(1.1, radius * 1.15);
      if (comets.has(pid)) {
        ctx.strokeStyle = "rgba(201,133,0,0.9)";
        ctx.lineWidth = 0.35;
        ctx.setLineDash([0.8, 0.8]);
        ctx.beginPath();
        ctx.arc(x, y, r + 1, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      const glow = ctx.createRadialGradient(x, y, 0, x, y, r * 2.2);
      glow.addColorStop(0, `${col}66`);
      glow.addColorStop(1, `${col}00`);
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(x, y, r * 2.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      // garrison label
      ctx.fillStyle = "rgba(242,244,251,0.92)";
      ctx.font = "600 2.4px ui-monospace, Menlo, monospace";
      ctx.fillText(String(ships), x, y - r - 0.9);
    }
  }, []);

  /* playback loop */
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const rep = replayRef.current;
      if (!rep) return;
      if (playingRef.current) {
        const dt = (now - last) / 1000;
        last = now;
        stepRef.current += dt * 12 * speedRef.current;
        if (stepRef.current >= rep.steps.length - 1) {
          stepRef.current = rep.steps.length - 1;
          playingRef.current = false;
          setPlaying(false);
        }
        setStepIdx(Math.floor(stepRef.current));
      } else {
        last = now;
      }
      draw(rep, Math.floor(stepRef.current));
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [draw]);

  const togglePlay = () => {
    const rep = replayRef.current;
    if (!rep) return;
    if (!playing && stepRef.current >= rep.steps.length - 1) stepRef.current = 0;
    playingRef.current = !playing;
    setPlaying(!playing);
  };

  const scrub = (v: number) => {
    stepRef.current = v;
    setStepIdx(v);
  };

  const setSpd = (s: number) => {
    speedRef.current = s;
    setSpeed(s);
  };

  /* live ship totals for the current step */
  const totals: { name: string; ships: number; color: string }[] = [];
  if (replay) {
    const step = replay.steps[Math.min(stepIdx, replay.steps.length - 1)];
    const colors = seatColors(replay.teams.length, replay.our_seat);
    const sums = new Array(replay.teams.length).fill(0);
    for (const p of step.p) if (p[1] >= 0) sums[p[1]] += p[5];
    for (const f of step.f) sums[f[0]] += f[4];
    replay.teams.forEach((name, s) => {
      totals.push({ name, ships: sums[s], color: colors[s] });
    });
  }
  const totalShips = totals.reduce((a, t) => a + t.ships, 0) || 1;

  const current = entries.find((e) => e.id === selected);

  return (
    <Section
      id="theater"
      kicker="live from the ladder"
      title="the replay theater"
      lede={
        <>
          These aren&apos;t simulations — they&apos;re real ranked games played by{" "}
          <span className="font-mono text-s-blue">Montana Schmeekler</span> on the
          Kaggle ladder, replayed from the raw episode logs. Blue is us. Watch the
          engine grind out a 2P win, or three rivals dismantle us in a 4P pod.
        </>
      }
      wide
    >
      <Reveal>
        <div className="mb-5 flex flex-wrap gap-2">
          {entries.map((e) => (
            <button
              key={e.id}
              onClick={() => selectReplay(e.id)}
              className={`rounded-full px-4 py-1.5 font-mono text-[11px] transition-colors ${
                selected === e.id
                  ? "bg-s-blue/20 text-s-blue outline outline-1 outline-s-blue/50"
                  : "bg-white/[0.04] text-ink-3 hover:text-ink-2"
              }`}
            >
              {e.label}
            </button>
          ))}
        </div>
      </Reveal>

      <Reveal>
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_290px]">
          <div className="card overflow-hidden p-3 sm:p-4">
            <div className="relative">
              <canvas
                ref={canvasRef}
                className="aspect-square w-full rounded-lg"
                role="img"
                aria-label={`Replay of game ${current?.label ?? ""}`}
              />
              {loading && (
                <div className="absolute inset-0 flex items-center justify-center bg-void/60 font-mono text-xs text-ink-2">
                  loading episode…
                </div>
              )}
            </div>

            {/* controls */}
            <div className="mt-3 flex items-center gap-3 px-1">
              <button
                onClick={togglePlay}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-s-blue/20 text-s-blue transition-colors hover:bg-s-blue/30"
                aria-label={playing ? "Pause" : "Play"}
              >
                {playing ? "❚❚" : "▶"}
              </button>
              <input
                type="range"
                min={0}
                max={(replay?.n_steps ?? 500) - 1}
                value={Math.min(stepIdx, (replay?.n_steps ?? 500) - 1)}
                onChange={(e) => scrub(Number(e.target.value))}
                className="w-full accent-[#3987e5]"
                aria-label="Scrub through game turns"
              />
              <span className="w-24 shrink-0 text-right font-mono text-[11px] tabular-nums text-ink-3">
                turn {String(stepIdx).padStart(3, "0")}/{(replay?.n_steps ?? 500) - 1}
              </span>
            </div>
            <div className="mt-2 flex items-center gap-1.5 px-1">
              <span className="mr-1 font-mono text-[10px] uppercase tracking-widest text-ink-3">speed</span>
              {[1, 2, 4, 8].map((s) => (
                <button
                  key={s}
                  onClick={() => setSpd(s)}
                  className={`rounded px-2 py-0.5 font-mono text-[11px] ${
                    speed === s ? "bg-s-blue/20 text-s-blue" : "text-ink-3 hover:text-ink-2"
                  }`}
                >
                  {s}×
                </button>
              ))}
            </div>
          </div>

          {/* scoreboard */}
          <div className="flex flex-col gap-3">
            <div className="card px-5 py-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
                episode
              </span>
              <p className="mt-1 font-mono text-xs leading-relaxed text-ink-2">
                #{current?.id} · {current?.players}P ·{" "}
                {current?.result === "win" ? (
                  <span className="text-[#2fc32f]">we won</span>
                ) : (
                  <span className="text-[#e66c6c]">placed {current?.placement}</span>
                )}
              </p>
            </div>
            <div className="card flex-1 px-5 py-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-3">
                live fleet strength
              </span>
              <div className="mt-3 flex flex-col gap-3">
                {totals.map((t) => (
                  <div key={t.name}>
                    <div className="flex items-baseline justify-between gap-2">
                      <span
                        className="truncate font-mono text-[11px]"
                        style={{ color: t.color }}
                      >
                        {t.name === (replay && replay.teams[replay.our_seat]) ? `▸ ${t.name}` : t.name}
                      </span>
                      <span className="font-mono text-[11px] tabular-nums text-ink-2">
                        {t.ships.toLocaleString()}
                      </span>
                    </div>
                    <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-white/[0.05]">
                      <div
                        className="h-full rounded-full transition-[width] duration-150"
                        style={{ width: `${(t.ships / totalShips) * 100}%`, background: t.color }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-4 border-t border-white/[0.06] pt-3 font-mono text-[10px] leading-relaxed text-ink-3">
                dashed amber ring = comet planet · triangle size ∝ fleet ships ·
                number = garrison
              </p>
            </div>
          </div>
        </div>
      </Reveal>
    </Section>
  );
}
