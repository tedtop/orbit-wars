"use client";
import { useState, useEffect, useRef, useCallback } from "react";

const OWNER_COLORS = ["#3b82f6", "#ef4444", "#a855f7", "#f59e0b"];
const NEUTRAL_COLOR = "#475569";
const SUN_COLOR = "#fbbf24";

interface Planet {
  id: number;
  owner: number;
  x: number;
  y: number;
  angular_velocity: number;
  garrison: number;
  production: number;
}

interface Fleet {
  id: number;
  owner: number;
  x: number;
  y: number;
  ships: number;
  dest_x?: number;
  dest_y?: number;
}

interface GameStep {
  planets: Planet[];
  fleets: Fleet[];
}

// Normalize coords to canvas space
function normalize(val: number, min: number, max: number, size: number, pad: number) {
  return pad + ((val - min) / (max - min)) * (size - 2 * pad);
}

export function OrbitalCanvas({ episodeId, teams }: { episodeId: string; teams: string[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [steps, setSteps] = useState<GameStep[]>([]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setStep(0);
    setPlaying(false);

    fetch(`/data/replays/${episodeId}.json`)
      .then(r => {
        if (!r.ok) throw new Error(`Replay not in local cache (${r.status})`);
        return r.json();
      })
      .then(d => {
        const parsed: GameStep[] = d.steps.map((s: Array<{observation: {planets: unknown[]; fleets: unknown[]}}>) => {
          const obs = s[0].observation;
          // planet: [id, owner, x, y, angular_velocity, garrison, production]
          const planets = (obs.planets as number[][]).map(p => ({
            id: p[0], owner: p[1], x: p[2], y: p[3],
            angular_velocity: p[4], garrison: p[5], production: p[6],
          }));
          // fleet: [id, owner, x, y, dest_x, dest_y, ships, ...]
          const fleets = (obs.fleets as number[][]).map(f => ({
            id: f[0], owner: f[1], x: f[2], y: f[3],
            dest_x: f[4], dest_y: f[5], ships: f[6],
          }));
          return { planets, fleets };
        });
        setSteps(parsed);
        setLoading(false);
      })
      .catch(e => {
        setError(e.message);
        setLoading(false);
      });
  }, [episodeId]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || steps.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const gameStep = steps[Math.min(step, steps.length - 1)];
    const PAD = 32;

    // Compute world bounds from first step
    const allX = steps[0].planets.map(p => p.x);
    const allY = steps[0].planets.map(p => p.y);
    const xMin = Math.min(...allX), xMax = Math.max(...allX);
    const yMin = Math.min(...allY), yMax = Math.max(...allY);
    const toX = (v: number) => normalize(v, xMin, xMax, W, PAD);
    const toY = (v: number) => normalize(v, yMin, yMax, H, PAD);

    // Background
    ctx.fillStyle = "#08080f";
    ctx.fillRect(0, 0, W, H);

    // Faint grid
    ctx.strokeStyle = "rgba(255,255,255,0.03)";
    ctx.lineWidth = 1;
    for (let gx = PAD; gx < W - PAD; gx += 40) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }
    for (let gy = PAD; gy < H - PAD; gy += 40) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }

    // Sun (approximate center)
    const cx = (toX(xMin) + toX(xMax)) / 2;
    const cy = (toY(yMin) + toY(yMax)) / 2;
    const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, 20);
    grd.addColorStop(0, "#fde68a");
    grd.addColorStop(0.5, "#fbbf24");
    grd.addColorStop(1, "transparent");
    ctx.fillStyle = grd;
    ctx.beginPath(); ctx.arc(cx, cy, 20, 0, Math.PI * 2); ctx.fill();

    // Fleets
    for (const fleet of gameStep.fleets) {
      const fx = toX(fleet.x), fy = toY(fleet.y);
      const color = fleet.owner >= 0 ? (OWNER_COLORS[fleet.owner % OWNER_COLORS.length]) : NEUTRAL_COLOR;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(fx, fy, 3, 0, Math.PI * 2);
      ctx.fill();
      // Direction line
      if (fleet.dest_x !== undefined && fleet.dest_y !== undefined) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 0.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(fx, fy);
        ctx.lineTo(toX(fleet.dest_x), toY(fleet.dest_y));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.globalAlpha = 1;
    }

    // Planets
    for (const planet of gameStep.planets) {
      const px = toX(planet.x), py = toY(planet.y);
      const r = Math.max(8, Math.min(20, 6 + planet.production * 1.5));
      const color = planet.owner >= 0 ? OWNER_COLORS[planet.owner % OWNER_COLORS.length] : NEUTRAL_COLOR;

      // Glow for owned planets
      if (planet.owner >= 0) {
        const glowR = ctx.createRadialGradient(px, py, 0, px, py, r + 6);
        glowR.addColorStop(0, `${color}40`);
        glowR.addColorStop(1, "transparent");
        ctx.fillStyle = glowR;
        ctx.beginPath(); ctx.arc(px, py, r + 6, 0, Math.PI * 2); ctx.fill();
      }

      // Planet body
      ctx.fillStyle = planet.owner >= 0 ? `${color}30` : "#1e293b";
      ctx.strokeStyle = color;
      ctx.lineWidth = planet.angular_velocity === 0 ? 2 : 1.5;
      ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();

      // Static planet marker (double ring)
      if (planet.angular_velocity === 0) {
        ctx.strokeStyle = `${color}50`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(px, py, r + 4, 0, Math.PI * 2); ctx.stroke();
      }

      // Garrison text
      ctx.fillStyle = "white";
      ctx.font = `bold ${Math.max(9, r - 4)}px var(--font-mono, monospace)`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(Math.round(planet.garrison).toString(), px, py);
    }

    // Step counter
    ctx.fillStyle = "#475569";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(`Step ${step + 1} / ${steps.length}`, 8, 6);
  }, [steps, step]);

  useEffect(() => { draw(); }, [draw]);

  // Playback
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (!playing) return;
    intervalRef.current = setInterval(() => {
      setStep(s => {
        if (s >= steps.length - 1) { setPlaying(false); return s; }
        return s + 1;
      });
    }, 1000 / speed);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, speed, steps.length]);

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-slate-500 text-sm">Loading replay…</div>
  );
  if (error) return (
    <div className="flex flex-col items-center justify-center h-40 gap-2">
      <span className="text-slate-500 text-sm">{error}</span>
      <a
        href={`https://www.kaggle.com/competitions/llm-20-questions/leaderboard?episode_id=${episodeId}`}
        target="_blank" rel="noopener noreferrer"
        className="text-xs text-blue-400 hover:text-blue-300"
      >
        View on Kaggle ↗
      </a>
    </div>
  );

  // Planet count trend
  const planetTrend = steps.map((s, i) => ({
    step: i,
    counts: OWNER_COLORS.map((_, oi) => s.planets.filter(p => p.owner === oi).length),
  }));

  return (
    <div className="space-y-4">
      {/* Team legend */}
      <div className="flex flex-wrap gap-3 text-xs">
        {teams.map((name, i) => (
          <span key={name} className="flex items-center gap-1.5 text-slate-400">
            <span className="w-3 h-3 rounded-full inline-block" style={{ background: OWNER_COLORS[i % OWNER_COLORS.length] }} />
            {name.slice(0, 24)}
          </span>
        ))}
        <span className="flex items-center gap-1.5 text-slate-600">
          <span className="w-3 h-3 rounded-full inline-block" style={{ background: NEUTRAL_COLOR }} />
          Neutral
        </span>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={560}
        height={420}
        className="w-full rounded-lg"
        style={{ maxHeight: 420, background: "#08080f" }}
      />

      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <button
          onClick={() => setPlaying(p => !p)}
          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors flex items-center gap-2"
        >
          {playing ? (
            <><svg width="12" height="12" viewBox="0 0 12 12"><rect x="2" y="1" width="3" height="10" fill="currentColor"/><rect x="7" y="1" width="3" height="10" fill="currentColor"/></svg>Pause</>
          ) : (
            <><svg width="12" height="12" viewBox="0 0 12 12"><path d="M2 1l9 5-9 5V1z" fill="currentColor"/></svg>Play</>
          )}
        </button>

        <input
          type="range"
          min={0}
          max={steps.length - 1}
          value={step}
          onChange={e => { setPlaying(false); setStep(Number(e.target.value)); }}
          className="flex-1 accent-blue-500"
        />

        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>Speed:</span>
          {[2,4,8,16].map(s => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`px-2 py-0.5 rounded font-mono transition-colors ${speed === s ? "bg-slate-700 text-white" : "text-slate-600 hover:text-slate-300"}`}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      {/* Mini planet count sparkline */}
      <div className="glass p-3">
        <div className="text-xs text-slate-600 mb-2">Planet ownership over time</div>
        <svg width="100%" height="60" viewBox={`0 0 ${steps.length} 60`} preserveAspectRatio="none">
          {[0,1].map(oi => {
            const pts = planetTrend.map(d => `${d.step},${60 - d.counts[oi] * 5}`).join(" ");
            return <polyline key={oi} points={pts} fill="none" stroke={OWNER_COLORS[oi]} strokeWidth={1} opacity={0.7} />;
          })}
          <line x1={step} y1={0} x2={step} y2={60} stroke="white" strokeWidth={0.5} opacity={0.4} />
        </svg>
      </div>
    </div>
  );
}
