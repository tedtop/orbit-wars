"use client";

import { useCallback, useMemo, useRef, useState } from "react";

export type Series = {
  id: string;
  label: string;
  color: string;
  dash?: string;
  points: { x: number; y: number }[];
};

export type Annotation = { x: number; label: string };

type Hover = { px: number; x: number; values: { label: string; color: string; y: number | null }[] };

/** Minimal SVG line chart: 2px lines, recessive grid, crosshair + tooltip. */
export default function LineChart({
  series,
  annotations = [],
  height = 340,
  yDomain,
  formatX,
  formatY = (v) => String(Math.round(v)),
  yLabel,
}: {
  series: Series[];
  annotations?: Annotation[];
  height?: number;
  yDomain?: [number, number];
  formatX: (x: number) => string;
  formatY?: (y: number) => string;
  yLabel?: string;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const W = 900;
  const H = height;
  const M = { top: 18, right: 16, bottom: 28, left: 48 };
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;

  const { x0, x1, y0, y1 } = useMemo(() => {
    const xs = series.flatMap((s) => s.points.map((p) => p.x));
    const ys = series.flatMap((s) => s.points.map((p) => p.y));
    return {
      x0: Math.min(...xs),
      x1: Math.max(...xs),
      y0: yDomain ? yDomain[0] : Math.min(...ys),
      y1: yDomain ? yDomain[1] : Math.max(...ys),
    };
  }, [series, yDomain]);

  const sx = useCallback((x: number) => M.left + ((x - x0) / (x1 - x0 || 1)) * iw, [M.left, x0, x1, iw]);
  const sy = useCallback((y: number) => M.top + (1 - (y - y0) / (y1 - y0 || 1)) * ih, [M.top, y0, y1, ih]);

  const paths = useMemo(
    () =>
      series.map((s) => ({
        ...s,
        d: s.points
          .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
          .join(""),
      })),
    [series, sx, sy],
  );

  const yTicks = useMemo(() => {
    const n = 5;
    return Array.from({ length: n + 1 }, (_, i) => y0 + ((y1 - y0) * i) / n);
  }, [y0, y1]);

  const xTicks = useMemo(() => {
    const n = 6;
    return Array.from({ length: n + 1 }, (_, i) => x0 + ((x1 - x0) * i) / n);
  }, [x0, x1]);

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const fx = ((e.clientX - rect.left) / rect.width) * W;
    if (fx < M.left || fx > W - M.right) return setHover(null);
    const dataX = x0 + ((fx - M.left) / iw) * (x1 - x0);
    const values = series.map((s) => {
      let best: { x: number; y: number } | null = null;
      for (const p of s.points) {
        if (!best || Math.abs(p.x - dataX) < Math.abs(best.x - dataX)) best = p;
      }
      return { label: s.label, color: s.color, y: best ? best.y : null };
    });
    setHover({ px: fx, x: dataX, values });
  };

  return (
    <div ref={wrapRef} className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={yLabel ?? "line chart"}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {/* grid */}
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" strokeWidth="1" />
            <text x={M.left - 8} y={sy(t) + 3} textAnchor="end" className="tick-label">
              {formatY(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={sx(t)} y={H - 8} textAnchor="middle" className="tick-label">
            {formatX(t)}
          </text>
        ))}
        <line x1={M.left} x2={W - M.right} y1={M.top + ih} y2={M.top + ih} stroke="var(--baseline)" strokeWidth="1" />

        {/* annotations */}
        {annotations.map((a) => (
          <g key={a.label}>
            <line x1={sx(a.x)} x2={sx(a.x)} y1={M.top} y2={M.top + ih} stroke="var(--baseline)" strokeWidth="1" strokeDasharray="3 4" />
            <text
              x={sx(a.x) + 4}
              y={M.top + 10}
              className="tick-label"
              style={{ fill: "var(--ink-2)" }}
            >
              {a.label}
            </text>
          </g>
        ))}

        {/* series */}
        {paths.map((s) => (
          <path key={s.id} d={s.d} fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} strokeLinejoin="round" />
        ))}

        {/* direct labels at line ends */}
        {paths.map((s) => {
          const last = s.points[s.points.length - 1];
          if (!last) return null;
          return (
            <text
              key={`${s.id}-lbl`}
              x={Math.min(sx(last.x) + 6, W - M.right)}
              y={sy(last.y) + 3}
              className="tick-label"
              style={{ fill: "var(--ink-2)" }}
              textAnchor={sx(last.x) > W - 120 ? "end" : "start"}
            >
              {s.label}
            </text>
          );
        })}

        {/* crosshair */}
        {hover && (
          <line x1={hover.px} x2={hover.px} y1={M.top} y2={M.top + ih} stroke="var(--ink-3)" strokeWidth="1" />
        )}
      </svg>

      {hover && (
        <div
          className="chart-tooltip"
          style={{
            left: `min(max(${(hover.px / W) * 100}% + 12px, 0px), calc(100% - 180px))`,
            top: 8,
          }}
        >
          <div className="mb-1 text-ink">{formatX(hover.x)}</div>
          {hover.values.map((v) => (
            <div key={v.label} className="flex items-center gap-2">
              <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: v.color }} />
              <span>{v.label}</span>
              <span className="ml-auto pl-3 text-ink">{v.y === null ? "—" : formatY(v.y)}</span>
            </div>
          ))}
        </div>
      )}

      {/* legend */}
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1">
        {series.map((s) => (
          <span key={s.id} className="flex items-center gap-2 font-mono text-[11px] text-ink-3">
            <span
              aria-hidden
              className="inline-block h-0.5 w-5 rounded"
              style={{ background: s.color }}
            />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
