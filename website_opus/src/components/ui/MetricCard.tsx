import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
  className?: string;
}

export function MetricCard({ label, value, sub, accent = "#3b82f6", className }: MetricCardProps) {
  return (
    <div className={cn("glass p-5 flex flex-col gap-1", className)}>
      <span className="text-xs font-medium uppercase tracking-widest text-slate-500">{label}</span>
      <span
        className="text-3xl font-bold font-[var(--font-space)] leading-none"
        style={{ color: accent }}
      >
        {value}
      </span>
      {sub && <span className="text-xs text-slate-500 mt-1">{sub}</span>}
    </div>
  );
}
