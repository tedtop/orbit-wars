export interface ScorePoint {
  time: string;       // ISO timestamp
  rank: number;
  score: number;
  prize_score: number | null;
  bot_name: string;
}

export interface TimelinePhase {
  phase: string;      // e.g. "Phase 0", "v6"
  date: string;
  title: string;
  headline: string;
  body_md: string;
  status: "submitted" | "dead_end" | "breakthrough" | "infrastructure" | "closed" | "ongoing";
}

export interface Experiment {
  epoch: number;
  name: string;
  base: string;
  track: string;
  verdict: "KEEP" | "DISCARD" | "IN PROGRESS";
  win_pct: number | null;
  delta: number | null;
  category: string;
  note: string;
}

export interface RLRun {
  run: string;
  update: number;
  steps: number;
  loss: number;
  clip_frac: number;
  explained_variance: number | null;
  entropy: number | null;
  eval_wr?: number | null;  // win rate vs greedy
  sps?: number;
}

export interface ReplayMeta {
  id: string;
  teams: string[];
  winner: string | null;
  our_result: "win" | "loss" | "unknown";
  steps: number;
  date: string;
}

export interface Scientist {
  name: string;
  field: string;
  cluster: "physics" | "life" | "math_cs" | "economics" | "other";
  description: string;
  rating?: number;
}
