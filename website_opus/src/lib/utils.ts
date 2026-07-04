import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatScore(score: number) {
  return score.toFixed(1);
}

export function formatRank(rank: number) {
  return `#${rank}`;
}
