import { useQuery } from "@tanstack/react-query";

const apiBase = `${import.meta.env.BASE_URL}api`.replace(/\/\//g, "/");

export interface StrategyBucket {
  strategyType: "ai-bots" | "dca-cb" | "buy-hold" | "trend-filter";
  label: string;
  agentCount: number;
  startingCapital: number;
  currentEquity: number;
  cash: number;
  invested: number;
  totalPnl: number;
  totalPnlPct: number;
  totalTrades: number;
  totalFees: number;
  peakValue: number;
  maxDrawdownPct: number;
  circuitBreakerActive?: boolean;
  latestDecision?: StrategyDecision;
}

export interface StrategyDecision {
  headline: string;
  detail?: string;
  tone: "good" | "warn" | "neutral";
  occurredAt?: string;
}

export interface SettingsChangeEntry {
  field: "drawdownTriggerPct" | "resumeLookbackDays" | "cycleDeployUsd";
  label: string;
  before: number;
  after: number;
  formattedBefore: string;
  formattedAfter: string;
}

export interface SettingsHistoryEvent {
  id: number;
  timestamp: string;
  changes: SettingsChangeEntry[];
}

export interface StrategyComparison {
  buckets: StrategyBucket[];
  equityCurves: Record<string, { timestamp: string; equity: number }[]>;
  settingsHistory: Record<string, SettingsHistoryEvent[]>;
  generatedAt: string;
}

export function useStrategyLab() {
  return useQuery<StrategyComparison>({
    queryKey: ["strategy-lab"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/crypto/strategy-lab`);
      if (!res.ok) throw new Error("Failed to fetch strategy lab data");
      return res.json();
    },
    refetchInterval: 15000,
  });
}
