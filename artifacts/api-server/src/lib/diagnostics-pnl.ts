/**
 * Phase 6 — Diagnostics: PnL breakdown by regime / model / coin / timeframe.
 *
 * Pure read aggregator over `trade_journal` joined to `prediction_journal`
 * for the model_version. Returns a denormalized matrix the dashboard can
 * pivot on. No mutation, no network calls.
 */
import {
  db,
  tradeJournalTable,
  predictionJournalTable,
} from "@workspace/db";
import { and, eq, gte, sql } from "drizzle-orm";

export interface PnlBreakdownBucket {
  regime: string;
  modelVersion: string;
  coinId: string;
  timeframe: string;
  nTrades: number;
  pnlUsd: number;
  winRate: number;
  avgPnlPct: number;
}

export interface PnlBreakdownResponse {
  windowHours: number;
  generatedAt: string;
  totals: {
    nTrades: number;
    pnlUsd: number;
    winRate: number;
  };
  buckets: PnlBreakdownBucket[];
}

export async function getPnlBreakdown(
  windowHours = 168,
): Promise<PnlBreakdownResponse> {
  const since = new Date(Date.now() - windowHours * 3600 * 1000);
  // Left join: trades whose linked prediction journal row carries the
  // model_version. Trades without a journal row still appear with
  // modelVersion="unknown".
  const rows = await db
    .select({
      regime: sql<string | null>`coalesce(${tradeJournalTable.regimeLabel}, 'unknown')`,
      modelVersion: sql<string | null>`coalesce(${predictionJournalTable.modelVersion}, 'unknown')`,
      coinId: tradeJournalTable.coinId,
      timeframe: tradeJournalTable.timeframe,
      pnlUsd: tradeJournalTable.realizedPnlUsd,
      pnlPct: tradeJournalTable.realizedPnlPct,
    })
    .from(tradeJournalTable)
    .leftJoin(
      predictionJournalTable,
      eq(predictionJournalTable.id, tradeJournalTable.predictionJournalId),
    )
    .where(
      and(
        gte(tradeJournalTable.createdAt, since),
        sql`${tradeJournalTable.realizedPnlUsd} IS NOT NULL`,
      ),
    );

  const map = new Map<string, PnlBreakdownBucket & { _pnlPctSum: number }>();
  let totalTrades = 0;
  let totalPnl = 0;
  let totalWins = 0;
  for (const r of rows) {
    const pnl = r.pnlUsd ?? 0;
    const pnlPct = r.pnlPct ?? 0;
    const key = `${r.regime}|${r.modelVersion}|${r.coinId}|${r.timeframe}`;
    const acc =
      map.get(key) ??
      ({
        regime: r.regime ?? "unknown",
        modelVersion: r.modelVersion ?? "unknown",
        coinId: r.coinId,
        timeframe: r.timeframe,
        nTrades: 0,
        pnlUsd: 0,
        winRate: 0,
        avgPnlPct: 0,
        _pnlPctSum: 0,
      } as PnlBreakdownBucket & { _pnlPctSum: number });
    acc.nTrades += 1;
    acc.pnlUsd += pnl;
    acc._pnlPctSum += pnlPct;
    if (pnl > 0) acc.winRate += 1;
    map.set(key, acc);

    totalTrades += 1;
    totalPnl += pnl;
    if (pnl > 0) totalWins += 1;
  }
  const buckets: PnlBreakdownBucket[] = [];
  for (const acc of map.values()) {
    const winRate = acc.nTrades > 0 ? acc.winRate / acc.nTrades : 0;
    const avgPnlPct = acc.nTrades > 0 ? acc._pnlPctSum / acc.nTrades : 0;
    buckets.push({
      regime: acc.regime,
      modelVersion: acc.modelVersion,
      coinId: acc.coinId,
      timeframe: acc.timeframe,
      nTrades: acc.nTrades,
      pnlUsd: Number(acc.pnlUsd.toFixed(4)),
      winRate: Number(winRate.toFixed(4)),
      avgPnlPct: Number(avgPnlPct.toFixed(4)),
    });
  }
  // Order by absolute pnl desc so the top movers are first.
  buckets.sort((a, b) => Math.abs(b.pnlUsd) - Math.abs(a.pnlUsd));
  return {
    windowHours,
    generatedAt: new Date().toISOString(),
    totals: {
      nTrades: totalTrades,
      pnlUsd: Number(totalPnl.toFixed(4)),
      winRate: totalTrades > 0 ? Number((totalWins / totalTrades).toFixed(4)) : 0,
    },
    buckets,
  };
}
