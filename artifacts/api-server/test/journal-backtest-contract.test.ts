import { describe, it, after } from "node:test";
import assert from "node:assert/strict";

import { db, predictionJournalTable, tradeJournalTable } from "@workspace/db";
import { eq } from "drizzle-orm";

import { writeBacktestJournalRows } from "../src/lib/journal-writer";

const TRACER = "backtest-contract-test";

async function cleanup(): Promise<void> {
  await db.delete(tradeJournalTable).where(eq(tradeJournalTable.coinId, TRACER));
  await db.delete(predictionJournalTable).where(eq(predictionJournalTable.coinId, TRACER));
}

describe("backtest journal contract parity with live", () => {
  after(cleanup);

  it("skipReason is populated from the structured gate, not 'backtest_skipped'", async () => {
    await cleanup();
    const result = await writeBacktestJournalRows([
      {
        coinId: TRACER,
        timeframe: "1h",
        modelId: "lightgbm",
        modelVersion: null,
        featureHash: null, featureVector: null,
        regimeLabel: null,
        direction: "stable",
        confidence: 0,
        probUp: null, probDown: null, probStable: null,
        expectedReturnPct: null, predictionStdPct: null,
        priceAtPrediction: 0,
        predictedPrice: null,
        actualPrice: null, realizedReturnPct: null, outcome: null,
        resolvesAt: new Date(),
        resolvedAt: new Date(),
        gatesApplied: { backtest: true, counter_trend_regime: true, detail: "bullish vs down" },
        simulatedTrade: null,
      },
    ]);
    assert.equal(result.predictionsInserted, 1);
    assert.equal(result.tradesInserted, 0);

    const [row] = await db
      .select()
      .from(predictionJournalTable)
      .where(eq(predictionJournalTable.coinId, TRACER))
      .limit(1);
    assert.ok(row);
    assert.equal(row.becameTrade, false);
    // The structured gate name must be propagated to the first-class
    // skipReason column — NOT the generic 'backtest_skipped' fallback.
    assert.equal(row.skipReason, "counter_trend_regime");
  });

  it("simulatedTrade payload produces a trade_journal row with positive MAE magnitude", async () => {
    await cleanup();
    const result = await writeBacktestJournalRows([
      {
        coinId: TRACER,
        timeframe: "1h",
        modelId: "lightgbm",
        modelVersion: null,
        featureHash: null, featureVector: null,
        regimeLabel: null,
        direction: "up",
        confidence: 0.7,
        probUp: null, probDown: null, probStable: null,
        expectedReturnPct: null, predictionStdPct: null,
        priceAtPrediction: 50_000,
        predictedPrice: 50_500,
        actualPrice: 50_500, realizedReturnPct: 1.0, outcome: "correct",
        resolvesAt: new Date(),
        resolvedAt: new Date(),
        gatesApplied: { backtest: true },
        simulatedTrade: {
          entryTime: new Date(), exitTime: new Date(),
          entryPriceRaw: 49_995, entryPriceAdj: 50_000,
          exitPriceRaw: 50_505, exitPriceAdj: 50_500,
          entryFee: 0.5, exitFee: 0.5, slippagePct: 0.0001,
          positionSizeUsd: 1_000,
          // Live convention: MAE is positive magnitude.
          mfePct: 1.5, maePct: 0.4,
          exitReason: "tp",
          realizedPnlUsd: 9.0, realizedPnlPct: 0.9,
        },
      },
    ]);
    assert.equal(result.predictionsInserted, 1);
    assert.equal(result.tradesInserted, 1);

    const [pj] = await db
      .select()
      .from(predictionJournalTable)
      .where(eq(predictionJournalTable.coinId, TRACER))
      .limit(1);
    assert.ok(pj);
    assert.equal(pj.becameTrade, true);
    assert.equal(pj.skipReason, null);

    const [tj] = await db
      .select()
      .from(tradeJournalTable)
      .where(eq(tradeJournalTable.coinId, TRACER))
      .limit(1);
    assert.ok(tj);
    // Positive magnitude — same as live trade_journal.maePct.
    assert.ok(tj.maePct !== null && Number(tj.maePct) >= 0, "MAE must be positive magnitude");
    assert.ok(tj.mfePct !== null && Number(tj.mfePct) >= 0);
    assert.equal(tj.exitReason, "tp");
  });
});
