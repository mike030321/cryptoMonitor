/**
 * Task #381 step 8 — fallback honesty test.
 *
 * Every documented failure path collapses to a neutral directive:
 *  - disabled (no env vars)
 *  - fetch network failure
 *  - HTTP 5xx
 *  - bad JSON
 *  - schema-invalid
 *  - allocation drift (sum != 1)
 *
 * The trading path must be unaffected: `isNeutralDirective` is true
 * and `getFamilySizeMultiplier` returns 1.0.
 */
import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";

import {
  postEvaluate,
  consumeFallbackCounters,
} from "../src/lib/meta-brain/client";
import {
  isNeutralDirective,
  neutralDirective,
} from "../src/lib/meta-brain/contract";
import {
  __setActiveDirectiveForTest,
  __resetAdapterState,
  getFamilySizeMultiplier,
} from "../src/lib/meta-brain/adapter";

function makeBatch() {
  return {
    slices: [
      {
        coin: "btc",
        timeframe: "5m",
        strategy_family: "momentum" as const,
        edge: 0.01,
        confidence: 0.6,
        calibrated_confidence: 0.6,
        risk_score: 0.4,
        recent_accuracy: 0.55,
        pnl_state: 0,
        drawdown_state: 0,
        disagreement: 0.1,
        prediction_error: 0,
        regime: "trend_up",
        volatility: 0.02,
        correlation_shift: 0,
        exposure: 0.1,
        turnover: 0,
        slippage_bps: 5,
        anomaly_flags: [],
      },
    ],
    portfolio: {
      total_drawdown: 0,
      realized_vol: 0,
      concentration: 0,
      leverage: 0,
      liquidity_stress: 0,
      correlation_shift: 0,
      active_risk_budget: 1,
      kill_switch_distance: 1,
      anomaly_flags: [],
    },
    timestamp: new Date().toISOString(),
  };
}

function startServer(handler: http.RequestListener): Promise<{
  url: string;
  close: () => Promise<void>;
}> {
  return new Promise((resolve) => {
    const s = http.createServer(handler);
    s.listen(0, "127.0.0.1", () => {
      const port = (s.address() as AddressInfo).port;
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () =>
          new Promise<void>((res) => s.close(() => res())),
      });
    });
  });
}

describe("meta-brain fallback paths (Task #381)", () => {
  beforeEach(() => {
    delete process.env.META_BRAIN_ENABLED;
    delete process.env.META_BRAIN_SHADOW;
    delete process.env.ML_ENGINE_URL;
    consumeFallbackCounters();
    __resetAdapterState();
  });
  afterEach(() => {
    delete process.env.META_BRAIN_ENABLED;
    delete process.env.META_BRAIN_SHADOW;
    delete process.env.ML_ENGINE_URL;
  });

  it("disabled → neutral, sizing untouched", async () => {
    const d = await postEvaluate(makeBatch());
    assert.ok(isNeutralDirective(d));
    __setActiveDirectiveForTest(d);
    assert.equal(getFamilySizeMultiplier("momentum"), 1.0);
    assert.equal(consumeFallbackCounters().disabled, 1);
  });

  it("fetch failure → neutral", async () => {
    process.env.META_BRAIN_ENABLED = "1";
    process.env.ML_ENGINE_URL = "http://127.0.0.1:1"; // port 1 → connection refused
    const d = await postEvaluate(makeBatch(), 500);
    assert.ok(isNeutralDirective(d));
  });

  it("HTTP 500 → neutral", async () => {
    const srv = await startServer((_req, res) => {
      res.writeHead(500).end();
    });
    process.env.META_BRAIN_ENABLED = "1";
    process.env.ML_ENGINE_URL = srv.url;
    const d = await postEvaluate(makeBatch());
    assert.ok(isNeutralDirective(d));
    await srv.close();
  });

  it("bad JSON body → neutral", async () => {
    const srv = await startServer((_req, res) => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end("not-json{{");
    });
    process.env.META_BRAIN_ENABLED = "1";
    process.env.ML_ENGINE_URL = srv.url;
    const d = await postEvaluate(makeBatch());
    assert.ok(isNeutralDirective(d));
    await srv.close();
  });

  it("schema-invalid → neutral", async () => {
    const srv = await startServer((_req, res) => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ tick_id: "abc" })); // missing required fields
    });
    process.env.META_BRAIN_ENABLED = "1";
    process.env.ML_ENGINE_URL = srv.url;
    const d = await postEvaluate(makeBatch());
    assert.ok(isNeutralDirective(d));
    await srv.close();
  });

  it("allocation drift (sum != 1) → neutral", async () => {
    const drifted = {
      tick_id: "real-uuid-1",
      trust_multiplier: { momentum: 1, mean_reversion: 1, breakout: 1, volatility_forecaster: 1, baseline: 1 },
      allocation_weight: { momentum: 0.5, mean_reversion: 0.5, breakout: 0.5, volatility_forecaster: 0.5, baseline: 0.5 },
      caution_level: 0.5,
      exploration_budget: 0.1,
      suppress_signal: false,
      defensive_mode: "off",
      suppressed_families: [],
      paused_slices: [],
      reason_codes: [],
    };
    const srv = await startServer((_req, res) => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(drifted));
    });
    process.env.META_BRAIN_ENABLED = "1";
    process.env.ML_ENGINE_URL = srv.url;
    const d = await postEvaluate(makeBatch());
    assert.ok(isNeutralDirective(d), "allocation drift must collapse to neutral");
    await srv.close();
  });

  it("shadow mode → tick_id prefixed shadow:, treated as neutral for sizing", async () => {
    const valid = {
      tick_id: "real-uuid-shadow",
      trust_multiplier: { momentum: 1.2, mean_reversion: 0.9, breakout: 1.0, volatility_forecaster: 1.0, baseline: 0.9 },
      allocation_weight: { momentum: 0.4, mean_reversion: 0.15, breakout: 0.15, volatility_forecaster: 0.15, baseline: 0.15 },
      caution_level: 0.5,
      exploration_budget: 0.1,
      suppress_signal: false,
      defensive_mode: "off",
      suppressed_families: [],
      paused_slices: [],
      reason_codes: [],
    };
    const srv = await startServer((_req, res) => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(valid));
    });
    process.env.META_BRAIN_SHADOW = "1";
    process.env.ML_ENGINE_URL = srv.url;
    const d = await postEvaluate(makeBatch());
    assert.match(d.tick_id, /^shadow:/);
    assert.ok(isNeutralDirective(d), "shadow tick_id must be treated as neutral");
    __setActiveDirectiveForTest(d);
    assert.equal(getFamilySizeMultiplier("momentum"), 1.0);
    await srv.close();
  });

  it("neutralDirective() always classifies as neutral", () => {
    for (const cause of ["disabled", "fetch_failed", "schema_invalid", "boot"]) {
      assert.ok(isNeutralDirective(neutralDirective(cause)));
    }
  });
});
