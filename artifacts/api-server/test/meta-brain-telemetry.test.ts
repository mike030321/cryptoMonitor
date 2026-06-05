/**
 * Task #383 — per-slice telemetry honesty test.
 *
 * `monitor.ts` now plumbs real `pnl_state`, `drawdown_state`, and
 * `exposure` per slice (computed from `paperPortfoliosTable` +
 * `paperTradesTable` once per cycle). This test exercises the
 * collector → flush boundary directly:
 *
 *  - When real (non-null) values are passed, the outgoing batch
 *    contains them verbatim and the slice's `anomaly_flags` does
 *    NOT include `missing:pnl_state`, `missing:drawdown_state`, or
 *    `missing:exposure`.
 *  - When null is passed (e.g. cold-start: no closed trades, no open
 *    position), the wire payload coerces to 0.0 AND the
 *    `missing:<field>` flag is appended so the brain can down-weight
 *    learning.
 */
import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";

import {
  collectSlice,
  setPortfolioTelemetry,
  flushTick,
  __resetAdapterState,
} from "../src/lib/meta-brain/adapter";

function startCapturingServer(): Promise<{
  url: string;
  close: () => Promise<void>;
  lastBody: () => unknown;
}> {
  let captured: unknown = null;
  return new Promise((resolve) => {
    const s = http.createServer((req, res) => {
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        try {
          captured = JSON.parse(body);
        } catch {
          captured = body;
        }
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            tick_id: "test-tick",
            trust_multiplier: {
              momentum: 1,
              mean_reversion: 1,
              breakout: 1,
              volatility_forecaster: 1,
              baseline: 1,
            },
            allocation_weight: {
              momentum: 0.2,
              mean_reversion: 0.2,
              breakout: 0.2,
              volatility_forecaster: 0.2,
              baseline: 0.2,
            },
            caution_level: 0.5,
            exploration_budget: 0.1,
            suppress_signal: false,
            defensive_mode: "off",
            suppressed_families: [],
            paused_slices: [],
            reason_codes: [],
          }),
        );
      });
    });
    s.listen(0, "127.0.0.1", () => {
      const port = (s.address() as AddressInfo).port;
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () => new Promise<void>((res) => s.close(() => res())),
        lastBody: () => captured,
      });
    });
  });
}

const baseSlice = {
  coin: "btc",
  timeframe: "5m",
  strategy_family: "momentum" as const,
  edge: 0.01,
  confidence: 0.6,
  calibrated_confidence: 0.6,
  risk_score: 0.4,
  recent_accuracy: 0.55,
  disagreement: 0.1,
  prediction_error: null,
  regime: "trend_up",
  volatility: 0.02,
  correlation_shift: null,
  turnover: null,
  slippage_bps: 5,
};

const basePortfolio = {
  total_drawdown: 0,
  realized_vol: 0,
  concentration: 0,
  leverage: 0,
  liquidity_stress: 0,
  correlation_shift: 0,
  active_risk_budget: 1,
  kill_switch_distance: 1,
};

describe("meta-brain per-slice telemetry (Task #383)", () => {
  let srv: Awaited<ReturnType<typeof startCapturingServer>> | null = null;

  beforeEach(async () => {
    __resetAdapterState();
    srv = await startCapturingServer();
    process.env.META_BRAIN_ENABLED = "1";
    process.env.ML_ENGINE_URL = srv.url;
  });

  afterEach(async () => {
    delete process.env.META_BRAIN_ENABLED;
    delete process.env.ML_ENGINE_URL;
    if (srv) await srv.close();
    srv = null;
  });

  it("real pnl_state / drawdown_state / exposure pass through with no missing flags", async () => {
    collectSlice({
      ...baseSlice,
      pnl_state: 0.0123,
      drawdown_state: 0.0456,
      exposure: 0.18,
    });
    setPortfolioTelemetry(basePortfolio);
    await flushTick();

    const body = srv!.lastBody() as {
      slices: Array<{
        pnl_state: number;
        drawdown_state: number;
        exposure: number;
        anomaly_flags: string[];
      }>;
    };
    assert.equal(body.slices.length, 1);
    const s = body.slices[0];
    assert.equal(s.pnl_state, 0.0123);
    assert.equal(s.drawdown_state, 0.0456);
    assert.equal(s.exposure, 0.18);
    assert.ok(
      !s.anomaly_flags.includes("missing:pnl_state"),
      "missing:pnl_state must not be flagged when real value present",
    );
    assert.ok(
      !s.anomaly_flags.includes("missing:drawdown_state"),
      "missing:drawdown_state must not be flagged when real value present",
    );
    assert.ok(
      !s.anomaly_flags.includes("missing:exposure"),
      "missing:exposure must not be flagged when real value present",
    );
  });

  it("null pnl_state / drawdown_state / exposure → 0.0 wire value with missing flags", async () => {
    collectSlice({
      ...baseSlice,
      pnl_state: null,
      drawdown_state: null,
      exposure: null,
    });
    setPortfolioTelemetry(basePortfolio);
    await flushTick();

    const body = srv!.lastBody() as {
      slices: Array<{
        pnl_state: number;
        drawdown_state: number;
        exposure: number;
        anomaly_flags: string[];
      }>;
    };
    const s = body.slices[0];
    assert.equal(s.pnl_state, 0);
    assert.equal(s.drawdown_state, 0);
    assert.equal(s.exposure, 0);
    assert.ok(s.anomaly_flags.includes("missing:pnl_state"));
    assert.ok(s.anomaly_flags.includes("missing:drawdown_state"));
    assert.ok(s.anomaly_flags.includes("missing:exposure"));
  });
});
