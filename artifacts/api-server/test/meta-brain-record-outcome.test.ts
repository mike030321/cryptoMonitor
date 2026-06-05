/**
 * Task #381 step 5 — record_outcome roundtrip + retry semantics.
 *
 * Verifies:
 *  - peekTickForTrade does NOT consume; binding survives a failed call
 *  - on `{ok: true}` the binding is cleared
 *  - shadow tick_ids are bound and the wire format strips the prefix
 *  - record_outcome on a `neutral:*` tick is rejected before the wire
 */
import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";

import {
  bindTradeToTick,
  clearTickBinding,
  peekTickForTrade,
  sendRecordOutcome,
  __resetAdapterState,
} from "../src/lib/meta-brain/adapter";
import { postRecordOutcome } from "../src/lib/meta-brain/client";

function startServer(handler: http.RequestListener) {
  return new Promise<{ url: string; close: () => Promise<void> }>((resolve) => {
    const s = http.createServer(handler);
    s.listen(0, "127.0.0.1", () => {
      const port = (s.address() as AddressInfo).port;
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () => new Promise<void>((res) => s.close(() => res())),
      });
    });
  });
}

const baseOutcome = {
  realized_pnl: 0.01,
  realized_drawdown: 0.005,
  realized_stability: 0.7,
  turnover_cost: 0.001,
  action_churn: null,
  correct_defense: null,
  correct_suppression: null,
  missed_edge_cost: null,
};

describe("meta-brain record_outcome (Task #381)", () => {
  beforeEach(() => {
    __resetAdapterState();
    process.env.META_BRAIN_ENABLED = "1";
  });
  afterEach(() => {
    delete process.env.META_BRAIN_ENABLED;
    delete process.env.ML_ENGINE_URL;
  });

  it("peekTickForTrade does not consume; clear is explicit", () => {
    bindTradeToTick(42, "real-uuid-foo");
    assert.equal(peekTickForTrade(42), "real-uuid-foo");
    assert.equal(peekTickForTrade(42), "real-uuid-foo"); // still there
    clearTickBinding(42);
    assert.equal(peekTickForTrade(42), undefined);
  });

  it("neutral tick_id is never bound", () => {
    bindTradeToTick(7, "neutral:disabled");
    assert.equal(peekTickForTrade(7), undefined);
  });

  it("shadow tick_id IS bound (so record_outcome can close the loop)", () => {
    bindTradeToTick(8, "shadow:real-uuid");
    assert.equal(peekTickForTrade(8), "shadow:real-uuid");
  });

  it("postRecordOutcome refuses to send on a neutral: tick", async () => {
    const ok = await postRecordOutcome({
      tick_id: "neutral:foo",
      timeframe: "1h",
      outcome: baseOutcome,
    });
    assert.equal(ok, false);
  });

  it("shadow: prefix is stripped on the wire", async () => {
    let receivedTickId: string | undefined;
    const srv = await startServer((req, res) => {
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        receivedTickId = JSON.parse(body).tick_id;
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
      });
    });
    process.env.ML_ENGINE_URL = srv.url;
    const ok = await postRecordOutcome({
      tick_id: "shadow:real-uuid-XYZ",
      timeframe: "1h",
      outcome: baseOutcome,
    });
    assert.equal(ok, true);
    assert.equal(receivedTickId, "real-uuid-XYZ");
    await srv.close();
  });

  it("sendRecordOutcome clears binding only on {ok: true}", async () => {
    let attempt = 0;
    const srv = await startServer((req, res) => {
      attempt++;
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        if (attempt === 1) {
          // First call → server returns ok:false (e.g. tick not in cache)
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify({ ok: false, reason: "tick_id_not_in_cache" }));
        } else {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify({ ok: true }));
        }
      });
    });
    process.env.ML_ENGINE_URL = srv.url;

    bindTradeToTick(99, "real-uuid-retry");
    await sendRecordOutcome(
      { tick_id: "real-uuid-retry", timeframe: "1h", outcome: baseOutcome },
      { tradeId: 99 },
    );
    // First attempt failed → binding must still be there for retry.
    assert.equal(peekTickForTrade(99), "real-uuid-retry");

    await sendRecordOutcome(
      { tick_id: "real-uuid-retry", timeframe: "1h", outcome: baseOutcome },
      { tradeId: 99 },
    );
    // Second attempt succeeded → binding cleared.
    assert.equal(peekTickForTrade(99), undefined);

    await srv.close();
  });

  it("null outcome sub-fields wire as 0.0", async () => {
    let body: any;
    const srv = await startServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        body = JSON.parse(raw);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
      });
    });
    process.env.ML_ENGINE_URL = srv.url;
    await postRecordOutcome({
      tick_id: "real-uuid-null",
      timeframe: "1h",
      outcome: baseOutcome,
    });
    assert.equal(body.outcome.action_churn, 0);
    assert.equal(body.outcome.correct_defense, 0);
    assert.equal(body.outcome.correct_suppression, 0);
    assert.equal(body.outcome.missed_edge_cost, 0);
    await srv.close();
  });
});
