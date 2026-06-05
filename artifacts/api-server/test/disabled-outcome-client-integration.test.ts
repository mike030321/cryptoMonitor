/**
 * Task #577 — end-to-end wiring check: when the meta-brain returns
 * `{ok:false, reason:"disabled_role_rejected"}` from
 * `/ml/meta-brain/record-outcome`, the api-server's client must:
 *   1. forward the optional `slice_id` on the wire if provided,
 *   2. push a structured event into the disabled-outcome notifier ring
 *      buffer with the offending tick_id, slice_id and timeframe so
 *      the dashboard banner + webhook payload can show them.
 *
 * This is the "alert me when an outcome lands on a disabled timeframe"
 * contract — the brain's existing `[disabled_outcome_received]` warn
 * stays in the logs, but every rejection now also lands in a place
 * the operator can actually see.
 *
 * NOTE: We deliberately do NOT mutate `shared/timeframe-roles.json`
 * here. The notifier reacts to the brain's RESPONSE (`reason ==
 * "disabled_role_rejected"`), not to what slice_role the api-server
 * resolved on the wire. Mocking the brain's response is enough; that
 * keeps this test isolated from the canonical roles registry and
 * means concurrent test runs (and Task #574's fixture writes) can't
 * race on the same file.
 */
import { describe, it, before, after, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";

import { db, appSettingsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { postRecordOutcome } from "../src/lib/meta-brain/client";
import {
  DISABLED_OUTCOME_NOTIFIER_STATE_KEY,
  __resetRingBuffer,
  __setInlineDispatchDisabled,
  __snapshotRingBuffer,
} from "../src/lib/disabled-outcome-notifier";

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

describe("disabled-outcome notifier — wired into postRecordOutcome (Task #577)", () => {
  before(() => {
    // Inline dispatch is fire-and-forget; suppress it so the unit
    // tests below own the ring buffer and don't bleed write traffic
    // into the persistent dedup state used by other tests.
    __setInlineDispatchDisabled(true);
  });
  after(async () => {
    __setInlineDispatchDisabled(false);
    await db
      .delete(appSettingsTable)
      .where(eq(appSettingsTable.key, DISABLED_OUTCOME_NOTIFIER_STATE_KEY));
  });
  beforeEach(async () => {
    process.env.META_BRAIN_ENABLED = "1";
    __resetRingBuffer();
    delete process.env.DISABLED_OUTCOME_ALERT_WEBHOOK_URL;
    delete process.env.DISABLED_OUTCOME_ALERT_SLACK_WEBHOOK_URL;
    delete process.env.SLACK_WEBHOOK_URL;
    await db
      .delete(appSettingsTable)
      .where(eq(appSettingsTable.key, DISABLED_OUTCOME_NOTIFIER_STATE_KEY));
  });
  afterEach(() => {
    delete process.env.META_BRAIN_ENABLED;
    delete process.env.ML_ENGINE_URL;
  });

  it("a disabled-timeframe rejection lands in the notifier ring buffer with tick_id + timeframe + slice_id", async () => {
    let received: { slice_id?: string; tick_id?: string } = {};
    const srv = await startServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        received = JSON.parse(raw);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: false, reason: "disabled_role_rejected" }));
      });
    });
    process.env.ML_ENGINE_URL = srv.url;

    const ok = await postRecordOutcome({
      tick_id: "uuid-leak-1",
      timeframe: "5m",
      sliceId: "slice-XYZ",
      outcome: baseOutcome,
    });

    // Brain rejected → caller sees ok=false (so the trade-tick binding
    // is preserved for retries / TTL sweep).
    assert.equal(ok, false);

    // Wire shape carries the slice_id and the tick_id we passed in.
    assert.equal(received.slice_id, "slice-XYZ");
    assert.equal(received.tick_id, "uuid-leak-1");

    // Notifier ring buffer captured the event with all three fields
    // operators need to find the leaking caller.
    const snap = __snapshotRingBuffer();
    assert.equal(snap.length, 1, "exactly one event recorded");
    assert.equal(snap[0].tickId, "uuid-leak-1");
    assert.equal(snap[0].sliceId, "slice-XYZ");
    assert.equal(snap[0].timeframe, "5m");

    await srv.close();
  });

  it("a NON-disabled rejection (e.g. tick_id_not_in_cache) does NOT enter the notifier", async () => {
    const srv = await startServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        void JSON.parse(raw);
        // Routine miss (trade-roled outcome the brain can't find in
        // its tick cache). Must NOT page the disabled-outcome alarm.
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: false, reason: "tick_id_not_in_cache" }));
      });
    });
    process.env.ML_ENGINE_URL = srv.url;

    await postRecordOutcome({
      tick_id: "uuid-missing-1",
      timeframe: "1d",
      outcome: baseOutcome,
    });
    assert.equal(
      __snapshotRingBuffer().length,
      0,
      "non-disabled rejection must not show up in the disabled-outcome notifier",
    );

    await srv.close();
  });

  it("an accepted outcome (ok:true) does NOT enter the notifier", async () => {
    const srv = await startServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        void JSON.parse(raw);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
      });
    });
    process.env.ML_ENGINE_URL = srv.url;

    await postRecordOutcome({
      tick_id: "uuid-fakeok",
      timeframe: "5m",
      outcome: baseOutcome,
    });
    assert.equal(__snapshotRingBuffer().length, 0);

    await srv.close();
  });
});
