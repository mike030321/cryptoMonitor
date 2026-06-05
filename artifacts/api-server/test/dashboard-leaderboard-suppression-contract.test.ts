/**
 * Task #532 / Rev 2.2 — contract test for `/crypto/dashboard`
 * leaderboard suppression.
 *
 * The previous behaviour was: when no agent had enough non-abstain
 * resolved samples to be ranked, the route still returned a synthetic
 * `bestAgent: { id: 0, name: "N/A", accuracy: 0, streak: 0, ... }`
 * placeholder. Operators read that as a real agent ranked first on
 * the leaderboard.
 *
 * Phase 7 of the audit replaced that placeholder with `null` and a
 * `signal: "insufficient_signal"` discriminator so the dashboard can
 * hide the leaderboard outright. Rev 2.2 additionally null-strips
 * `streak`, `streakType`, and `score` from the *non-null* leaderboard
 * payloads so the abstain-era contaminated badges cannot ride along
 * with the recomputed accuracy.
 *
 * This test runs both as a static contract over the source and as a
 * runtime probe against the live api-server (skipped if the server is
 * not on the expected port — it is in this workspace's standard dev
 * stack).
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROUTE_FILE = join(__dirname, "..", "src", "routes", "crypto", "index.ts");

describe("Task #532 — /crypto/dashboard leaderboard suppression contract", () => {
  const src = readFileSync(ROUTE_FILE, "utf8");

  it("source declares the `signal` discriminator and the two valid values", () => {
    assert.match(
      src,
      /dashboardSignal\s*:\s*"ok"\s*\|\s*"insufficient_signal"/,
      "expected `dashboardSignal: \"ok\" | \"insufficient_signal\"` in /dashboard route source",
    );
    assert.match(
      src,
      /dashboardSignal\s*=\s*"ok"/,
      "expected dashboardSignal to be set to 'ok' when there is a real-signal candidate set",
    );
  });

  it("source returns null bestAgent/worstAgent via cleanRankingPayload (no fake placeholder)", () => {
    assert.match(
      src,
      /cleanRankingPayload\(bestAgent\)/,
      "bestAgent must flow through cleanRankingPayload (which returns null when input is null)",
    );
    assert.match(
      src,
      /cleanRankingPayload\(worstAgent\)/,
      "worstAgent must flow through cleanRankingPayload",
    );
    // Verify cleanRankingPayload null-strips the abstain-era badges
    assert.match(
      src,
      /streak:\s*null,\s*streakType:\s*null,\s*score:\s*null/,
      "cleanRankingPayload must null-strip streak / streakType / score so the abstain-era values cannot ride the leaderboard payload",
    );
  });

  it("live /crypto/dashboard returns null leaderboard + signal=insufficient_signal in the current brain-offline state", async (t) => {
    const port = process.env.API_PORT ?? "8080";
    let res: Response;
    try {
      res = await fetch(`http://localhost:${port}/api/crypto/dashboard`);
    } catch {
      t.skip(`api-server not running on :${port}; skipping live runtime probe`);
      return;
    }
    assert.equal(res.status, 200);
    const body = (await res.json()) as Record<string, unknown>;
    // The brain is OFF in this workspace by design; therefore there
    // must be zero ranked agents, which the contract requires to
    // surface as null + insufficient_signal.
    assert.equal(
      body.bestAgent,
      null,
      `expected dashboard.bestAgent === null when brain is offline; got ${JSON.stringify(body.bestAgent)}`,
    );
    assert.equal(
      body.worstAgent,
      null,
      `expected dashboard.worstAgent === null when brain is offline; got ${JSON.stringify(body.worstAgent)}`,
    );
    assert.equal(
      body.signal,
      "insufficient_signal",
      `expected dashboard.signal === "insufficient_signal"; got ${JSON.stringify(body.signal)}`,
    );
  });
});

describe("Task #532 — /crypto/meta-brain/status nullable-on-failure contract", () => {
  const src = readFileSync(ROUTE_FILE, "utf8");

  it("closedTrades24h is typed `number | null` and set to `null` on DB failure", () => {
    assert.match(
      src,
      /let\s+closedTrades24h\s*:\s*number\s*\|\s*null/,
      "closedTrades24h must be typed `number | null` so it can be null on probe failure",
    );
    assert.match(
      src,
      /catch\s*\{\s*closedTrades24h\s*=\s*null;?\s*\}/,
      "catch path must set closedTrades24h = null (not 0) per the no-fake-zero contract",
    );
  });

  it("trustStateChanges24h is typed `number | null` and starts as null when stats probe failed", () => {
    assert.match(
      src,
      /let\s+trustStateChanges24h\s*:\s*number\s*\|\s*null\s*=\s*stats\s*===\s*null\s*\?\s*null\s*:\s*0/,
      "trustStateChanges24h must initialise to null when stats === null (ml-engine /stats unreachable)",
    );
  });
});
