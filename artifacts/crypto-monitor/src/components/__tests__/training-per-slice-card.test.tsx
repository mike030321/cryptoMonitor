import { describe, expect, test, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";

import { TrainingPerSliceCard } from "@/components/training-per-slice-card";

// Task #615 — render guard for the live-gated replay surface on the
// per-slice training card. Verifies the four-way verdict pill, the
// loose-vs-live PnL line, and the dominant rejection reason that
// appears for bleeding/dormant slices. Uses a fetch mock so the test
// is self-contained.

interface FetchCall {
  url: string;
  init?: RequestInit;
}

function jsonResponse(body: unknown, init: ResponseInit = { status: 200 }): Response {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: { "content-type": "application/json", ...(init.headers ?? {}) },
  });
}

function installFetch(handler: (url: string) => Response): {
  calls: FetchCall[];
  restore: () => void;
} {
  const calls: FetchCall[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    calls.push({ url, init });
    return handler(url);
  }) as typeof fetch;
  return {
    calls,
    restore: () => {
      globalThis.fetch = original;
    },
  };
}

const TRAINING_REPORT = {
  status: "ok",
  generated_at: "2026-04-29T10:00:00Z",
  timeframes: {
    "5m": {
      per_coin: {
        bonk: {
          per_class_holdout_breakdown: { UP: 1, DOWN: 1, STABLE: 1 },
          per_class_accuracy: {
            UP: { n: 1, accuracy: 0.5 },
            DOWN: { n: 1, accuracy: 0.5 },
            STABLE: { n: 1, accuracy: 0.5 },
          },
          pnl_after_fees: {
            n_trades: 4,
            net_pct_mean: -0.05,
            net_pct_total: -1.23,
            gross_pct_mean: 0.02,
            round_trip_cost_pct: 0.07,
            win_rate: 0.25,
          },
        },
      },
    },
    "1h": {
      per_coin: {
        pepe: {
          per_class_holdout_breakdown: { UP: 1, DOWN: 1, STABLE: 1 },
          per_class_accuracy: {
            UP: { n: 1, accuracy: 0.6 },
            DOWN: { n: 1, accuracy: 0.6 },
            STABLE: { n: 1, accuracy: 0.6 },
          },
          pnl_after_fees: {
            n_trades: 12,
            net_pct_mean: 0.025,
            net_pct_total: 0.45,
            gross_pct_mean: 0.05,
            round_trip_cost_pct: 0.025,
            win_rate: 0.6,
          },
        },
      },
    },
  },
  verification: { per_slice: [] },
};

const LIVE_GATED_OK = {
  status: "ok",
  run_dir: "training_run_20260429T000000Z",
  generated_at: "2026-04-29T00:00:00Z",
  per_slice: {
    "bonk/5m": {
      loose_post_fee_pct_total: -1.23,
      live_trade_count: 0,
      live_net_pnl_pct: null,
      dominant_rejection_reason: "directional_edge",
      live_replay_status: "ok",
      economic_verdict: "dormant",
      economic_verdict_phrase: "dormant / no-edge under production gates",
    },
    "pepe/1h": {
      loose_post_fee_pct_total: 0.45,
      live_trade_count: 12,
      live_net_pnl_pct: 0.30,
      dominant_rejection_reason: null,
      live_replay_status: "ok",
      economic_verdict: "tradeable",
      economic_verdict_phrase: "tradeable / positive under production gates",
    },
  },
  verdict_counts: { bleeding: 0, dormant: 1, tradeable: 1, inconclusive: 0 },
  bleeding_slices: [],
  dormant_slices: ["bonk/5m"],
  tradeable_slices: ["pepe/1h"],
};

describe("TrainingPerSliceCard — live-gated replay surface (Task #615)", () => {
  let restore: (() => void) | null = null;

  beforeEach(() => {
    vi.useRealTimers();
  });

  afterEach(() => {
    cleanup();
    restore?.();
    restore = null;
    vi.restoreAllMocks();
  });

  test("renders verdict pill, loose-vs-live PnL, and dominant rejection for a dormant slice", async () => {
    const { restore: r } = installFetch((url) => {
      if (url.endsWith("/api/crypto/training/live-gated-replay")) {
        return jsonResponse(LIVE_GATED_OK);
      }
      if (url.endsWith("/api/crypto/quant-training-report")) {
        return jsonResponse(TRAINING_REPORT);
      }
      return jsonResponse({ status: "missing" });
    });
    restore = r;

    render(<TrainingPerSliceCard />);

    // Dormant pill + loose-vs-live PnL line + dominant rejection reason.
    await waitFor(() => {
      expect(
        screen.getByTestId("training-per-slice-live-gated-verdict-5m-bonk"),
      ).toHaveTextContent("dormant");
    });
    expect(
      screen.getByTestId("training-per-slice-live-gated-pnl-5m-bonk"),
    ).toHaveTextContent("loose -1.23% · live —");
    expect(
      screen.getByTestId("training-per-slice-live-gated-rejection-5m-bonk"),
    ).toHaveTextContent("rejected: directional_edge");

    // Tradeable pill + loose-vs-live PnL line + no rejection chip.
    expect(
      screen.getByTestId("training-per-slice-live-gated-verdict-1h-pepe"),
    ).toHaveTextContent("tradeable");
    expect(
      screen.getByTestId("training-per-slice-live-gated-pnl-1h-pepe"),
    ).toHaveTextContent("loose +0.45% · live +0.30% (n=12)");
    expect(
      screen.queryByTestId("training-per-slice-live-gated-rejection-1h-pepe"),
    ).toBeNull();

    // Footer shows the per-verdict counts and the source run folder.
    const footer = screen.getByTestId("training-per-slice-live-gated-footer");
    expect(footer).toHaveTextContent("tradeable 1");
    expect(footer).toHaveTextContent("dormant 1");
    expect(footer).toHaveTextContent("bleeding 0");
    expect(footer).toHaveTextContent("training_run_20260429T000000Z");
  });

  test("renders an em-dash and a missing footer when no live-gated block is on disk", async () => {
    const { restore: r } = installFetch((url) => {
      if (url.endsWith("/api/crypto/training/live-gated-replay")) {
        return jsonResponse({ status: "missing" });
      }
      if (url.endsWith("/api/crypto/quant-training-report")) {
        return jsonResponse(TRAINING_REPORT);
      }
      return jsonResponse({ status: "missing" });
    });
    restore = r;

    render(<TrainingPerSliceCard />);

    await waitFor(() => {
      expect(
        screen.getByTestId("training-per-slice-live-gated-empty-5m-bonk"),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByTestId("training-per-slice-live-gated-footer-missing"),
    ).toHaveTextContent("no campaign run");
    // Verdict pill must not render when there is no entry for the slice.
    expect(
      screen.queryByTestId("training-per-slice-live-gated-verdict-5m-bonk"),
    ).toBeNull();
  });

  test("renders an empty footer when the latest summary predates Task #613", async () => {
    const { restore: r } = installFetch((url) => {
      if (url.endsWith("/api/crypto/training/live-gated-replay")) {
        return jsonResponse({
          status: "empty",
          run_dir: "training_run_20260101T000000Z",
          per_slice: {},
        });
      }
      if (url.endsWith("/api/crypto/quant-training-report")) {
        return jsonResponse(TRAINING_REPORT);
      }
      return jsonResponse({ status: "missing" });
    });
    restore = r;

    render(<TrainingPerSliceCard />);

    await waitFor(() => {
      expect(
        screen.getByTestId("training-per-slice-live-gated-footer-empty"),
      ).toHaveTextContent("training_run_20260101T000000Z");
    });
  });
});
