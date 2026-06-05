import { describe, expect, test, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { ShortTfCalibrationVerdictCard } from "@/components/short-tf-calibration-verdict-card";
import type {
  ShortTfCalibrationVerdictResponse,
  ShortTfCalibrationVerdictState,
} from "@/hooks/use-news";

// Task #599 — render guard for the dashboard card.
//
// Pins the three operationally meaningful states (ok, error, unknown)
// so a future refactor of the hook payload or the state-pill mapping
// cannot silently blank the dashboard.

vi.mock("@/hooks/use-news", async () => {
  const actual =
    await vi.importActual<typeof import("@/hooks/use-news")>(
      "@/hooks/use-news",
    );
  return {
    ...actual,
    useShortTfCalibrationVerdict: vi.fn(),
  };
});

const { useShortTfCalibrationVerdict } = await import("@/hooks/use-news");
const useShortTfCalibrationVerdictMock = vi.mocked(
  useShortTfCalibrationVerdict,
);

function renderCard() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <ShortTfCalibrationVerdictCard />
    </QueryClientProvider>,
  );
}

function makeResponse(
  state: ShortTfCalibrationVerdictState,
  overrides: Partial<ShortTfCalibrationVerdictResponse> = {},
): ShortTfCalibrationVerdictResponse {
  const baseShortTf =
    state === "unknown"
      ? null
      : {
          lastStatus: state === "ok" ? "ok" : state,
          lastAttemptAt: "2026-04-29T04:00:00Z",
          lastSuccessAt: state === "ok" ? "2026-04-29T04:00:00Z" : null,
          lastError: state === "ok" ? null : "boom",
          lastElapsedSeconds: 12.5,
          triggerTimeframes: ["1h"],
          timeoutSeconds: 600,
          command: "python -m scripts.task592_parallel_stage2",
          lastMdPath:
            state === "ok"
              ? "reports/20260429T040000Z-task592-1h2h-stage2-verdict.md"
              : null,
          lastJsonPath:
            state === "ok"
              ? "reports/20260429T040000Z-task592-1h2h-stage2-verdict.json"
              : null,
          summary:
            state === "ok"
              ? {
                  capturedAt: "2026-04-29T04:00:00Z",
                  roundTripCostPct: 0.003,
                  timeframesSubset: ["1h", "2h"],
                  wallTimeSeconds: 11.2,
                  nWorkers: 2,
                  nWorkUnits: 18,
                  off: {
                    nSlices: 9,
                    nPassingGate: 5,
                    nInTradeShareBand: 4,
                    meanTradeShare: 0.62,
                    meanDaLift: 0.0123,
                    sumPnlPctTotalAug: 1.4,
                  },
                  on: {
                    nSlices: 9,
                    nPassingGate: 6,
                    nInTradeShareBand: 5,
                    meanTradeShare: 0.71,
                    meanDaLift: 0.0234,
                    sumPnlPctTotalAug: 2.7,
                  },
                }
              : null,
        };
  return {
    state,
    statusFileExists: state !== "unknown",
    statusReadError: null,
    shortTf: baseShortTf,
    markdownPath:
      state === "ok"
        ? "reports/20260429T040000Z-task592-1h2h-stage2-verdict.md"
        : null,
    jsonPath:
      state === "ok"
        ? "reports/20260429T040000Z-task592-1h2h-stage2-verdict.json"
        : null,
    markdownTail: state === "ok" ? "## task #592 stage-2 calibration\n…" : null,
    markdownReadError: null,
    fetchedAt: "2026-04-29T04:00:01Z",
    ...overrides,
  };
}

describe("ShortTfCalibrationVerdictCard", () => {
  beforeEach(() => {
    useShortTfCalibrationVerdictMock.mockReset();
  });
  afterEach(() => cleanup());

  test("renders OK state with stage aggregates and OK pill", () => {
    useShortTfCalibrationVerdictMock.mockReturnValue({
      data: makeResponse("ok"),
      isLoading: false,
      isError: false,
      error: null,
    } as ReturnType<typeof useShortTfCalibrationVerdict>);

    renderCard();

    const card = screen.getByTestId("short-tf-calibration-verdict-card");
    expect(card).toBeTruthy();
    expect(card.getAttribute("data-verdict-state")).toBe("ok");

    const pill = screen.getByTestId("short-tf-calibration-verdict-state");
    expect(pill.textContent).toMatch(/OK/);

    expect(
      screen.getByTestId("short-tf-calibration-verdict-stage-off"),
    ).toBeTruthy();
    expect(
      screen.getByTestId("short-tf-calibration-verdict-stage-on"),
    ).toBeTruthy();
    expect(
      screen.getByTestId("short-tf-calibration-verdict-details-toggle"),
    ).toBeTruthy();
  });

  test("renders ERROR state with error pill and surfaces lastError", () => {
    useShortTfCalibrationVerdictMock.mockReturnValue({
      data: makeResponse("error"),
      isLoading: false,
      isError: false,
      error: null,
    } as ReturnType<typeof useShortTfCalibrationVerdict>);

    renderCard();

    const card = screen.getByTestId("short-tf-calibration-verdict-card");
    expect(card.getAttribute("data-verdict-state")).toBe("error");

    const pill = screen.getByTestId("short-tf-calibration-verdict-state");
    expect(pill.textContent).toMatch(/ERROR/);

    expect(
      screen.getByTestId("short-tf-calibration-verdict-error").textContent,
    ).toMatch(/boom/);
  });

  test("renders UNKNOWN state when no shortTf block has been written yet", () => {
    useShortTfCalibrationVerdictMock.mockReturnValue({
      data: makeResponse("unknown"),
      isLoading: false,
      isError: false,
      error: null,
    } as ReturnType<typeof useShortTfCalibrationVerdict>);

    renderCard();

    const card = screen.getByTestId("short-tf-calibration-verdict-card");
    expect(card.getAttribute("data-verdict-state")).toBe("unknown");

    const pill = screen.getByTestId("short-tf-calibration-verdict-state");
    expect(pill.textContent).toMatch(/(N\/A|UNKNOWN)/i);

    // No summary block when shortTf is null.
    expect(
      screen.queryByTestId("short-tf-calibration-verdict-stage-off"),
    ).toBeNull();
    expect(
      screen.queryByTestId("short-tf-calibration-verdict-stage-on"),
    ).toBeNull();

    // Toggle still rendered so operators can confirm there's no report yet.
    expect(
      screen.getByTestId("short-tf-calibration-verdict-details-toggle"),
    ).toBeTruthy();
  });
});
