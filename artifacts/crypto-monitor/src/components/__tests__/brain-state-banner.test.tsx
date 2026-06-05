/**
 * Task #681 — brain-state-banner: explicit unknown-source fallback.
 *
 * Pins five render branches:
 *  1. brainSource="default"     → existing "never been enabled" copy
 *  2. brainSource="env"         → existing "env force-off" copy
 *  3. brainSource="auto_revert" → existing "auto-reverted" copy
 *  4. brainSource="manual"      → existing "operator flipped it OFF" copy
 *  5. brainSource="some_unrecognized_value"
 *                                → new "source unknown" fallback that
 *                                  surfaces the raw string verbatim
 *
 * Cases 1-4 are byte-identical to today's behaviour (Task #532 / C-3).
 * Case 5 is the new behaviour added in Task #681.
 *
 * Also asserts the meta-line at the bottom of the banner continues to
 * print `source: <raw>` for the unknown case (the existing operator
 * fallback the audit pointed out at lines 172-174 in the pre-#681
 * file).
 */
import { describe, expect, test, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { BrainStateBanner } from "@/components/brain-state-banner";
import type { BrainRuntimeStatus } from "@/hooks/use-news";

vi.mock("wouter", () => ({
  Link: ({ children, ...rest }: { children: React.ReactNode } & Record<string, unknown>) => {
    const { href, ...attrs } = rest as { href?: string };
    return (
      <a href={href} {...(attrs as Record<string, unknown>)}>
        {children}
      </a>
    );
  },
}));

vi.mock("@/hooks/use-news", async () => {
  const actual = await vi.importActual<typeof import("@/hooks/use-news")>(
    "@/hooks/use-news",
  );
  return {
    ...actual,
    useBrainRuntimeStatus: vi.fn(),
  };
});

const { useBrainRuntimeStatus } = await import("@/hooks/use-news");

function setStatus(
  brainSource: string,
  promotionGateRetries?: BrainRuntimeStatus["promotionGateRetries"],
) {
  vi.mocked(useBrainRuntimeStatus).mockReturnValue({
    data: {
      state: "offline_disabled",
      brainEnabled: false,
      // The hook contract is a fixed enum today; the banner widens
      // its local typedef to `string` so unknown values fall through
      // to the new "source unknown" copy. Cast through unknown so
      // this test can exercise the unknown-source case without
      // editing the hook contract (Option A in task-681).
      brainSource: brainSource as BrainRuntimeStatus["brainSource"],
      mlAvailabilitySnapshotReady: true,
      recentAbstainReasons: {},
      recentNonAbstainCount: 0,
      lastSuccessfulDecisionAt: null,
      currentRunDir: null,
      windowMinutes: 30,
      promotionGateRetries,
    },
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof useBrainRuntimeStatus>);
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  cleanup();
});

describe("BrainStateBanner — known source copy is unchanged (Task #532 regression pin)", () => {
  test("source=default → 'never been enabled' headline", () => {
    setStatus("default");
    render(<BrainStateBanner />);
    expect(screen.getByTestId("brain-state-banner")).toBeInTheDocument();
    expect(
      screen.getByText(/QUANT DISABLED — quant brain has never been enabled/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /There is no `app_settings\.quant_brain_enabled=true` row in the database\./,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(/source: default/)).toBeInTheDocument();
  });

  test("source=env → 'env force-off' headline", () => {
    setStatus("env");
    render(<BrainStateBanner />);
    expect(
      screen.getByText(/QUANT DISABLED — env force-off/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/QUANT_BRAIN_FORCE_OFF=1 in the environment\./),
    ).toBeInTheDocument();
    expect(screen.getByText(/source: env/)).toBeInTheDocument();
  });

  test("source=auto_revert → 'auto-reverted' headline", () => {
    setStatus("auto_revert");
    render(<BrainStateBanner />);
    expect(
      screen.getByText(/QUANT DISABLED — auto-reverted/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Auto-revert tripped \(drift_low or consecutive losses\)/,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(/source: auto_revert/)).toBeInTheDocument();
  });

  test("source=manual → 'operator flipped it OFF' headline", () => {
    setStatus("manual");
    render(<BrainStateBanner />);
    expect(
      screen.getByText(/QUANT DISABLED — operator flipped it OFF/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/An operator set quant_brain_enabled=false\./),
    ).toBeInTheDocument();
    expect(screen.getByText(/source: manual/)).toBeInTheDocument();
  });
});

describe("BrainStateBanner — unknown source fallback (Task #681)", () => {
  test("source='some_unrecognized_value' → 'source unknown' headline + raw value in subtitle", () => {
    setStatus("some_unrecognized_value");
    render(<BrainStateBanner />);

    // Title contains the literal text "source unknown" (case-insensitive).
    const banner = screen.getByTestId("brain-state-banner");
    expect(banner.textContent ?? "").toMatch(/source unknown/i);

    // Subtitle includes the raw brainSource string verbatim.
    expect(
      screen.getByText(/Reported source: some_unrecognized_value/),
    ).toBeInTheDocument();

    // Meta-line at the bottom of the banner still prints the raw
    // `source: <raw>` field — this is the operator fallback the
    // codex audit pointed out and Task #681 explicitly preserves.
    // Use a regex anchored to the meta-line wording (no "Reported"
    // prefix) so this assertion does not double-match the subtitle.
    expect(banner.textContent ?? "").toMatch(
      /(?:^|[^a-zA-Z])source: some_unrecognized_value/,
    );

    // Headline does NOT silently use the generic
    // "quant_brain_enabled is false" copy as the headline.
    expect(banner.textContent ?? "").not.toMatch(
      /QUANT DISABLED — quant brain disabled/,
    );
  });

  test("future enum value (e.g. 'shadow') is surfaced verbatim", () => {
    setStatus("shadow");
    render(<BrainStateBanner />);
    const banner = screen.getByTestId("brain-state-banner");
    expect(banner.textContent ?? "").toMatch(/source unknown/i);
    expect(
      screen.getByText(/Reported source: shadow/),
    ).toBeInTheDocument();
  });
});

// ────────────────────────────────────────────────────────────────────
// Task #686 — promotion-gate retry chip on the offline banner
// ────────────────────────────────────────────────────────────────────
//
// Pins three render branches:
//  1. `promotionGateRetries` undefined (older api-server) → no chip
//  2. `count === 0`                                       → no chip
//  3. `count > 0`                                         → chip with
//        count + most-recent-reason verbatim
describe("BrainStateBanner — promotion-gate retry chip (Task #686)", () => {
  test("no chip when promotionGateRetries is missing on the wire", () => {
    setStatus("manual", undefined);
    render(<BrainStateBanner />);
    expect(
      screen.queryByTestId("brain-state-banner-promotion-gate-retries"),
    ).toBeNull();
  });

  test("no chip when count is zero", () => {
    setStatus("manual", {
      count: 0,
      windowMs: 60 * 60 * 1000,
      mostRecentAt: null,
      mostRecentReason: null,
      mostRecentAttempt: null,
    });
    render(<BrainStateBanner />);
    expect(
      screen.queryByTestId("brain-state-banner-promotion-gate-retries"),
    ).toBeNull();
  });

  test("chip surfaces count + most-recent retry_failure_reason verbatim", () => {
    setStatus("manual", {
      count: 3,
      windowMs: 60 * 60 * 1000,
      mostRecentAt: new Date().toISOString(),
      mostRecentReason: "non_2xx_status_503",
      mostRecentAttempt: 3,
    });
    render(<BrainStateBanner />);
    const chip = screen.getByTestId("brain-state-banner-promotion-gate-retries");
    expect(chip).toBeInTheDocument();
    expect(chip.getAttribute("data-promotion-gate-retry-count")).toBe("3");
    expect(chip.textContent ?? "").toMatch(/promotion-gate retries: 3/);
    expect(chip.textContent ?? "").toMatch(/in last 60m/);
    // Operator must see the literal failure reason so they can tell
    // a network blip apart from an admin-key 401.
    expect(chip.textContent ?? "").toMatch(/non_2xx_status_503/);
    expect(chip.textContent ?? "").toMatch(/attempt 3/);
  });

  test("single-shot retry (count=1) is distinguishable from all-attempts-failed (count=3)", () => {
    setStatus("manual", {
      count: 1,
      windowMs: 60 * 60 * 1000,
      mostRecentAt: new Date().toISOString(),
      mostRecentReason: "network_error",
      mostRecentAttempt: 1,
    });
    render(<BrainStateBanner />);
    const chip = screen.getByTestId("brain-state-banner-promotion-gate-retries");
    expect(chip.getAttribute("data-promotion-gate-retry-count")).toBe("1");
    expect(chip.textContent ?? "").toMatch(/promotion-gate retries: 1/);
    expect(chip.textContent ?? "").toMatch(/network_error/);
  });
});
