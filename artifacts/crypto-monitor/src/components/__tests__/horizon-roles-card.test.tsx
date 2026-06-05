import { describe, expect, test, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  HorizonRolesCard,
  badgeClassesFor,
  badgeTextFor,
  isClickableEvidence,
  normalizeForRender,
  summaryLine,
  TIMEFRAME_ORDER,
  tradeBadgeText,
} from "@/components/horizon-roles-card";
import type {
  TimeframeRoleEntry,
  TimeframeRolesDoc,
} from "@/hooks/use-timeframe-roles";

vi.mock("@/hooks/use-timeframe-roles", async () => {
  const actual = await vi.importActual<
    typeof import("@/hooks/use-timeframe-roles")
  >("@/hooks/use-timeframe-roles");
  return {
    ...actual,
    useTimeframeRoles: vi.fn(),
  };
});

vi.mock("@/hooks/use-news", async () => {
  const actual = await vi.importActual<typeof import("@/hooks/use-news")>(
    "@/hooks/use-news",
  );
  return {
    ...actual,
    useBrainRuntimeStatus: vi.fn(),
  };
});

const { useTimeframeRoles } = await import("@/hooks/use-timeframe-roles");
const { useBrainRuntimeStatus } = await import("@/hooks/use-news");

function tfEntry(overrides: Partial<TimeframeRoleEntry> = {}): TimeframeRoleEntry {
  return {
    role: "context",
    context_subkind: "filter",
    disabled_reason: null,
    reason: "stub reason",
    evidence_ref: "shared/evidence/path.md",
    last_reviewed_at: "2026-04-28T12:00:00.000Z",
    promoted_slices_in_tf: [],
    ...overrides,
  };
}

function makeMixedDoc(): TimeframeRolesDoc {
  return {
    schema_version: 1,
    generated_at: "2026-04-28T17:30:00.000Z",
    generated_by_task: "test",
    timeframes: {
      "1m": tfEntry({
        role: "trade",
        context_subkind: null,
        reason: "1m trade reason",
        promoted_slices_in_tf: ["bonk:1m:v3"],
      }),
      "5m": tfEntry({
        role: "disabled",
        context_subkind: null,
        disabled_reason: "by_data",
        reason: "5m disabled reason",
      }),
      "1h": tfEntry({
        role: "shadow",
        context_subkind: null,
        reason: "1h shadow reason",
      }),
      "2h": tfEntry({
        role: "context",
        context_subkind: "filter",
        reason: "2h context filter reason",
      }),
      "6h": tfEntry({
        role: "context",
        context_subkind: "filter",
        reason: "6h context filter reason",
      }),
      "1d": tfEntry({
        role: "context",
        context_subkind: "regime",
        reason: "1d regime reason",
      }),
    },
  };
}

function makeFailClosedDoc(): TimeframeRolesDoc {
  const timeframes: Record<string, TimeframeRoleEntry> = {};
  for (const tf of TIMEFRAME_ORDER) {
    timeframes[tf] = tfEntry({
      role: "disabled",
      context_subkind: null,
      disabled_reason: "by_safety",
      reason: "fail-closed default",
      evidence_ref: "fail-closed-default",
    });
  }
  return {
    schema_version: 1,
    generated_at: "2026-04-28T17:30:00.000Z",
    generated_by_task: "fail-closed",
    timeframes,
  };
}

function renderCard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <HorizonRolesCard />
    </QueryClientProvider>,
  );
}

function setHooks(opts: {
  doc?: TimeframeRolesDoc;
  summary?: { trade: number; shadow: number; context: number; disabled: number };
  isLoading?: boolean;
  isError?: boolean;
  brainEnabled?: boolean;
}) {
  vi.mocked(useTimeframeRoles).mockReturnValue({
    data:
      opts.isLoading || opts.isError
        ? undefined
        : {
            document: opts.doc!,
            summary: opts.summary ?? {
              trade: 0,
              shadow: 0,
              context: 0,
              disabled: 0,
            },
          },
    isLoading: opts.isLoading ?? false,
    isError: opts.isError ?? false,
    error: null,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof useTimeframeRoles>);
  vi.mocked(useBrainRuntimeStatus).mockReturnValue({
    data: {
      brainEnabled: opts.brainEnabled ?? false,
      state: opts.brainEnabled ? "online" : "offline_disabled",
      brainSource: "default",
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

describe("badge wording rule", () => {
  test("tradeBadgeText reads TRADE-ELIGIBLE when brain is OFF", () => {
    expect(tradeBadgeText(false)).toBe("TRADE-ELIGIBLE");
  });

  test("tradeBadgeText reads TRADING when brain is ON", () => {
    expect(tradeBadgeText(true)).toBe("TRADING");
  });

  test("badgeTextFor flips trade-only wording on brain enable", () => {
    expect(badgeTextFor("trade", false)).toBe("TRADE-ELIGIBLE");
    expect(badgeTextFor("trade", true)).toBe("TRADING");
    expect(badgeTextFor("shadow", true)).toBe("SHADOW");
    expect(badgeTextFor("context", true)).toBe("CONTEXT");
    expect(badgeTextFor("disabled", true)).toBe("DISABLED");
  });
});

describe("badge color palette contract", () => {
  test("each role maps to its expected color token", () => {
    expect(badgeClassesFor("trade")).toMatch(/emerald/);
    expect(badgeClassesFor("shadow")).toMatch(/amber/);
    expect(badgeClassesFor("context")).toMatch(/slate/);
    expect(badgeClassesFor("disabled")).toMatch(/muted/);
  });
});

describe("evidence_ref clickability", () => {
  test("non-empty paths are clickable", () => {
    expect(isClickableEvidence("artifacts/ml-engine/reports/foo.json")).toBe(
      true,
    );
    expect(isClickableEvidence("https://example.com/x")).toBe(true);
  });

  test("the fail-closed sentinel is NOT clickable", () => {
    expect(isClickableEvidence("fail-closed-default")).toBe(false);
  });

  test("empty refs are not clickable", () => {
    expect(isClickableEvidence("")).toBe(false);
  });
});

describe("normalizeForRender", () => {
  test("returns six rows in fixed order when the doc has all six TFs", () => {
    const doc = makeMixedDoc();
    const result = normalizeForRender(doc, {
      trade: 1,
      shadow: 1,
      context: 3,
      disabled: 1,
    });
    expect(result.renderedEntries.map((r) => r.tf)).toEqual([
      "1m",
      "5m",
      "1h",
      "2h",
      "6h",
      "1d",
    ]);
    expect(result.synthesizedMissingTfs).toEqual([]);
    expect(result.effectiveSummary).toEqual({
      trade: 1,
      shadow: 1,
      context: 3,
      disabled: 1,
    });
  });

  test("synthesizes fail-closed disabled rows for missing TFs and lists them", () => {
    const partial: TimeframeRolesDoc = {
      schema_version: 1,
      generated_at: "2026-04-28T17:30:00.000Z",
      generated_by_task: "partial",
      timeframes: { "1m": tfEntry({ role: "trade", context_subkind: null }) },
    };
    const result = normalizeForRender(partial, {
      trade: 1,
      shadow: 0,
      context: 0,
      disabled: 0,
    });
    expect(result.synthesizedMissingTfs).toEqual([
      "5m",
      "1h",
      "2h",
      "6h",
      "1d",
    ]);
    for (const tf of ["5m", "1h", "2h", "6h", "1d"]) {
      const row = result.renderedEntries.find((r) => r.tf === tf)!;
      expect(row.synthesized).toBe(true);
      expect(row.entry.role).toBe("disabled");
      expect(row.entry.disabled_reason).toBe("by_safety");
      expect(row.entry.evidence_ref).toBe("fail-closed-default");
    }
  });

  test("recomputes summary counts from the rendered six rows when synthesizing", () => {
    const partial: TimeframeRolesDoc = {
      schema_version: 1,
      generated_at: "2026-04-28T17:30:00.000Z",
      generated_by_task: "partial",
      timeframes: { "1m": tfEntry({ role: "trade", context_subkind: null }) },
    };
    // Server lies: claims 0 disabled. Card MUST recompute.
    const result = normalizeForRender(partial, {
      trade: 1,
      shadow: 0,
      context: 0,
      disabled: 0,
    });
    expect(result.effectiveSummary).toEqual({
      trade: 1,
      shadow: 0,
      context: 0,
      disabled: 5,
    });
  });

  test("yields six fail-closed rows when given an empty doc", () => {
    const empty: TimeframeRolesDoc = {
      schema_version: 1,
      generated_at: "2026-04-28T17:30:00.000Z",
      generated_by_task: "empty",
      timeframes: {},
    };
    const result = normalizeForRender(empty, {
      trade: 0,
      shadow: 0,
      context: 0,
      disabled: 0,
    });
    expect(result.synthesizedMissingTfs).toEqual([...TIMEFRAME_ORDER]);
    expect(result.renderedEntries.every((r) => r.synthesized)).toBe(true);
    expect(result.effectiveSummary).toEqual({
      trade: 0,
      shadow: 0,
      context: 0,
      disabled: 6,
    });
  });
});

describe("summaryLine", () => {
  test("uses 'Trade-eligible on' verb when brain is OFF", () => {
    const doc = makeMixedDoc();
    const line = summaryLine(
      doc,
      { trade: 1, shadow: 1, context: 3, disabled: 1 },
      false,
    );
    expect(line).toContain("Trade-eligible on 1 timeframe(s) (1m)");
    expect(line).toContain("1 shadow");
    expect(line).toContain("3 context");
    expect(line).toContain("1 disabled");
  });

  test("flips to 'Trading on' verb when brain is ON", () => {
    const doc = makeMixedDoc();
    const line = summaryLine(
      doc,
      { trade: 1, shadow: 1, context: 3, disabled: 1 },
      true,
    );
    expect(line).toContain("Trading on 1 timeframe(s) (1m)");
  });

  test("renders '0 timeframes' when no TF is trade-roled", () => {
    const doc = makeFailClosedDoc();
    expect(
      summaryLine(doc, { trade: 0, shadow: 0, context: 0, disabled: 6 }, false),
    ).toContain("Trade-eligible on 0 timeframes");
  });
});

describe("<HorizonRolesCard/>", () => {
  test("renders skeleton while loading", () => {
    setHooks({ isLoading: true });
    renderCard();
    expect(screen.getByTestId("horizon-roles-card")).toBeInTheDocument();
    expect(screen.getByTestId("horizon-roles-card-loading")).toBeInTheDocument();
  });

  test("renders error fallback when the roles query errors", () => {
    setHooks({ isError: true });
    renderCard();
    expect(screen.getByTestId("horizon-roles-card-error")).toBeInTheDocument();
    expect(screen.getByTestId("horizon-roles-card-error")).toHaveTextContent(
      /defaulting to refused/,
    );
  });

  test("renders the full mixed-roles view with TRADE-ELIGIBLE wording when brain is OFF", () => {
    setHooks({
      doc: makeMixedDoc(),
      summary: { trade: 1, shadow: 1, context: 3, disabled: 1 },
      brainEnabled: false,
    });
    renderCard();
    expect(screen.getByTestId("horizon-roles-card")).toBeInTheDocument();
    for (const tf of TIMEFRAME_ORDER) {
      expect(screen.getByTestId(`horizon-role-row-${tf}`)).toBeInTheDocument();
    }
    expect(screen.getByTestId("horizon-role-badge-1m")).toHaveTextContent(
      "TRADE-ELIGIBLE",
    );
    expect(screen.getByTestId("horizon-role-badge-1m").className).toMatch(
      /emerald/,
    );
    expect(screen.getByTestId("horizon-role-badge-1h").className).toMatch(
      /amber/,
    );
    expect(screen.getByTestId("horizon-role-badge-1d").className).toMatch(
      /slate/,
    );
    expect(screen.getByTestId("horizon-role-badge-5m").className).toMatch(
      /muted/,
    );
    expect(screen.getByTestId("horizon-role-sublabel-5m")).toHaveTextContent(
      "by_data",
    );
    expect(screen.getByTestId("horizon-role-sublabel-1d")).toHaveTextContent(
      "regime",
    );
    expect(screen.getByTestId("horizon-roles-summary")).toHaveTextContent(
      /Trade-eligible on 1 timeframe\(s\) \(1m\)/,
    );
    expect(
      screen.queryByTestId("horizon-roles-fail-closed-banner"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("horizon-roles-partial-doc-banner"),
    ).not.toBeInTheDocument();
  });

  test("flips trade badge text to TRADING when brain is ON", () => {
    setHooks({
      doc: makeMixedDoc(),
      summary: { trade: 1, shadow: 1, context: 3, disabled: 1 },
      brainEnabled: true,
    });
    renderCard();
    expect(screen.getByTestId("horizon-role-badge-1m")).toHaveTextContent(
      "TRADING",
    );
    expect(screen.getByTestId("horizon-roles-summary")).toHaveTextContent(
      /Trading on 1 timeframe\(s\) \(1m\)/,
    );
  });

  test("renders evidence cells as anchors with target=_blank for non-sentinel refs", () => {
    setHooks({
      doc: makeMixedDoc(),
      summary: { trade: 1, shadow: 1, context: 3, disabled: 1 },
      brainEnabled: false,
    });
    renderCard();
    for (const tf of TIMEFRAME_ORDER) {
      const evidence = screen.getByTestId(`horizon-role-evidence-${tf}`);
      expect(evidence.tagName).toBe("A");
      expect(evidence).toHaveAttribute("target", "_blank");
      expect(evidence).toHaveAttribute("href", expect.stringMatching(/.+/));
    }
  });

  test("renders the fail-closed banner when every TF is the fail-closed sentinel", () => {
    setHooks({
      doc: makeFailClosedDoc(),
      summary: { trade: 0, shadow: 0, context: 0, disabled: 6 },
      brainEnabled: false,
    });
    renderCard();
    const banner = screen.getByTestId("horizon-roles-fail-closed-banner");
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveTextContent(/All timeframes refused \(fail-closed\)/);
    for (const tf of TIMEFRAME_ORDER) {
      const row = screen.getByTestId(`horizon-role-row-${tf}`);
      expect(row).toHaveAttribute("data-role", "disabled");
      const evidence = within(row).getByTestId(`horizon-role-evidence-${tf}`);
      expect(evidence.tagName).toBe("SPAN");
      expect(evidence).toHaveTextContent("fail-closed-default");
    }
  });

  test("renders the fail-closed banner with 'role registry is empty' message for empty docs and synthesizes all six rows", () => {
    const emptyDoc: TimeframeRolesDoc = {
      schema_version: 1,
      generated_at: "2026-04-28T17:30:00.000Z",
      generated_by_task: "empty",
      timeframes: {},
    };
    setHooks({
      doc: emptyDoc,
      summary: { trade: 0, shadow: 0, context: 0, disabled: 0 },
      brainEnabled: false,
    });
    renderCard();
    const banner = screen.getByTestId("horizon-roles-fail-closed-banner");
    expect(banner).toHaveTextContent(/role registry is empty/);
    for (const tf of TIMEFRAME_ORDER) {
      const row = screen.getByTestId(`horizon-role-row-${tf}`);
      expect(row).toHaveAttribute("data-role", "disabled");
      expect(row).toHaveAttribute("data-synthesized", "true");
    }
  });

  test("renders partial-doc banner when only some TFs are present and recomputes counts", () => {
    const partial: TimeframeRolesDoc = {
      schema_version: 1,
      generated_at: "2026-04-28T17:30:00.000Z",
      generated_by_task: "partial",
      timeframes: {
        "1m": tfEntry({
          role: "trade",
          context_subkind: null,
          reason: "1m trade reason",
        }),
      },
    };
    setHooks({
      doc: partial,
      // Server-supplied counts deliberately under-report disabled.
      summary: { trade: 1, shadow: 0, context: 0, disabled: 0 },
      brainEnabled: false,
    });
    renderCard();
    const partialBanner = screen.getByTestId(
      "horizon-roles-partial-doc-banner",
    );
    expect(partialBanner).toHaveTextContent(/missing 5 required timeframe/);
    expect(
      screen.queryByTestId("horizon-roles-fail-closed-banner"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("horizon-role-row-1m")).toHaveAttribute(
      "data-synthesized",
      "false",
    );
    for (const tf of ["5m", "1h", "2h", "6h", "1d"]) {
      expect(screen.getByTestId(`horizon-role-row-${tf}`)).toHaveAttribute(
        "data-synthesized",
        "true",
      );
    }
    // Recomputed: 1 trade + 5 disabled, not the lying server summary.
    expect(screen.getByTestId("horizon-roles-summary")).toHaveTextContent(
      /5 disabled/,
    );
  });
});
