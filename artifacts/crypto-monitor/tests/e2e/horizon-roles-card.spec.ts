/**
 * Task #551 — `<HorizonRolesCard/>` e2e test.
 *
 * Stubs the two endpoints the card consumes (`/api/crypto/timeframe-roles`
 * and `/api/crypto/brain/runtime-status`) so the dashboard renders
 * deterministically across the role / brain combinations the task
 * requires us to assert:
 *
 *   - mixed roles render the right badge color, sub-label, and
 *     reason for each TF;
 *   - an empty / fail-closed document renders the fail-closed banner;
 *   - the `trade` badge wording flips between TRADE-ELIGIBLE and
 *     TRADING based on `quant_brain_enabled` (the central honesty rule
 *     of the task);
 *   - context_subkind and disabled_reason render as plain-text
 *     sub-labels rather than inflating the 4-role enum.
 *
 * Other dashboard cards make their own API calls; we let those 404
 * silently — the card under test has its own dedicated routes that we
 * fully stub. This mirrors the pattern in strategy-lab.spec.ts.
 */
import { test, expect, type Route } from "@playwright/test";

const ROLES_API = "**/api/crypto/timeframe-roles";
const BRAIN_API = "**/api/crypto/brain/runtime-status";

interface BrainOpts {
  brainEnabled: boolean;
  state?: "online" | "offline_no_model" | "offline_disabled";
  brainSource?: "default" | "manual" | "auto_revert" | "env";
}

function makeBrainStatus(opts: BrainOpts) {
  return {
    state: opts.state ?? (opts.brainEnabled ? "online" : "offline_disabled"),
    brainEnabled: opts.brainEnabled,
    brainSource: opts.brainSource ?? "default",
    mlAvailabilitySnapshotReady: true,
    recentAbstainReasons: {},
    recentNonAbstainCount: 0,
    lastSuccessfulDecisionAt: null,
    currentRunDir: null,
    windowMinutes: 60,
  };
}

function tfEntry(overrides: Partial<{
  role: "trade" | "shadow" | "context" | "disabled";
  context_subkind: "filter" | "regime" | "risk_state" | null;
  disabled_reason: "by_data" | "by_gate" | "by_operator" | "by_safety" | null;
  reason: string;
  evidence_ref: string;
  promoted_slices_in_tf: string[];
  last_reviewed_at: string;
}>) {
  return {
    role: "context",
    context_subkind: "filter",
    disabled_reason: null,
    reason: "stubbed reason",
    evidence_ref: "stubbed/evidence/path.md",
    last_reviewed_at: "2026-04-28T12:00:00.000Z",
    promoted_slices_in_tf: [] as string[],
    ...overrides,
  };
}

function makeMixedDoc() {
  return {
    schema_version: 1 as const,
    generated_at: "2026-04-28T17:30:00.000Z",
    generated_by_task: "#551-test",
    timeframes: {
      "1m": tfEntry({
        role: "trade",
        context_subkind: null,
        reason: "1m trade-eligible reason",
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

function makeMixedSummary() {
  return { trade: 1, shadow: 1, context: 3, disabled: 1 };
}

function makeFailClosedDoc() {
  // Mirrors the server-side `makeFailClosedDocument()` shape so the
  // card's `allRefused` check (every TF disabled+by_safety with the
  // fail-closed-default sentinel) flips on.
  const tfs = ["1m", "5m", "1h", "2h", "6h", "1d"];
  const timeframes: Record<string, ReturnType<typeof tfEntry>> = {};
  for (const tf of tfs) {
    timeframes[tf] = tfEntry({
      role: "disabled",
      context_subkind: null,
      disabled_reason: "by_safety",
      reason: "fail-closed default",
      evidence_ref: "fail-closed-default",
    });
  }
  return {
    schema_version: 1 as const,
    generated_at: "2026-04-28T17:30:00.000Z",
    generated_by_task: "fail-closed",
    timeframes,
  };
}

async function stubJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

async function stubAll(
  page: import("@playwright/test").Page,
  doc: ReturnType<typeof makeMixedDoc> | ReturnType<typeof makeFailClosedDoc>,
  summary: { trade: number; shadow: number; context: number; disabled: number },
  brain: BrainOpts,
) {
  await page.route(ROLES_API, (r) => stubJson(r, { document: doc, summary }));
  await page.route(BRAIN_API, (r) => stubJson(r, makeBrainStatus(brain)));
}

test.describe("Horizon Roles card", () => {
  test("renders all six timeframes with the right badge / sub-label / reason for each role", async ({ page }) => {
    await stubAll(page, makeMixedDoc(), makeMixedSummary(), {
      brainEnabled: false,
    });
    await page.goto("/");

    const card = page.getByTestId("horizon-roles-card");
    await expect(card).toBeVisible();

    // 1m: trade. Brain is OFF → badge text MUST be TRADE-ELIGIBLE.
    const trade = page.getByTestId("horizon-role-row-1m");
    await expect(trade).toHaveAttribute("data-role", "trade");
    await expect(page.getByTestId("horizon-role-badge-1m")).toHaveText(
      "TRADE-ELIGIBLE",
    );
    await expect(trade).toContainText("1m trade-eligible reason");
    await expect(page.getByTestId("horizon-role-promoted-1m")).toHaveText(
      "1 promoted",
    );

    // 5m: disabled, by_data. Sub-label must show by_data.
    const disabled = page.getByTestId("horizon-role-row-5m");
    await expect(disabled).toHaveAttribute("data-role", "disabled");
    await expect(page.getByTestId("horizon-role-badge-5m")).toHaveText(
      "DISABLED",
    );
    await expect(page.getByTestId("horizon-role-sublabel-5m")).toHaveText(
      "by_data",
    );

    // 1h: shadow. No sub-label.
    const shadow = page.getByTestId("horizon-role-row-1h");
    await expect(shadow).toHaveAttribute("data-role", "shadow");
    await expect(page.getByTestId("horizon-role-badge-1h")).toHaveText(
      "SHADOW",
    );

    // 1d: context, regime. Sub-label must show regime.
    const context = page.getByTestId("horizon-role-row-1d");
    await expect(context).toHaveAttribute("data-role", "context");
    await expect(page.getByTestId("horizon-role-badge-1d")).toHaveText(
      "CONTEXT",
    );
    await expect(page.getByTestId("horizon-role-sublabel-1d")).toHaveText(
      "regime",
    );

    // Summary line lists the trade-eligible TFs and per-role counts.
    const summary = page.getByTestId("horizon-roles-summary");
    await expect(summary).toContainText("Trade-eligible on 1 timeframe(s) (1m)");
    await expect(summary).toContainText("1 shadow");
    await expect(summary).toContainText("3 context");
    await expect(summary).toContainText("1 disabled");

    // Badge color palette is part of the contract: trade=emerald,
    // shadow=amber, context=slate, disabled=muted. We assert that the
    // expected color token appears in each badge's class list so a
    // future palette swap can't quietly recolor "DISABLED" green.
    await expect(page.getByTestId("horizon-role-badge-1m")).toHaveClass(
      /emerald/,
    );
    await expect(page.getByTestId("horizon-role-badge-1h")).toHaveClass(
      /amber/,
    );
    await expect(page.getByTestId("horizon-role-badge-1d")).toHaveClass(
      /slate/,
    );
    await expect(page.getByTestId("horizon-role-badge-5m")).toHaveClass(
      /muted/,
    );

    // The evidence cell must be a navigable anchor (operator can click
    // through to verify the source). The fail-closed sentinel is the
    // only ref allowed to render as plain text — no row in this mixed
    // doc carries that sentinel, so every evidence cell must be an
    // <a> tag.
    for (const tf of ["1m", "5m", "1h", "2h", "6h", "1d"]) {
      const evidence = page.getByTestId(`horizon-role-evidence-${tf}`);
      await expect(evidence).toBeVisible();
      await expect(evidence).toHaveAttribute("target", "_blank");
      await expect(evidence).toHaveAttribute("href", /.+/);
    }
  });

  test("flips trade badge text to TRADING when quant_brain_enabled is true", async ({ page }) => {
    await stubAll(page, makeMixedDoc(), makeMixedSummary(), {
      brainEnabled: true,
    });
    await page.goto("/");

    await expect(page.getByTestId("horizon-roles-card")).toBeVisible();
    await expect(page.getByTestId("horizon-role-badge-1m")).toHaveText(
      "TRADING",
    );

    // Summary verb flips too so the dashboard never reads "Trade-eligible
    // on 1 timeframe(s)" while the brain is actively trading.
    await expect(page.getByTestId("horizon-roles-summary")).toContainText(
      "Trading on 1 timeframe(s) (1m)",
    );
  });

  test("renders the fail-closed banner when every timeframe is disabled by_safety with the fail-closed-default sentinel", async ({ page }) => {
    await stubAll(
      page,
      makeFailClosedDoc(),
      { trade: 0, shadow: 0, context: 0, disabled: 6 },
      { brainEnabled: false },
    );
    await page.goto("/");

    await expect(page.getByTestId("horizon-roles-card")).toBeVisible();
    const banner = page.getByTestId("horizon-roles-fail-closed-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText("All timeframes refused (fail-closed)");

    // Each row still renders so the operator can see WHICH TFs were
    // refused — the banner doesn't replace the list.
    for (const tf of ["1m", "5m", "1h", "2h", "6h", "1d"]) {
      const row = page.getByTestId(`horizon-role-row-${tf}`);
      await expect(row).toHaveAttribute("data-role", "disabled");
      await expect(page.getByTestId(`horizon-role-sublabel-${tf}`)).toHaveText(
        "by_safety",
      );
    }

    // The fail-closed sentinel evidence_ref must NOT render as a
    // clickable link — it isn't a real path. The card renders it as
    // plain text instead so the operator never tries to "open" it.
    for (const tf of ["1m", "5m", "1h", "2h", "6h", "1d"]) {
      const evidence = page.getByTestId(`horizon-role-evidence-${tf}`);
      await expect(evidence).toContainText("fail-closed-default");
      await expect(evidence).not.toHaveAttribute("href", /.+/);
    }
  });

  test("synthesizes fail-closed disabled rows for any required timeframe missing from the document", async ({ page }) => {
    // Partial document: only the 1m row is published. The card MUST
    // still render exactly six rows in the fixed horizon order — the
    // server-side gate treats every missing TF as fail-closed
    // disabled (`getRoleEntryForTimeframe`), and the dashboard must
    // mirror that so a refused horizon never silently disappears.
    const partialDoc = {
      schema_version: 1 as const,
      generated_at: "2026-04-28T17:30:00.000Z",
      generated_by_task: "partial-doc-test",
      timeframes: {
        "1m": tfEntry({
          role: "trade",
          context_subkind: null,
          reason: "1m trade-eligible reason",
          promoted_slices_in_tf: ["bonk:1m:v3"],
        }),
      },
    };
    await stubAll(
      page,
      partialDoc,
      // The server-supplied summary in this case lies (only counts the
      // 1m trade) — the card MUST recompute counts from the rendered
      // six rows so it reports 5 disabled, not 0.
      { trade: 1, shadow: 0, context: 0, disabled: 0 },
      { brainEnabled: false },
    );
    await page.goto("/");

    await expect(page.getByTestId("horizon-roles-card")).toBeVisible();

    // Exactly the six required timeframes render, in fixed order, no
    // more no less.
    for (const tf of ["1m", "5m", "1h", "2h", "6h", "1d"]) {
      await expect(page.getByTestId(`horizon-role-row-${tf}`)).toBeVisible();
    }

    // The 1m row uses the published entry (real, not synthesized).
    const real = page.getByTestId("horizon-role-row-1m");
    await expect(real).toHaveAttribute("data-role", "trade");
    await expect(real).toHaveAttribute("data-synthesized", "false");

    // The five missing rows are synthesized as fail-closed disabled,
    // marked with data-synthesized="true" so an operator (and any
    // assertion) can tell they came from the loader's fail-closed
    // default rather than the published JSON.
    for (const tf of ["5m", "1h", "2h", "6h", "1d"]) {
      const row = page.getByTestId(`horizon-role-row-${tf}`);
      await expect(row).toHaveAttribute("data-role", "disabled");
      await expect(row).toHaveAttribute("data-synthesized", "true");
      await expect(page.getByTestId(`horizon-role-sublabel-${tf}`)).toHaveText(
        "by_safety",
      );
      await expect(page.getByTestId(`horizon-role-evidence-${tf}`)).toContainText(
        "fail-closed-default",
      );
    }

    // A partial-doc warning banner appears (distinct from the
    // all-refused fail-closed banner) and lists which TFs are missing.
    const partialBanner = page.getByTestId("horizon-roles-partial-doc-banner");
    await expect(partialBanner).toBeVisible();
    await expect(partialBanner).toContainText("missing 5 required timeframe");
    await expect(partialBanner).toContainText("5m");
    await expect(partialBanner).toContainText("1d");

    // Summary line is recomputed from the six rendered rows so a
    // partial document can NOT under-report disabled horizons (the
    // server's lying summary said 0 disabled; the recomputed count
    // shows 5).
    const summary = page.getByTestId("horizon-roles-summary");
    await expect(summary).toContainText("Trade-eligible on 1 timeframe(s) (1m)");
    await expect(summary).toContainText("5 disabled");
  });

  test("renders the fail-closed banner when the role document is empty", async ({ page }) => {
    const emptyDoc = {
      schema_version: 1 as const,
      generated_at: "2026-04-28T17:30:00.000Z",
      generated_by_task: "empty-doc-test",
      timeframes: {} as Record<string, ReturnType<typeof tfEntry>>,
    };
    await stubAll(
      page,
      emptyDoc,
      { trade: 0, shadow: 0, context: 0, disabled: 0 },
      { brainEnabled: false },
    );
    await page.goto("/");

    await expect(page.getByTestId("horizon-roles-card")).toBeVisible();
    const banner = page.getByTestId("horizon-roles-fail-closed-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText("All timeframes refused (fail-closed)");
    await expect(banner).toContainText("role registry is empty");

    // All six rows still render — synthesized as fail-closed disabled.
    for (const tf of ["1m", "5m", "1h", "2h", "6h", "1d"]) {
      const row = page.getByTestId(`horizon-role-row-${tf}`);
      await expect(row).toBeVisible();
      await expect(row).toHaveAttribute("data-role", "disabled");
      await expect(row).toHaveAttribute("data-synthesized", "true");
    }
    await expect(page.getByTestId("horizon-roles-summary")).toContainText(
      "Trade-eligible on 0 timeframes",
    );
  });
});
