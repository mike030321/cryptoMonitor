import { test, expect, type Route } from "@playwright/test";

const STRATEGY_LAB_API = "**/api/crypto/strategy-lab*";

function makeStubPayload() {
  const now = Date.now();
  const point = (k: string, offsetMin: number, equity: number) => ({
    timestamp: new Date(now - offsetMin * 60_000).toISOString(),
    equity,
  });
  return {
    generatedAt: new Date(now).toISOString(),
    buckets: [
      {
        strategyType: "ai-bots",
        label: "Quant Fleet",
        agentCount: 5,
        startingCapital: 1000,
        currentEquity: 1080,
        cash: 200,
        invested: 880,
        totalPnl: 80,
        totalPnlPct: 8.0,
        totalTrades: 14,
        totalFees: 2.31,
        peakValue: 1090,
        maxDrawdownPct: 1.2,
      },
      {
        strategyType: "dca-cb",
        label: "DCA + Breaker",
        agentCount: 0,
        startingCapital: 1000,
        currentEquity: 1042,
        cash: 100,
        invested: 942,
        totalPnl: 42,
        totalPnlPct: 4.2,
        totalTrades: 21,
        totalFees: 1.55,
        peakValue: 1051,
        maxDrawdownPct: 0.9,
        circuitBreakerActive: false,
      },
      {
        strategyType: "buy-hold",
        label: "Buy & Hold",
        agentCount: 0,
        startingCapital: 1000,
        currentEquity: 1015,
        cash: 0,
        invested: 1015,
        totalPnl: 15,
        totalPnlPct: 1.5,
        totalTrades: 1,
        totalFees: 1.0,
        peakValue: 1030,
        maxDrawdownPct: 1.5,
      },
    ],
    equityCurves: {
      "ai-bots": [
        point("ai-bots", 60, 1000),
        point("ai-bots", 45, 1020),
        point("ai-bots", 30, 1050),
        point("ai-bots", 15, 1070),
        point("ai-bots", 0, 1080),
      ],
      "dca-cb": [
        point("dca-cb", 60, 1000),
        point("dca-cb", 45, 1010),
        point("dca-cb", 30, 1025),
        point("dca-cb", 15, 1035),
        point("dca-cb", 0, 1042),
      ],
      "buy-hold": [
        point("buy-hold", 60, 1000),
        point("buy-hold", 45, 1005),
        point("buy-hold", 30, 1010),
        point("buy-hold", 15, 1012),
        point("buy-hold", 0, 1015),
      ],
    },
  };
}

async function stubStrategyLab(route: Route) {
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(makeStubPayload()),
  });
}

test.describe("Strategy Lab", () => {
  test("sidebar nav link routes to /lab", async ({ page }) => {
    await page.route(STRATEGY_LAB_API, stubStrategyLab);
    await page.goto("/");

    const navLink = page.getByTestId("nav-strategy-lab");
    await expect(navLink).toBeVisible();
    await expect(navLink).toContainText("Strategy Lab");

    await navLink.click();

    await page.waitForURL("**/lab");
    expect(new URL(page.url()).pathname).toBe("/lab");
  });

  test("lab page renders all three bucket cards and the equity-curves panel", async ({ page }) => {
    await page.route(STRATEGY_LAB_API, stubStrategyLab);
    await page.goto("/lab");

    await expect(page.getByTestId("strategy-lab-page")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Strategy Lab" })).toBeVisible();

    await expect(page.getByTestId("bucket-ai-bots")).toBeVisible();
    await expect(page.getByTestId("bucket-dca-cb")).toBeVisible();
    await expect(page.getByTestId("bucket-buy-hold")).toBeVisible();

    await expect(page.getByText("Equity curves")).toBeVisible();
  });

  test("equity-curve chart renders an SVG with one line per bucket when snapshots exist", async ({ page }) => {
    await page.route(STRATEGY_LAB_API, stubStrategyLab);
    await page.goto("/lab");

    await expect(page.getByTestId("strategy-lab-page")).toBeVisible();

    const chartSvg = page.locator(".recharts-responsive-container svg").first();
    await expect(chartSvg).toBeVisible();

    await expect(page.locator(".recharts-line")).toHaveCount(3);

    await expect(
      page.getByText("Collecting data — equity curves will appear after a few cycles."),
    ).toHaveCount(0);
  });

  test("leader callout highlights the bucket with the highest pnl%", async ({ page }) => {
    await page.route(STRATEGY_LAB_API, stubStrategyLab);
    await page.goto("/lab");

    await expect(page.getByTestId("strategy-lab-page")).toBeVisible();

    const callout = page.getByTestId("leader-callout");
    await expect(callout).toBeVisible();
    await expect(callout).toContainText("Quant Fleet");
    await expect(callout).toContainText("+8.00%");
  });

  test("page survives a hard reload", async ({ page }) => {
    await page.route(STRATEGY_LAB_API, stubStrategyLab);
    await page.goto("/lab");
    await expect(page.getByTestId("strategy-lab-page")).toBeVisible();

    await page.reload();

    await expect(page.getByTestId("strategy-lab-page")).toBeVisible();
    await expect(page.getByTestId("bucket-ai-bots")).toBeVisible();
    await expect(page.getByTestId("bucket-dca-cb")).toBeVisible();
    await expect(page.getByTestId("bucket-buy-hold")).toBeVisible();
  });
});
