import { useEffect, useMemo, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  ChevronDown,
  ChevronUp,
  ChevronsUpDown,
  RefreshCw,
  Wallet,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface PerClassBreakdown {
  UP?: number;
  DOWN?: number;
  STABLE?: number;
}

interface PerClassAccuracyEntry {
  n?: number | null;
  accuracy?: number | null;
}

interface PerClassAccuracy {
  UP?: PerClassAccuracyEntry;
  DOWN?: PerClassAccuracyEntry;
  STABLE?: PerClassAccuracyEntry;
}

interface PredictionCollapse {
  collapse_gap?: number | null;
  predicted_top_class_share?: number | null;
  label_top_class_share?: number | null;
}

interface PnlAfterFees {
  n_trades?: number | null;
  trade_share?: number | null;
  gross_pct_mean?: number | null;
  round_trip_cost_pct?: number | null;
  net_pct_mean?: number | null;
  net_pct_total?: number | null;
  win_rate?: number | null;
}

interface SliceReport {
  status?: string;
  n_rows?: number | null;
  per_class_holdout_breakdown?: PerClassBreakdown | null;
  per_class_accuracy?: PerClassAccuracy | null;
  prediction_collapse?: PredictionCollapse | null;
  pnl_after_fees?: PnlAfterFees | null;
}

interface TimeframeReport {
  status?: string;
  per_coin?: Record<string, SliceReport> | null;
  pooled?: SliceReport | null;
}

// Task #401 / #416 — verification block schema fragment. Each per_slice
// verdict carries `min_directional_accuracy_applied` so the dashboard
// can render the per-tf floor that actually decided promotion (the
// default 0.50 vs the 1d-only 0.530 override). Older reports may omit
// the per-tf map and the per-slice field; we fall back to the legacy
// single floor in that case.
interface VerificationVerdict {
  coin?: string;
  timeframe?: string;
  kind?: string;
  reason?: string;
  promoted?: boolean;
  directional_accuracy?: number | null;
  min_directional_accuracy_applied?: number | null;
}

interface VerificationBlock {
  min_directional_accuracy?: number | null;
  min_directional_accuracy_per_tf?: Record<string, number> | null;
  per_slice?: VerificationVerdict[] | null;
}

interface TrainingReport {
  status?: string;
  generated_at?: string;
  timeframes?: Record<string, TimeframeReport> | null;
  verification?: VerificationBlock | null;
}

// Task #615 — per-slice live-gated replay block from the most recent
// `phase7_summary.json`. Produced by Task #613 in
// `scripts/run_full_training_campaign.py`. Lets the dashboard surface
// the four-way verdict pill plus loose-vs-live PnL so operators don't
// have to open the run folder to spot a bleeding/dormant slice.
type EconomicVerdict =
  | "bleeding"
  | "dormant"
  | "tradeable"
  | "inconclusive";

interface LiveGatedSlice {
  loose_post_fee_pct_total?: number | null;
  live_trade_count?: number | null;
  live_net_pnl_pct?: number | null;
  dominant_rejection_reason?: string | null;
  live_replay_status?: string | null;
  economic_verdict?: EconomicVerdict | string | null;
  economic_verdict_phrase?: string | null;
}

interface LiveGatedReplay {
  status?: string;
  run_dir?: string | null;
  generated_at?: string | null;
  per_slice?: Record<string, LiveGatedSlice> | null;
  verdict_counts?: Partial<Record<EconomicVerdict, number>> | null;
}

interface FlatRow {
  key: string;
  tf: string;
  coin: string;
  slice: SliceReport;
}

const apiUrl = (path: string) => `${import.meta.env.BASE_URL}api${path}`;

const TF_ORDER = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"];
const tfRank = (tf: string) => {
  const i = TF_ORDER.indexOf(tf);
  return i === -1 ? 99 : i;
};

const COLLAPSE_THRESHOLD = 0.15;

// Task #401 — legacy DA floor used as a fallback when the report doesn't
// carry the per-slice `min_directional_accuracy_applied` field yet (i.e.
// reports written before #401 landed). Mirrors verification.py's
// MIN_DIRECTIONAL_ACCURACY constant.
const DEFAULT_DA_FLOOR = 0.5;

type SortKey = "pnl" | "tf" | "coin" | "trades" | "collapse" | "da";
type SortDir = "asc" | "desc";

function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${Number(v).toFixed(digits)}%`;
}

function fmtNum(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return String(v);
}

function fmtShare(n: number | null | undefined, total: number): string {
  if (n == null || total <= 0) return "—";
  return `${((n / total) * 100).toFixed(0)}%`;
}

export function TrainingPerSliceCard() {
  const [data, setData] = useState<TrainingReport | null>(null);
  const [liveGated, setLiveGated] = useState<LiveGatedReplay | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("pnl");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  async function refresh() {
    setLoading(true);
    try {
      // Task #615 — fetch the per-slice training diagnostics and the
      // most-recent campaign's live-gated replay block in parallel.
      // The live-gated proxy never throws on missing data (it returns
      // `{status: "missing" | "empty"}`) so a fresh project still
      // renders the table without the verdict pills.
      const [reportRes, liveRes] = await Promise.all([
        fetch(apiUrl(`/crypto/quant-training-report`)),
        fetch(apiUrl(`/crypto/training/live-gated-replay`)),
      ]);
      if (!reportRes.ok) throw new Error(`HTTP ${reportRes.status}`);
      setData((await reportRes.json()) as TrainingReport);
      if (liveRes.ok) {
        setLiveGated((await liveRes.json()) as LiveGatedReplay);
      } else {
        setLiveGated({ status: "missing" });
      }
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
    const id = setInterval(() => void refresh(), 60_000);
    return () => clearInterval(id);
  }, []);

  // Index live-gated entries by `coin/tf` slug so per-row lookup is O(1).
  const liveGatedByKey = useMemo(() => {
    const m = new Map<string, LiveGatedSlice>();
    const ps = liveGated?.per_slice ?? {};
    for (const [slug, entry] of Object.entries(ps)) {
      m.set(slug, entry);
    }
    return m;
  }, [liveGated]);

  function lookupLiveGated(tf: string, coin: string): LiveGatedSlice | undefined {
    return liveGatedByKey.get(`${coin}/${tf}`);
  }

  const rows: FlatRow[] = useMemo(() => {
    const out: FlatRow[] = [];
    const tfs = data?.timeframes ?? {};
    for (const [tf, tfRep] of Object.entries(tfs)) {
      if (!tfRep) continue;
      const perCoin = tfRep.per_coin ?? {};
      for (const [coin, slc] of Object.entries(perCoin)) {
        out.push({ key: `${tf}::${coin}`, tf, coin, slice: slc });
      }
      if (tfRep.pooled) {
        out.push({
          key: `${tf}::__pooled__`,
          tf,
          coin: "__pooled__",
          slice: tfRep.pooled,
        });
      }
    }
    return out;
  }, [data]);

  // Index per_slice verdicts by (tf, coin, kind) for row lookup.
  const verdictByKey = useMemo(() => {
    const m = new Map<string, VerificationVerdict>();
    const list = data?.verification?.per_slice ?? [];
    for (const v of list) {
      if (!v?.timeframe || !v?.coin) continue;
      const kind = v.kind ?? (v.coin === "__pooled__" ? "pooled" : "per_coin");
      m.set(`${v.timeframe}::${v.coin}::${kind}`, v);
    }
    return m;
  }, [data]);

  function lookupVerdict(tf: string, coin: string): VerificationVerdict | undefined {
    const kind = coin === "__pooled__" ? "pooled" : "per_coin";
    return verdictByKey.get(`${tf}::${coin}::${kind}`);
  }

  // Resolve the floor to render in the header next to a tf when no
  // per_slice verdict is available (e.g. report missing the per-tf
  // map entirely).
  function floorForTf(tf: string): number {
    const map = data?.verification?.min_directional_accuracy_per_tf ?? null;
    if (map && typeof map[tf] === "number") return Number(map[tf]);
    const fallback = data?.verification?.min_directional_accuracy;
    return typeof fallback === "number" ? fallback : DEFAULT_DA_FLOOR;
  }

  // Floor list for the header chip strip — every tf the report shows
  // gets a chip; the chip text reads "tf · floor" (e.g. "1d · 0.530"
  // or "1h · 0.500"). Tfs with an explicit override are highlighted so
  // operators can spot the looser/tighter ones at a glance.
  const tfsInReport = useMemo(() => {
    const tfs = Object.keys(data?.timeframes ?? {});
    tfs.sort((a, b) => tfRank(a) - tfRank(b));
    return tfs;
  }, [data]);

  const overrideMap = data?.verification?.min_directional_accuracy_per_tf ?? null;
  const defaultFloor =
    typeof data?.verification?.min_directional_accuracy === "number"
      ? data!.verification!.min_directional_accuracy!
      : DEFAULT_DA_FLOOR;

  const sorted = useMemo(() => {
    const cmpNum = (a: number | null | undefined, b: number | null | undefined) => {
      const av = a == null || Number.isNaN(a) ? Number.NEGATIVE_INFINITY : a;
      const bv = b == null || Number.isNaN(b) ? Number.NEGATIVE_INFINITY : b;
      return av - bv;
    };
    const arr = [...rows];
    arr.sort((a, b) => {
      let d = 0;
      switch (sortKey) {
        case "pnl":
          d = cmpNum(
            a.slice.pnl_after_fees?.net_pct_mean,
            b.slice.pnl_after_fees?.net_pct_mean,
          );
          break;
        case "trades":
          d = cmpNum(
            a.slice.pnl_after_fees?.n_trades,
            b.slice.pnl_after_fees?.n_trades,
          );
          break;
        case "collapse":
          d = cmpNum(
            a.slice.prediction_collapse?.collapse_gap,
            b.slice.prediction_collapse?.collapse_gap,
          );
          break;
        case "da": {
          // Sort by lift over the per-slice DA floor so 1d slices that
          // sit just above the legacy 0.50 but below the 0.530 1d floor
          // sort *below* a 1h slice that genuinely cleared its floor.
          const va = lookupVerdict(a.tf, a.coin);
          const vb = lookupVerdict(b.tf, b.coin);
          const liftA =
            typeof va?.directional_accuracy === "number"
              ? va.directional_accuracy -
                (typeof va.min_directional_accuracy_applied === "number"
                  ? va.min_directional_accuracy_applied
                  : floorForTf(a.tf))
              : null;
          const liftB =
            typeof vb?.directional_accuracy === "number"
              ? vb.directional_accuracy -
                (typeof vb.min_directional_accuracy_applied === "number"
                  ? vb.min_directional_accuracy_applied
                  : floorForTf(b.tf))
              : null;
          d = cmpNum(liftA, liftB);
          break;
        }
        case "tf":
          d = tfRank(a.tf) - tfRank(b.tf);
          if (d === 0) d = a.coin.localeCompare(b.coin);
          break;
        case "coin":
          d = a.coin.localeCompare(b.coin);
          if (d === 0) d = tfRank(a.tf) - tfRank(b.tf);
          break;
      }
      return sortDir === "asc" ? d : -d;
    });
    return arr;
  }, [rows, sortKey, sortDir, verdictByKey, data]);

  const isMissing = data?.status === "missing";

  function toggleSort(k: SortKey) {
    if (sortKey === k) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(k);
      setSortDir(k === "tf" || k === "coin" ? "asc" : "desc");
    }
  }

  function SortIcon({ k }: { k: SortKey }) {
    if (sortKey !== k) {
      return <ChevronsUpDown className="inline h-3 w-3 opacity-50" />;
    }
    return sortDir === "asc" ? (
      <ChevronUp className="inline h-3 w-3" />
    ) : (
      <ChevronDown className="inline h-3 w-3" />
    );
  }

  return (
    <Card data-testid="training-per-slice-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <Wallet className="h-5 w-5" />
          Per-slice accuracy &amp; post-fee PnL
          {data?.generated_at && (
            <Badge variant="outline" className="ml-2 text-[10px]">
              {new Date(data.generated_at).toLocaleString()}
            </Badge>
          )}
        </CardTitle>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => void refresh()}
          disabled={loading}
          aria-label="Refresh per-slice diagnostics"
        >
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
        </Button>
      </CardHeader>

      <CardContent className="space-y-3">
        {err && (
          <div className="text-xs text-red-400" data-testid="training-per-slice-error">
            Error: {err}
          </div>
        )}

        {isMissing && (
          <div className="text-xs text-muted-foreground">
            No training report yet. Run a training pass to populate this card.
          </div>
        )}

        {!isMissing && rows.length === 0 && !err && (
          <div className="text-xs text-muted-foreground">
            Latest report has no per-slice data yet.
          </div>
        )}

        {rows.length > 0 && (
          <GateConstantsHeader
            tfsInReport={tfsInReport}
            defaultFloor={defaultFloor}
            overrideMap={overrideMap}
            haveVerification={Boolean(data?.verification)}
          />
        )}

        {rows.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs" data-testid="training-per-slice-table">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="py-1 pr-3">
                    <button
                      type="button"
                      onClick={() => toggleSort("tf")}
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      aria-label="Sort by timeframe"
                      data-testid="training-per-slice-sort-tf"
                    >
                      TF <SortIcon k="tf" />
                    </button>
                  </th>
                  <th className="py-1 pr-3">
                    <button
                      type="button"
                      onClick={() => toggleSort("coin")}
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      aria-label="Sort by coin"
                      data-testid="training-per-slice-sort-coin"
                    >
                      Coin <SortIcon k="coin" />
                    </button>
                  </th>
                  <th className="py-1 pr-3">
                    <button
                      type="button"
                      onClick={() => toggleSort("collapse")}
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      title="Per-class accuracy on the holdout slice (UP / DOWN / STABLE) — share of true rows in each class that the calibrated model predicted correctly, with the sample count. Pulled from per_class_accuracy. A 'collapse' chip appears when the model's predicted top-class share exceeds the empirical top-class share by ≥ 0.15. Sort orders slices by predicted-class collapse gap."
                      aria-label="Sort by per-class collapse gap"
                      data-testid="training-per-slice-sort-collapse"
                    >
                      Per-class accuracy (n) <SortIcon k="collapse" />
                    </button>
                  </th>
                  <th className="py-1 pr-3">
                    <button
                      type="button"
                      onClick={() => toggleSort("da")}
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      title="Calibrated directional accuracy on the holdout, with the per-timeframe gate floor that decided promotion. Sort orders slices by lift over their own floor — a 1d slice at DA 0.516 is below the 1d 0.530 floor even though it would have cleared the legacy 0.50."
                      aria-label="Sort by directional accuracy lift over the per-tf gate floor"
                      data-testid="training-per-slice-sort-da"
                    >
                      DA · gate <SortIcon k="da" />
                    </button>
                  </th>
                  <th
                    className="py-1 pr-3 text-right"
                    title="Mean realized gross % return per simulated trade on the holdout slice, before round-trip costs."
                  >
                    Gross %
                  </th>
                  <th
                    className="py-1 pr-3 text-right"
                    title="Round-trip transaction-cost % subtracted from each trade (matches the live frictions contract)."
                  >
                    Fee %
                  </th>
                  <th
                    className="py-1 pr-3 text-right"
                  >
                    <button
                      type="button"
                      onClick={() => toggleSort("pnl")}
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      title="Mean net % return per trade after fees. Computed by replaying the live entry rule (argmax≠STABLE, |p_up - p_down| ≥ min_directional_edge, expected magnitude ≥ min_expected_return_pct when a regression head exists) on the calibrated holdout, then subtracting the contract round-trip cost. Default sort."
                      aria-label="Sort by net post-fee PnL"
                      data-testid="training-per-slice-sort-pnl"
                    >
                      Net % <SortIcon k="pnl" />
                    </button>
                  </th>
                  <th className="py-1 pr-3 text-right">
                    <button
                      type="button"
                      onClick={() => toggleSort("trades")}
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      title="Number of simulated trades the live entry rule fired on the holdout slice."
                      aria-label="Sort by trade count"
                      data-testid="training-per-slice-sort-trades"
                    >
                      Trades <SortIcon k="trades" />
                    </button>
                  </th>
                  <th
                    className="py-1 pr-3 text-right"
                    title="Share of simulated trades whose net % return is strictly positive after the round-trip fee. Computed on the same trade set as Net %."
                  >
                    Win rate
                  </th>
                  <th
                    className="py-1 pr-3"
                    title="Live-gated replay verdict from the most recent campaign's phase7_summary.json. Bleeding = loose holdout PnL is negative AND the live-gated replay still trades and loses (n≥5). Dormant = loose holdout PnL is negative but production gates abstain (live n<5), so no live edge can be realised. Tradeable = live-gated replay is profitable with n≥5. Inconclusive = signals disagree or the live diagnostic is missing for the slice."
                  >
                    Live-gated verdict
                  </th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => (
                  <SliceRow
                    key={r.key}
                    row={r}
                    verdict={lookupVerdict(r.tf, r.coin)}
                    fallbackFloor={floorForTf(r.tf)}
                    liveGated={lookupLiveGated(r.tf, r.coin)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}

        {rows.length > 0 && (
          <LiveGatedFooter liveGated={liveGated} />
        )}

        <div className="text-[10px] text-muted-foreground">
          Net % mirrors the live entry rule: argmax ≠ STABLE, |p_up - p_down| ≥
          min_directional_edge, and expected magnitude ≥ min_expected_return_pct
          when the slice has a regression head. Gross % is the realized return
          per simulated trade; net % subtracts the contract round-trip cost.
          Win rate is the share of those simulated trades that closed positive
          after fees. A{" "}
          <span className="text-amber-300">collapse</span> chip means the
          predicted top-class share exceeds the empirical top-class share by ≥{" "}
          {COLLAPSE_THRESHOLD.toFixed(2)} (the failure-analysis threshold). The{" "}
          <span className="font-mono">DA · gate</span> column shows the slice's
          calibrated directional accuracy and whether it cleared the
          per-timeframe gate floor — see the chip strip above the table for the
          floor applied to each timeframe in this run. Em-dashes mark legacy
          slices written before the per-slice diagnostic surface landed. The{" "}
          <span className="font-mono">Live-gated verdict</span> column reads
          the most recent campaign's <span className="font-mono">phase7_summary.json</span>{" "}
          live-gated replay block (loose holdout PnL vs live-gated replay PnL)
          so a bleeding or dormant slice is visible without opening the run
          folder. The dominant rejection reason is shown when the verdict is
          bleeding or dormant.
        </div>
      </CardContent>
    </Card>
  );
}

function PerClassBadge({
  symbol,
  label,
  entry,
  countFallback,
  colorClass,
}: {
  symbol: string;
  label: "UP" | "DOWN" | "STABLE";
  entry: PerClassAccuracyEntry | undefined;
  countFallback: number | undefined;
  colorClass: string;
}) {
  const n = entry?.n ?? countFallback ?? 0;
  const acc =
    typeof entry?.accuracy === "number" ? entry.accuracy : null;
  const accLabel = acc === null ? "—" : `${(acc * 100).toFixed(0)}%`;
  const title =
    acc === null
      ? `${label}: ${n} holdout rows. Per-class accuracy not available for this slice.`
      : `${label}: ${(acc * 100).toFixed(1)}% accuracy on ${n} holdout rows ` +
        `(share of true ${label} rows the calibrated model predicted correctly).`;
  return (
    <Badge
      variant="outline"
      className={cn("text-[10px]", colorClass)}
      title={title}
      data-testid={`training-per-slice-acc-${label.toLowerCase()}`}
    >
      {symbol} {accLabel} (n={fmtNum(n)})
    </Badge>
  );
}

function SliceRow({
  row,
  verdict,
  fallbackFloor,
  liveGated,
}: {
  row: FlatRow;
  verdict: VerificationVerdict | undefined;
  fallbackFloor: number;
  liveGated: LiveGatedSlice | undefined;
}) {
  const { tf, coin, slice } = row;
  const bd = slice.per_class_holdout_breakdown ?? null;
  const acc = slice.per_class_accuracy ?? null;
  const collapse = slice.prediction_collapse ?? null;
  const pnl = slice.pnl_after_fees ?? null;

  const totalN = bd
    ? (bd.UP ?? 0) + (bd.DOWN ?? 0) + (bd.STABLE ?? 0)
    : 0;

  const collapseGap =
    typeof collapse?.collapse_gap === "number" ? collapse.collapse_gap : null;
  const showCollapse =
    collapseGap != null && collapseGap >= COLLAPSE_THRESHOLD;

  const net = pnl?.net_pct_mean;
  const netClass =
    net == null || Number.isNaN(net)
      ? "text-muted-foreground"
      : net > 0
        ? "text-emerald-300"
        : "text-red-400";

  const isPooled = coin === "__pooled__";
  const coinLabel = isPooled ? "pooled" : coin;

  return (
    <tr
      className="border-t border-border/30 align-top"
      data-testid={`training-per-slice-row-${tf}-${coin}`}
    >
      <td className="py-2 pr-3 font-mono">{tf}</td>
      <td className="py-2 pr-3 font-mono">
        {isPooled ? (
          <Badge variant="outline" className="text-[10px]">
            pooled
          </Badge>
        ) : (
          coinLabel
        )}
      </td>
      <td className="py-2 pr-3" data-testid={`training-per-slice-classes-${tf}-${coin}`}>
        {(!bd || totalN === 0) && !acc ? (
          <span className="text-muted-foreground">—</span>
        ) : (
          <div className="flex flex-wrap items-center gap-1">
            <PerClassBadge
              symbol="↑"
              label="UP"
              entry={acc?.UP}
              countFallback={bd?.UP}
              colorClass="border-emerald-500/40 text-emerald-300"
            />
            <PerClassBadge
              symbol="↓"
              label="DOWN"
              entry={acc?.DOWN}
              countFallback={bd?.DOWN}
              colorClass="border-red-500/40 text-red-300"
            />
            <PerClassBadge
              symbol="="
              label="STABLE"
              entry={acc?.STABLE}
              countFallback={bd?.STABLE}
              colorClass="border-muted-foreground/40 text-muted-foreground"
            />
            {showCollapse && (
              <Badge
                variant="outline"
                className="text-[10px] border-amber-500/50 bg-amber-500/10 text-amber-300"
                title={`Predicted top-class share exceeds empirical top-class share by ${(collapseGap! * 100).toFixed(1)}% (threshold ${(COLLAPSE_THRESHOLD * 100).toFixed(0)}%). Model is collapsing onto one class.`}
                data-testid={`training-per-slice-collapse-${tf}-${coin}`}
              >
                collapse {(collapseGap! * 100).toFixed(0)}%
              </Badge>
            )}
          </div>
        )}
      </td>
      <td className="py-2 pr-3" data-testid={`training-per-slice-da-${tf}-${coin}`}>
        <DaGateCell verdict={verdict} fallbackFloor={fallbackFloor} tf={tf} coin={coin} />
      </td>
      <td className="py-2 pr-3 text-right tabular-nums text-muted-foreground">
        {fmtPct(pnl?.gross_pct_mean, 3)}
      </td>
      <td className="py-2 pr-3 text-right tabular-nums text-muted-foreground">
        {pnl?.round_trip_cost_pct == null
          ? "—"
          : `−${Number(pnl.round_trip_cost_pct).toFixed(3)}%`}
      </td>
      <td
        className={cn("py-2 pr-3 text-right tabular-nums font-medium", netClass)}
        data-testid={`training-per-slice-net-${tf}-${coin}`}
      >
        {fmtPct(pnl?.net_pct_mean, 3)}
      </td>
      <td className="py-2 pr-3 text-right tabular-nums">
        {fmtNum(pnl?.n_trades)}
      </td>
      <td className="py-2 pr-3 text-right tabular-nums text-muted-foreground">
        {pnl?.win_rate == null
          ? "—"
          : `${(Number(pnl.win_rate) * 100).toFixed(1)}%`}
      </td>
      <td className="py-2 pr-3" data-testid={`training-per-slice-live-gated-${tf}-${coin}`}>
        <LiveGatedCell tf={tf} coin={coin} entry={liveGated} />
      </td>
    </tr>
  );
}

const VERDICT_TONE: Record<EconomicVerdict, string> = {
  bleeding: "border-rose-500/60 bg-rose-500/15 text-rose-200",
  dormant: "border-amber-500/60 bg-amber-500/15 text-amber-200",
  tradeable: "border-emerald-500/60 bg-emerald-500/15 text-emerald-200",
  inconclusive: "border-zinc-500/40 bg-zinc-500/10 text-zinc-300",
};

const VERDICT_LABEL: Record<EconomicVerdict, string> = {
  bleeding: "bleeding",
  dormant: "dormant",
  tradeable: "tradeable",
  inconclusive: "inconclusive",
};

function isEconomicVerdict(v: unknown): v is EconomicVerdict {
  return v === "bleeding" || v === "dormant" || v === "tradeable" || v === "inconclusive";
}

function fmtPnl(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${Number(v).toFixed(digits)}%`;
}

function LiveGatedCell({
  tf,
  coin,
  entry,
}: {
  tf: string;
  coin: string;
  entry: LiveGatedSlice | undefined;
}) {
  if (!entry) {
    return (
      <span
        className="text-muted-foreground"
        data-testid={`training-per-slice-live-gated-empty-${tf}-${coin}`}
      >
        —
      </span>
    );
  }
  const rawVerdict = entry.economic_verdict;
  const verdict: EconomicVerdict = isEconomicVerdict(rawVerdict)
    ? rawVerdict
    : "inconclusive";
  const tone = VERDICT_TONE[verdict];
  const label = VERDICT_LABEL[verdict];
  const phrase = entry.economic_verdict_phrase ?? label;
  const loose = entry.loose_post_fee_pct_total;
  const live = entry.live_net_pnl_pct;
  const liveN = entry.live_trade_count;
  const looseTxt = fmtPnl(loose ?? null);
  const liveTxt = fmtPnl(live ?? null);
  const liveStatus = entry.live_replay_status ?? "missing";
  const dominant = entry.dominant_rejection_reason ?? null;
  const showDominant = (verdict === "bleeding" || verdict === "dormant") && !!dominant;
  const title =
    `Live-gated replay verdict: ${phrase}.\n` +
    `Loose holdout post-fee PnL: ${looseTxt}.\n` +
    `Live-gated PnL: ${liveTxt} (n=${liveN ?? "—"}, status=${liveStatus}).` +
    (dominant ? `\nDominant rejection reason: ${dominant}.` : "");
  return (
    <div
      className="flex flex-col gap-0.5"
      data-testid={`training-per-slice-live-gated-cell-${tf}-${coin}`}
    >
      <Badge
        variant="outline"
        className={cn("text-[10px] w-fit", tone)}
        title={title}
        data-testid={`training-per-slice-live-gated-verdict-${tf}-${coin}`}
      >
        {label}
      </Badge>
      <span
        className="text-[10px] tabular-nums text-muted-foreground"
        data-testid={`training-per-slice-live-gated-pnl-${tf}-${coin}`}
      >
        loose {looseTxt} · live {liveTxt}
        {liveN != null ? ` (n=${liveN})` : ""}
      </span>
      {showDominant && (
        <span
          className="text-[10px] text-amber-300"
          data-testid={`training-per-slice-live-gated-rejection-${tf}-${coin}`}
          title={`Most-common reason the production gates rejected a candidate trade on this slice during the live-gated replay.`}
        >
          rejected: {dominant}
        </span>
      )}
    </div>
  );
}

function LiveGatedFooter({
  liveGated,
}: {
  liveGated: LiveGatedReplay | null;
}) {
  if (!liveGated) {
    return null;
  }
  const status = liveGated.status;
  if (status === "missing") {
    return (
      <div
        className="text-[10px] text-muted-foreground"
        data-testid="training-per-slice-live-gated-footer-missing"
      >
        Live-gated replay: no campaign run with{" "}
        <span className="font-mono">phase7_summary.json</span> on disk yet.
      </div>
    );
  }
  if (status === "empty") {
    return (
      <div
        className="text-[10px] text-muted-foreground"
        data-testid="training-per-slice-live-gated-footer-empty"
      >
        Live-gated replay: latest run{" "}
        {liveGated.run_dir ? (
          <span className="font-mono">{liveGated.run_dir}</span>
        ) : null}{" "}
        has no per-slice block (campaign predates Task #613).
      </div>
    );
  }
  if (status === "error") {
    return (
      <div
        className="text-[10px] text-red-400"
        data-testid="training-per-slice-live-gated-footer-error"
      >
        Live-gated replay: failed to read latest summary
        {liveGated.run_dir ? (
          <>
            {" "}from <span className="font-mono">{liveGated.run_dir}</span>
          </>
        ) : null}
        .
      </div>
    );
  }
  const counts = liveGated.verdict_counts ?? {};
  const total =
    (counts.bleeding ?? 0) +
    (counts.dormant ?? 0) +
    (counts.tradeable ?? 0) +
    (counts.inconclusive ?? 0);
  return (
    <div
      className="flex flex-wrap items-center gap-1 text-[10px] text-muted-foreground"
      data-testid="training-per-slice-live-gated-footer"
    >
      <span className="mr-1">Live-gated replay verdicts ({total} slices):</span>
      <Badge
        variant="outline"
        className={cn("text-[10px]", VERDICT_TONE.tradeable)}
        title="Slices where the live-gated replay is profitable with n≥5 trades."
        data-testid="training-per-slice-live-gated-count-tradeable"
      >
        tradeable {counts.tradeable ?? 0}
      </Badge>
      <Badge
        variant="outline"
        className={cn("text-[10px]", VERDICT_TONE.dormant)}
        title="Slices where loose holdout PnL is negative but production gates abstain (live n<5)."
        data-testid="training-per-slice-live-gated-count-dormant"
      >
        dormant {counts.dormant ?? 0}
      </Badge>
      <Badge
        variant="outline"
        className={cn("text-[10px]", VERDICT_TONE.bleeding)}
        title="Slices where loose holdout PnL is negative AND the live-gated replay still trades and loses (n≥5)."
        data-testid="training-per-slice-live-gated-count-bleeding"
      >
        bleeding {counts.bleeding ?? 0}
      </Badge>
      <Badge
        variant="outline"
        className={cn("text-[10px]", VERDICT_TONE.inconclusive)}
        title="Signals disagree or the live diagnostic is missing for the slice."
        data-testid="training-per-slice-live-gated-count-inconclusive"
      >
        inconclusive {counts.inconclusive ?? 0}
      </Badge>
      {liveGated.run_dir && (
        <span className="ml-2 opacity-70">
          source: <span className="font-mono">{liveGated.run_dir}</span>
        </span>
      )}
    </div>
  );
}

function GateConstantsHeader({
  tfsInReport,
  defaultFloor,
  overrideMap,
  haveVerification,
}: {
  tfsInReport: string[];
  defaultFloor: number;
  overrideMap: Record<string, number> | null;
  haveVerification: boolean;
}) {
  if (!haveVerification) {
    return (
      <div
        className="text-[11px] text-muted-foreground"
        data-testid="training-per-slice-gate-constants-missing"
      >
        Verification block missing — DA gate falls back to{" "}
        {defaultFloor.toFixed(3)} for every slice.
      </div>
    );
  }

  return (
    <div
      className="flex flex-wrap items-center gap-1 text-[11px]"
      data-testid="training-per-slice-gate-constants"
    >
      <span className="text-muted-foreground mr-1">DA gate floor:</span>
      <Badge
        variant="outline"
        className="text-[10px] border-zinc-500/40 text-zinc-300"
        title="Default directional-accuracy gate."
        data-testid="training-per-slice-gate-default"
      >
        default · {defaultFloor.toFixed(3)}
      </Badge>
      {tfsInReport.map((tf) => {
        const override = overrideMap && typeof overrideMap[tf] === "number"
          ? Number(overrideMap[tf])
          : null;
        const floor = override ?? defaultFloor;
        // Higher floor = stricter gate (tighter); lower floor = looser.
        const tighter = override !== null && override > defaultFloor;
        const looser = override !== null && override < defaultFloor;
        const tone = tighter
          ? "border-amber-500/50 bg-amber-500/10 text-amber-300"
          : looser
            ? "border-sky-500/50 bg-sky-500/10 text-sky-300"
            : "border-zinc-500/40 text-zinc-400";
        const annotation = tighter ? "tighter" : looser ? "looser" : null;
        const title = override !== null
          ? `${tf} override: floor ${floor.toFixed(3)} (${annotation} than default ${defaultFloor.toFixed(3)}).`
          : `${tf}: default floor ${floor.toFixed(3)}.`;
        return (
          <Badge
            key={tf}
            variant="outline"
            className={cn("text-[10px]", tone)}
            title={title}
            data-testid={`training-per-slice-gate-${tf}`}
          >
            {tf} · {floor.toFixed(3)}
            {annotation && (
              <span className="ml-1 opacity-70">({annotation})</span>
            )}
          </Badge>
        );
      })}
    </div>
  );
}

function DaGateCell({
  verdict,
  fallbackFloor,
  tf,
  coin,
}: {
  verdict: VerificationVerdict | undefined;
  fallbackFloor: number;
  tf: string;
  coin: string;
}) {
  const da =
    typeof verdict?.directional_accuracy === "number"
      ? verdict.directional_accuracy
      : null;
  if (da === null) {
    return (
      <span
        className="text-muted-foreground"
        data-testid={`training-per-slice-da-empty-${tf}-${coin}`}
      >
        —
      </span>
    );
  }
  const floor =
    typeof verdict?.min_directional_accuracy_applied === "number"
      ? verdict.min_directional_accuracy_applied
      : fallbackFloor;
  // verification.classify_slice fails slices with DA <= floor.
  const aboveGate = da > floor;
  const tone = aboveGate
    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
    : "border-rose-500/50 bg-rose-500/10 text-rose-300";
  const label = aboveGate
    ? `above gate (${tf} floor ${floor.toFixed(3)})`
    : `below gate (${tf} floor ${floor.toFixed(3)})`;
  const title = `DA ${da.toFixed(3)} ${aboveGate ? ">" : "≤"} ${tf} floor ${floor.toFixed(3)}.`;
  return (
    <div
      className="flex flex-col gap-0.5"
      data-testid={`training-per-slice-da-cell-${tf}-${coin}`}
    >
      <span className="font-mono tabular-nums">{da.toFixed(3)}</span>
      <Badge
        variant="outline"
        className={cn("text-[10px] w-fit", tone)}
        title={title}
        data-testid={`training-per-slice-da-badge-${tf}-${coin}`}
      >
        {label}
      </Badge>
    </div>
  );
}
