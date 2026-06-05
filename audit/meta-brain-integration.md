# Meta-Brain Integration — Task #373

Supervisory `market_meta_brain` layer vendored into `ml-engine` and wired
as an advisory signal above the deterministic LightGBM quant brain.

## Invariant: No price prediction, no trade placement

The meta-brain **shapes** trust/sizing/caution/suppression/defensive-mode.
It **does not** predict price, produce signals, or place trades. All
LLM / news / sentiment remain excluded from live execution by the
existing guards (`no-llm-fields-*` parity tests).

## Wiring

### Python (ml-engine)

- `vendor/market_meta_brain/` — the vendored package (not edited here).
- `app/meta_brain.py` — singleton service, path injection,
  `Checkpointer` persistence under `models/meta_brain_state/`, LRU
  cache of tick_id → directive (size 2048).
- Endpoints (mounted at `/ml`):
  - `GET  /ml/meta-brain/health`
  - `POST /ml/meta-brain/evaluate`   (batch of slices → directive)
  - `POST /ml/meta-brain/record-outcome` (closes the learning loop)
  - `GET  /ml/meta-brain/stats`
- Service lifecycle managed in `app/main.py` lifespan (init + shutdown).

### TypeScript (api-server)

- `src/lib/meta-brain/contract.ts` — Zod schemas for batch payload +
  directive, with finite/bounded validation. `MetaBrainDirective` is
  clamped `0..1.5` for trust_multiplier, `0..1` for caution, etc.
- `src/lib/meta-brain/client.ts` — 250 ms timeout HTTP client; never
  blocks the tick loop. Rate-limited logging on failure.
- `src/lib/meta-brain/fallback.ts` — neutral directive (trust=1,
  caution=0, exploration=0, no suppressions) returned on any error.
- `src/lib/meta-brain/adapter.ts` — **single source of truth** for
  personality → strategy_family mapping. Keys: `momentum`,
  `mean_reversion`, `breakout`, `volatility_forecaster`, `baseline`.
- `src/lib/meta-brain/index.ts` — public surface:
  `collectSlice`, `flushTick`, `getFamilySizeMultiplier`,
  `resolveStrategyFamily`, `getActiveDirective`, `bindTradeToTick`,
  `resolveTickForTrade`, `sendRecordOutcome`.
- `src/lib/monitor.ts`:
  - `collectSlice(...)` per prediction (right after shadow recording,
    before the no-trade-zone branch).
  - `flushTick()` once at end of cycle, before the `monitoringState`
    update and budget log.
- `src/lib/paper-trader.ts`:
  - `getFamilySizeMultiplier(family)` composed with the existing quant
    `metaSizeMultiplier` — both subject to the same `clampMetaSizeMultiplier`
    `[0.5, 1.5]` band so the brain cannot widen risk beyond existing limits.
  - `bindTradeToTick(tradeId, tickId)` right after paper-position insert.
  - `sendRecordOutcome({...})` fired after the trade journal's
    `closedTradeRow` lookup on close — tick_id resolved from the open-time
    binding; missing outcome sub-metrics passed as `0.0`.

## Safety properties

1. **Gated.** `META_BRAIN_ENABLED=1` required; otherwise every call is
   a neutral-fallback no-op.
2. **Shadow mode.** `META_BRAIN_SHADOW=1` records + learns but returns
   neutral directives to sizing — zero P&L impact while validating.
3. **Non-blocking.** Every wiring site is wrapped in try/catch and
   bounded by a 250 ms client timeout. The trade path never awaits a
   failing meta-brain call.
4. **One-tick lag.** The directive produced by tick N shapes tick N+1's
   sizing. Justified because the 30 s monitor cadence is << 5 m (the
   shortest active timeframe).
5. **Clamp preservation.** Size multipliers pass through
   `clampMetaSizeMultiplier([0.5, 1.5])`; no change to the outer
   position-cap constants (`MAX_POSITION_PCT`, `MAX_CASH_PER_POSITION_PCT`).
6. **No DB column added.** Trade → tick_id binding is an in-memory
   `Map` on the api-server. On restart the map is empty; any trades
   opened before restart simply don't feed `record_outcome` (the brain
   handles sparse reward by design).

## Payload fields and TODOs

Per-slice (19 fields in the evaluate batch):
- `coin`, `timeframe`, `strategy_family`, `edge`, `confidence`,
  `calibrated_confidence`, `risk_score`, `disagreement`, `regime`,
  `volatility`, `anomaly_flags` — all sourced from the existing
  quant output / regime detector.
- Real per-slot/per-coin telemetry (Task #383): `pnl_state` and
  `drawdown_state` are computed once per cycle from the most recent
  ≤20 closed paper trades on each (coin, timeframe) within a 7-day
  window — `pnl_state` = sum(pnl)/sum(positionSize), `drawdown_state`
  = max peak-to-trough cumulative fractional return drop. `exposure`
  is per-coin (timeframe-agnostic): sum of open notional / fleet
  total equity, clamped to [0,1]. All three are nullable; null →
  0.0 + `missing:<field>` flag in `anomaly_flags`.
- TODO (still passed as null + `missing:<field>` flag):
  `correlation_shift`, `turnover`, `prediction_error`. (`recent_accuracy`
  and `slippage_bps` already real per Task #381.)

Portfolio-level (9 fields) is filled from the paper-portfolio snapshot.

Per-outcome (on trade close):
- Derived: `realized_pnl` (pnl/positionSize), `realized_drawdown`
  (MAE%/100), `realized_stability` (`1 - |mfe-mae|/(mfe+mae)`),
  `turnover_cost` ((entryFee+exitFee)/positionSize).
- TODO (passed as 0.0): `action_churn`, `correct_defense`,
  `correct_suppression`, `missed_edge_cost` — all require
  counterfactual or cross-tick tracking not yet in place. The brain's
  bounded plasticity handles sparse signal.

## Testing

Parity invariants preserved by existing tests:
- `no-llm-fields-in-trade-decisions.test.ts`
- `no-llm-fields-runtime.test.ts`
- `decision-engine-parity.test.ts`

End-to-end smoke: both `GET /ml/meta-brain/health` and
`GET /ml/meta-brain/stats` return 200 with the expected shape on a
fresh boot.

## Task #381 — Hardening for SHADOW-mode enable

The original Task #373 wiring shipped neutral-by-default but contained
several silent bugs that would have produced wrong learning the moment
`META_BRAIN_ENABLED=1` flipped on. Task #381 fixes those without
adding features.

### Telemetry honesty

- `monitor.ts` now sends real `recent_accuracy` (from
  `getAgentPastAccuracy`), real `slippage_bps` (`SLIPPAGE_PCT*10000`),
  real `calibrated_confidence` (the calibrated probability max — the
  previous code mis-labelled `q.rawConfidence` here), and pushes one
  fleet-level `setPortfolioTelemetry` snapshot per cycle (drawdown,
  concentration/exposure, active risk budget, kill-switch distance).
- Anything not yet truthfully measured (`pnl_state`, `drawdown_state`,
  `correlation_shift`, `exposure` at slice level, `turnover`,
  `prediction_error`, several portfolio fields) is sent as `null`. The
  adapter and the Python side both translate `null → 0.0` for the
  vendored `float`-typed dataclass and append `missing:<field>` to
  `anomaly_flags` so the trust updater can down-weight learning.
- `record_outcome` outcome sub-metrics that require counterfactual or
  fleet-level tracking (`action_churn`, `correct_defense`,
  `correct_suppression`, `missed_edge_cost`) are sent as `null`
  (wire-translated to `0.0`) instead of fabricated zeros.

### Allocation scaling

`getFamilySizeMultiplier` was previously `trust × alloc × N_families`,
which under any non-uniform softmax pinned losers to the `0.5` floor
and saturated winners at the `1.5` ceiling — turning a smooth
allocation signal into a binary on/off. Replaced with
`trust × (alloc / alloc_mean)` so a small allocation shift produces a
small sizing shift. Validated by `meta-brain-clamp.test.ts`.

### Suppression wiring

- `isFamilySuppressed(family, coin?, timeframe?)` is now wired into the
  paper-trader gate. When the brain emits `suppress_signal`,
  `suppressed_families`, or a `paused_slices` entry covering the
  current `(coin, timeframe)`, the trade is routed through the
  existing `recordSkip("meta_brain_suppress", …)` skip path — no new
  execution branch.
- Shadow-mode directives are treated as neutral by both
  `isFamilySuppressed` and `getFamilySizeMultiplier`, so suppression
  cannot fire while the brain is in shadow mode.

### Defensive mode

`clampMetaSizeMultiplier` keeps the `[0.5, 1.5]` band by default. When
the active directive's `defensive_mode === "hard"` the floor relaxes
to `0.0` so the brain can route to suppression via `mult = 0` — the
single explicit branch documented at the call site. `soft` damps by
0.7; `off` is a no-op.

### Shadow-mode tick_id

The brain returns a real `uuid` for every successful evaluate. In
shadow mode the client wraps it as `shadow:<uuid>` before caching as
the active directive. `isNeutralDirective` and `isFamilySuppressed`
both treat `shadow:*` as neutral for sizing — yet `bindTradeToTick`
preserves shadow bindings so `record_outcome` can close the learning
loop on the same tick that authorized the entry. The wire layer in
`postRecordOutcome` strips the `shadow:` prefix so the brain only
ever sees the underlying uuid it issued.

### Tick-binding retry semantics

Replaced the consume-on-read `resolveTickForTrade` with explicit
`peekTickForTrade` + `clearTickBinding`. The binding is removed only
on a confirmed `{ok: true}` response from the brain; transient
failures leave the binding in place so the next close attempt (or the
24h TTL sweep) can retry. `postRecordOutcome` now returns `boolean`,
and the adapter exposes per-cycle hit/miss counters.

### Persistence

`TRADE_TO_TICK` is snapshotted to
`<workspace>/.cache/meta_brain_state/trade_to_tick.json` (the
default; override via `META_BRAIN_STATE_DIR`) on every change
(coalesced single-flight write); `hydrateBindings()` is called on
api-server boot. The Python-side `_tick_cache` stays in-memory by
design (full directive objects), and the api-server side carries the
small int → string mapping that's enough to feed the learning loop
across either side restarting.

The Python side now hydrates `trust_by_family` from
`trust_model.json` on init (the rest of the brain components — regime
memory, episodic memory — deliberately start neutral and re-derive
from one cycle of telemetry, logged as `starts_neutral`).

### Observability

`flushTick` emits a structured `meta_brain_cycle_stats` log line every
tick with: slice count, active bindings, record_outcome cache hit/miss
counts, suppressed-tick count, defensive-mode breakdown, per-cause
fallback counters (consumed from the client), and evaluate-latency
p50/p95 (sampled, capped at 256 samples). Operators can grep this to
audit shadow-mode behaviour without enabling sizing.

### Test surface

- TS (`pnpm --filter @workspace/api-server exec node --import tsx --test`):
  - `meta-brain-fallback.test.ts` — every documented failure path
    collapses to neutral (disabled, fetch failure, HTTP 5xx, bad JSON,
    schema-invalid, allocation drift, shadow mode).
  - `meta-brain-clamp.test.ts` — the sizing clamp under each defensive
    mode and the smooth allocation factor.
  - `meta-brain-direction.test.ts` — multiplier never goes negative
    (the brain cannot flip trade direction); `suppress_signal` and
    `paused_slices` semantics; shadow/neutral never suppress.
  - `meta-brain-family-mapping.test.ts` — the (personality,
    specialist) → strategy_family mapping snapshot.
  - `meta-brain-record-outcome.test.ts` — peek/clear semantics, retry
    on transient failure, shadow prefix wire-stripping, neutral
    refusal, null-as-0 wire encoding.
- Python (`pytest tests/test_meta_brain_http.py`):
  - null-numeric-field acceptance (with auto missing flags),
  - allocation sums to 1,
  - unknown tick → `{ok: false}` (200, not 5xx),
  - record_outcome roundtrip success,
  - `trust_model.json` checkpoint hydration,
  - `/stats` endpoint surface.

All bundled in the `meta-brain-tests` validation workflow so CI runs
both halves on every change.
