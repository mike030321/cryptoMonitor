# Overview

The Crypto AI Monitor (Nexus) is a real-time dashboard for advanced crypto market analysis and prediction. It features 10 AI agents that forecast cryptocurrency price movements for 10 high-risk/high-reward altcoins. The platform leverages AI consensus and robust data infrastructure to provide actionable insights and demonstrate AI capabilities in financial forecasting. Its primary purpose is to deliver timely market intelligence and serve as a tool for understanding AI-driven investment strategies in the volatile cryptocurrency market.

# User Preferences

I prefer iterative development with clear communication on major changes. Please ask before implementing significant architectural shifts or feature alterations. I value clear, concise explanations and well-structured code.

# System Architecture

## Core Technologies
- **Monorepo**: pnpm workspaces
- **Backend**: Node.js 24, TypeScript 5.9, Express 5
- **Database**: PostgreSQL with Drizzle ORM
- **Validation**: Zod
- **Frontend**: React, Vite, Tailwind CSS, shadcn/ui, Recharts
- **AI/ML**: LightGBM for quantitative predictions, a supervisory Meta-Brain for sizing/suppression.

## Architectural Design

### Monitoring Engine
Operates every 60 seconds, using primary/fallback data sources, and supports multi-timeframe predictions (5m, 1h, 2h, 6h, 1d). It integrates the Fear & Greed Index, seeds historical OHLCV data, specializes agents per coin, and performs pattern analysis (RSI, MACD, Bollinger Bands, EMA, ATR).

### Prediction Resolution & Scoring
Predictions are resolved based on price movements against thresholds, categorized as correct, wrong, or neutral, with a scoring system rewarding accuracy and confidence.

### Confidence Calibration & Weighted Model Voting
AI confidence is adjusted based on historical accuracy, and model votes are weighted by per-coin historical accuracy.

### Temporal Pattern Chains & Regime Detection
A regime detector classifies market conditions (bull/bear/sideways/volatile) and injects this, along with 2-step and 3-step sequence patterns, into AI prompts.

### Trend-Aware Direction Bias
A post-prediction filter penalizes counter-trend predictions and boosts with-trend predictions.

### Agent Specialization
Calculates per-agent, per-coin affinity scores based on rolling accuracy, PnL, and prediction count for dynamic coin allocation.

### Best Pick Consensus Engine
Weighs predictions by agent score, accuracy, confidence, timeframe weight, and assesses timeframe alignment for success probability.

### AI Agents
Ten distinct AI agents specialize in specific trading strategies.

### Paper Trading System
Agents use virtual portfolios with conservative position sizing, minimum confidence thresholds, Kelly criterion-based sizing, real-world fees, ATR-based stop-loss/take-profit, and a circuit breaker for consecutive losses.

### Prediction Resolution Thresholds
Lower thresholds for meme coin volatility, with a neutral zone and calibrated confidence blending.

### Data Persistence
All learning data is persisted in the PostgreSQL database.

### Validation Framework
Generates reports comparing system performance against baselines, tracking calibration error, profit factor, and accuracy.

### ML Engine Sidecar
A separate Python FastAPI service handles numeric feature engineering and trained-model inference, including health checks and automated retraining.

### 5m Candle Data Pipeline
The 5m timeframe powers the deepest training window (305-day contiguous gate
in `_evaluate_5m_gate`). Two daemon threads inside ml-engine keep it healthy:

- **Daily HEAD top-up** (`app/scheduled_5m_topup.py`) — pulls the last 7 days
  every 24h so the dashboard never silently drifts.
- **Weekly TAIL backfill** (`app/weekly_5m_tail_backfill.py`) — runs
  `scripts/backfill_5m_extend.py` on a 7d cadence (and once on every
  ml-engine startup, after a 120s delay) to extend the deep historical bar
  back up to 320 days. Holds a Postgres advisory lock distinct from the
  daily top-up's so the two daemons never argue. Lives in-process because
  the project is at the 10-workflow registration ceiling.

Both daemons use the smart fetcher `fetch_5m_smart` from
`scripts/backfill_history.py`, which routes per-coin to the deepest
available source:

- **Coinbase Exchange** is primary for the 9 coins that list there
  (PEPE, BONK, FLOKI, WIF, SEI, RNDR, INJ, TIA, WLD). Coinbase has ~10y of
  5m history vs OKX's ~60d, so it's the only path to the 305d gate.
- **OKX** is the fallback for `jupiter-exchange-solana` (no Coinbase
  product listing). JUP is the canonical member of
  `KNOWN_5M_PARTIAL_COINS` in `run_full_training_campaign.py` — exempt
  from the contiguous_days clause but still subject to density,
  gap_rate, and synthetic_rows checks. The verdict is tagged
  `partial_history_exempt=True` so audits show the waiver explicitly.

The actual source label flows through `insert_candles_batch(..., source=...)`
and the progress journal's per-coin `"source"` field, so an operator can
always see who served what.

### Top-Up Counter Persistence
The 5m top-up scheduler's run counters (`runs_total`, `ticks_total`,
`rows_inserted_total`, `alerts_emitted_total`, `skips_locked_total`,
`stuck_replica_alerts_total`) are persisted to a single `app_settings`
row via `_save_counters_to_settings` and hydrated on startup via
`_load_counters_from_settings` so the admin panel reflects the true
lifetime totals across ml-engine restarts. Counter writes happen inside
the same advisory lock the tick already holds; DB write failures fall
back to in-memory.

### 5m Contiguity Tolerance (Task #604)
Coinbase Exchange suffered a single platform-level outage on
**2025-10-25** lasting ~6h-6.5h that affected ALL nine of the
Coinbase-served coins simultaneously. Without tolerance, that one
event collapses the strict-consecutive `contiguous_days` measure for
BONK / PEPE / CELESTIA from ~320d to 66-129d and they fail the 305d
gate even though their density is ≥99.5% and gap_rate ≤0.7%.

`app/contiguity.py::CONTIGUITY_TOLERANCE_SECONDS = {"5m": 25200}`
(7h = 84 missing 5m buckets) is the tightest principled value that
absorbs the worst observed outage (390min on WLD) with ~30 min of
margin. The shared helper `compute_longest_contiguous_run` is used
by BOTH `_evaluate_5m_gate` (training campaign) and
`measure_contiguous_5m` (top-up health alert) so the dashboard and
the campaign never disagree about the same number.

Tolerance ONLY loosens the longest-run measurement. Density,
gap_rate, synthetic_rows, and the 305d threshold remain strict —
DOGWIF / FLOKI / WLD / INJ / RENDER still fail honestly on
density/gap_rate. Audit reasons surface BOTH numbers
(`days=320 [strict=128, tolerance=420m]`) so an operator can always
see what drove the verdict.

Post-Task-#604 5m universe (verified 2026-04-29):
- **PASS (4)**: jupiter-exchange-solana (exempt), bonk, pepe,
  celestia (tolerated outage)
- **FAIL (5, honest)**: dogwifcoin, floki-inu, injective-protocol,
  render-token, worldcoin-wld

Follow-up work tracks back-filling the 2025-10-25 outage from a
non-Coinbase source so this tolerance can later be tightened or
removed entirely.

### Brain Convention
- **Quant Brain**: LightGBM model for paper trades.
- **Meta-model**: Supervisory layer for sizing/suppression, not direct trading.
- **Meta-Brain**: Colloquial term for the Meta-model and its API server adapter.

### Deterministic Quant Only
The system is purely deterministic, relying solely on quantitative inputs for trade decisions. No LLM, news, or sentiment input is used in the trade-decision path or dashboard.

### Supervisory Meta-Brain
The `market_meta_brain` package is a supervisory layer that shapes trust, sizing, caution, suppression, and defensive mode, without directly predicting price or placing trades.

### Quant Training Contract
Uses real `price_history` rows only, ensures point-in-time features, uses walk-forward validation, and targets `net_pnl_after_costs_pct` and `realized_vol_next_horizon`.

### Deterministic Agent Registry
A typed, deterministic strategy-profile registry defines agent behaviors and retirement rules, ensuring consistent execution.

### Dashboard Rework (4-executor fleet)
The dashboard is organized around four deterministic executors (`momentum_core`, `mean_reversion_core`, `breakout_core`, `volatility_defensive`), including a `FamilyFleet` summary, drill-down pages, and `BenchmarksPanel`.

### Per-Timeframe Role Layer
Each timeframe carries an explicit role (`trade`, `shadow`, `context`, or `disabled`) stored in `shared/timeframe-roles.json`, enforced at both the brain promotion gate and paper-trader execution path.

### LightGBM 3-Class Architecture — Structural Ceiling Reached (Outcome B, 2026-04-30)
Task #640 ran the final max-honest-data retrain (1100-day 6h window backfilled from OKX → 18,836 new candles; 1100-day 1d window already at source maximum) and produced **0 champions / 18 attempted slices** under the unmodified MTTM gate (`MIN_DIRECTIONAL_ACCURACY=0.50`, `MIN_DIRECTIONAL_ACCURACY_PER_TF.1d=0.53`, `MIN_HOLDOUT_ROWS=200`, `PREDICTION_COLLAPSE_TOP_SHARE=0.85`). Best DA observed across all 18 slices: pepe/6h widened 0.4476, bonk/1d 0.4313 — both well below their respective gates. 16/18 slices below baseline DA, 11/18 with collapsed call_share >0.85, 4/18 with AUC <0.50 (worse than random). The 2.3× expansion of 6h training rows (12 mo → 28-30 mo, two market cycles) produced **no directional-accuracy improvement** vs the 12 mo runs from earlier today on any of the 9 widened-window 6h slices — confirming that data depth is **not** the binding constraint. Structural verdict: the current 3-class `{UP, STABLE, DOWN}` label scheme with vol-scaled threshold is the binding ceiling. Verdict report at `artifacts/ml-engine/models/reports/20260429T230000Z/task-640-final-max-honest-data-verdict.md`; sibling next-research proposal at `next-research-proposal-quintile-labels.md` (registered as project task #642). The configuration change made (adding `ML_LOOKBACK_DAYS_6H=1100` to `artifacts/ml-engine/.replit-artifact/artifact.toml` env blocks) is preserved — it routes the trainer to the already-backfilled 6h depth and does **not** weaken any gate. `app_settings.quant_brain_enabled` remains `false`. BTC/ETH/SOL ML expansion deferred at Checkpoint 1 (OKX 95-day funding/OI cap → would fail Checkpoint 2 NaN-share gate; needs `OKX_SYMBOLS`/`OKX_SWAP_BASE`/`CROSS_MARKET_LIQ_SOURCES` entries plus a `btc_lead_ret_5m` self-leak guard in `app/training/labels.py` before the majors can be trained honestly). Next research direction: label re-engineering (quintile-based labels with explicit abstain class) — see follow-up #642.

### DS health probe — Task #670
The diagnostic-sandbox auto-disable evaluator computes peak-to-trough drawdown over **every** closed BTC/5m trade since `enabledAt` and trips once it eats through the −5% floor — by which point the lane is already off and operators only have an after-the-fact `mttm_disable_reason` to explain it. The B4 sweep showed the same model can move from −4.52% to −5.55% holdout drawdown with a 12-minute window shift, so once the lane is running live a small drift can silently push the champion past the floor. `getDiagnosticSandboxHealth()` (in `artifacts/api-server/src/lib/mttm.ts`) exposes a cheap **trailing-window** drawdown probe: same `paper_trades` stream the evaluator already watches, sliced to the most-recent `nNegPnl` closed DS trades (default 50). Returns `trailing_drawdown_pct`, `headroom_pct = trailing − floor`, and a `needs_refit` flag that flips at `floor × 0.8` (default −4%) so a re-fit can be staged before the floor trips. 30 s in-process cache keyed on (mode, enabled, enabledAt, btcVersion, floor, window). Surfaced at `GET /api/diagnostic-sandbox/health` (OpenAPI schema `DiagnosticSandboxHealth`) and rendered as a third color-coded chip in `DiagnosticSandboxBanner` (emerald = healthy, amber = needs refit, red = floor breached). No model fit, no inference, no Python round-trip.

### Diagnostic Paper Sandbox (BTC/5m, family-C, beta-calibrated) — Task #659
A second MTTM lane (`mttm_mode='diagnostic_sandbox'`) hard-pinned to a single `bitcoin/5m` slot at fixed 0.5% sizing. Lives alongside the default 16-slot lane and is mutually exclusive at the data level (`getMttmConfig()` collapses universe + sizing inside the cache primer). Auto-disables on either drawdown ≤ −5% or n ≥ 50 trades with negative aggregate PnL. Operator-toggleable via admin-key gated routes `POST /api/diagnostic-sandbox/{mode,btc-version,evaluate}`; status visible at public `GET /api/diagnostic-sandbox/status` and the teal `DiagnosticSandboxBanner` rendered just below `MttmBanner` on the dashboard. Registry now stamps `beta_calibration` (long/short Platt {a,b,c}), `calibration_status` enum, and `scope_constraint` payload; serving applies a `BETA_EPS=1e-6` clamp + `_enforce_scope` guard via `predict_one(X, *, coin_id, timeframe)` so a calibrated model cannot be asked to predict outside its declared scope. Phases 1, 2, 4, 5 SHIPPED; Phase 3 (re-fit + `promote_shadow_to_serving`) and Phase 6 (10 paper proofs) BLOCKED on no provisioned Postgres + no on-disk BTC artifact in this branch — the sample-data B3 re-fit returned DD=−5.44% which is past the DS auto-disable floor and triggers the spec's "drifted out of acceptable B3 bands" termination clause. Verdict report: `artifacts/ml-engine/reports/task-C-BTC-diagnostic-20260430T151032Z.md`. Hard rules respected: `quant_brain_enabled` untouched (still false), no raw-SQL promotion path introduced, no ETH in DS universe, `defaultState()` and `trading-frictions.json` unchanged.

### UI/UX Decisions
The frontend, built with React, Vite, Tailwind CSS, and shadcn/ui, provides a dashboard with agent cards, performance charts, a filterable prediction feed, market data, and dedicated pages for family drill-downs and archived agents.

# External Dependencies

- **CoinMarketCap Pro API**: Primary source for cryptocurrency market data.
- **CoinGecko API**: Fallback for cryptocurrency market data.