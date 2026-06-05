# Why the quant brain (and family) is not trading — final diagnosis 2026-04-29 21:00 UTC

## TL;DR

The quant brain is producing dead predictions (`direction='stable'`,
`confidence=0`) for every coin because **no LightGBM model has ever been
promoted to `champion` in `model_registry`**. All 254 active rows are in
state `shadow`. The trader correctly does nothing with zero-confidence
"stable" signals, and silently — no skip event is even written, which is
why the last `skip_events` row is from 2026-04-23 15:17 UTC, six days ago.

The reason no model has been promoted is **not** a configuration / flag
problem and **cannot** be fixed by toggling switches. Every recent retrain
has measured directional accuracy below the 0.50 MTTM gate on every coin
on every horizon tested. Per the prior
`mttm-6h-retrain-verdict.md` (architect, same day), this is a
"feature/label problem, not a hyperparameter problem".

The data-pipeline fix landed earlier this session (BTC/ETH/SOL
`okx_backfill_mid_v1` rows; `btc_lead_ret_5m` / `eth_lead_ret_5m` went
100 % NaN → 0 % NaN; merged 8-coin parquet
`models/datasets/6h_20260429T205500Z.parquet` 11 168 rows) was a
prerequisite to give retrains *correct* features, but it does not by
itself create model edge.

## Evidence

### 1. `model_registry`: zero champions, ever

```
SELECT state, is_active, COUNT(*) FROM model_registry GROUP BY 1,2;
 state  | is_active | count
--------+-----------+-------
 shadow | t         |   254
```

There is no `champion` row, period. Including from today's retrains
(2026-04-29 14:17–18:27 UTC, all 1h/2h/6h slots + meta + specialists +
pooled): every single one is `shadow`.

### 2. The 4 active "Core" agents emit dead predictions

```
SELECT id, name, is_active FROM agents WHERE is_active = true;
 28 | Momentum Core         | t
 29 | Mean Reversion Core   | t
 30 | Breakout Core         | t
 31 | Volatility Defensive  | t
```

Last 24 h: 5 336 predictions written, 100 % with
`direction='stable'`, `confidence=0`, `source=NULL`,
`raw_confidence=NULL`. Sample:

```
 64169 | Momentum Core | worldcoin-wld | stable | 0 | NULL | 2026-04-29 21:04:35
 64168 | Momentum Core | render-token  | stable | 0 | NULL | 2026-04-29 21:04:35
 64166 | Momentum Core | worldcoin-wld | stable | 0 | NULL | 2026-04-29 21:03:35
```

The other 24 agents (the original `ai-bots` family — Momentum Max,
Hybrid-X-Y, Pattern Alpha, Scalper Steve, Volume Victor, Breakout
Bella, etc.) were mass-archived at 2026-04-24 16:54:20 UTC.

### 3. Trade flow is dead, not blocked

Daily paper-trade counts:

```
 2026-04-21 | 110
 2026-04-22 | 147
 2026-04-23 |  37   ← last day quant family traded
 2026-04-24 |  10   ← only DCA + Circuit Breaker from here on
 2026-04-25 |  10
 2026-04-28 |  10
 2026-04-29 |  10
```

The 10/day from Apr 24 onward are all from agent id 17 ("DCA + Circuit
Breaker"), which is a non-brain rule-based strategy. Quant + family
trade count: zero for six days.

The last `skip_events` row is also Apr 23 15:17. Zero in the last six
days. Zero in the last 24 h. The trader is not even *attempting* to
trade quant signals — it sees `confidence=0` and falls out of the loop
before recording a skip.

### 4. The gate is failing on edge, not on plumbing

From the prior architect verdict
(`models/reports/20260429T182338Z/mttm-6h-retrain-verdict.md`), 6h
universe (8 coins, post-#633 ML_TINY_SLICE_THRESHOLD=600 experiment):

| Coin     | DA (need ≥0.50) | call_share (need ≤0.95) |
|---------:|:---------------:|:------------------------:|
| bonk         | 0.417 | 0.957 ✗ |
| celestia     | 0.396 | 0.975 ✗ |
| dogwifcoin   | 0.409 | 0.936 |
| floki-inu    | 0.397 | 0.971 ✗ |
| injective    | 0.419 | 0.911 |
| jupiter      | 0.392 | 0.939 |
| pepe         | 0.419 | 0.650 |
| render-token | 0.434 | 0.957 ✗ |

0/8 pass DA. Mean 0.41. The hyperparameter lever made log-loss WORSE
on all 8 (mean +0.45 nats).

1h is not better. Spot-check on bonk 1h
(`models/bonk/1h/20260429T140238Z/manifest.json`):

```
metrics.directional_accuracy = 0.388
metrics.auc                  = 0.520
metrics.log_loss             = 2.64
directional_call_share       = 0.745  (holdout, n=1715)
```

DA is 0.388 — **worse** than 6h.

1d (Task #634, separate run) — same outcome: 0/8 cleared, all flagged
`below_coinflip` or `directional_call_regression`.

### 5. `quant_brain_enabled` is unset

```
SELECT key, value FROM app_settings WHERE key = 'quant_brain_enabled';
(0 rows)
```

Even if it were set to `true`, the served predictor would still emit
`confidence=0` because there is no champion model to back it.

## What this means

The quant brain is not trading because **the LightGBM models being
trained right now genuinely cannot predict 5m / 1h / 6h / 1d direction
better than a coin flip on this 8-coin universe with the current feature
set and label scheme**. The MTTM gate (DA ≥ 0.50, call_share ≤ 0.95) is
correctly refusing to promote them.

This is not a bug. It is the gate doing its job.

## What I will NOT do (per stated policy)

- Will not flip `quant_brain_enabled` (no flag flips).
- Will not lower the DA / call-share gates (no gate weakening).
- Will not force-promote shadow → champion (no friction edits, no meta-brain authority).
- Will not synthesize features or labels to coax DA above 0.50 (no synthetic data).
- Will not unarchive the original ai-bots agents to "look like" quant is trading.

## Legitimate paths forward (for the user to choose from)

1. **Feature engineering** — the architect's verdict is explicit: this is
   a feature/label problem. New informative features (cross-asset,
   regime-aware, microstructure beyond what we already have) are the
   only honest path to crossing 0.50 DA.

2. **Label re-engineering** — the current 3-class
   {DOWN, STABLE, UP} with vol-scaled thresholds may be miscalibrated
   for these horizons / coins. A binary {UP, NOT-UP} or quintile-based
   label may behave better; this is research, not a config change.

3. **Universe change** — the 8-coin meme/L1-alt universe is
   particularly noisy. Adding BTC/ETH/SOL or majors as quant-tradable
   slots could help (data is now there after today's backfill).

4. **Lower-frequency / event-driven horizons** — funding-rate flips,
   macro releases, on-chain liquidation cascades. Different problem,
   different model class.

5. **Explicit user override** — if the user authorizes a temporary
   gate relaxation for diagnostic purposes (e.g. DA ≥ 0.45 for paper-
   only on 6h), that becomes a deliberate, logged policy change rather
   than a back-door fix. I will not make that change without explicit
   instruction.

## Files referenced

- `artifacts/api-server/src/lib/brain-promotion-gate.ts` — gate criteria
- `artifacts/ml-engine/app/registry_lifecycle.py` — shadow → champion logic
- `artifacts/ml-engine/models/reports/20260429T182338Z/mttm-6h-retrain-verdict.md`
- `artifacts/ml-engine/models/reports/20260429T205500Z/data-pipeline-fix-verdict.md`
- `artifacts/ml-engine/models/datasets/6h_20260429T205500Z.parquet`
