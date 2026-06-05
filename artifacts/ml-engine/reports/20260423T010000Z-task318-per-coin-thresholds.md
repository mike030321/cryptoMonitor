# Task #318 — Per-coin 5m label thresholds calibrated from realized vol

**Date:** 2026-04-23
**Scope:** 5m timeframe, 5 target coins (celestia, pepe, bonk, worldcoin-wld, dogwifcoin)
**Source dataset:** `artifacts/ml-engine/models/datasets/5m_20260423T080219Z.parquet`
**Verification report:** `artifacts/ml-engine/models/report.json` (generated 2026-04-23T08:01:37Z)

## 1. Problem

The previous per-coin 5m `label_thresholds_percent_per_coin` override (task #120, 2026-04-22) collapsed every
coin to a single tight 0.04 % band so the 10 quietest coins would generate enough UP/DOWN labels to learn
direction. On the 5 coins targeted by the verification gate this back-fired:

| coin           | training-window MAD of \|forward\_return\| | old override | training rows in DOWN/UP |
|----------------|--------------------------------------------|--------------|--------------------------|
| celestia       | 0.85 %                                     | 0.04 %       | 96 %                     |
| pepe           | 0.81 %                                     | 0.04 %       | 95 %                     |
| bonk           | 0.95 %                                     | 0.04 %       | 96 %                     |
| worldcoin-wld  | 0.97 %                                     | 0.04 %       | 96 %                     |
| dogwifcoin     | 0.93 %                                     | 0.04 %       | 95 %                     |

The 0.04 % band was orders of magnitude below the typical 5m move, so almost every training row landed in
DOWN or UP and the modal class share never crossed 0.50. The multinomial-logistic baseline therefore parked
near a coin-flip and the booster had no consolidation to lift on the held-out tail.

## 2. Methodology

For each of the 5 target coins:

```
threshold_pct = round(0.25 × MAD(|forward_return_5m_pct|, training 80 % window), 2)
clipped to [0.10 %, 0.30 %]
```

Constraints honoured:

- Each per-coin override stays strictly less than `outcome_thresholds_percent["5m"] = 0.35` so a model
  trade still has to clear the round-trip cost to count as "correct"
  (`test_per_coin_label_thresholds_below_outcome_thresholds` continues to pass).
- The other 5 coins (floki-inu, injective-protocol, jupiter-exchange-solana, render-token, sei-network)
  retain the 0.04 % override from task #120 because their realized 5m vol is low enough that a wider band
  would push them under the minimum-rows-per-class floor.

Resulting overrides (`shared/trading-frictions.json` →
`label_thresholds_percent_per_coin`):

| coin           | new threshold |
|----------------|---------------|
| bonk           | 0.22          |
| celestia       | 0.21          |
| dogwifcoin     | 0.24          |
| pepe           | 0.20          |
| worldcoin-wld  | 0.24          |

## 3. Test relaxation

`artifacts/ml-engine/tests/test_training.py::test_resolve_label_threshold_uses_per_coin_override` previously
asserted every override was *strictly less than* the timeframe default. Per-coin calibration explicitly
allows wider bands (when MAD justifies it), so the assertion was relaxed to "uses the override exactly".
The companion guard `test_per_coin_label_thresholds_below_outcome_thresholds` (override < outcome
threshold) is unchanged and still enforces the round-trip-cost contract.

## 4. Verification gate result (post-retrain)

Triggered via `POST /ml/admin/retrain {"timeframes":["5m"]}` against the running ML Engine. New 5m models
were promoted between 08:05 and 08:38 UTC; report regenerated at 09:10 UTC.

| coin                        | threshold | n_rows | model DA | baseline DA | Δ        |
|-----------------------------|-----------|--------|----------|-------------|----------|
| celestia                    | 0.21      | 2484   | 0.3580   | 0.4406      | -0.0826  |
| pepe                        | 0.20      | 2525   | 0.3967   | 0.4204      | -0.0238  |
| bonk                        | 0.22      | 2526   | 0.3511   | 0.4504      | -0.0993  |
| worldcoin-wld               | 0.24      | 2484   | 0.4464   | 0.3449      | **+0.1014** |
| dogwifcoin                  | 0.24      | 2484   | 0.3338   | 0.4599      | -0.1261  |
| floki-inu (control)         | 0.04      | 2484   | 0.4536   | 0.4744      | -0.0208  |
| injective-protocol (control)| 0.04      | 2484   | 0.4522   | 0.4560      | -0.0039  |
| jupiter-exchange-solana (ctrl)| 0.04    | 2484   | 0.4565   | 0.4754      | -0.0188  |
| render-token (control)      | 0.04      | 2484   | 0.4667   | 0.4556      | **+0.0111** |
| sei-network (control)       | 0.04      | 2028   | 0.4527   | 0.4621      | -0.0095  |

### Gate vs. acceptance bar

The published verification gate expects, per slice:

1. baseline DA ≥ 0.55, **and**
2. model DA > baseline DA + 0.01, **and**
3. model DA > 0.50 on a holdout of ≥ 200 rows.

None of the 10 5m slices clear bar (1) — every coin's baseline DA on the 5m holdout sits in the 0.34 – 0.48
range. Per-coin label calibration cannot move the baseline DA floor; that is a function of the
holdout-window distribution (median |r| ≈ 0.15 % on the most recent 20 % of bars) and the structural
train/holdout regime mismatch flagged in `20260423T000000Z-failure-analysis.md`. Per the task brief
("Acceptable to keep slices red if bar not met") we accept the red gate and document the outcome here
rather than over-fit thresholds to the holdout.

### What the change DID move

- **worldcoin-wld**: model DA jumped from a coin-flip baseline (0.345) to 0.446, a +0.10 lift — the only
  target coin where the booster now clearly beats the baseline on the holdout.
- **render-token** (control, untouched): also flipped to model > baseline (+0.011), which is consistent
  with a regime-shift hypothesis rather than a threshold artifact.
- For the other 4 target coins the booster regressed below the (now stronger) baseline — expected when
  the new threshold makes the modal STABLE class dominant on training but the holdout remains
  almost-entirely-STABLE under both the old and new band, so the model has less directional signal to
  learn while the baseline benefits from the cleaner class structure.

## 5. Next steps (deferred — not in scope for #318)

- The structural train/holdout regime mismatch is the dominant driver of the red gate. A walk-forward
  holdout that mirrors training-window vol, or a recency-weighted training scheme, would do more to lift
  baseline DA above 0.55 than further threshold tuning.
- worldcoin-wld is now a candidate to graduate from the red list — its 5m holdout already clears the
  n ≥ 200 row criterion (n_rows = 2484 across folds, ~497 per fold) and model DA 0.446 beats baseline
  0.345, but model DA still sits below the absolute 0.50 floor. A follow-up task can either lower the
  absolute floor for slices that show a structurally robust lift, or wait for the next regime where
  model DA also clears 0.50, before promoting the slice off the red list.
