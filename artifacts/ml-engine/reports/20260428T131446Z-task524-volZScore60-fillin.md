# Task #524 — fill-in retrain & verification report

_Generated 20260428T131446Z • elapsed 438.4s_

Fills in the slices the main retrain skipped:
- `sei-network` @ 1d/2h/1h (the size-picker chose sei-less parquets)
- every default coin + `__pooled__` @ 5m/1m (not in the main retrain's tf list)

## Versions trained

| coin | tf | version | status |
|---|---|---|---|
| sei-network | 1d | `20260428T131525Z` | trained |
| sei-network | 2h | `20260428T131539Z` | trained |
| sei-network | 1h | `20260428T131543Z` | trained |
| pepe | 5m | `20260428T131547Z` | trained |
| bonk | 5m | `20260428T131552Z` | trained |
| floki-inu | 5m | `20260428T131556Z` | trained |
| dogwifcoin | 5m | `20260428T131601Z` | trained |
| render-token | 5m | `20260428T131605Z` | trained |
| injective-protocol | 5m | `20260428T131608Z` | trained |
| celestia | 5m | `20260428T131612Z` | trained |
| worldcoin-wld | 5m | `20260428T131619Z` | trained |
| jupiter-exchange-solana | 5m | `20260428T131623Z` | trained |
| __pooled__ | 5m | `20260428T131900Z` | trained |
| pepe | 1m | `20260428T131905Z` | trained |
| bonk | 1m | `20260428T131910Z` | trained |
| floki-inu | 1m | `20260428T131918Z` | trained |
| dogwifcoin | 1m | `20260428T131923Z` | trained |
| render-token | 1m | `20260428T131932Z` | trained |
| injective-protocol | 1m | `20260428T131946Z` | trained |
| celestia | 1m | `20260428T132005Z` | trained |
| worldcoin-wld | 1m | `20260428T132016Z` | trained |
| jupiter-exchange-solana | 1m | `20260428T132022Z` | trained |
| __pooled__ | 1m | `20260428T132204Z` | trained |

## Verification

- passed: **True**
- slices_promoted: 7
- slices_no_lift: 0
- slices_below_coinflip: 7
- slices_insufficient_sample: 0
- slices_contract_failed: 0
- slices_untrained: 29
- slices_cadence_mixed: 0
- slices_directional_call_regression: 9
- slices_pooled_vocab_too_small: 0
- slices_promoted_baseline: 1
- per-slice verdicts persisted: written=23 skipped=29
