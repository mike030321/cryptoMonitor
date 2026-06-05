# Task #667 — B4 promote + DS validate

- run_id: `20260430T204958Z`
- target: `bitcoin/5m` version `20260430T204702Z` (model_id `lightgbm`)
- base_url: `http://localhost:80/api`

## Manifest check
- served_predictor_kind: `dual_binary_head`
- calibration_method: `beta`
- calibration_status: `under_confident_documented`
- label_family: `C_post_cost`
- abstain_tau: `0.3015189769273562`
- friction_threshold_pct: `0.5`
- scope_constraint: `{"coin_id": "bitcoin", "timeframe": "5m", "candidate": "C_post_cost", "label_family": "C_post_cost", "allowed_universe": ["bitcoin:5m"]}`

## Promotion
- shadow_row_id: `263`
- promoted_id: `263`
- previous_champion_id: `None`

## HTTP

- POST btc-version status=`200` body=`{"btcVersion": "20260430T204702Z", "ready": false}`
- POST mode status=`200` body=`{"mode": "diagnostic_sandbox", "universe": [{"coinId": "bitcoin", "timeframe": "5m", "version": "20260430T204702Z"}], "maxPositionPct": 0.005, "ready": true}`

## 10 paper proofs (POST /diagnostic-sandbox/evaluate)

| i | status | tripped | kind | reason |
|---:|---:|:--|:--|:--|
| 1 | 200 | False |  |  |
| 2 | 200 | False |  |  |
| 3 | 200 | False |  |  |
| 4 | 200 | False |  |  |
| 5 | 200 | False |  |  |
| 6 | 200 | False |  |  |
| 7 | 200 | False |  |  |
| 8 | 200 | False |  |  |
| 9 | 200 | False |  |  |
| 10 | 200 | False |  |  |

- any_tripped: `False`
- all_clean: `True`
