# Task #586 — backfill verification report
_Generated at_ **2026-04-28T22:16:30.508496+00:00**
_Run by_ `artifacts/ml-engine/scripts/task586_verify_backfill.py`
Backfill source tags audited: `okx_backfill_funding_v1`, `okx_backfill_oi_v1`, `okx_backfill_mid_v1`
## Forbidden-features manifest (must be unchanged)
`shared/forbidden-features.json` — 2544 bytes — SHA-256 `ed9eafc9f6399e39b0302fc9881d4357bcb7f75194c459edec4478f2f6e9f897`
If this hash differs from the value in the previous campaign's report, the backfill **must not be merged**: the campaign specifies the manifest stays byte-identical.
## Backfilled rows in `market_signals`
_Audit timestamp_ **2026-04-28T22:16:29.954960+00:00** (UTC)
| coin_id | source | rows | earliest | latest |
|---|---|---:|---|---|
| bonk | okx_backfill_funding_v1 | 271 | 2026-01-28T16:00:00+00:00 | 2026-04-28T16:00:00+00:00 |
| bonk | okx_backfill_oi_v1 | 1444 | 2026-02-27T17:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| btc | okx_backfill_mid_v1 | 105118 | 2025-04-28T21:00:00+00:00 | 2026-04-28T20:45:00+00:00 |
| celestia | okx_backfill_funding_v1 | 543 | 2026-01-28T12:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| celestia | okx_backfill_oi_v1 | 1444 | 2026-02-27T17:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| dogwifcoin | okx_backfill_funding_v1 | 543 | 2026-01-28T12:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| dogwifcoin | okx_backfill_oi_v1 | 1444 | 2026-02-27T17:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| eth | okx_backfill_mid_v1 | 105117 | 2025-04-28T21:05:00+00:00 | 2026-04-28T20:45:00+00:00 |
| floki-inu | okx_backfill_funding_v1 | 271 | 2026-01-28T16:00:00+00:00 | 2026-04-28T16:00:00+00:00 |
| floki-inu | okx_backfill_oi_v1 | 1444 | 2026-02-27T17:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| injective-protocol | okx_backfill_funding_v1 | 271 | 2026-01-28T16:00:00+00:00 | 2026-04-28T16:00:00+00:00 |
| injective-protocol | okx_backfill_oi_v1 | 1444 | 2026-02-27T17:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| jupiter-exchange-solana | okx_backfill_funding_v1 | 543 | 2026-01-28T12:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| jupiter-exchange-solana | okx_backfill_oi_v1 | 1444 | 2026-02-27T17:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| pepe | okx_backfill_funding_v1 | 271 | 2026-01-28T16:00:00+00:00 | 2026-04-28T16:00:00+00:00 |
| pepe | okx_backfill_oi_v1 | 1444 | 2026-02-27T17:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| render-token | okx_backfill_funding_v1 | 543 | 2026-01-28T12:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| render-token | okx_backfill_oi_v1 | 1444 | 2026-02-27T17:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| sei-network | okx_backfill_funding_v1 | 543 | 2026-01-28T12:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| sei-network | okx_backfill_oi_v1 | 1444 | 2026-02-27T17:00:00+00:00 | 2026-04-28T20:00:00+00:00 |
| sol | okx_backfill_mid_v1 | 105117 | 2025-04-28T21:10:00+00:00 | 2026-04-28T20:50:00+00:00 |
| worldcoin-wld | okx_backfill_funding_v1 | 271 | 2026-01-28T16:00:00+00:00 | 2026-04-28T16:00:00+00:00 |
| worldcoin-wld | okx_backfill_oi_v1 | 1444 | 2026-02-27T17:00:00+00:00 | 2026-04-28T20:00:00+00:00 |

**Anti-leak check**: future-dated rows = `0` (must be 0).
## Cached training-dataset coverage
Each block summarises the freshest `models/datasets/<tf>_<TS>.parquet` post-refresh. Columns covered: `funding_rate`, `open_interest_z`, `liquidations_1h_usd`, `bid_ask_spread_bps`, `btc_lead_ret_5m`, `eth_lead_ret_5m`.

### tf=`1m` — `models/datasets/1m_20260428T202028Z.parquet` (51087 rows) — **stale (predates backfill)** ⚠
| feature | rows nonzero | frac | q05 | q50 | q95 |
|---|---:|---:|---:|---:|---:|
| funding_rate | 37342 | 0.7309 | -0.00039100397843867536 | 1.9459417671896517e-05 | 9.999999747378752e-05 |
| open_interest_z | 37331 | 0.7307 | -2.1742058968376545 | -0.496652023431879 | 2.223233350661122 |
| liquidations_1h_usd | 10309 | 0.2018 | 3.380000114440918 | 764.5999755859375 | 37752.4453125 |
| bid_ask_spread_bps | 37342 | 0.7309 | 1.5816528797149658 | 3.021604537963867 | 5.805515289306641 |
| btc_lead_ret_5m | 37272 | 0.7296 | -0.1743010009679595 | -0.0014081461253350634 | 0.15164636849036606 |
| eth_lead_ret_5m | 37242 | 0.729 | -0.19789994473538777 | -0.0017291196935662514 | 0.17887415552414054 |

_Per-coin nonzero fraction:_

| coin_id | bid_ask_spread_bps | btc_lead_ret_5m | eth_lead_ret_5m | funding_rate | liquidations_1h_usd | open_interest_z |
|---|---:|---:|---:|---:|---:|---:|
| bonk | 0.7053 | 0.704 | 0.7034 | 0.7053 | 0.0699 | 0.7051 |
| celestia | 0.7379 | 0.7365 | 0.7359 | 0.7379 | 0.2333 | 0.7377 |
| dogwifcoin | 0.7378 | 0.7364 | 0.7358 | 0.7378 | 0.3156 | 0.7376 |
| floki-inu | 0.7399 | 0.7385 | 0.7379 | 0.7399 | 0.0938 | 0.7397 |
| injective-protocol | 0.7378 | 0.7364 | 0.7358 | 0.7378 | 0.1571 | 0.7374 |
| jupiter-exchange-solana | 0.7379 | 0.7365 | 0.7359 | 0.7379 | 0.0936 | 0.7377 |
| pepe | 0.711 | 0.7096 | 0.7091 | 0.711 | 0.4692 | 0.7108 |
| render-token | 0.7378 | 0.7364 | 0.7358 | 0.7378 | 0.0346 | 0.7376 |
| sei-network | 0.7284 | 0.727 | 0.7264 | 0.7284 | 0.1736 | 0.7282 |
| worldcoin-wld | 0.7379 | 0.7365 | 0.7359 | 0.7379 | 0.3734 | 0.7377 |

### tf=`5m` — `models/datasets/5m_20260428T202028Z.parquet` (187091 rows) — **stale (predates backfill)** ⚠
| feature | rows nonzero | frac | q05 | q50 | q95 |
|---|---:|---:|---:|---:|---:|
| funding_rate | 15738 | 0.0841 | -0.0004718862473964691 | -8.140991667460185e-06 | 9.999999747378752e-05 |
| open_interest_z | 15738 | 0.0841 | -2.569139675295171 | -0.38081053486653405 | 1.9710423453842474 |
| liquidations_1h_usd | 3116 | 0.0167 | 4.6072001457214355 | 797.7985229492188 | 32050.90234375 |
| bid_ask_spread_bps | 15738 | 0.0841 | 1.6014091968536377 | 3.069838762283325 | 5.782017707824707 |
| btc_lead_ret_5m | 15728 | 0.0841 | -0.10865463623443644 | 0.03941760563153505 | 0.10294537715067567 |
| eth_lead_ret_5m | 15718 | 0.084 | -0.12902640781102925 | 0.06608669536785729 | 0.12471132335712151 |

_Per-coin nonzero fraction:_

| coin_id | bid_ask_spread_bps | btc_lead_ret_5m | eth_lead_ret_5m | funding_rate | liquidations_1h_usd | open_interest_z |
|---|---:|---:|---:|---:|---:|---:|
| bonk | 0.0904 | 0.0903 | 0.0902 | 0.0904 | 0.0042 | 0.0904 |
| celestia | 0.0904 | 0.0903 | 0.0902 | 0.0904 | 0.0125 | 0.0904 |
| dogwifcoin | 0.0904 | 0.0903 | 0.0902 | 0.0904 | 0.0243 | 0.0904 |
| floki-inu | 0.0903 | 0.0903 | 0.0902 | 0.0903 | 0.0053 | 0.0903 |
| injective-protocol | 0.0904 | 0.0903 | 0.0902 | 0.0904 | 0.0515 | 0.0904 |
| jupiter-exchange-solana | 0.0904 | 0.0903 | 0.0903 | 0.0904 | 0.005 | 0.0904 |
| pepe | 0.0903 | 0.0903 | 0.0902 | 0.0903 | 0.0338 | 0.0903 |
| render-token | 0.0904 | 0.0903 | 0.0903 | 0.0904 | 0.0019 | 0.0904 |
| sei-network | 0.0239 | 0.0239 | 0.0238 | 0.0239 | 0.0072 | 0.0239 |
| worldcoin-wld | 0.0903 | 0.0903 | 0.0902 | 0.0903 | 0.0201 | 0.0903 |

### tf=`1h` — `models/datasets/1h_20260428T212141Z.parquet` (81105 rows) — post-backfill ✅
| feature | rows nonzero | frac | q05 | q50 | q95 |
|---|---:|---:|---:|---:|---:|
| funding_rate | 20310 | 0.2504 | -0.00030256836907938123 | 2.5281013222411275e-05 | 9.999999747378752e-05 |
| open_interest_z | 13070 | 0.1611 | -1.2947174180241607 | -0.08451112448663639 | 1.6782977698931936 |
| liquidations_1h_usd | 37 | 0.0005 | 23.80387935638428 | 2593.7177734375 | 81539.61874999969 |
| bid_ask_spread_bps | 70 | 0.0009 | 1.5848021924495697 | 3.0845158100128174 | 5.890074324607849 |
| btc_lead_ret_5m | 80940 | 0.998 | -0.21555637969210845 | -0.0009306286871219663 | 0.20432268950014007 |
| eth_lead_ret_5m | 81017 | 0.9989 | -0.334881023412264 | -0.00166596446830982 | 0.3154874969709761 |

_Per-coin nonzero fraction:_

| coin_id | bid_ask_spread_bps | btc_lead_ret_5m | eth_lead_ret_5m | funding_rate | liquidations_1h_usd | open_interest_z |
|---|---:|---:|---:|---:|---:|---:|
| bonk | 0.0008 | 0.9979 | 0.999 | 0.2362 | 0.0002 | 0.1522 |
| celestia | 0.0008 | 0.9979 | 0.999 | 0.2367 | 0.0006 | 0.1522 |
| dogwifcoin | 0.0008 | 0.9979 | 0.999 | 0.2367 | 0.0007 | 0.1522 |
| floki-inu | 0.0008 | 0.9979 | 0.999 | 0.2362 | 0.0002 | 0.1522 |
| injective-protocol | 0.0008 | 0.9979 | 0.999 | 0.2362 | 0.0 | 0.1522 |
| jupiter-exchange-solana | 0.0008 | 0.9979 | 0.999 | 0.2367 | 0.0005 | 0.1522 |
| pepe | 0.0008 | 0.9979 | 0.999 | 0.2362 | 0.0005 | 0.1522 |
| render-token | 0.0008 | 0.9979 | 0.999 | 0.2367 | 0.0003 | 0.1522 |
| sei-network | 0.0018 | 0.9992 | 0.9982 | 0.5344 | 0.0016 | 0.3436 |
| worldcoin-wld | 0.0008 | 0.9979 | 0.999 | 0.2362 | 0.0006 | 0.1522 |

### tf=`2h` — `models/datasets/2h_20260428T212141Z.parquet` (40368 rows) — post-backfill ✅
| feature | rows nonzero | frac | q05 | q50 | q95 |
|---|---:|---:|---:|---:|---:|
| funding_rate | 10140 | 0.2512 | -0.00030256836907938123 | 2.5474180802120827e-05 | 9.999999747378752e-05 |
| open_interest_z | 6520 | 0.1615 | -1.2942208364162715 | -0.08452296948879852 | 1.6695877260162193 |
| liquidations_1h_usd | 10 | 0.0002 | 135.23457036018374 | 11078.9248046875 | 90931.89550781238 |
| bid_ask_spread_bps | 20 | 0.0005 | 1.5783497631549834 | 3.071413040161133 | 5.83883945941925 |
| btc_lead_ret_5m | 40295 | 0.9982 | -0.21845689341164687 | -0.0019288355192736218 | 0.21050398465171194 |
| eth_lead_ret_5m | 40319 | 0.9988 | -0.34264695904996717 | -0.005901462236347805 | 0.3088796380484534 |

_Per-coin nonzero fraction:_

| coin_id | bid_ask_spread_bps | btc_lead_ret_5m | eth_lead_ret_5m | funding_rate | liquidations_1h_usd | open_interest_z |
|---|---:|---:|---:|---:|---:|---:|
| bonk | 0.0005 | 0.9981 | 0.9988 | 0.2369 | 0.0002 | 0.1525 |
| celestia | 0.0005 | 0.9981 | 0.9988 | 0.2374 | 0.0002 | 0.1525 |
| dogwifcoin | 0.0005 | 0.9981 | 0.9988 | 0.2374 | 0.0002 | 0.1525 |
| floki-inu | 0.0005 | 0.9981 | 0.9988 | 0.2369 | 0.0002 | 0.1525 |
| injective-protocol | 0.0005 | 0.9981 | 0.9988 | 0.2369 | 0.0 | 0.1525 |
| jupiter-exchange-solana | 0.0005 | 0.9981 | 0.9988 | 0.2374 | 0.0002 | 0.1525 |
| pepe | 0.0005 | 0.9981 | 0.9988 | 0.2369 | 0.0002 | 0.1525 |
| render-token | 0.0005 | 0.9981 | 0.9988 | 0.2374 | 0.0002 | 0.1525 |
| sei-network | 0.0011 | 0.9995 | 0.9979 | 0.5387 | 0.0011 | 0.3461 |
| worldcoin-wld | 0.0005 | 0.9981 | 0.9988 | 0.2369 | 0.0002 | 0.1525 |

### tf=`6h` — `models/datasets/6h_20260428T212141Z.parquet` (13203 rows) — post-backfill ✅
| feature | rows nonzero | frac | q05 | q50 | q95 |
|---|---:|---:|---:|---:|---:|
| funding_rate | 3350 | 0.2537 | -0.00030520797008648515 | 1.8940389054478146e-05 | 9.999999747378752e-05 |
| open_interest_z | 2140 | 0.1621 | -1.3109075330393303 | -0.08444621748429351 | 1.6820857407998062 |
| liquidations_1h_usd | 0 | 0.0 | — | — | — |
| bid_ask_spread_bps | 0 | 0.0 | — | — | — |
| btc_lead_ret_5m | 13203 | 1.0 | -0.22021832892870213 | -0.0002682597341574272 | 0.2161374645596156 |
| eth_lead_ret_5m | 13183 | 0.9985 | -0.33662496987138135 | 0.003396753552791919 | 0.32813189997178327 |

_Per-coin nonzero fraction:_

| coin_id | bid_ask_spread_bps | btc_lead_ret_5m | eth_lead_ret_5m | funding_rate | liquidations_1h_usd | open_interest_z |
|---|---:|---:|---:|---:|---:|---:|
| bonk | 0.0 | 1.0 | 0.9986 | 0.2393 | 0.0 | 0.1529 |
| celestia | 0.0 | 1.0 | 0.9986 | 0.2393 | 0.0 | 0.1529 |
| dogwifcoin | 0.0 | 1.0 | 0.9986 | 0.2393 | 0.0 | 0.1529 |
| floki-inu | 0.0 | 1.0 | 0.9986 | 0.2393 | 0.0 | 0.1529 |
| injective-protocol | 0.0 | 1.0 | 0.9986 | 0.2393 | 0.0 | 0.1529 |
| jupiter-exchange-solana | 0.0 | 1.0 | 0.9986 | 0.2393 | 0.0 | 0.1529 |
| pepe | 0.0 | 1.0 | 0.9986 | 0.2393 | 0.0 | 0.1529 |
| render-token | 0.0 | 1.0 | 0.9986 | 0.2393 | 0.0 | 0.1529 |
| sei-network | 0.0 | 1.0 | 0.9967 | 0.5556 | 0.0 | 0.3549 |
| worldcoin-wld | 0.0 | 1.0 | 0.9986 | 0.2393 | 0.0 | 0.1529 |

### tf=`1d` — `models/datasets/1d_20260428T133304Z.parquet` (7864 rows) — **stale (predates backfill)** ⚠
| feature | rows nonzero | frac | q05 | q50 | q95 |
|---|---:|---:|---:|---:|---:|
| funding_rate | 36 | 0.0046 | -0.00035927780118072405 | 5.002581019653007e-06 | 8.780576899880543e-05 |
| open_interest_z | 36 | 0.0046 | -2.028539608871378 | 0.10203659660386605 | 2.0363019966514906 |
| liquidations_1h_usd | 11 | 0.0014 | 13.602200150489807 | 419.097900390625 | 2400.7090454101562 |
| bid_ask_spread_bps | 36 | 0.0046 | 1.58982914686203 | 3.03035831451416 | 5.782017707824707 |
| btc_lead_ret_5m | 36 | 0.0046 | 0.03941760563153505 | 0.05280108936200198 | 0.09737790473977018 |
| eth_lead_ret_5m | 36 | 0.0046 | 0.06608669536785729 | 0.08734299377354399 | 0.10991220230888998 |

_Per-coin nonzero fraction:_

| coin_id | bid_ask_spread_bps | btc_lead_ret_5m | eth_lead_ret_5m | funding_rate | liquidations_1h_usd | open_interest_z |
|---|---:|---:|---:|---:|---:|---:|
| bonk | 0.005 | 0.005 | 0.005 | 0.005 | 0.0012 | 0.005 |
| celestia | 0.0046 | 0.0046 | 0.0046 | 0.0046 | 0.0011 | 0.0046 |
| dogwifcoin | 0.0056 | 0.0056 | 0.0056 | 0.0056 | 0.0014 | 0.0056 |
| floki-inu | 0.0038 | 0.0038 | 0.0038 | 0.0038 | 0.0009 | 0.0038 |
| injective-protocol | 0.0047 | 0.0047 | 0.0047 | 0.0047 | 0.0035 | 0.0047 |
| jupiter-exchange-solana | 0.0051 | 0.0051 | 0.0051 | 0.0051 | 0.0 | 0.0051 |
| pepe | 0.0038 | 0.0038 | 0.0038 | 0.0038 | 0.0019 | 0.0038 |
| render-token | 0.0065 | 0.0065 | 0.0065 | 0.0065 | 0.0 | 0.0065 |
| sei-network | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| worldcoin-wld | 0.0041 | 0.0041 | 0.0041 | 0.0041 | 0.0021 | 0.0041 |

## Notes & known limitations
- `liquidations_1h_usd` and `bid_ask_spread_bps` remain at their poller-only coverage. OKX does not expose a free historical liquidations or order-book-depth endpoint with the 365-day reach the dataset needs, so synthesising a value would violate the campaign's *real-data-only* contract. See the `backfill_market_signals.py` docstring for the audit trail.
- `funding_rate` covers ~90 days back (OKX funding-rate-history retention); `open_interest_z` covers ~60 days back (OKX OI history retention); `btc/eth/sol` `mid_price` covers ~365 days back (chunked through the OKX history-candles endpoint). The older portion of the 1d / 1100-day window therefore stays at 0 for funding/OI; we record it explicitly here so a future ablation does not mistake retention for a bug.
- The retention pruner in `artifacts/api-server/src/lib/market-signals-retention.ts` was updated to exempt rows with `source LIKE 'okx_backfill_%'` so the live poller's 30-day trim does not erase this campaign's writes within an hour. A regression test guards this exemption.
