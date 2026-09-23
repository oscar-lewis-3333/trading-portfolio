# Development execution data

The saved bundle covers 1 January 2022 through 31 December 2024. It contains
26,820 asset/date references for all 60 development configurations and their
matched benchmarks, including assets that may need to be sold. No holdout
returns or strategy performance were evaluated in this data-resolution step.

## Reference rule

Keep targets fixed at midnight. Starting at 00:05 UTC, check each minute until
01:00. Choose the first time when every required asset has a completed,
positive-volume minute candle whose start is at most five minutes old. Use
the latest such candle's close for each asset. Recheck all assets when the
time moves later; never use a later price at an earlier execution timestamp.

The common time covers the union of requirements across the entire development
grid and benchmarks. This deliberately gives all configurations the same
data-availability convention. It can delay a particular portfolio because of
an asset that another configuration needs. Actual holdings must also be checked
by the future simulator; the request plan assumes earlier rebalances completed.

Historical trade marks are not executable bid/ask quotes. Availability of a
minute candle at its end is an assumption; feed latency and achievable fills
remain unverified. Transaction costs and any execution stress tests belong in
the portfolio simulation, applied only to each asset's absolute net trade.

## Resolution results

| Item | Count |
|---|---:|
| Required references | 26,820 |
| Original missing 00:05 references | 82 |
| Execution dates | 1,096 |
| Dates retaining 00:05 | 1,025 |
| Dates with later execution | 71 |
| Dates requiring one-hour captures | 5 |
| New one-hour responses | 105 |
| Unresolved references | 0 |

Latest execution is 00:41 UTC on 20 August 2023, a 36-minute delay from 00:05.
The 105 extended captures agree exactly with the original first-15-minute
windows. No original captures, assets, or required dates were removed or
filled with invented prices. The original 82 audit flags remain recorded.

## Saved files and replay

`data/processed/development_execution_v1/` contains:

- `references.csv`: prices, actual execution times, candle availability and source paths.
- `schedule.csv`: one common execution time per development date.
- `audit.csv`: the original 00:05 reference audit.
- `manifest.json`: policy, plan/protocol fingerprints, and SHA-256 hashes of
  every used raw response and exported table.

`execution_resolution.load_execution_data` validates those hashes and the
reference coverage/timing without network requests. The notebook loads this
bundle; its earlier small sample audit is also configured for cached data only.
Source paths currently refer to the local project and its sibling
`crypto_cross_sectional` cache, which must remain available.

The data layer is ready for the next jointly implemented research step. The
existing lookback features and target grid were preserved; no additional
features, parameter selection, bootstrap inference or return evaluation were
introduced here.
