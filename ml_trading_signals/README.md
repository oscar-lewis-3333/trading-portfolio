# MACHINE LEARNING TRADING SIGNALS

## OVERVIEW

A project split into 3 parts, first part we tried rigorously to find signals from technical indicators, which ultimately was largely unsuccessful. Part 2 we tried a reformulation and eventually found a cross-sectional momentum signal, and part 3 is an ongoing test on stock fundamentals

## FEATURES
- OHLCV-based feature engineering
- Triple barrier / simple labelling
- Pooled multi-ticker walk-forward validation system with an embargo
- Statistical significance t-tests with predicted forward returns against a baseline 
- Classification/regression evaluation, comparison with 'buy and hold' strategy
- Cross-sectional excess-return target (relative to universe to cancel market drift)
- Momentum ranking rule, swept over lookback and selection sizes, with nested walk-forward selection and out-of-sample HAC validation
- Machine learning enhancement testing alongside a shown indicator
- Forward looking value test (ongoing due to limited data)
- 11 test functions to detect implementation errors

## KEY DESIGN DECISIONS

- This project was designed to be about ML when discusses trading, so making the data any less-complex would remove the objective we were chasing.
- The real baseline we are competing with is the 'buy and hold' strategy, not accuracy. This is due to this being the easiest and most accessible option, and any trades we do make encounter trading costs in real life, so we need to ensure they make a noticeable (statistically significant) difference.
- The embargo is chosen to match the horizon. This is due to leakage of data, since for each label we look forward by $h$ days (forward_returns formula), so value at $t$ depends on $t+h$. If $h \leq 21$ then day $t$'s data depends on day $t+h$ which is inside the testing window, hence leaking data. So we enforce an embargo the same as the horizon (could be greater than, but then we miss out on useful data, so choose infimum).
- Momentum method within this project have been adapted so that rankings are done at close on one day, with forward returns being calculated from the following open, onwards. This has reduced look-forward bias from previous versions as it is impossible to buy at close whilst knowing the close price, which is what was previously implemented.
- Walk-forward in Part 2 does not fix any given specific 'day' momentum or top fraction, but rather selects the configuration which has the largest t-statistic when compared to the universe. 

## RESULTS

### Part 1 — Classification label outcomes

| Method | Positive rate | Labels (1 / 0) | Avg holding period | Avg return (label=1) | Avg return (label=0) |
|---|---:|---:|---:|---:|---:|
| Simple label (5-day forward return > 0) | 0.5856 | 1468 / 1039 | — | — | — |
| Triple barrier (10-day, 2:1 reward/risk) | 0.4281 | 1062 / 1419 | 3.47 days | 3.21% | -1.75% |

### Part 1 — Single-ticker walk-forward classification and long-only results

| Metric | Value |
|---|---:|
| Mean ROC-AUC | 0.5364 |
| Mean accuracy | 0.5397 |
| Mean coverage | 33.28% |
| Mean selected-long return | 0.405% |
| Mean long-only return | 0.164% |
| Mean benchmark return | 0.389% |
| Mean conditional selection lift | 0.094% |
| Mean long-only excess | -0.226% |
| Long-only excess vs benchmark (HAC) | t=-2.604, p=0.0093 |

### Part 1 — Pooled classifier mean metrics (20 tickers)

| Metric | Mean value |
|---|---:|
| ROC-AUC | 0.5013 |
| Average precision | 0.3853 |
| Positive rate | 0.3811 |
| Brier score | 0.2391 |

### Part 1 — Pooled top-20% daily selection

| Metric | Mean value |
|---|---:|
| Top label rate | 37.4380% |
| Universe label rate | 38.1101% |
| Label lift | -0.6721 percentage points |
| Top return | 0.2888% |
| Universe return | 0.2534% |
| Excess return | 0.0353% |
| Excess vs universe (HAC) | t=0.527, p=0.5982 |

### Part 1 — Probability-quintile diagnostic

| Probability bucket | Dates | Mean stocks | Mean probability | Label rate | Mean return | HAC t | HAC p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2016 | 4.0 | 32.4438% | 38.1324% | 0.2334% | — | — |
| 2 | 2016 | 4.0 | 35.7068% | 38.3061% | 0.2294% | — | — |
| 3 | 2016 | 4.0 | 37.5721% | 38.9509% | 0.2806% | — | — |
| 4 | 2016 | 4.0 | 39.5813% | 37.7232% | 0.2350% | — | — |
| 5 | 2016 | 4.0 | 43.7716% | 37.4380% | 0.2888% | — | — |
| 5 minus 1 return spread | — | — | — | — | 0.0554% | 0.528 | 0.5976 |

### Part 2 — Full-feature random-forest ranked portfolio

| Metric | Value |
|---|---:|
| Total test days | 2016 |
| Mean top-selected return | 2.4426% |
| Mean bottom-avoided return | 1.8022% |
| Mean universe return | 1.8896% |
| Top vs universe (HAC) | t=1.570, p=0.1167 |
| Top vs bottom (HAC) | t=1.117, p=0.2642 |

### Part 2 — Fixed 21-day momentum baseline (top 20%)

| Metric | Value |
|---|---:|
| Mean top return | 2.9045% |
| Mean bottom return | 2.0132% |
| Mean universe return | 2.0429% |
| Top minus bottom | 0.8913% |
| Top minus universe | 0.8616% |
| Top vs bottom (HAC) | t=1.692, p=0.0908 |

### Part 2 — Holm-significant momentum sweep results vs universe

| Lookback | Top fraction | Test days | Top return | Bottom return | Universe return | Top minus bottom | Top minus universe | Holm p (top vs bottom) | Holm p (top vs universe) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| return_63d | 0.1 | 2428 | 4.0579% | 1.9491% | 2.0308% | 2.1088% | 2.0271% | 0.3524 | 0.0295 |
| return_63d | 0.2 | 2428 | 3.2114% | 1.9010% | 2.0308% | 1.3104% | 1.1806% | 0.5954 | 0.0365 |

### Part 2 — Adaptive momentum walk-forward out-of-sample summary

| Metric | Value |
|---|---:|
| `return_10d`, top 10% selected | 7 folds |
| `return_63d`, top 10% selected | 15 folds |
| `return_63d`, top 20% selected | 8 folds |
| Mean out-of-sample top return | 3.6320% |
| Mean out-of-sample universe return | 2.0551% |
| Mean out-of-sample excess | 1.5769% |
| Excess vs 0 (HAC) | t=2.384, p=0.0172 |
| Test days where top beat universe | 52.33% |
| Folds where top beat universe | 73.33% |

### Part 2 — ML feature-pair enhancement tests

| Additional feature paired with `return_21d` | Mean excess | HAC t | Raw p | Holm p |
|---|---:|---:|---:|---:|
| `return_63d` | 0.2929% | 1.4369 | 0.1509 | 0.5367 |
| `volume_trend` | 0.1748% | 0.8985 | 0.3690 | 0.7381 |
| `rel_strength_21d` | 0.2668% | 1.4984 | 0.1342 | 0.5367 |
| `vol_ratio` | 0.1450% | 0.8266 | 0.4086 | 0.7381 |

### Part 2 — S&P 500 adaptive momentum walk-forward out-of-sample summary

| Metric | Value |
|---|---:|
| `return_10d`, top 10% selected | 8 folds |
| `return_21d`, top 10% selected | 2 folds |
| Mean out-of-sample top return | 2.7187% |
| Mean out-of-sample universe return | 1.5778% |
| Mean out-of-sample excess | 1.1408% |
| Excess vs 0 (HAC) | t=2.063, p=0.0395 |
| Test days where top beat universe | 61.59% |
| Folds where top beat universe | 70.00% |

## KEY FINDINGS

- It was very difficult to find any form of signal on daily horizon using ML with the technical indicators. Almost all tests came to be of statistical insignificance from 0% return, with any values being on the negative side anyway. This was tested on one ticker, and then on a large family of them, and each time it failed in comparison to a classic 'buy and hold' strategy.
- After adjusting p-values, only form of statistical significance found was ranking our universe by 63 day momentum every 21 days, and selecting the top 10% or 20%. The 20% approach is what is built in the equity curve, which has further testing in risk_management
- Walk-forward approach on the universe found statistically significant results, but has been changed since previous iterations of this project. Now the each approach in the sweep (10, 20, 30, 50%), (5d, 10d, 21d, 63d) is tested strictly on a test period, with the approach having the largest t-statistic measuring top selection return vs universe return being chosen. Over 30 folds, 63d momentum was chosen 23 times (10% top fraction 15 times, 20% top fraction 8 times), with the only other approach chosen being 10d momentum with 10% top fraction. This shows a bias on the universe to choose the 63d momentum, aligning with our results from the sweep.
- The book referenced heavily favoured simplicity of features, which is what has somewhat been shown here. ML was trying to train to noise, which is why it ended up failing. ML succeeds when combining and filtering known signals, instead of creating them.
- The S&P 500 sweep came up entirely insignificant in contrary to the smaller universe. This leads us to believe that there is something special about that universe of tickers which favours specifically 63d momentum. When performing the same walk-forward approach as for the universe, statistically significant excess results were replicated, despite a varying range of momenta (10d, 21d) and only 10% top fraction being selected. This leads us to believe the general momentum principle still applies to the S&P 500, just not a specific momenta, although the amount of folds tested is only 10, comparitively .

## FORWARD TEST STATUS

Test was begun on 13/08/2026, on 19 tickers which are ranked by a composite signal which encourages high RoE and low P-E ratio. Ranked on a quartile basis. To be updated in the future.

## LIMITATIONS

- Our universe to measure market drift was only 20 stocks in size, and largely consisted of large-cap US stocks and equities. In reality, market has a lot more stocks/equities and many in other regions which could hold statistical significance (UPDATED FINDINGS IN KEY FINDINGS)
- Part 3 cannot be backtested due to a limited amount of historical data on each stocks P/E ratio and RoE, so the only statistically viable way to draw conclusions is to wait and see what occurs in the future. Ideally would back-test, but not possible with given data.
- Results for S&P 500 are tested on a smaller amount of folds purely for computational reasons. Amount of folds is 10, compared to 30 for the other universe due to the period being 5 years compared to 10 years. Potential for different statistical results in that section over the longer time-period.

## LIBRARIES USED
- scikit-learn — classification, linear regression, ensemble models (random forest, gradient boosting), evaluation metrics
- pandas, numpy — feature engineering (OHLCV-based), pooled multi-ticker data handling
- scipy — hypothesis/t-tests throughout to check significance
- yfinance — price data (via past_market_analysis project) and P-E/RoE in part 3
- matplotlib — trade outcome scatter, momentum sweep visualisation, forward test setup plot

## PROJECT STRUCTURE
    src/
        features.py — feature engineering, multi-ticker pooling, fundamental features, forward-test tracking, momentum equity curve
        labels.py — simple/triple-barrier labelling and cross-sectional excess-return target
        walk_forward.py — chronological split generation, out-of-sample evaluation, ranked portfolios, momentum sweeps and nested walk-forward selection
        plotting_ml_trading_signals.py — out-of-sample selection, momentum sweep and forward-test visualisations
    notebooks/
        ml_signals.ipynb — main analysis of src notebooks (Part 1: ML-feature testing, Part 2: momentum ranking (added subsection for S&P 500), Part 3: fundamentals and forward test)
    tests/
        test_equity_curve.py — testing everything about the construction of the equity curve
        test_labels_targets.py — testing that labels and targets perform as they're supposed to
        test_momentum_validation.py — testing that ranking by 63d momentum works as intended
        test_oos_evaluation.py — testing the out of sample methods perform as expected

## USAGE
```python
from features import build_multi_ticker_dataset, build_fundamental_features, start_forward_tracking, forward_evaluation_tracking
from labels import excess_return_target
from walk_forward import momentum_baseline_sweep, walk_forward_momentum_rule
from plotting_ml_trading_signals import plot_momentum_sweep

#build pooled dataset of any chosen tickers, this counts as 'universe'
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "JNJ", "XOM", "WMT", "PG", "HD", "DIS", "NFLX", "AMD", "INTC", "CSCO", "ADBE", "CRM"]
pooled = build_multi_ticker_dataset(tickers, period="10y", horizon=10, ranking_horizon=21)

#cross-sectional excess-return target (how well a ticker performs relative to the market)
pooled_v2 = excess_return_target(pooled)

#sweep top_frac, lookback_cols to find the best balance for a signal
sweep = momentum_baseline_sweep(pooled_v2)
plot_momentum_sweep(sweep)

#nested walk-forward: select the best rule on each training window, then test it later
mom_wf_folds, mom_wf_daily = walk_forward_momentum_rule(pooled_v2, train_size=504, test_size=63, horizon=21, embargo=21, expanding=True)

#forward-looking fundamentals test (requires waiting calendar time)
fund = build_fundamental_features(tickers)
log = start_forward_tracking(tickers, fund, output_path='../data/forward_test_log.csv')
#to be revisited later:
forward_evaluation_tracking(log_path='../data/forward_test_log.csv')
```
