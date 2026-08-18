# MACHINE LEARNING TRADING SIGNALS

## OVERVIEW

A project split into 3 parts, first part we tried rigorously to find signals from technical indicators, which ultimately was largely unsuccessful. Part 2 we tried a reformulation and eventually found a cross-sectional momentum signal, and part 3 is an ongoing test on stock fundamentals

## FEATURES
- OHLCV-based feature engineering
- Triple barrier / simple labelling
- Pooled multi-ticker walk-forward validation system with an embargo
- Statistical significance t-tests with predicted forward returns against a baseline 
- Classification/regression evaluation, comparison with 'buy and hold' strategy - not
- Cross-sectional excess-return target (relative to universe to cancel market drift)
- Momentum ranking rule, swept over lookback and selection sizes, with walk-forward validation
- Machine learning enhancement testing alongside a shown indicator
- Forward looking value test (ongoing due to limited data)

## KEY DESIGN DECISIONS

- This project was designed to be about ML when discusses trading, so making the data any less-complex would remove the objective we were chasing.
- The real baseline we are competing with is the 'buy and hold' strategy, not accuracy. This is due to this being the easiest and most accessible option, and any trades we do make encounter trading costs in real life, so we need to ensure they make a noticeable difference.
- The embargo is chosen to match the horizon. This is due to leakage of data, since for each label we look forward by $h$ days (forward_returns formula), so value at $t$ depends on $t+h$. If $h \leq 21$ then day $t$'s data depends on day $t+h$ which is inside the testing window, hence leaking data. So we enforce an embargo the same as the horizon (could be greater than, but then we miss out on useful data, so choose infimum).

## RESULTS

### Part 1 — Classification labelling comparison

| Method | Positive rate | Class balance |
|---|---|---|
| Simple label (5-day fwd return > 0) | 0.5845 | 1466 / 1042 |
| Triple barrier (10-day, 2:1 reward/risk) | 0.5032 | 1249 / 1233 |

Triple barrier avg holding period: 4.57 days. Avg return when label=1: +3.83%, when label=0: -2.71%.

### Part 1 — Single-ticker walk-forward classification

| Metric | Value |
|---|---|
| Mean ROC-AUC | 0.5191 |
| Mean accuracy | 0.5289 |
| Mean strategy return per trade | 0.220% |

### Part 1 — Pooled dataset (20 tickers)

| Metric | Value |
|---|---|
| Mean ROC-AUC | 0.4856 |
| ROC-AUC vs 0.5 | t=-1.740, p=0.0919 |
| Strategy return vs 0 | t=-2.145, p=0.0399 |
| Mean confident ROC-AUC | 0.4852 |
| Confident ROC-AUC vs 0.5 | t=-0.885, p=0.3829 |
| Confident strategy return vs 0 | t=-2.743, p=0.0100 |

### Part 1 — Confidence threshold sweep

| Threshold | Folds | Mean return | t | p |
|---|---|---|---|---|
| 0.55 | 32 | -0.307% | -2.520 | 0.0171 |
| 0.58 | 32 | -0.388% | -2.949 | 0.0060 |
| 0.60 | 32 | -0.540% | -2.743 | 0.0100 |
| 0.65 | 31 | -0.670% | -0.992 | 0.3291 |

### Part 1 — Strategy vs buy-and-hold

| Metric | Value |
|---|---|
| Mean strategy return | -0.540% |
| Mean buy & hold return | 0.406% |
| Mean excess return | -0.946% |
| Excess vs 0 | t=-3.016, p=0.0051 |

### Part 1 — Buy vs sell prediction accuracy

| Direction | Mean return | t | p |
|---|---|---|---|
| Buy | 0.1829% | 0.385 | 0.7036 |
| Sell (treated as long) | 0.8621% | 2.852 | 0.0077 |

24/32 folds favourable when sell predictions treated as buy signals.

### Part 2 — Ranked portfolio, full feature set (no improvement over Part 1)

| Metric | Value |
|---|---|
| Total test days | 2016 |
| Mean top-selected return | 0.4186% |
| Mean bottom-avoided return | 0.3866% |
| Mean universe return | 0.4156% |
| Top vs universe | t=0.054, p=0.9573 |
| Top vs bottom | t=0.364, p=0.7157 |

### Part 2 — Momentum sweep (significant results only, p<0.05)

| Lookback | Top frac | Top return | Bottom return | Universe | Top−Bottom | p (top vs bottom) |
|---|---|---|---|---|---|---|
| return_10d | 0.1 | 0.8042% | 0.4412% | 0.4430% | 0.3630 | 0.0014 |
| return_21d | 0.1 | 0.9172% | 0.3667% | 0.4430% | 0.5505 | 0.0000 |
| return_21d | 0.2 | 0.7027% | 0.4219% | 0.4430% | 0.2808 | 0.0002 |
| return_21d | 0.3 | 0.5794% | 0.4000% | 0.4430% | 0.1794 | 0.0027 |
| return_21d | 0.5 | 0.4890% | 0.3969% | 0.4430% | 0.0921 | 0.0358 |
| return_63d | 0.1 | 0.8737% | 0.4822% | 0.4517% | 0.3915 | 0.0014 |
| return_63d | 0.2 | 0.6965% | 0.4466% | 0.4517% | 0.2499 | 0.0032 |
| return_63d | 0.3 | 0.5790% | 0.4475% | 0.4517% | 0.1314 | 0.0441 |
| return_63d | 0.5 | 0.5319% | 0.3716% | 0.4517% | 0.1603 | 0.0005 |

return_5d was not significant at any top_frac tested.

### Part 2 — Momentum walk-forward validation vs ML enhancement

| Approach | Top return | Excess vs universe | p |
|---|---|---|---|
| Pure momentum (return_21d, top 25%) | 0.7102% | 0.2142%* | 0.0167* |
| ML (gradient boosting, 5 features) | 0.4922% | 0.0625% | 0.1289 |
| return_21d + return_63d | — | 0.0310% | 0.4430 |
| return_21d + volume_trend | — | 0.0746% | 0.0632 |
| return_21d + rel_strength_21d | — | 0.0423% | 0.2646 |
| return_21d + vol_ratio | — | 0.0667% | 0.0940 |

*Fold-level walk-forward result (t=2.517, p=0.0167), distinct from the pooled single-split figures above it.

## KEY FINDINGS

- It was very difficult to find any form of signal on daily horizon using ML with the technical indicators. Almost all tests came to be of statistical insignificance from 0% return, with any values being on the negative side anyway. This was tested on one ticker, and then on a large family of them, and each time it failed in comparison to a classic 'buy and hold' strategy.
- The only indicator that could potentially hold (would need more rigourous and diverse statistical testing) is an anti-correlation of our sell signals, which did produce results of statistical significance. In layman's terms, our machine was predicting 'sell' signals when it should've been predicting 'buy' signals. 
- A momentum signal succeeded where the process above failed, producing a statistically significant difference between top-performing tickers and the universe baseline (and bottom performing tickers). When we tried to combine this with any other technical indicator, the strength of the signal reduced to being statistical insignificant from the universe average
- The book referenced heavily favoured simplicity of features, which is what has somewhat been shown here. ML was trying to train to noise, which is why it ended up failing. ML succeeds when combining and filtering known signals, instead of creating them.
- (UPDATE) Ran momentum system with universe as S&P 500. Found no significance across any returns window or any top fraction selection. When momentum walk forward tested, excess returns were insignificant from 0. This implies that there was potentially something special about the properties of the universe we selected, whether that be liquidity, volume, market-cap or any other property.

## FORWARD TEST STATUS

Test was begun on 13/08/2026, on 19 tickers which are ranked by a composite signal which encourages high RoE and low P-E ratio. Ranked on a quartile basis. To be updated in the future.

## LIMITATIONS

- Our universe to measure market drift was only 20 stocks in size, and largely consisted of large-cap US stocks and equities. In reality, market has a lot more stocks/equities and many in other regions which could hold statistical significance (UPDATED FINDINGS IN KEY FINDINGS)
- Part 3 cannot be backtested due to a limited amount of historical data on each stocks P/E ratio and RoE, so the only statistically viable way to draw conclusions is to wait and see what occurs in the future. Ideally would back-test, but not possible with given data.

## LIBRARIES USED
- scikit-learn — classification, linear regression, ensemble models (random forest, gradient boosting), evaluation metrics
- pandas, numpy — feature engineering (OHLCV-based), pooled multi-ticker data handling
- scipy — hypothesis/t-tests throughout to check significance
- yfinance — price data (via past_market_analysis project) and P-E/RoE in part 3
- matplotlib — trade outcome scatter, momentum sweep visualisation, forward test setup plot

## PROJECT STRUCTURE
    src/
        features.py     — feature engineering (momentum, volatility, mean-reversion, volume, cross-sectional), multi-ticker pooling,fundamental features, forward test tracking
        labels.py        — simple/ triple-barrier labelling, cross-sectional excess-return target
        walk_forward.py  — pooled walk-forward split generation, walk_forward methods for ranked portfolio, confidence filtering and momentum rule,  direction-split diagnostics, momentum baseline sweep
        plotting.py       — buy/sell trade outcome scatter, momentum sweep visualisation, forward test setup plot to visualise quartiles
    notebooks/
        ml_signals.ipynb — main analysis of src notebooks (Part 1: ML-feature testing, Part 2: momentum ranking, Part 3: fundamentals and forward test)

## USAGE
```python
from features import build_multi_ticker_dataset, build_fundamental_features, start_forward_tracking, forward_evaluation_tracking
from labels import simple_labels, triple_barrier_labels, excess_return_target
from walk_forward import walk_forward_evaluate_pooled, walk_forward_confident_only, evaluate_by_direction, strategy_vs_buy_hold, walk_forward_ranked_portfolio, momentum_baseline_sweep, walk_forward_momentum_rule
from plotting import plot_trade_outcomes, plot_momentum_sweep, plot_forward_test_setup

#build pooled dataset of any chosen tickers, this counts as 'universe'
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", ...]
pooled = build_multi_ticker_dataset(tickers, period="10y", horizon=21)

#cross-sectional excess-return target (how well a ticker performs relative to the market)
pooled_v2 = excess_return_target(pooled, horizon=21)

#sweep top_frac, lookback_cols to find the best balance for a signal
sweep = momentum_baseline_sweep(pooled_v2)
plot_momentum_sweep(sweep)

#walk-forward using momentum rule, can choose lookback, top_frac based on plot above
mom_wf = walk_forward_momentum_rule(pooled_v2, lookback_col='return_21d', top_frac=0.25, train_size=250, test_size=63, embargo=21)

#forward-looking fundamentals test (requires waiting calendar time)
fund = build_fundamental_features(tickers)
log = start_forward_tracking(tickers, fund, output_path='../data/forward_test_log.csv')
#to be revisited later:
forward_evaluation_tracking(log_path='../data/forward_test_log.csv')
```
