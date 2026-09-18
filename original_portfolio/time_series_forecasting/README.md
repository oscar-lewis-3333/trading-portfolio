# TIME SERIES FORECASTING

## OVERVIEW
There are two main models used here, ARIMA (AutoRegression Integrated Moving Averaege) to model future returns

## FEATURES
- ADF stationary testing to determine time dependence
- ARIMA model order selection using AIC grid search
- GARCH, EGARCH, GJR-GARCH volatility modelling and forecasting
- Walk forward backtesting of volatility model
- GARCH vs implied volatility comparison
- Cross ticker comparison/validation

## KEY DESIGN DECISIONS
- AIC grid chosen of pmdarima due to a lack of combatability with Numpy 2.x
- Returns are chosen to be modelled instead of close prices due to returns being stationary by the ADF test and close prices being non-stationary by the same test
- The choice of refit_every is because fitting a GARCH model is a very slow process, so if done every day over a 5 day period, processing time would be extremely slow

## RESULTS

### Cross-ticker comparisons
| Ticker | ARIMA_order | GARCH_alpha_beta | GJR_gamma | GJR_gamma_pval | AIC_improvement |
|--------|-------------|------------------|-----------|----------------|-----------------|
| AAPL   | (1, 0, 3)   | 0.9463           | 0.1279    | 0.0150         | 19.14           |
| NVDA   | (1, 0, 0)   | 0.9597           | 0.2668    | 0.0732         | 20.81           |
| SPY    | (0, 0, 3)   | 0.9244           | 0.1807    | 0.6388         | 48.33           |
| JNJ    | (3, 0, 2)   | 0.7623           | -0.0164   | 0.8772         | -1.93           |
| TSLA   | (0, 0, 0)   | 0.9895           | 0.0282    | 0.1582         | 2.83            |

### GARCH Walk-forward backtesting

| Metric | Value |
|---|---|
| Backtest observations | 231 |
| Mean absolute error | 0.0612 |
| RMSE | 0.0727 |
| Correlation | 0.0621 |

### Different measures of volatility

| Measure | Value |
|---|---|
| 5-year realised volatility | 25.85% |
| 5-day rolling volatility | 55.12% |
| GARCH day-1 forecast | 40.64% |
| Market implied volatility | 28.92% |

## KEY FINDINGS
- Looking at the Cross-ticker comparison, we see no correlation between the ARIMA orders chosen per ticker, confirming the fact that it is not a general measure, but a model that applies differently to different tickers. Also we have $\alpha + \beta$ is very close to one for most tickers, implying that periods with more shock take longer to return to standard volatility. We also see that across these tickers the GJR-Model was generally the more effective method at forecasting volatility when measured by AIC improvement.

- We see that our backtesting walk-forward method was not particularly effective, as it had limited correlation with the actual volatility path. This could be for a few reasons. The refit_every option cannot be too small otherwise computation takes a significant amount of time due to the slow nature of fitting a GARCH model, so we're stuck with relatively large refit_every's, which potentially reduce accurary. It is also true that the GARCH volatility forecasting accurately is a difficult process with limited overall accuracy.

- When compared with the implied volatility measure from options pricing, we see immediately that investors are (implicitly) predicting a much lower volatility compared to the GARCH forecast. At the time of this model running (31/07/2026) AAPL had a sharp shock and the GARCH prediction seems to reflect this alongside the 5-day rolling volatility, however a significant call option bought after the drop is predicting a drop back towards AAPL's steady volatility level.

- Observing plots in the notebook, it is clear that the GARCH forecast often sits inbetween the 10 and 30 day rolling volatility averages, with the first acting as true recency bias, the latter 'smoothing' out the volatility and GARCH being somewhere in between these, valuing recent volatility more, but not too much.
 
## LIMITATIONS
- Walk forward back-testing was not successful when measured by correlation to the actual volatilty, the reasons for this have already been discussed in key findings.

- In key results, we see that often $\alpha + \beta$ are close to 1. When this occurs, the GARCH model starts to strain and predict nonsense volatility projection (specifically for $\alpha + \beta \geq 1$), possibly a reason for such a lack of correlation when backtesting our model on various tickers, as in these time periods almost all tickers have had periods of massive volatility spikes.

- Often these statistics change significantly based upon what time period we look over, meaning that the GARCH model's 'performance' depends significantly on the time period it was tested over, some models are better at predicting large spikes, others with a more steady market. In our case, these correspond to different time periods, as some contain large volatility spikes depending on many years one chooses to investigate.

## LIBRARIES USED
- statsmodels — ADF stationarity test and ARIMA model fitting
- arch — GARCH, GJR-GARCH, EGARCH volatility models
- pandas — time series data handling and rolling ticker statistics
- numpy — numerical computation
- matplotlib — visualisation of volatility forecasts
- yfinance — historical price data (via past_market_analysis)

## PROJECT STRUCTURE
    src/
        arima_model.py — 
        garch_model.py — GARCH/GJR fitting, volatility forecasting, walk-forward backtesting, implied volatility comparison, cross-ticker validation
        plotting.py    — volatility forecast plots with variable rolling windows, historical volatility baseline, zoom feature
    notebooks/
        forecasting.ipynb — analysis/examples of src python files

## USAGE
```python
from past_market_analysis.data_loader import fetch_price_data
from past_marker_analysis.analysis import summary_statistics
from arima_model import test_stationarity, fit_arima
from garch_model import fit_garch, forecast_volatility, fit_asymmetric_garch, backtest_garch, compare_garch_vs_implied, cross_ticker_validation
from plotting import plot_garch_forecast

ticker = "AAPL"
df = fetch_price_data(ticker, period="5y")
df = compute_returns(df)

# Confirm returns are stationary before modelling
test_stationarity(df['Return'], label=f"{ticker} returns")

# ARIMA — automatic (p,d,q) selection by AIC grid search
arima = fit_arima(df['Return'])
print(arima.summary())

# GARCH — volatility model and 10-day forecast
garch = fit_garch(df['Return'])
vol_forecast = forecast_volatility(garch, horizon=10)
plot_garch_forecast(df, vol_forecast, realised_vol, ticker, zoom_days=60)

# GJR-GARCH — captures the leverage effect
gjr = fit_asymmetric_garch(df['Return'], model_type='GJR')
print(f"AIC improvement over plain GARCH: {garch.aic - gjr.aic:.2f}")

# Walk-forward forecast validation
bt = backtest_garch(df['Return'], train_size=500, refit_every=20)

# Compare GARCH forecast against market implied volatility
compare_garch_vs_implied(ticker, expiry_index=5)

# Check findings generalise across assets
summary = cross_ticker_validation(["AAPL", "NVDA", "SPY", "JNJ", "TSLA"])
```