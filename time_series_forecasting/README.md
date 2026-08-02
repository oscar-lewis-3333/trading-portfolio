# TIME SERIES FORECASTING

## OVERVIEW
[ARIMA for returns, GARCH for volatility, with forecast validation]

## FEATURES
[ADF stationarity testing, automated ARIMA order selection via AIC grid search,
GARCH and GJR-GARCH fitting, volatility forecasting, walk-forward backtesting,
GARCH vs market implied volatility comparison, cross-ticker validation]

## KEY DESIGN DECISIONS
[Why the manual AIC grid search instead of pmdarima — NumPy 2.x incompatibility,
and the value of understanding the selection process. Why returns rather than
prices — stationarity. Why refit_every rather than daily refitting.]

## RESULTS
[ARIMA orders by ticker, GARCH persistence values, GJR gamma significance,
backtest error metrics, the live GARCH vs implied volatility example]

## KEY FINDINGS
[The honest story: models describe volatility persistence robustly, but
return predictability and leverage asymmetry are asset-specific and
sample-sensitive. Near-zero forecast correlation. AAPL's gamma significance
disappearing on a larger sample.]

## LIMITATIONS
[Noisy realised volatility proxy in backtesting. Modest sample sizes. Single
market period. alpha+beta at/above 1.0 for JNJ and SPY suggesting model strain.
GARCH forecast validation is genuinely hard to do well.]

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