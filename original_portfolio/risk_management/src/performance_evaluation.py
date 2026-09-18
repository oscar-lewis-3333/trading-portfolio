#create tables showcasing comparison of risk methods across various statistics, not just by the equity curve.
import numpy as np
import pandas as pd


def build_returns_by_method(raw_returns, drawdown_returns, volatility_returns, barrier_returns):
    returns = pd.concat({
        'Raw momentum': raw_returns,
        'Drawdown overlay': drawdown_returns,
        'Volatility overlay': volatility_returns,
        'Position barriers': barrier_returns}, axis=1, join='inner')

    returns = returns.apply(pd.to_numeric, errors='raise').astype(float)

    if returns.empty:
        raise ValueError("No common return dates were found")
    if returns.index.has_duplicates:
        raise ValueError("Return dates cannot be duplicated")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Returns must be chronological")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns must be finite")
    if (returns < -1).any().any():
        raise ValueError("Returns cannot be below -100%")

    return returns


def build_exposure_by_method(returns_by_method, drawdown_exposure, volatility_exposure, barrier_cash_weight):
    if not isinstance(returns_by_method, pd.DataFrame):
        raise TypeError("Returns by method must be a DataFrame")

    barrier_cash_weight = pd.to_numeric(barrier_cash_weight.copy(), errors='raise').astype(float)

    exposures = pd.concat({
        'Raw momentum': pd.Series(1.0, index=returns_by_method.index),
        'Drawdown overlay': drawdown_exposure,
        'Volatility overlay': volatility_exposure,
        'Position barriers': 1 - barrier_cash_weight}, axis=1)

    exposures = exposures.reindex(returns_by_method.index)
    exposures = exposures.apply(pd.to_numeric, errors='raise').astype(float)

    if exposures.isna().any().any():
        raise ValueError("Exposure dates do not align with return dates")
    if not np.isfinite(exposures.to_numpy()).all():
        raise ValueError("Exposures must be finite")
    if not exposures.ge(0).all().all() or not exposures.le(1).all().all():
        raise ValueError("Exposures must be between 0 and 1")

    return exposures


def compare_risk_methods(returns_by_method, exposure_by_method, periods_per_year=252): #by periods a 'trading period' or 'trading day' is meant.
    if not isinstance(returns_by_method, pd.DataFrame):
        raise TypeError("Returns by method must be a DataFrame")
    if not isinstance(exposure_by_method, pd.DataFrame):
        raise TypeError("Exposure by method must be a DataFrame")
    if returns_by_method.empty:
        raise ValueError("Returns cannot be empty")
    if not returns_by_method.index.equals(exposure_by_method.index):
        raise ValueError("Return and exposure dates must match")
    if not returns_by_method.columns.equals(exposure_by_method.columns):
        raise ValueError("Return and exposure methods must match")
    if not isinstance(periods_per_year, (int, np.integer)) or isinstance(periods_per_year, bool) or periods_per_year < 1:
        raise ValueError("Periods per year must be a positive integer (hence greater than or equal to 1)")

    dates = pd.to_datetime(returns_by_method.index, errors='raise')
    years = (dates[-1] - dates[0]).days / 365.25

    if years <= 0:
        raise ValueError("At least two distinct dates are required")

    rows = []

    for method in returns_by_method.columns:
        returns = returns_by_method[method]
        equity = (1 + returns).cumprod()

        running_peak = equity.cummax().clip(lower=1.0)
        drawdown = equity / running_peak - 1

        annual_vol = returns.std() * np.sqrt(periods_per_year)
        cagr = equity.iloc[-1] ** (1 / years) - 1 #new metric, compound annual growth rate measuring mean annual growth rate.
        max_drawdown = drawdown.min()

        sharpe = returns.mean() / returns.std()* np.sqrt(periods_per_year) if returns.std() > 0 else np.nan

        calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

        rows.append({
            'Method': method,
            'Start': dates[0].date(),
            'End': dates[-1].date(),
            'Observations': len(returns),
            'Final equity': equity.iloc[-1],
            'CAGR (%)': cagr * 100,
            'Annual volatility (%)': annual_vol * 100,
            'Maximum drawdown (%)': max_drawdown * 100,
            'Sharpe': sharpe,
            'Calmar': calmar,
            'Average exposure (%)': exposure_by_method[method].mean() * 100
        })

    return pd.DataFrame(rows).set_index('Method').round(3)