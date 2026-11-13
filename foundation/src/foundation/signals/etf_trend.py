#fixed ETF lookback builders, with decisions only using observed closes

import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
RISK_CLASSES = ("equities", "bonds", "real_assets")
LOOKBACKS = (6, 9, 12)

def build_trend_features(monthly_prices, lookback_months=12):

    #return trailing returns and month-end long/cash signals

    if monthly_prices.empty:
        raise ValueError("Provide nonempty monthly prices.")
    if (isinstance(lookback_months, bool) or not isinstance(lookback_months, int) or lookback_months < 1):
        raise ValueError("lookback_months must be a positive integer.")

    months = monthly_prices.index.to_period("M")
    expected = pd.period_range(months.min(), months.max(), freq="M")

    if not months.equals(expected):
        raise ValueError("Prices must contain consecutive, ordered months.")

    prices = monthly_prices.astype(float)
    invalid = prices.notna() & ((prices <= 0) | ~np.isfinite(prices))
    if invalid.any().any():
        raise ValueError("Observed prices must be positive and finite.")

    trailing_return = prices / prices.shift(lookback_months) - 1

    #12 monthly returns require 13 consecutive price observations
    complete_window = prices.notna().rolling(lookback_months + 1, min_periods=lookback_months + 1).sum().eq(lookback_months + 1)

    trailing_return = trailing_return.where(complete_window)
    signal = trailing_return.gt(0).astype(float).where(trailing_return.notna())

    return trailing_return, signal

def _positive(values):
    return np.isfinite(values) & (values > 0)


def _validate_inputs(inputs):
    required = {"adjusted_close", "execution_open", "valuation_open", "valuation_close", "cash", "cash_execution_allowed", "bank_rate_hurdle", "universe"}
    if not required.issubset(inputs):
        raise ValueError(f"Missing inputs: {sorted(required.difference(inputs))}.")
    
    prices = inputs["adjusted_close"]
    dates, tickers = prices.index, prices.columns
    if (prices.empty or not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing or dates.tz is not None or not dates.equals(dates.normalize()) or not tickers.is_unique or "CASH" in tickers):
        raise ValueError("Prices require unique tickers and ordered daily dates.")
    for key in ("execution_open", "valuation_open", "valuation_close"):
        panel = inputs[key]
        if not panel.index.equals(dates) or not panel.columns.equals(tickers):
            raise ValueError(f"{key} must match the adjusted-close panel.")
        
    cash = inputs["cash"]
    if not cash.index.equals(dates) or not {"cash_reference_open", "cash_reference_close", "cash_source"}.issubset(cash):
        raise ValueError("Joined cash references must match the price dates.")
    if not _positive(cash[["cash_reference_open", "cash_reference_close"]].to_numpy(dtype=float)).all():
        raise ValueError("Cash valuation references must be positive and finite.")
    
    allowed = inputs["cash_execution_allowed"]
    if not allowed.index.equals(dates) or allowed.isna().any() or not allowed.isin([True, False]).all():
        raise ValueError("Cash execution flags must cover every session.")
    
    bank = inputs["bank_rate_hurdle"]
    if not bank.index.equals(dates) or not _positive(bank.to_numpy(dtype=float)).all():
        raise ValueError("Bank Rate hurdle must cover every session.")
    universe = inputs["universe"]
    if not universe.index.is_unique or not tickers.isin(universe.index).all():
        raise ValueError("Universe must identify every ticker uniquely.")
    if not universe.loc[tickers, "asset_class"].isin(RISK_CLASSES).all():
        raise ValueError("Risky holdings require equities, bonds or real_assets.")
        
    return dates, tickers


def _erc_weights(covariance):
    #equal risk contributuions (risk parity)
    
    covariance = np.asarray(covariance, dtype=float)
    count = len(covariance)
    if count == 1:
        return np.ones(1)
    scale = float(np.trace(covariance))
    if not np.isfinite(scale) or scale <= 0:
        return None
    
    covariance = covariance / scale

    def objective(weights):
        contributions = weights * (covariance @ weights)
        variance = float(contributions.sum())
        if variance <= 0:
            return 1e10
        
        return float(((contributions / variance - 1 / count) ** 2).sum())

    initial = 1 / np.sqrt(np.diag(covariance))
    initial /= initial.sum()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds",
            category=RuntimeWarning, module=r"scipy\.optimize\._slsqp_py")
        fit = minimize(objective, initial, method="SLSQP", bounds=[(1e-8, 1)] * count,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-14, "maxiter": 1000})
    if not fit.success or not np.isfinite(fit.x).all() or objective(fit.x) > 1e-8:
        return None
    
    return fit.x / fit.x.sum()


def build_allocation_targets(inputs):
    #build matched allocations, and 6/9/12 month excess cash trend targers

    dates, tickers = _validate_inputs(inputs)
    adjusted = inputs["adjusted_close"].astype(float)
    adjusted = adjusted.where(_positive(adjusted))

    returns = adjusted.pct_change(fill_method=None)
    returns = returns.where(np.isfinite(returns))
    observed = returns.notna().rolling(252, min_periods=1).sum()
    volatility = returns.rolling(252, min_periods=240).std(ddof=1) * np.sqrt(252)

    session_series = pd.Series(dates, index=dates)
    month_ends = pd.DatetimeIndex(session_series.groupby(dates.to_period("M")).max())

    #each decisions needs an observed following session to be executed on
    month_ends = month_ends[month_ends < dates[-1]]
    execution_dates = dates[dates.get_indexer(month_ends) + 1]
    schedule = pd.DataFrame({"execution_date": execution_dates}, index=month_ends)

    schedule.index.name = "decision_date"
    monthly = adjusted.reindex(month_ends)
    monthly_valid = monthly.notna().rolling(13, min_periods=13).sum().eq(13)
    history_age = pd.DataFrame(False, index=month_ends, columns=tickers)

    for ticker in tickers:
        first = adjusted[ticker].first_valid_index()
        if first is not None:
            history_age[ticker] = month_ends >= first + pd.DateOffset(months=12)

    eligibility = (monthly_valid & history_age & observed.reindex(month_ends).ge(240) & volatility.reindex(month_ends).gt(0) & np.isfinite(volatility.reindex(month_ends)))
    classes = inputs["universe"].loc[tickers, "asset_class"]
    baseline = pd.DataFrame(0.0, index=month_ends, columns=list(tickers) + ["CASH"])
    diagnostics = []

    for decision in month_ends:
        active = eligibility.loc[decision]
        trailing = returns.loc[:decision].tail(252)
        within, sleeve_returns = {}, {}
        for asset_class in RISK_CLASSES:
            names = tickers[active & classes.eq(asset_class)]
            if len(names):
                inverse = 1 / volatility.loc[decision, names]
                within[asset_class] = inverse / inverse.sum()

                #no bridging. all assets must have a return on that day
                complete = trailing[names].dropna(how="any")
                sleeve_returns[asset_class] = complete @ within[asset_class]
        active_classes = list(within)
        paired_count, fallback = 0, "none"
        if not active_classes:
            class_weights = np.array([])
            fallback = "no_eligible_lines"
        elif len(active_classes) == 1:
            class_weights = np.array([1 / 3])
            fallback = "single_active_class"

        else:
            paired = pd.DataFrame(sleeve_returns).dropna(how="any")
            paired_count = len(paired)
            erc = None
            if paired_count >= 240:
                covariance = LedoitWolf().fit(paired.to_numpy()).covariance_
                erc = _erc_weights(covariance)

            if erc is None:
                erc = np.full(len(active_classes), 1 / len(active_classes))
                fallback = "paired_history_below_240" if paired_count < 240 else "erc_solver_fallback"

            class_weights = erc * len(active_classes) / 3
        for asset_class, class_weight in zip(active_classes, class_weights):
            baseline.loc[decision, within[asset_class].index] = within[asset_class] * class_weight
        baseline.loc[decision, "CASH"] = max(0.0, 1 - baseline.loc[decision, tickers].sum())

        diagnostics.append({"decision_date": decision, "eligible_lines": int(active.sum()),
            "active_classes": len(active_classes), "paired_sleeve_returns": paired_count,
            "allocation_fallback": fallback,
            **{f"{c}_weight": float(baseline.loc[decision, tickers[classes.eq(c)]].sum()) for c in RISK_CLASSES}})

    decision_weights = {"baseline": baseline}
    bank_monthly = inputs["bank_rate_hurdle"].reindex(month_ends)
    signals = {}
    for lookback in LOOKBACKS:
        excess = monthly.div(monthly.shift(lookback)).sub(1).sub(bank_monthly.div(bank_monthly.shift(lookback)).sub(1), axis=0)
        signal = excess.gt(0) & eligibility
        signals[lookback] = signal

        trend = baseline.copy()
        trend.loc[:, tickers] = baseline[tickers] * signal
        trend["CASH"] = (1 - trend[tickers].sum(axis=1)).clip(lower=0.0, upper=1.0)
        decision_weights[f"trend_{lookback}"] = trend

    any_eligible = eligibility.any(axis=1)
    if not any_eligible.any():
        raise ValueError("No line reaches the declared eligibility rules.")
    
    start = any_eligible.index[any_eligible][0]
    schedule = schedule.loc[start:]
    targets = {}
    for name, weights in decision_weights.items():
        weights = weights.loc[start:].copy()
        decision_weights[name] = weights
        executable = weights.copy()
        executable.index = pd.DatetimeIndex(schedule["execution_date"], name="Date")
        targets[name] = executable

    return {"targets": targets, "decision_weights": decision_weights,
        "decision_schedule": schedule, "eligibility": eligibility.loc[start:],
        "signals": {key: value.loc[start:] for key, value in signals.items()},
        "volatility": volatility.reindex(schedule.index),
        "observed_returns": observed.reindex(schedule.index),
        "diagnostics": pd.DataFrame(diagnostics).set_index("decision_date").loc[start:]}
