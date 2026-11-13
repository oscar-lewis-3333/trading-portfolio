import numpy as np
import pandas as pd


def _clean(returns):
    #if a metric is computed over silently dropped observations, it isn't real
    if not isinstance(returns, pd.Series):
        raise ValueError("Returns must be a Series.")
    series = returns.astype(float)
    if series.empty:
        raise ValueError("The return series is empty.")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("Returns must be finite; handle gaps before measuring.")
    if series.le(-1).any():
        raise ValueError("A return of -100% or worse cannot be compounded.")

    return series


def total_return(returns):
    #compounded returns
    return float((1.0 + _clean(returns)).prod() - 1.0)


def cagr(returns, periods_per_year=365):
    series = _clean(returns)
    years = len(series) / periods_per_year
    if years <= 0:
        raise ValueError("The window must cover a positive length of time.")

    return float((1.0 + total_return(series)) ** (1.0 / years) - 1.0)


def annualised_volatility(returns, periods_per_year=365):
    series = _clean(returns)
    if len(series) < 2:
        raise ValueError("Volatility needs at least two observations.")

    return float(series.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns, risk_free_annual=0.0, periods_per_year=365):
    #excess returns measured over cash
    series = _clean(returns)
    daily_rf = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = series - daily_rf
    deviation = excess.std(ddof=1)
    if not np.isfinite(deviation) or deviation <= 0:
        return float("nan")

    return float(excess.mean() / deviation * np.sqrt(periods_per_year))


def max_drawdown(returns):
    curve = (1.0 + _clean(returns)).cumprod()
    peak = curve.cummax()

    return float((curve / peak - 1.0).min())


def information_ratio(active_return, periods_per_year=365):
    #the active stream's Sharpe against zero

    return sharpe(active_return, 0.0, periods_per_year)


def summary(returns, risk_free_annual=0.0, periods_per_year=365):
    #one row per signal, identical fields for every signal
    series = _clean(returns)

    return {
        "observations": int(len(series)),
        "total_return": total_return(series),
        "cagr": cagr(series, periods_per_year),
        "annualised_volatility": annualised_volatility(series, periods_per_year),
        "sharpe": sharpe(series, risk_free_annual, periods_per_year),
        "max_drawdown": max_drawdown(series)}


def block_bootstrap_difference(strategy, benchmark, block_days=28, resamples=10000, seed=0, periods_per_year=365):
    #using circular block boostrap for the Sharpe difference significance

    paired = pd.concat([_clean(strategy).rename("s"), _clean(benchmark).rename("b")], axis=1).dropna()
    observations = len(paired)
    if observations < block_days * 2:
        raise ValueError("The sample is too short for the requested block length.")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("resamples must be an integer of at least 100.")

    values = paired.to_numpy(dtype=float)
    blocks = int(np.ceil(observations / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, observations, size=(resamples, blocks))
    offsets = np.arange(block_days)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(resamples, -1) % observations
    indices = indices[:, :observations]

    drawn = values[indices]
    means = drawn.mean(axis=1)
    deviations = drawn.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(deviations > 0, means / deviations * np.sqrt(periods_per_year), np.nan)

    differences = sharpes[:, 0] - sharpes[:, 1]
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        raise ValueError("No usable bootstrap resamples.")

    estimate = sharpe(paired["s"], 0.0, periods_per_year) - sharpe(paired["b"], 0.0, periods_per_year)
    return {"estimate": float(estimate),
        "ci_lower": float(np.percentile(differences, 2.5)),
        "ci_upper": float(np.percentile(differences, 97.5)),
        "block_days": int(block_days),
        "resamples": int(resamples)}


def block_bootstrap_ir(active_return, block_days=28, resamples=10000, seed=0, periods_per_year=365, level=0.95):
    #one sided lower bound on the active IR using circular blocks

    values = _clean(active_return).to_numpy(dtype=float)
    observations = len(values)
    if observations < block_days * 2:
        raise ValueError("The sample is too short for the requested block length.")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("resamples must be an integer of at least 100.")
    if not 0.5 < level < 1:
        raise ValueError("level must lie in (0.5, 1).")

    blocks = int(np.ceil(observations / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, observations, size=(resamples, blocks))
    offsets = np.arange(block_days)
    indices = ((starts[:, :, None] + offsets[None, None, :]).reshape(resamples, -1) % observations)[:, :observations]

    drawn = values[indices]
    means = drawn.mean(axis=1)
    deviations = drawn.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        irs = np.where(deviations > 0, means / deviations * np.sqrt(periods_per_year), np.nan)

    irs = irs[np.isfinite(irs)]
    if irs.size == 0:
        raise ValueError("No usable bootstrap resamples.")

    return {"estimate": float(information_ratio(active_return, periods_per_year)),
        "lower": float(np.percentile(irs, 100 * (1 - level))),
        "upper": float(np.percentile(irs, 100 * level)),
        "level": level,
        "block_days": int(block_days),
        "resamples": int(resamples)}


def shrunk_information_ratio(ir, years, prior_sd=0.30):
    #empirical Bayes posterior mean towards zero. kappa = tau^2 / (tau^2 + 1/years)

    if not np.isfinite(ir) or years <= 0 or prior_sd <= 0:
        return float("nan")
    kappa = prior_sd ** 2 / (prior_sd ** 2 + 1.0 / years)

    return float(kappa * ir)


def deflated_sharpe_ratio(active_return, trial_sharpes, periods_per_year=365):
    #probability that true Sharpe exceeds the best of N null trials

    from math import erf, sqrt

    values = _clean(active_return).to_numpy(dtype=float)
    n = len(values)
    trials = np.asarray([t for t in trial_sharpes if np.isfinite(t)], dtype=float)
    if n < 30 or trials.size < 1:
        raise ValueError("Need at least 30 observations and one recorded trial.")

    scale = np.sqrt(periods_per_year)
    observed = values.mean() / values.std(ddof=1)
    per_period = trials / scale
    spread = per_period.std(ddof=1) if trials.size > 1 else 0.0

    count = trials.size
    gamma = 0.5772156649015329
    normal_inv = _normal_quantile
    expected_max = spread * ((1 - gamma) * normal_inv(1 - 1 / count) + gamma * normal_inv(1 - 1 / (count * np.e))) if count > 1 else 0.0

    centred = values - values.mean()
    skew = float((centred ** 3).mean() / values.std(ddof=0) ** 3)
    kurt = float((centred ** 4).mean() / values.std(ddof=0) ** 4)
    denominator = np.sqrt(1 - skew * observed + (kurt - 1) / 4 * observed ** 2)
    z = (observed - expected_max) * np.sqrt(n - 1) / denominator

    return {"deflated_sharpe": float(0.5 * (1 + erf(z / sqrt(2)))),
        "observed_sharpe": float(observed * scale),
        "expected_max_null_sharpe": float(expected_max * scale),
        "trials": int(count)}


def _normal_quantile(p):
    #Acklam's rational approximation to get around a scipy dependency

    if not 0 < p < 1:
        raise ValueError("p must lie in (0, 1).")
    
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > high:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
