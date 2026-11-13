#extension using risk weighting techiniques, with inverse volatility in place of 10% fixed pies.
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

import etf_backtest
import etf_cash
import etf_data
import etf_features
import etf_research

VOL_WINDOWS = (63, 126)
MIN_OBSERVED_FRACTION = 0.9


def _load_risk_metrics():
    #reuse assets from portfolio_construction
    root = Path(__file__).resolve().parents[3]
    path = root / "original_portfolio" / "portfolio_construction" / "src" / "risk_metrics.py"
    spec = importlib.util.spec_from_file_location("portfolio_construction_risk_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def minimum_observations(window, min_fraction=MIN_OBSERVED_FRACTION):
    #57 of 63 and 114 of 126 at the agreed 90%
    return math.ceil(round(min_fraction * window, 9))


def trailing_volatility(adjusted_close, decision_dates, *, window, min_fraction=MIN_OBSERVED_FRACTION, annualisation=252):
    #annualised std (volatility) of daily returns (measured close-to-close as usual) over the past window sessions

    if (isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2):
        raise ValueError("window must be an integer of at least 2.")
    if not adjusted_close.index.is_unique or not adjusted_close.index.is_monotonic_increasing:
        raise ValueError("Prices must have ordered, unique dates.")
    decision_dates = pd.DatetimeIndex(decision_dates)
    if not decision_dates.isin(adjusted_close.index).all():
        raise ValueError("Every decision date must be a price session.")

    returns = adjusted_close.astype(float).pct_change(fill_method=None)
    min_obs = minimum_observations(window, min_fraction)
    volatility = returns.rolling(window, min_periods=min_obs).std(ddof=1) * np.sqrt(annualisation)
    observed = returns.notna().rolling(window, min_periods=1).sum()

    #window cannot go back before the test starts
    position = pd.Series(np.arange(len(returns)), index=returns.index)
    too_early = position < window
    volatility = volatility.mask(too_early, np.nan)
    return volatility.loc[decision_dates], observed.loc[decision_dates].astype(int)


def inverse_vol_weights(signal, volatility):

    #w_i = signal_i * (1/sigma_i) / sum_j (1/sigma_j) over all funds. remainder in cash. fund without valid vol is not available, keeping its 1/N(N=10) in cash. funds available share remaining (N_available/N) between them
    if not signal.index.equals(volatility.index) or not signal.columns.equals(volatility.columns):
        raise ValueError("Signal and volatility must have matching dates and funds.")
    if "CASH" in signal.columns or not signal.columns.is_unique:
        raise ValueError("Provide unique fund columns; CASH is reserved.")
    if signal.isna().any().any() or not signal.isin([0, 1]).all().all():
        raise ValueError("Signals must be complete and contain only 0 or 1.")
    if ((volatility <= 0) | np.isinf(volatility)).any().any():
        raise ValueError("Volatilities must be positive and finite where available.")

    n_funds = len(signal.columns)
    unavailable = volatility.isna()
    inverse = 1 / volatility
    available_count = (~unavailable).sum(axis=1)
    share = inverse.div(inverse.sum(axis=1), axis=0).mul(available_count / n_funds, axis=0)

    weights = (share.fillna(0.0) * signal.astype(float))
    weights["CASH"] = 1 - weights.sum(axis=1)
    if (weights["CASH"] < -1e-12).any():
        raise ValueError("Inverse-volatility weights exceed full investment.")
    weights["CASH"] = weights["CASH"].clip(lower=0.0)
    return weights, unavailable


def constant_mix_weights(volatility):
    #this is the weights assuming each ticker is available
    signal = pd.DataFrame(1.0, index=volatility.index, columns=volatility.columns)
    return inverse_vol_weights(signal, volatility)


def to_execution_weights(decision_weights, decision_schedule):
    #move the weights formed at the month ending's close to following trading days' open
    weights = decision_weights.reindex(decision_schedule.index)
    if weights.isna().any().any():
        raise ValueError("Weights are missing on some decision dates.")
    weights = weights.copy()
    weights.index = pd.DatetimeIndex(decision_schedule["execution_date"], name="Date")
    return weights


def build_targets(data, period, *, vol_window, lookback_months=6):
    #weights for the signal date and the rebalance date for the four compared portfolios
    inputs = etf_research.period_inputs(data, period)
    schedule = inputs["decision_schedule"]
    decisions = schedule.index
    funds = data["adjusted_close"].columns

    _, signal = etf_features.build_trend_features(data["monthly_prices"], lookback_months=lookback_months)
    signal = signal.reindex(decisions)
    volatility, observed = trailing_volatility(data["adjusted_close"], decisions, window=vol_window)

    equal_weight = etf_backtest.build_target_weights(signal)
    inverse_trend, unavailable = inverse_vol_weights(signal, volatility)
    constant_mix, _ = constant_mix_weights(volatility)

    buy_and_hold = pd.DataFrame(1 / len(funds), index=decisions[:1], columns=funds)
    buy_and_hold["CASH"] = 0.0

    decision_weights = {
        "equal_weight_trend": equal_weight,
        "inverse_vol_trend": inverse_trend,
        "inverse_vol_constant_mix": constant_mix,
        "buy_and_hold": buy_and_hold
    }
    execution_weights = {name: to_execution_weights(weights, schedule.loc[weights.index]) for name, weights in decision_weights.items()}
    return {
        "inputs": inputs,
        "signal": signal,
        "volatility": volatility,
        "observed_returns": observed,
        "unavailable": unavailable,
        "decision_weights": decision_weights,
        "execution_weights": execution_weights
    }


def unavailable_events(unavailable):
    #logging the unavailable funds and at what date they were unavailable as a pair
    stacked = unavailable.stack()
    return stacked[stacked].index.to_frame(index=False, name=["decision_date", "ticker"])


def run_portfolios(targets, settings):
    #running each portfolio under same conditions
    inputs = targets["inputs"]
    execution_dates = pd.DatetimeIndex(inputs["decision_schedule"]["execution_date"])
    marked_open, marked_close, stale_quotes = etf_data.prepare_valuation_marks(inputs["open_prices"], inputs["close_prices"], inputs["excluded_quotes"], execution_dates=execution_dates)
    results = {}
    for name, weights in targets["execution_weights"].items():
        result = etf_backtest.run_backtest(marked_open, marked_close, weights, **settings)
        held = result["holdings"].drop(columns="CASH").gt(0)
        result["daily"]["stale_valuation"] = (stale_quotes & held).any(axis=1)
        results[name] = result

    return results


def monthly_returns(daily_returns):
    #compounding daily (net) returns monthly
    return (1 + daily_returns).groupby(daily_returns.index.to_period("M")).prod() - 1


def monthly_excess_returns(results, cash_monthly):
    #portfolio monthly return against return on cash
    frame = pd.DataFrame({name: monthly_returns(result["daily"]["net_return"]) for name, result in results.items()})
    if not frame.index.equals(cash_monthly.index):
        raise ValueError("Portfolio and cash months must match.")
    return frame.sub(cash_monthly, axis=0)


def sharpe_ratio(monthly_excess, periods_per_year=12):
    #Sharpe ratio (annualised) for monthly excess
    values = np.asarray(monthly_excess, dtype=float)
    std = values.std(axis=-1, ddof=1)
    if np.any(std <= 0):
        raise ValueError("Zero-variance returns make the Sharpe ratio undefined.")
    return values.mean(axis=-1) / std * np.sqrt(periods_per_year)


def annual_traded_fraction(daily, *, exclude_initial=True):
    #buys + sells as a fraction of NAV at each rebalance, summed and annualised.
    #10bp per side costs 0.10% of NAV per year for each 1.0 of this figure.
    rebalances = daily.loc[daily["rebalanced"]]
    if exclude_initial:
        rebalances = rebalances.iloc[1:]
    fraction = (rebalances["traded_notional"] / rebalances["nav_open_before_cost"]).sum()
    years = ((daily.index[-1] - daily.index[0]).days + 1) / 365.25
    return fraction / years


def summarise(results, cash_monthly, initial_capital_gbp):
    #summarising results (and Sharpe) in a table
    summary = etf_research.summarise_portfolios(results, initial_capital_gbp)
    excess = monthly_excess_returns(results, cash_monthly)
    summary["sharpe_excess"] = [float(sharpe_ratio(excess[name])) for name in summary.index]
    summary["traded_pct_nav_pa"] = [annual_traded_fraction(results[name]["daily"]) * 100 for name in summary.index]
    summary["total_interest_gbp"] = [results[name]["daily"]["interest"].sum() for name in summary.index]
    summary["rebalances"] = [int(results[name]["daily"]["rebalanced"].sum()) for name in summary.index]
    return summary


def select_vol_window(candidates, *, tolerance=0.05, default_window=126):
    #sweeping volatility windows, window decided by development Sharpe in excess of cash

    if set(candidates.index) != set(VOL_WINDOWS):
        raise ValueError(f"Provide exactly the windows {VOL_WINDOWS}.")
    sharpe = candidates["sharpe_excess"]
    turnover = candidates["traded_pct_nav_pa"]
    gap = abs(sharpe.loc[63] - sharpe.loc[126])

    if gap > tolerance:
        chosen = int(sharpe.idxmax())
        reason = f"Sharpe gap {gap:.3f} > {tolerance}: higher development Sharpe"
    elif not np.isclose(turnover.loc[63], turnover.loc[126], rtol=0, atol=1e-9):
        chosen = int(turnover.idxmin())
        reason = f"Sharpe gap {gap:.3f} <= {tolerance}: lower turnover"
    else:
        chosen = default_window
        reason = f"Sharpe gap {gap:.3f} <= {tolerance} and equal turnover: default {default_window}"
    return chosen, reason


#following two functions are bootstrap testing for Sharpe difference significance. first is the sampling, second is the testing itself.
def _circular_block_indices(n_obs, block_length, n_bootstrap, rng):
    blocks_per_sample = (n_obs + block_length - 1) // block_length
    starts = rng.integers(0, n_obs, size=(n_bootstrap, blocks_per_sample))
    indices = (starts[:, :, None] + np.arange(block_length)) % n_obs
    return indices.reshape(n_bootstrap, -1)[:, :n_obs]


def paired_sharpe_bootstrap(strategy_excess, benchmark_excess, *, block_months=3, n_bootstrap=10_000, confidence_level=0.95, seed=42):

    for series in (strategy_excess, benchmark_excess):
        if not isinstance(series.index, pd.PeriodIndex):
            raise ValueError("Use a monthly PeriodIndex.")
    if not strategy_excess.index.equals(benchmark_excess.index):
        raise ValueError("Strategy and benchmark months must match.")
    expected = pd.period_range(strategy_excess.index.min(), strategy_excess.index.max(), freq="M")
    if not strategy_excess.index.equals(expected):
        raise ValueError("Months must be consecutive, ordered and unique.")
    for value in (block_months, n_bootstrap):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("Block length and sample count must be positive integers.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must lie between zero and one.")

    a = strategy_excess.to_numpy(dtype=float)
    b = benchmark_excess.to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Excess returns must be finite.")
    n_months = len(a)
    if n_months < 2 * block_months:
        raise ValueError("Provide at least two blocks of observations.")

    sharpe_a = float(sharpe_ratio(a))
    sharpe_b = float(sharpe_ratio(b))
    observed = sharpe_a - sharpe_b

    rng = np.random.default_rng(seed)
    indices = _circular_block_indices(n_months, block_months, n_bootstrap, rng)
    draws = sharpe_ratio(a[indices]) - sharpe_ratio(b[indices])
    errors = draws - observed

    alpha = 1 - confidence_level
    lower_error, upper_error = np.quantile(errors, [alpha / 2, 1 - alpha / 2])
    #h_0: Sharpe diff <= 0.
    p_value = (1 + np.count_nonzero(errors >= observed)) / (n_bootstrap + 1)

    return pd.Series({
        "months": n_months,
        "block_months": block_months,
        "strategy_sharpe": sharpe_a,
        "benchmark_sharpe": sharpe_b,
        "sharpe_difference": observed,
        "lower_ci": observed - upper_error,
        "upper_ci": observed - lower_error,
        "one_sided_p_value": p_value
    })


def gate_a_verdict(development_difference, holdout_difference, *, threshold=0.10):
    #designing whether our results are useful in any way
    if development_difference > threshold and holdout_difference > threshold:
        return "helped"
    if development_difference < -threshold and holdout_difference < -threshold:
        return "hurt"
    return "no difference"


def class_risk_shares(decision_weights, adjusted_close, asset_groups, *, window):
    #describing share of the invested portfolios' risk from each asset at each signal date. uses sample covariance from window
    risk_metrics = _load_risk_metrics()
    funds = [column for column in decision_weights.columns if column != "CASH"]
    groups = pd.Series(asset_groups).reindex(funds)
    if groups.isna().any():
        raise ValueError("Every fund needs an asset group.")

    returns = adjusted_close[funds].astype(float).pct_change(fill_method=None)
    positions = pd.Series(np.arange(len(returns)), index=returns.index)
    records = {}
    for date, row in decision_weights[funds].iterrows():
        weights = row.to_numpy(dtype=float)
        if weights.sum() <= 0:
            records[date] = {**{group: np.nan for group in groups.unique()}, "ex_ante_vol_pct": np.nan, "invested_pct": 0.0}
            continue
        end = positions.loc[date]
        sample = returns.iloc[max(end - window + 1, 0):end + 1].dropna()
        covariance = sample.cov().to_numpy() * 252
        _, shares = risk_metrics.risk_contributions(weights, covariance)
        by_group = pd.Series(shares, index=funds).groupby(groups).sum()
        records[date] = {
            **by_group.to_dict(),
            "ex_ante_vol_pct": float(np.sqrt(weights @ covariance @ weights)) * 100,
            "invested_pct": weights.sum() * 100
        }
    return pd.DataFrame.from_dict(records, orient="index").rename_axis("decision_date")


COMPARISONS = {
    "weighting: inverse-vol trend vs equal-weight trend": ("inverse_vol_trend", "equal_weight_trend"),
    "trend rule: inverse-vol trend vs inverse-vol constant mix": ("inverse_vol_trend", "inverse_vol_constant_mix")
}


def run_period(data, period, *, vol_window, settings):
    #do all above for one period and all portfolios
    targets = build_targets(data, period, vol_window=vol_window)
    runs = run_portfolios(targets, settings)
    sessions = targets["inputs"]["open_prices"].index
    cash_monthly = etf_cash.monthly_cash_returns(settings["cash_rate_annual"], sessions)
    return {
        "targets": targets,
        "runs": runs,
        "cash_monthly": cash_monthly,
        "excess": monthly_excess_returns(runs, cash_monthly),
        "summary": summarise(runs, cash_monthly, settings["initial_capital_gbp"]),
    }


def compare_pairs(excess, *, block_lengths=(3, 2, 6), n_bootstrap=10_000, confidence_level=0.95, seed=42):
    #bootstrap test of Sharpe difference for each strategy and its benchmark, with sensitivities
    rows = []
    for label, (strategy, benchmark) in COMPARISONS.items():
        for block in block_lengths:
            result = paired_sharpe_bootstrap(
                excess[strategy], excess[benchmark], block_months=block,
                n_bootstrap=n_bootstrap, confidence_level=confidence_level, seed=seed,
            )
            rows.append({"comparison": label, "primary": block == block_lengths[0], **result.to_dict()})
    return pd.DataFrame(rows).set_index(["comparison", "block_months"])