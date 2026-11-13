
#weekly observation beta. no gaps filled
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

def build_lse_schedule(start, end):
    #build LSE trading schedule [start, end)

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    if start >= end:
        raise ValueError("start must be earlier than end.")

    schedule = mcal.get_calendar("LSE").schedule(start_date=start, end_date=end - pd.Timedelta(days=1))
    session = pd.Timestamp("2022-05-30")

    if start <= session < end and session not in schedule.index:
        schedule.loc[session, ["market_open", "market_close"]] = [
            pd.Timestamp("2022-05-30 08:00", tz="Europe/London").tz_convert("UTC"),
            pd.Timestamp("2022-05-30 16:30", tz="Europe/London").tz_convert("UTC")]

    return schedule.sort_index().rename_axis("Date")


def build_weekly_return_panels(prices, proxy_raw, schedule):
    #build aligned weekly stock and market returns without filling gaps

    sessions = pd.DatetimeIndex(schedule.index, name="Date")

    if prices.index.names != ["Date", "ticker"] or not prices.index.is_unique:
        raise ValueError("Expected unique (Date, ticker) price rows.")
    if not proxy_raw.index.is_unique:
        raise ValueError("Market-proxy dates must be unique.")

    #extend calendar to identify an incomplete final week
    calendar_end = (sessions[-1].to_period("W-FRI").end_time.normalize() + pd.Timedelta(days=1))
    full_sessions = build_lse_schedule(sessions[0], calendar_end).index

    week_ends = full_sessions.to_series().groupby(full_sessions.to_period("W-FRI")).max()
    week_ends = pd.DatetimeIndex(week_ends.loc[week_ends <= sessions[-1]].to_numpy(), name="Date")

    stock_valid = (prices["close_reference_usable"].eq(True) & prices["positive_reported_volume"].eq(True) & np.isfinite(prices["adj_close"]) & prices["adj_close"].gt(0))
    stock_close = prices["adj_close"].where(stock_valid).unstack("ticker").reindex(week_ends)
    proxy_valid = (np.isfinite(proxy_raw["Adj Close"])& proxy_raw["Adj Close"].gt(0)& proxy_raw["Volume"].gt(0))
    proxy_close = proxy_raw["Adj Close"].where(proxy_valid).reindex(week_ends)

    stock_returns = stock_close.pct_change(fill_method=None)
    market_returns = proxy_close.pct_change(fill_method=None).rename("market_return")

    return stock_returns, market_returns


def build_rolling_beta(stock_returns, market_returns, lookback=104, min_observations=78):
    #estimate beta using matched returns within fixed (weekly) windows
    
    if not 2 <= min_observations <= lookback:
        raise ValueError("Require 2 <= min_observations <= lookback.")
    if not stock_returns.index.equals(market_returns.index):
        raise ValueError("Stock and market dates must match exactly.")
    if (not stock_returns.index.is_unique or not stock_returns.index.is_monotonic_increasing or not stock_returns.columns.is_unique):
        raise ValueError("Require unique, ordered dates and unique tickers.")

    betas = pd.DataFrame(np.nan, index=stock_returns.index, columns=stock_returns.columns)
    paired_counts = pd.DataFrame(0, index=stock_returns.index, columns=stock_returns.columns, dtype=int)

    for ticker in stock_returns.columns:
        stock = stock_returns[ticker]
        paired = np.isfinite(stock) & np.isfinite(market_returns)

        stock = stock.where(paired)
        market = market_returns.where(paired)

        counts = paired.rolling(lookback, min_periods=1).sum()
        paired_counts[ticker] = counts.astype(int)

        covariance = stock.rolling(lookback, min_periods=min_observations).cov(market)
        market_variance = market.rolling(lookback, min_periods=min_observations).var()

        #count weekly rows from first return
        history_ready = paired.cummax().cumsum().ge(lookback)

        beta = covariance.div(market_variance.where(market_variance.gt(0)))
        betas[ticker] = beta.where(history_ready & counts.ge(min_observations) & np.isfinite(beta))

    return betas, paired_counts


def build_low_beta_sweep_features(stock_returns, market_returns, schedule, liquidity_eligible, parameter_grid):
    expected_index = pd.MultiIndex.from_product([schedule.index, stock_returns.columns], names=["Date", "ticker"])
    if not liquidity_eligible.index.equals(expected_index):
        raise ValueError("Liquidity must match the scheduled stock panel.")

    specifications = parameter_grid[["beta_lookback_weeks", "beta_min_observations"]].drop_duplicates()
    if specifications.empty:
        raise ValueError("The parameter grid is empty.")

    common_eligible = liquidity_eligible.eq(True).copy()
    feature_panels = {}

    for lookback, minimum in specifications.itertuples(index=False, name=None):
        weekly_beta, _ = build_rolling_beta(stock_returns, market_returns, lookback=lookback, min_observations=minimum)

        daily_beta = weekly_beta.reindex(schedule.index, method="ffill")
        beta = daily_beta.rename_axis(index="Date", columns="ticker").stack(future_stack=True)

        common_eligible &= np.isfinite(beta)
        feature_panels[(lookback, minimum)] = beta.to_frame("beta")

    #apply completed intersection to every estimation window
    for panel in feature_panels.values():
        panel["eligible"] = common_eligible

    return feature_panels, common_eligible.rename("eligible")
