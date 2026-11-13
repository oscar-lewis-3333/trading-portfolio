
#testing on event-driven signals, which fire on news, hold for some time, and cash else.
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Market:
    sessions: pd.DatetimeIndex        #dates
    close_time: np.ndarray            #close times
    open_px: pd.DataFrame             #valid adjusted opens
    close_px: pd.DataFrame            #valid adjusted closes
    traded_value: pd.DataFrame        #GBP traded
    tradable: pd.DataFrame            #eligibile assets

    @property
    def tickers(self):
        return self.close_px.columns


def build_market(prices, schedule):
    #gives the market prices and relevant statistics (adj open/close, open/close usable, known suspended periods, etc.) indexed by dates

    def wide(column):
        return prices[column].unstack("ticker").reindex(schedule.index)

    blocked = (wide("known_suspended").fillna(False).astype(bool)
               | wide("close_suspended").fillna(False).astype(bool)
               | wide("unit_quarantine").fillna(False).astype(bool))
    open_ok = wide("open_reference_usable").fillna(False).astype(bool) & ~blocked
    close_ok = wide("close_reference_usable").fillna(False).astype(bool) & ~blocked
    open_px = wide("adj_open").astype(float).where(open_ok)
    close_px = wide("adj_close").astype(float).where(close_ok)

    for frame in (open_px, close_px):
        frame[frame <= 0] = np.nan
    close_time = pd.DatetimeIndex(schedule["market_close"]).tz_convert("UTC").asi8
    if not np.all(np.diff(close_time) > 0):
        raise ValueError("Session closes must be strictly increasing.")
    
    return Market(schedule.index, close_time, open_px, close_px, wide("traded_value_gbp").astype(float), ~blocked)


def day_zero(event_times, market):
    #index of the first session whose close lies strictly after each event

    times = pd.DatetimeIndex(pd.to_datetime(event_times, utc=True)).asi8
    return np.searchsorted(market.close_time, times, side="right")


def abnormal_close_returns(market):
    #close to close returns after an event, with a gap on either side giving NaN

    returns = market.close_px.pct_change(fill_method=None)
    universe = returns.mean(axis=1)
    return returns.sub(universe, axis=0), universe


def trailing_liquidity(market, window=60, min_obs=40):
    #median traded value over the window ending the session before
    return market.traded_value.rolling(window, min_periods=min_obs).median().shift(1)


def event_reactions(events, market, *, vol_window=60, min_vol_obs=40, liq_window=60, min_liq_obs=40):
    #frame with the ticker, event time, abnormal reaction, z, and liquidity on the following day

    abnormal, _ = abnormal_close_returns(market)
    liquidity = trailing_liquidity(market, liq_window, min_liq_obs)
    columns = {t: j for j, t in enumerate(market.tickers)}

    ab, liq, trad = abnormal.to_numpy(), liquidity.to_numpy(), market.tradable.to_numpy()
    d0 = day_zero(events["time"], market)
    rows = []

    for (idx, event), i0 in zip(events.iterrows(), d0):
        j = columns.get(event["ticker"])
        row = dict(event_id=idx, ticker=event["ticker"], time=event["time"], day0=int(i0),
                   reaction=np.nan, sigma=np.nan, z=np.nan, liquidity=np.nan, reason="")
        if j is None:
            row["reason"] = "no_price_history"
        elif i0 < vol_window + 1 or i0 >= len(market.sessions) - 1:
            row["reason"] = "outside_price_history"
        elif not (trad[i0, j] and trad[i0 - 1, j]) or np.isnan(ab[i0, j]):
            row["reason"] = "reaction_not_observable"
        else:
            history = ab[i0 - vol_window:i0, j]
            history = history[~np.isnan(history)]
            if len(history) < min_vol_obs:
                row["reason"] = "short_volatility_history"
            else:
                sigma = history.std(ddof=1)
                row.update(reaction=ab[i0, j], sigma=sigma, z=ab[i0, j] / sigma if sigma > 0 else np.nan,
                           liquidity=liq[i0, j], reason="ok")
        rows.append(row)
    return pd.DataFrame(rows).set_index("event_id")


def _exit_index(open_px, close_px, entry, target, max_wait, j):
    last = len(open_px) - 1
    if target > last:
        return None, None, "censored"
    for i in range(target, min(target + max_wait, last) + 1):
        if not np.isnan(open_px[i, j]):
            return i, open_px[i, j], "open"
    for i in range(min(target + max_wait, last), entry, -1):
        if not np.isnan(close_px[i, j]):
            return i, close_px[i, j], "last_close"
    return None, None, "no_exit_price"


def hold_events(reactions, market, holding, *, eligible=None, max_wait=10):
    #enter at open on day 1, exit holding days later. benchmark set as equal-weight open-to-open return over the same session

    columns = {t: j for j, t in enumerate(market.tickers)}
    op, cl = market.open_px.to_numpy(), market.close_px.to_numpy()
    elig = np.ones_like(op, dtype=bool) if eligible is None else eligible.reindex_like(market.open_px).fillna(False).to_numpy(bool)
    out = []
    for idx, ev in reactions.iterrows():
        row = dict(event_id=idx, entry=np.nan, exit=np.nan, gross=np.nan, benchmark=np.nan, exit_kind="")
        j, entry = columns.get(ev["ticker"]), int(ev["day0"]) + 1
        if j is None or entry >= len(op) or np.isnan(op[entry, j]):
            row["exit_kind"] = "entry_open_unusable"
            out.append(row)
            continue

        exit_i, exit_px, kind = _exit_index(op, cl, entry, entry + holding, max_wait, j)
        row["exit_kind"] = kind
        if exit_i is not None:
            both = elig[entry] & ~np.isnan(op[entry]) & ~np.isnan(op[exit_i])
            bench = np.nanmean(op[exit_i, both] / op[entry, both] - 1.0) if both.any() else np.nan
            row.update(entry=entry, exit=exit_i, gross=exit_px / op[entry, j] - 1.0, benchmark=bench)

        out.append(row)
    return pd.DataFrame(out).set_index("event_id")


def net_return(gross, cost_in, cost_out):
    #per-side costs as a fraction of traded value

    return (1.0 - cost_out) * (1.0 + gross) / (1.0 + cost_in) - 1.0


def month_cluster_bootstrap(values, months, resamples=10000, seed=0, level=0.95):
    #mean of values with a one-sided bootstrap significance test

    frame = pd.DataFrame({"v": np.asarray(values, float), "m": np.asarray(months)}).dropna()
    groups = frame.groupby("m")["v"]
    sums, counts = groups.sum().to_numpy(), groups.count().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(resamples, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return dict(mean=float(frame["v"].mean()), lower=float(np.quantile(means, 1 - level)),
                upper=float(np.quantile(means, level)), events=int(len(frame)), months=int(len(sums)))


def _marks(market):
    #last usable close. only carried forward for valuing a held position

    return market.close_px.ffill().to_numpy()


def slot_portfolio(trades, market, n_slots=15, start=None, end=None):
    #input all information defined above, and get back the return daily closing value and all trades taken

    columns = {t: j for j, t in enumerate(market.tickers)}
    op, marks = market.open_px.to_numpy(), _marks(market)
    first = 0 if start is None else int(start)
    last = len(op) - 1 if end is None else int(end)

    by_entry = {i: g.sort_values("time") for i, g in trades.groupby("entry")}
    cash, held, values, taken = 1.0, {}, [], []
    for i in range(first, last + 1):
        for key in [k for k, p in held.items() if p["exit"] == i]:
            p = held.pop(key)
            cash += p["units"] * p["exit_price"] * (1.0 - p["cost_out"])

        if i in by_entry:
            for tid, t in by_entry[i].iterrows():
                if len(held) >= n_slots or any(p["ticker"] == t["ticker"] for p in held.values()):
                    continue

                j = columns[t["ticker"]]
                value = cash + sum(p["units"] * (op[i, p["j"]] if not np.isnan(op[i, p["j"]]) else marks[i - 1, p["j"]]) for p in held.values())
                spend = min(value / n_slots, cash)
                if spend <= 0:
                    continue

                cash -= spend
                held[tid] = dict(ticker=t["ticker"], j=j, exit=int(t["exit"]), exit_price=float(t["exit_price"]),
                                 cost_out=float(t["cost_out"]),
                                 units=spend / (1.0 + t["cost_in"]) / op[i, j])
                taken.append(tid)
        values.append(cash + sum(p["units"] * marks[i, p["j"]] for p in held.values()))
    return pd.Series(values, index=market.sessions[first:last + 1], name="value"), taken


def monthly_equal_weight(market, eligible, cost_per_side, start=None, end=None):
    #the benchmark for our experiment, buy all eligible shares at first open of each month and hold

    op, marks = market.open_px.to_numpy(), _marks(market)
    elig = eligible.reindex_like(market.open_px).fillna(False).to_numpy(bool)
    cost = np.broadcast_to(np.asarray(cost_per_side if np.isscalar(cost_per_side) else cost_per_side.reindex_like(market.open_px).to_numpy(), float), op.shape)
    first = 0 if start is None else int(start)
    
    last = len(op) - 1 if end is None else int(end)
    months = market.sessions.to_period("M")
    units = np.zeros(op.shape[1])
    cash, values = 1.0, []
    for i in range(first, last + 1):
        if i == first or months[i] != months[i - 1]:
            price = np.where(np.isnan(op[i]), marks[i - 1] if i > 0 else np.nan, op[i])
            held_value = np.nan_to_num(units * price)
            value = cash + held_value.sum()
            tradable = ~np.isnan(op[i])
            members = elig[i] & tradable

            if members.any():
                #shares with unusable open cannot trade today and are left alone
                frozen = np.nansum(np.where(tradable, 0.0, held_value))
                budget = value - frozen
                target = np.where(members, budget / members.sum(), 0.0)

                fees = np.nansum(np.where(tradable, np.abs(target - held_value), 0.0) * np.nan_to_num(cost[i]))
                target = np.where(members, (budget - fees) / members.sum(), 0.0)
                units = np.where(tradable, target / np.where(tradable, op[i], 1.0), units)
                cash = 0.0

        values.append(cash + np.nansum(units * marks[i]))
    return pd.Series(values, index=market.sessions[first:last + 1], name="value")
