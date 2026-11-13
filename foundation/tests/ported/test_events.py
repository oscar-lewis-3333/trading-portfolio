import numpy as np
import pandas as pd
import pytest

from foundation._research.events import build_market, day_zero, event_reactions, hold_events, month_cluster_bootstrap, monthly_equal_weight, net_return, slot_portfolio, trailing_liquidity


def synthetic(n=120, tickers=("AAA", "BBB", "CCC"), seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    schedule = pd.DataFrame({"market_open": dates.tz_localize("UTC") + pd.Timedelta(hours=8),
                             "market_close": dates.tz_localize("UTC") + pd.Timedelta(hours=16, minutes=30)}, index=dates)
    rows = []
    for t in tickers:
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        opens = close * (1 + rng.normal(0, 0.005, n))
        for d, o, c in zip(dates, opens, close):
            rows.append(dict(Date=d, ticker=t, adj_open=o, adj_close=c, traded_value_gbp=100_000.0,
                             open_reference_usable=True, close_reference_usable=True,
                             known_suspended=False, close_suspended=False, unit_quarantine=False))
    prices = pd.DataFrame(rows).set_index(["Date", "ticker"])
    return prices, schedule


def test_day_zero_before_and_after_close():
    prices, schedule = synthetic()
    m = build_market(prices, schedule)
    d = schedule.index[70]

    before_open = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=7)
    after_close = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=17)
    exactly_close = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=16, minutes=30)
    assert list(day_zero([before_open, after_close, exactly_close], m)) == [70, 71, 71]


def test_reaction_ignores_everything_after_day_zero_close():
    prices, schedule = synthetic()
    events = pd.DataFrame({"ticker": ["AAA"], "time": [pd.Timestamp(schedule.index[80], tz="UTC") + pd.Timedelta(hours=7)]})
    first = event_reactions(events, build_market(prices, schedule))
    changed = prices.copy()

    later = changed.index.get_level_values("Date") > schedule.index[80]
    changed.loc[later, ["adj_open", "adj_close"]] *= 3.0
    second = event_reactions(events, build_market(changed, schedule))

    assert first.loc[0, "reason"] == "ok"
    assert first.loc[0, "z"] == pytest.approx(second.loc[0, "z"])
    assert first.loc[0, "liquidity"] == pytest.approx(second.loc[0, "liquidity"])


def test_reaction_is_abnormal_to_the_universe():
    prices, schedule = synthetic()
    m = build_market(prices, schedule)
    events = pd.DataFrame({"ticker": ["BBB"], "time": [pd.Timestamp(schedule.index[90], tz="UTC")]})
    r = event_reactions(events, m).iloc[0]
    returns = m.close_px.pct_change(fill_method=None).iloc[90]
    assert r["reaction"] == pytest.approx(returns["BBB"] - returns.mean())


def test_liquidity_uses_only_prior_sessions():
    prices, schedule = synthetic()
    prices.loc[(schedule.index[100], "AAA"), "traded_value_gbp"] = 1e12
    liq = trailing_liquidity(build_market(prices, schedule))
    assert liq.iloc[100]["AAA"] == pytest.approx(100_000.0)


def test_hold_returns_open_to_open_and_benchmark():
    prices, schedule = synthetic()
    m = build_market(prices, schedule)
    events = pd.DataFrame({"ticker": ["CCC"], "time": [pd.Timestamp(schedule.index[70], tz="UTC")]})
    held = hold_events(event_reactions(events, m), m, holding=10).iloc[0]

    op = m.open_px
    assert (held["entry"], held["exit"]) == (71, 81)
    assert held["gross"] == pytest.approx(op["CCC"].iloc[81] / op["CCC"].iloc[71] - 1)
    assert held["benchmark"] == pytest.approx((op.iloc[81] / op.iloc[71] - 1).mean())


def test_missing_exit_open_waits_then_uses_last_close():
    prices, schedule = synthetic()
    for i in range(81, 95):
        prices.loc[(schedule.index[i], "CCC"), "open_reference_usable"] = False
    m = build_market(prices, schedule)
    events = pd.DataFrame({"ticker": ["CCC"], "time": [pd.Timestamp(schedule.index[70], tz="UTC")]})
    held = hold_events(event_reactions(events, m), m, holding=10, max_wait=5).iloc[0]

    assert held["exit_kind"] == "last_close" and held["exit"] == 86
    assert held["gross"] == pytest.approx(m.close_px["CCC"].iloc[86] / m.open_px["CCC"].iloc[71] - 1)


def test_unusable_entry_open_skips_the_event():
    prices, schedule = synthetic()
    prices.loc[(schedule.index[71], "CCC"), "open_reference_usable"] = False
    m = build_market(prices, schedule)
    
    events = pd.DataFrame({"ticker": ["CCC"], "time": [pd.Timestamp(schedule.index[70], tz="UTC")]})
    held = hold_events(event_reactions(events, m), m, holding=10).iloc[0]
    assert held["exit_kind"] == "entry_open_unusable" and np.isnan(held["gross"])


def test_net_return_charges_both_sides():
    assert net_return(0.0, 0.01, 0.01) == pytest.approx(0.99 / 1.01 - 1)
    assert net_return(0.10, 0.0, 0.0) == pytest.approx(0.10)


def test_slot_portfolio_single_trade_matches_hand_calculation():
    prices, schedule = synthetic()
    m = build_market(prices, schedule)
    op = m.open_px["AAA"]
    trades = pd.DataFrame([dict(ticker="AAA", entry=71, exit=81, exit_price=op.iloc[81], cost_in=0.02,
                                cost_out=0.01, time=schedule.index[70])], index=["t1"])
    values, taken = slot_portfolio(trades, m, n_slots=4, start=60, end=90)
    stake = 0.25
    expected = 0.75 + stake / 1.02 / op.iloc[71] * op.iloc[81] * 0.99
    assert taken == ["t1"]
    assert values.iloc[-1] == pytest.approx(expected)
    assert values.loc[schedule.index[65]] == pytest.approx(1.0)


def test_slot_portfolio_respects_slot_limit_and_duplicates():
    prices, schedule = synthetic()
    m = build_market(prices, schedule)
    op = m.open_px
    rows = [dict(ticker=t, entry=71, exit=81, exit_price=op[t].iloc[81], cost_in=0, cost_out=0,
                 time=schedule.index[70] + pd.Timedelta(minutes=k)) for k, t in enumerate(["AAA", "BBB", "CCC"])]
    rows.append(dict(ticker="AAA", entry=75, exit=85, exit_price=op["AAA"].iloc[85], cost_in=0, cost_out=0,
                     time=schedule.index[74]))
    trades = pd.DataFrame(rows, index=["a", "b", "c", "a_again"])
    _, taken = slot_portfolio(trades, m, n_slots=2)
    assert taken == ["a", "b"]


def test_monthly_equal_weight_without_costs_tracks_the_average():
    prices, schedule = synthetic(n=40)
    m = build_market(prices, schedule)
    eligible = pd.DataFrame(True, index=m.open_px.index, columns=m.open_px.columns)
    values = monthly_equal_weight(m, eligible, 0.0, start=0, end=15)
    op, cl = m.open_px.iloc[0], m.close_px.iloc[15]
    assert values.iloc[-1] == pytest.approx((cl / op).mean())


def test_month_bootstrap_keeps_months_together():
    out = month_cluster_bootstrap([1.0, 1.0, -1.0, -1.0], ["a", "a", "b", "b"], resamples=2000)
    assert out["mean"] == 0.0 and out["months"] == 2 and out["lower"] == pytest.approx(-1.0)
