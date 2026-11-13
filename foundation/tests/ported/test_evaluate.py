import numpy as np
import pandas as pd
import pytest

from foundation._research.contracts import CASH, all_cash
from foundation._research.costs import per_side_rate, round_trip_bps
from foundation._research.evaluate import active_returns, estimate_beta, run_portfolio
from foundation._research.metrics import block_bootstrap_difference, max_drawdown, sharpe, total_return


def _frame(dates, values):
    return pd.DataFrame(values, index=dates)


def test_all_cash_earns_only_the_cash_rate():
    dates = pd.date_range("2022-01-03", periods=30, freq="D", tz="UTC")
    returns = _frame(dates, {"BTC-GBP": np.full(30, 0.05)})
    targets = all_cash(dates[:1], ["BTC-GBP"])

    ledger = run_portfolio(targets, returns, cost_mode="pessimistic", cash_rate=0.0365)
    assert ledger["one_way_turnover"].sum() == pytest.approx(0.0)
    #30 days of 3.65% annualized interest is (1.0365 ** (30 / 365)) - 1.0

    assert total_return(ledger["net_return"]) == pytest.approx((1.0365 ** (30 / 365)) - 1.0, rel=1e-9)


def test_fully_invested_matches_the_asset_less_entry_cost():
    dates = pd.date_range("2022-01-03", periods=10, freq="D", tz="UTC")
    returns = _frame(dates, {"BTC-GBP": np.full(10, 0.01)})
    targets = pd.DataFrame({"BTC-GBP": [1.0], CASH: [0.0]}, index=dates[:1])

    ledger = run_portfolio(targets, returns, cost_mode="pessimistic", cash_rate=0.0)
    rate = per_side_rate("pessimistic")
    expected = (1.0 - rate) * (1.01 ** 10) - 1.0
    assert total_return(ledger["net_return"]) == pytest.approx(expected, rel=1e-12)
    assert ledger["one_way_turnover"].sum() == pytest.approx(1.0)


def test_costs_scale_with_turnover_and_mode():
    dates = pd.date_range("2022-01-03", periods=6, freq="D", tz="UTC")
    returns = _frame(dates, {"A-GBP": np.zeros(6), "B-GBP": np.zeros(6)})
    schedule = dates[[0, 2, 4]]
    targets = pd.DataFrame({
        "A-GBP": [1.0, 0.0, 1.0],
        "B-GBP": [0.0, 1.0, 0.0],
        CASH: [0.0, 0.0, 0.0],
    }, index=schedule)

    pessimistic = run_portfolio(targets, returns, cost_mode="pessimistic", cash_rate=0.0)
    optimistic = run_portfolio(targets, returns, cost_mode="optimistic", cash_rate=0.0)

    #one entry plus two full switches: 1.0 + 2.0 + 2.0 of one-way notional

    assert pessimistic["one_way_turnover"].sum() == pytest.approx(5.0)
    assert pessimistic["cost_fraction"].sum() > optimistic["cost_fraction"].sum()
    assert round_trip_bps("pessimistic") == 25.0
    assert round_trip_bps("optimistic") == 7.0


def test_missing_return_for_a_held_asset_is_an_error():
    dates = pd.date_range("2022-01-03", periods=5, freq="D", tz="UTC")
    values = np.zeros(5)
    values[3] = np.nan
    returns = _frame(dates, {"A-GBP": values})
    targets = pd.DataFrame({"A-GBP": [1.0], CASH: [0.0]}, index=dates[:1])

    with pytest.raises(ValueError, match="Missing return"):
        run_portfolio(targets, returns, cash_rate=0.0)


def test_beta_is_estimated_on_training_data_only():
    dates = pd.date_range("2022-01-03", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    benchmark = pd.Series(rng.normal(0, 0.02, 400), index=dates)
    signal = 0.7 * benchmark + pd.Series(rng.normal(0, 0.001, 400), index=dates)

    train_end = dates[300]
    beta = estimate_beta(signal, benchmark, train_end)
    assert beta == pytest.approx(0.7, abs=0.05)

    active = active_returns(signal, benchmark, beta)
    assert abs(active.corr(benchmark)) < 0.2


def test_beta_needs_enough_training_observations():
    dates = pd.date_range("2022-01-03", periods=40, freq="D", tz="UTC")
    series = pd.Series(np.linspace(-0.01, 0.01, 40), index=dates)
    with pytest.raises(ValueError):
        estimate_beta(series, series, dates[-1])


def test_max_drawdown_and_sharpe_are_signed_as_expected():
    dates = pd.date_range("2022-01-03", periods=4, freq="D", tz="UTC")
    series = pd.Series([0.10, -0.20, 0.05, 0.02], index=dates)
    assert max_drawdown(series) < 0
    assert np.isfinite(sharpe(series))


def test_bootstrap_returns_an_interval_containing_the_estimate():
    dates = pd.date_range("2022-01-03", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    strategy = pd.Series(rng.normal(0.0004, 0.02, 400), index=dates)
    benchmark = pd.Series(rng.normal(0.0001, 0.02, 400), index=dates)

    result = block_bootstrap_difference(strategy, benchmark, block_days=28, resamples=500, seed=7)
    assert result["ci_lower"] < result["ci_upper"]
    assert result["block_days"] == 28


def _executions(decisions, days, fills):
    frame = pd.DataFrame(fills, index=decisions)
    frame.insert(0, "execution_day", pd.DatetimeIndex(days))
    return frame


def test_same_day_zero_fill_execution_matches_the_default():
    dates = pd.date_range("2022-01-03", periods=14, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    returns = _frame(dates, {"A-GBP": rng.normal(0, 0.02, 14), "B-GBP": rng.normal(0, 0.02, 14)})
    decisions = dates[[0, 7]]
    targets = pd.DataFrame({"A-GBP": [0.5, 0.2], "B-GBP": [0.5, 0.3], CASH: [0.0, 0.5]}, index=decisions)
    executions = _executions(decisions, decisions, {"A-GBP": [0.0, 0.0], "B-GBP": [0.0, 0.0]})

    default = run_portfolio(targets, returns, cash_rate=0.02)
    executed = run_portfolio(targets, returns, cash_rate=0.02, executions=executions)
    pd.testing.assert_frame_equal(default, executed, check_exact=False, rtol=1e-12, atol=1e-14)


def test_next_day_execution_splits_the_day_at_the_fill():
    dates = pd.date_range("2022-01-03", periods=4, freq="D", tz="UTC")
    returns = _frame(dates, {"A-GBP": [0.0, 0.10, 0.0, 0.0]})
    targets = pd.DataFrame({"A-GBP": [1.0], CASH: [0.0]}, index=dates[:1])

    #decision on day 0, fill on day 1 after 4% move from close.
    executions = _executions(dates[:1], dates[1:2], {"A-GBP": [0.04]})

    ledger = run_portfolio(targets, returns, cost_mode="pessimistic", cash_rate=0.0, executions=executions)
    rate = per_side_rate("pessimistic")
    assert ledger["net_return"].iloc[0] == pytest.approx(0.0)
    assert ledger["net_return"].iloc[1] == pytest.approx((1.0 - rate) * (1.10 / 1.04) - 1.0, rel=1e-12)
    assert ledger["one_way_turnover"].iloc[1] == pytest.approx(1.0)


def test_selling_at_the_fill_keeps_the_move_before_it():
    dates = pd.date_range("2022-01-03", periods=3, freq="D", tz="UTC")
    returns = _frame(dates, {"A-GBP": [0.0, -0.20, 0.0]})
    decisions = dates[[0, 1]]
    targets = pd.DataFrame({"A-GBP": [1.0, 0.0], CASH: [0.0, 1.0]}, index=decisions)
    #bought on day 0, sold on day 1 after 1% fall. later 19% fall avoided.
    executions = _executions(decisions, decisions, {"A-GBP": [0.0, -0.01]})

    ledger = run_portfolio(targets, returns, cost_mode="optimistic", cash_rate=0.0, executions=executions)
    rate = per_side_rate("optimistic")
    assert ledger["net_return"].iloc[1] == pytest.approx(0.99 * (1.0 - rate * 1.0) - 1.0, rel=1e-12)


def test_missing_fill_for_a_traded_asset_is_an_error():
    dates = pd.date_range("2022-01-03", periods=3, freq="D", tz="UTC")
    returns = _frame(dates, {"A-GBP": np.zeros(3)})
    targets = pd.DataFrame({"A-GBP": [1.0], CASH: [0.0]}, index=dates[:1])
    executions = _executions(dates[:1], dates[:1], {"A-GBP": [np.nan]})
    with pytest.raises(ValueError, match="Missing fill"):
        run_portfolio(targets, returns, executions=executions)


def test_overlapping_executions_are_rejected():
    dates = pd.date_range("2022-01-03", periods=5, freq="D", tz="UTC")
    returns = _frame(dates, {"A-GBP": np.zeros(5)})
    decisions = dates[[0, 1]]
    targets = pd.DataFrame({"A-GBP": [1.0, 0.0], CASH: [0.0, 1.0]}, index=decisions)
    executions = _executions(decisions, dates[[2, 3]], {"A-GBP": [0.0, 0.0]})
    with pytest.raises(ValueError, match="before the next decision"):
        run_portfolio(targets, returns, executions=executions)


def test_per_pair_costs_charge_each_trade_at_its_own_rate(tmp_path):
    import pandas as pd
    import pytest
    from foundation._research.contracts import CASH
    from foundation._research.costs import load_cost_table
    from foundation._research.evaluate import run_portfolio

    days = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC", name="decision_at")
    returns = pd.DataFrame({"A-GBP": [0.0, 0.0, 0.0], "B-GBP": [0.0, 0.0, 0.0]}, index=days)
    targets = pd.DataFrame({"A-GBP": [0.5], "B-GBP": [0.5], CASH: [0.0]}, index=days[:1])
    path = tmp_path / "costs_test_20260923.csv"
    pd.DataFrame({"product_id": ["A-GBP", "B-GBP"], "per_side_bps": [10.0, 50.0]}).to_csv(path, index=False)
    table = load_cost_table(path)

    ledger = run_portfolio(targets, returns, cash_rate=0.0, cost_table=table)
    assert ledger["cost_fraction"].iloc[0] == pytest.approx(0.5 * 0.0010 + 0.5 * 0.0050)
    assert table.attrs["source"] == "costs_test_20260923.csv"


def test_a_pair_without_a_measured_cost_is_an_error(tmp_path):
    import pandas as pd
    import pytest
    from foundation._research.contracts import CASH
    from foundation._research.costs import load_cost_table
    from foundation._research.evaluate import run_portfolio

    days = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC", name="decision_at")
    returns = pd.DataFrame({"A-GBP": [0.0, 0.0], "C-GBP": [0.0, 0.0]}, index=days)
    targets = pd.DataFrame({"A-GBP": [0.5], "C-GBP": [0.5], CASH: [0.0]}, index=days[:1])
    path = tmp_path / "costs.csv"
    pd.DataFrame({"product_id": ["A-GBP"], "per_side_bps": [10.0]}).to_csv(path, index=False)
    
    with pytest.raises(ValueError, match="No measured cost"):
        run_portfolio(targets, returns, cost_table=load_cost_table(path))
