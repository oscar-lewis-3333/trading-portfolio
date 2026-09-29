"""Economic/accounting checks for the consolidated research presentation."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT.parent / "etf_trend/src"),
    str(PROJECT.parent / "equity_momentum/src"),
]

import etf_data
import etf_momentum_backtest as backtest
import etf_momentum_features as features
import etf_momentum_research as research


def test_missing_cache_fails_without_downloading(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline research must not download replacement history.")
    monkeypatch.setattr(etf_data, "load_etf_history", forbidden)
    with pytest.raises(FileNotFoundError, match="saved study file"):
        research._read_cache(tmp_path, "IUSA.L", "repaired", research.study_protocol())


def test_momentum_requires_internal_history_and_is_causal():
    dates = pd.date_range("2020-01-31", periods=7, freq=pd.offsets.MonthEnd())
    prices = pd.DataFrame({"A": [100., 110., 121., 133.1, 140., 150., 160.]}, index=dates)
    scores = features.build_momentum_scores(prices, 3)
    assert scores.iloc[3, 0] == pytest.approx(.331)
    assert scores.iloc[:3].isna().all().all()
    changed = prices.copy()
    changed.iloc[-2:] = 100000.
    pd.testing.assert_frame_equal(features.build_momentum_scores(changed, 3).iloc[:5], scores.iloc[:5])
    missing = prices.copy()
    missing.iloc[1] = np.nan
    assert features.build_momentum_scores(missing, 3).iloc[3:5].isna().all().all()


def test_ties_and_insufficient_eligibility():
    scores = pd.DataFrame({"B": [1., np.nan], "A": [1., np.nan]})
    weights = backtest.build_target_weights(scores, top_n=1)
    assert weights.loc[0, "A"] == 1
    assert weights.loc[0, "B"] == 0
    assert weights.loc[1, "CASH"] == 1


def test_future_returns_cannot_change_annual_selections():
    dates = pd.bdate_range("2017-01-02", "2021-12-31", name="Date")
    noise = np.where(np.arange(len(dates)) % 2, .001, -.001)
    returns = pd.DataFrame({
        0: np.where(dates.year <= 2019, .004, -.03) + noise,
        1: np.where(dates.year <= 2019, .001, .03) + noise,
    }, index=dates)
    grid = pd.DataFrame({
        "lookback_months": [3, 9], "top_n": [1, 1],
    }, index=pd.Index([0, 1], name="configuration_id"))

    data = {"schedule": pd.DataFrame(index=dates)}
    candidates = {"returns": returns, "parameter_grid": grid}
    expected = research.select_annual_choices(data, candidates, research.study_protocol())

    assert expected["configuration_id"].tolist() == [0, 1]
    altered = deepcopy(candidates)
    altered["returns"].loc["2021", 0] = 10.
    altered["returns"].loc["2021", 1] = -.99
    actual = research.select_annual_choices(data, altered, research.study_protocol())
    pd.testing.assert_frame_equal(expected, actual)


def _target_fixture():
    months = pd.date_range("2019-12-31", periods=3, freq=pd.offsets.MonthEnd())
    monthly = pd.DataFrame({"A": [100., 110., 121.], "B": [100., 105., 110.]}, index=months)
    calendar = pd.DataFrame({"execution_date": pd.to_datetime(["2020-02-03", "2020-03-02"])}, index=months[1:])
    choices = pd.DataFrame({
        "selection_date": [months[1]], "test_start": [pd.Timestamp("2020-02-03")],
        "test_end": [pd.Timestamp("2020-03-31")],
        "lookback_months": [1], "top_n": [1]
    })
    data = {
        "monthly_prices": monthly, "calendar": calendar,
        #A is the stronger asset, but isnt available to common universe
        "common_eligibility": pd.DataFrame({"A": False, "B": True}, index=months),
    }
    return data, choices


def test_walk_forward_uses_common_eligibility_and_next_open():
    data, choices = _target_fixture()
    targets = research.build_walk_forward_targets(data, choices)
    assert targets["B"].eq(1).all()
    assert targets["A"].eq(0).all()
    assert targets.index[0] == pd.Timestamp("2020-02-03")


def test_walk_forward_rejects_same_day_configuration_selection():
    data, choices = _target_fixture()
    choices.loc[0, "selection_date"] = choices.loc[0, "test_start"]
    with pytest.raises(ValueError, match="precede"):
        research.build_walk_forward_targets(data, choices)


def test_year_change_preserves_capital_and_charges_both_sides_of_rotation():
    dates = pd.to_datetime(["2020-01-02", "2020-12-31", "2021-01-04"])
    prices = pd.DataFrame(100., index=dates, columns=["A", "B"])
    targets = pd.DataFrame(
        {"A": [1., 0.], "B": [0., 1.], "CASH": [0., 0.]},
        index=dates[[0, 2]]
    )
    result = backtest.run_backtest(prices, prices, targets, initial_capital_gbp=10000., cost_per_side_bps=100.)
    entered = 10000 / 1.01
    assert result["daily"]["nav_close"].iloc[1] == pytest.approx(entered)
    assert result["daily"]["nav_close"].iloc[2] == pytest.approx(entered * .99 / 1.01)
    assert result["daily"]["cost"].iloc[1] == 0


def test_attribution_keeps_overnight_loss_of_a_sold_holding():
    dates = pd.bdate_range("2024-02-01", periods=3)
    opens = pd.DataFrame({"A": [100., 80., 90.], "B": [10., 12., 9.]}, index=dates)
    closes = pd.DataFrame({"A": [102., 88., 92.], "B": [11., 13., 10.]}, index=dates)
    targets = pd.DataFrame({"A": [1., 0.], "B": [0., 1.], "CASH": [0., 0.]}, index=dates[:2])

    settings = research.study_protocol()["backtest"]
    result = backtest.run_backtest(opens, closes, targets, **settings)
    attribution, net = research.attribute_profit(result, settings)

    units = 10000 / (100 * 1.001)
    overnight_pnl = units * (80 - 102)
    sale_fee = units * 80 * .001
    assert net.loc[dates[1], "A"] == pytest.approx(overnight_pnl - sale_fee)
    assert attribution["net_pnl_gbp"].sum() == pytest.approx(result["daily"]["nav_close"].iloc[-1] - 10000)


def test_valuation_mark_is_previous_close_and_other_quotes_are_unchanged():
    dates = pd.bdate_range("2025-10-20", periods=5)
    quotes = pd.DataFrame({"A": [10., 11., np.nan, 14., 15.]}, index=dates)
    flags = quotes.isna()
    marked_open, marked_close, _ = etf_data.prepare_valuation_marks(quotes, quotes, flags, execution_dates=dates[:1])
    assert marked_close.iloc[2, 0] == 11.
    pd.testing.assert_frame_equal(marked_open.mask(flags), quotes)
    pd.testing.assert_frame_equal(marked_close.mask(flags), quotes)


@pytest.mark.parametrize("case", ["execution", "terminal", "unreviewed", "consecutive"])
def test_unsafe_valuation_substitution_fails(case):
    dates = pd.bdate_range("2025-10-20", periods=5)
    quotes = pd.DataFrame({"A": [10., 11., np.nan, 14., 15.]}, index=dates)
    executions = dates[:1]
    if case == "terminal":
        quotes.iloc[-1] = np.nan
    if case == "consecutive":
        quotes.iloc[3] = np.nan
    flags = quotes.isna()
    if case == "unreviewed":
        flags.iloc[2] = False
    if case == "execution":
        executions = dates[[0, 2]]
    with pytest.raises(ValueError):
        etf_data.prepare_valuation_marks(quotes, quotes, flags, execution_dates=executions)
