
#ensure transaction fees apply correctly

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

from test_uk_data_preparation import CACHE, NOTEBOOK, defines, run_cells


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import reversal_backtest as backtest


def fixture(n=9, tickers=("A.L", "B.L")):
    # An irregular calendar exposes accidental calendar-day arithmetic.
    dates = pd.bdate_range("2020-01-02", periods=n + 1).delete(3)
    dates.name = "Date"
    schedule = pd.DataFrame({
        "market_open": dates.tz_localize("UTC") + pd.Timedelta(hours=8),
        "market_close": dates.tz_localize("UTC") + pd.Timedelta(hours=16),
    }, index=dates)
    index = pd.MultiIndex.from_product([dates, tickers], names=["Date", "ticker"])
    market = pd.DataFrame({
        "open_gbp": 10., "close_gbp": 10., "dividend_gbp": 0.,
        "known_suspended": False, "open_reference_usable": True,
        "positive_reported_volume": True,
    }, index=index)
    return schedule, market


def decisions(schedule, choices):
    rows = []
    for position, allocations in choices:
        date = schedule.index[position]
        for ticker, weight in allocations.items():
            rows.append({"Date": date, "ticker": ticker, "target_weight": weight,
                         "signal_time": schedule.at[date, "market_close"]})
    return pd.DataFrame(rows).set_index(["Date", "ticker"]).sort_index()


def simulate(schedule, market, choices, horizon=1, cost=0.):
    return backtest.run_execution_backtest(
        decisions(schedule, choices), market, schedule,
        initial_capital_gbp=1000., holding_sessions=horizon,
        cost_per_side_bps=cost,
    )


class ExecutionArithmeticTests(unittest.TestCase):
    def test_round_trip_costs_match_closed_form_and_do_not_borrow(self):
        schedule, market = fixture()
        market.loc[(schedule.index[2], "A.L"), "open_gbp"] = 12.
        result = simulate(schedule, market, [(0, {"A.L": 1.})], cost=100.)
        expected = 1000. * (12. / 10.) * (1. - .01) / (1. + .01)
        self.assertAlmostEqual(result["daily"]["equity_gbp"].iloc[-1], expected)
        orders = result["orders"]
        np.testing.assert_allclose(orders["cost_gbp"], [.01 * 1000 / 1.01,
                                                     .01 * 1200 / 1.01])
        self.assertTrue(result["daily"]["cash_gbp"].ge(0).all())
        self.assertEqual(list(orders["Date"]), [schedule.index[1], schedule.index[2]])

    def test_budgets_use_signal_close_despite_next_open_gain(self):
        schedule, market = fixture()
        market.loc[(schedule.index[2], "A.L"), "open_gbp"] = 20.
        result = simulate(schedule, market, [(0, {"A.L": 1.}), (1, {"B.L": 1.})])
        b_buy = result["orders"].loc[lambda x: x["ticker"].eq("B.L") & x["side"].eq("buy")].iloc[0]
        self.assertEqual(b_buy["notional_gbp"], 1000.)
        self.assertEqual(result["daily"].at[schedule.index[2], "cash_gbp"], 1000.)
        self.assertEqual(b_buy["Date"], schedule.index[2])

    def test_dividend_entry_exit_entitlement_and_reserve_not_reinvested(self):
        schedule, market = fixture()
        for position, dividend, price in [(1, 7., 10.), (2, 1., 9.), (3, 1., 8.)]:
            market.loc[(schedule.index[position], "A.L"),
                       ["dividend_gbp", "open_gbp", "close_gbp"]] = [dividend, price, price]
        result = simulate(schedule, market, [(0, {"A.L": 1.}), (3, {"B.L": 1.})], horizon=2)
        daily = result["daily"]
        self.assertEqual(daily.at[schedule.index[1], "dividend_reserve_gbp"], 0.)
        self.assertEqual(daily.at[schedule.index[2], "dividend_reserve_gbp"], 100.)
        self.assertEqual(daily.at[schedule.index[3], "dividend_reserve_gbp"], 200.)
        self.assertEqual(daily.at[schedule.index[3], "cash_gbp"], 800.)
        self.assertEqual(daily["equity_gbp"].iloc[-1], 1000.)
        b_buy = result["orders"].loc[lambda x: x["ticker"].eq("B.L") & x["side"].eq("buy")].iloc[0]
        self.assertEqual(b_buy["notional_gbp"], 800.)

    def test_suspended_exit_keeps_capital_locked_and_realises_reopening_loss(self):
        schedule, market = fixture()
        for date in schedule.index[2:4]:
            # Even positive volume and an incorrectly permissive reference flag
            # must not override a confirmed suspension.
            market.loc[(date, "A.L"), ["known_suspended", "open_gbp", "close_gbp"]] = [True, 99., 99.]
        market.loc[(schedule.index[4], "A.L"), ["open_gbp", "close_gbp"]] = 6.
        result = simulate(schedule, market, [(0, {"A.L": 1.}), (1, {"B.L": 1.})])
        daily, orders = result["daily"], result["orders"]
        self.assertEqual(daily.at[schedule.index[2], "cash_gbp"], 0.)
        self.assertEqual(daily.at[schedule.index[2], "stale_value_gbp"], 1000.)
        self.assertEqual(int(orders["status"].eq("blocked").sum()), 2)
        b = orders.loc[orders["ticker"].eq("B.L")].iloc[0]
        self.assertEqual((b["status"], b["reason"]), ("cancelled", "no_cash"))
        sale = orders.loc[orders["side"].eq("sell") & orders["status"].eq("assumed_fill")].iloc[0]
        self.assertEqual(sale["Date"], schedule.index[4])
        self.assertEqual(daily["equity_gbp"].iloc[-1], 600.)

    def test_dividend_during_suspension_does_not_double_count_stale_value(self):
        schedule, market = fixture()
        market.loc[(schedule.index[2], "A.L"), ["known_suspended", "dividend_gbp"]] = [True, 1.]
        market.loc[(schedule.index[3], "A.L"), ["open_gbp", "close_gbp"]] = 9.
        result = simulate(schedule, market, [(0, {"A.L": 1.})])
        suspended = result["daily"].loc[schedule.index[2]]
        self.assertEqual(suspended["stale_value_gbp"], 900.)
        self.assertEqual(suspended["dividend_reserve_gbp"], 100.)
        self.assertEqual(suspended["equity_gbp"], 1000.)
        self.assertEqual(result["daily"]["equity_gbp"].iloc[-1], 1000.)

    def test_cancelled_buy_keeps_its_slot_cash_without_later_retry(self):
        schedule, market = fixture()
        market.loc[(schedule.index[1], "A.L"), "known_suspended"] = True
        result = simulate(schedule, market, [(0, {"A.L": .5, "B.L": .5})], horizon=2)
        self.assertEqual(result["daily"].at[schedule.index[1], "cash_gbp"], 500.)
        a_orders = result["orders"].loc[lambda x: x["ticker"].eq("A.L")]
        self.assertEqual(len(a_orders), 1)
        self.assertEqual(a_orders.iloc[0]["status"], "cancelled")
        self.assertEqual(result["orders"].loc[lambda x: x["ticker"].eq("B.L") & x["side"].eq("buy"), "notional_gbp"].iloc[0], 500.)

    def test_unknown_open_blocks_exit_until_observed(self):
        schedule, market = fixture()
        market.loc[(schedule.index[2], "A.L"), "open_gbp"] = np.nan
        result = simulate(schedule, market, [(0, {"A.L": 1.})])
        sells = result["orders"].loc[lambda x: x["side"].eq("sell")]
        self.assertEqual(list(sells["status"]), ["blocked", "assumed_fill"])
        self.assertEqual(sells.iloc[1]["Date"], schedule.index[3])

    def test_unresolved_terminal_position_is_not_liquidated_or_dropped(self):
        schedule, market = fixture()
        market.loc[(slice(schedule.index[2], None), "A.L"), "known_suspended"] = True
        result = simulate(schedule, market, [(0, {"A.L": 1.})])
        self.assertEqual(result["final_positions"]["A.L"]["units"], 100.)
        self.assertEqual(result["daily"]["cash_gbp"].iloc[-1], 0.)
        self.assertEqual(result["daily"]["stale_value_gbp"].iloc[-1], 1000.)
        self.assertEqual(result["daily"]["pending_exits"].iloc[-1], 1)

    def test_all_cash_run_has_no_orders_or_holdings(self):
        schedule, market = fixture()
        result = simulate(schedule, market, [(0, {"A.L": 0.})])
        self.assertTrue(result["daily"]["equity_gbp"].eq(1000.).all())
        self.assertTrue(result["orders"].empty)
        self.assertTrue(result["holdings"].empty)

    def test_zero_volume_fill_is_explicitly_flagged_in_price_proxy_scenario(self):
        schedule, market = fixture()
        market["positive_reported_volume"] = False
        result = simulate(schedule, market, [(0, {"A.L": 1.})])
        self.assertTrue(result["orders"]["status"].eq("assumed_fill").all())
        self.assertTrue(result["orders"]["volume_warning"].all())

    def test_future_price_dividend_and_signal_changes_leave_past_unchanged(self):
        schedule, market = fixture()
        choices = [(0, {"A.L": 1.}), (4, {"A.L": 1., "B.L": 0.})]
        baseline = simulate(schedule, market, choices, horizon=2)
        changed = market.copy()
        future = changed.index.get_level_values("Date") > schedule.index[3]
        changed.loc[future, ["open_gbp", "close_gbp"]] *= 2.
        changed.loc[future, "dividend_gbp"] = .5
        alternate = simulate(schedule, changed, [(0, {"A.L": 1.}), (4, {"A.L": 0., "B.L": 1.})], horizon=2)
        cutoff = schedule.index[3]
        pd.testing.assert_frame_equal(baseline["daily"].loc[:cutoff], alternate["daily"].loc[:cutoff])
        pd.testing.assert_frame_equal(baseline["orders"].loc[lambda x: x["Date"] <= cutoff],
                                      alternate["orders"].loc[lambda x: x["Date"] <= cutoff])

    def test_missing_held_dividend_or_incomplete_calendar_fails_explicitly(self):
        schedule, market = fixture()
        market.loc[(schedule.index[2], "A.L"), "dividend_gbp"] = np.nan
        with self.assertRaisesRegex(ValueError, "Unknown or invalid dividend"):
            simulate(schedule, market, [(0, {"A.L": 1.})])
        with self.assertRaisesRegex(ValueError, "every ticker-session"):
            simulate(schedule, market.iloc[1:], [(0, {"A.L": 1.})])

    def test_helper_preserves_caller_state_and_does_not_duplicate_blocked_holding(self):
        schedule, market = fixture()
        session = schedule.index[2]
        day = market.xs(session, level="Date").copy()
        day.loc["A.L", "known_suspended"] = True
        holdings = {"A.L": {"units": 100., "exit_session": session}}
        original = copy.deepcopy(holdings)
        original_day = day.copy(deep=True)
        result = backtest.execute_open_orders(0., holdings, day, {"A.L": 1000.},
                                             session, schedule.index[4])
        self.assertEqual(holdings, original)
        pd.testing.assert_frame_equal(day, original_day)
        self.assertEqual(result["positions"], original)
        self.assertEqual(list(result["orders"]["reason"]), ["suspended", "already_held"])


class UKExecutionNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not NOTEBOOK.exists() or not CACHE.exists():
            raise unittest.SkipTest("The UK notebook and frozen local cache are required")
        cells = []
        for number, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            cells.append((number, source))
            if defines(source, "uk_gross_backtest"):
                break
        else:
            raise AssertionError("No saved UK execution-backtest cell found")
        protected = [NOTEBOOK, CACHE, PROJECT / "src" / "reversal_backtest.py"]
        cls.hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        cls.ns = {"__name__": "__main__", "display": lambda *a, **k: None}
        run_cells(cells, cls.ns)
        cls.result = cls.ns["uk_gross_backtest"]
        cls.daily = cls.result["daily"]
        cls.orders = cls.result["orders"]
        cls.holdings = cls.result["holdings"]

    def test_fresh_replay_preserves_files_dates_and_all_cash_flows(self):
        for path, digest in self.hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        schedule = self.ns["uk_schedule"]
        expected_dates = schedule.loc[self.ns["uk_decision_dates"][0]:].index
        pd.testing.assert_index_equal(self.daily.index, expected_dates)
        changes = self.orders.groupby("Date")["cash_change_gbp"].sum().reindex(expected_dates, fill_value=0.)
        cash = self.ns["UK_BASELINE"]["initial_capital_gbp"] + changes.cumsum()
        np.testing.assert_allclose(self.daily["cash_gbp"], cash, rtol=1e-10, atol=1e-6)
        self.assertTrue(self.daily["cash_gbp"].ge(0).all())
        self.assertTrue((expected_dates < pd.Timestamp(self.ns["UK_PERIODS"]["holdout_start"])).all())

    def test_inventory_and_wealth_reconcile_independently(self):
        fills = self.orders.loc[self.orders["status"].eq("assumed_fill")].copy()
        fills["signed_units"] = fills["units"] * fills["side"].map({"buy": 1., "sell": -1.})
        movements = fills.groupby(["Date", "ticker"])["signed_units"].sum().unstack("ticker", fill_value=0.)
        inventory = movements.reindex(self.daily.index, fill_value=0.).cumsum()
        actual = self.holdings["units"].unstack("ticker").reindex(index=self.daily.index, columns=inventory.columns).fillna(0.)
        np.testing.assert_allclose(actual, inventory, rtol=1e-10, atol=1e-6)
        values = self.holdings["value_gbp"].groupby(level="Date").sum().reindex(self.daily.index, fill_value=0.)
        np.testing.assert_allclose(values, self.daily["stock_value_gbp"], rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(self.daily["equity_gbp"],
                                   self.daily["cash_gbp"] + values + self.daily["dividend_reserve_gbp"], rtol=1e-12)
        self.assertEqual(self.result["final_positions"], {})

    def test_dividend_reserve_matches_previous_session_inventory(self):
        quantities = self.holdings["units"].unstack("ticker").reindex(self.daily.index).fillna(0.)
        overnight = quantities.shift(1, fill_value=0.)
        dividends = self.ns["uk_execution_market"]["dividend_gbp"].unstack("ticker").reindex(index=overnight.index, columns=overnight.columns)
        entitled = (overnight * dividends).where(overnight.gt(0), 0.)
        self.assertTrue(np.isfinite(entitled.to_numpy()).all())
        expected = entitled.sum(axis=1, skipna=False)
        np.testing.assert_allclose(expected, self.daily["dividends_accrued_gbp"], atol=1e-8)
        np.testing.assert_allclose(expected.cumsum(), self.daily["dividend_reserve_gbp"], atol=1e-8)

    def test_fills_respect_prior_decisions_budgets_and_suspensions(self):
        ns = self.ns
        fills = self.orders.loc[self.orders["status"].eq("assumed_fill")]
        keys = pd.MultiIndex.from_frame(fills[["Date", "ticker"]])
        references = ns["uk_execution_market"].reindex(keys)
        self.assertFalse(references["known_suspended"].any())
        np.testing.assert_allclose(fills["price_gbp"], references["open_gbp"], rtol=0, atol=0)
        prior_date = dict(zip(ns["uk_schedule"].index[1:], ns["uk_schedule"].index[:-1]))
        for buy in fills.loc[fills["side"].eq("buy")].itertuples(index=False):
            signal = prior_date[buy.Date]
            decision = ns["uk_decisions"].loc[(signal, buy.ticker)]
            self.assertTrue(decision["selected"])
            equity = self.daily.at[signal, "cash_gbp"] + self.daily.at[signal, "stock_value_gbp"]
            budget = equity * decision["target_weight"]
            self.assertLessEqual(buy.notional_gbp + buy.cost_gbp, budget + 1e-6)

    def test_audited_suspensions_delay_exits_to_actual_restoration_dates(self):
        for ticker, signal, planned_exit, restoration in [
            ("PREM.L", "2017-06-23", "2017-07-03", "2017-07-12"),
            ("EAH.L", "2020-12-29", "2021-01-07", "2021-02-04"),
        ]:
            with self.subTest(ticker=ticker):
                self.assertTrue(self.ns["uk_decisions"].loc[(pd.Timestamp(signal), ticker), "selected"])
                orders = self.orders.loc[self.orders["ticker"].eq(ticker)
                                         & self.orders["side"].eq("sell")
                                         & self.orders["Date"].ge(planned_exit)]
                first_fill = orders.loc[orders["status"].eq("assumed_fill")].iloc[0]
                self.assertEqual(first_fill["Date"], pd.Timestamp(restoration))
                during = self.holdings.xs(ticker, level="ticker").loc[planned_exit:]
                during = during.loc[during.index < pd.Timestamp(restoration)]
                self.assertTrue(during["stale_mark"].all())
                self.assertTrue(during["units"].eq(first_fill["units"]).all())


if __name__ == "__main__":
    unittest.main()
