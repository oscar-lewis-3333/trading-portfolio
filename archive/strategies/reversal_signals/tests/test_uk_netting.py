
#ensuring the order netting (if we are selling 3 and buying 2, then only sell 1 instead of 2 seperate orders)
import copy
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from test_uk_data_preparation import CACHE, NOTEBOOK, defines, run_cells
from test_uk_execution_checkpoint import PROJECT, backtest, decisions, fixture


def run_netted(schedule, market, choices, horizon=1, cost=0.):
    return backtest.run_execution_backtest(
        decisions(schedule, choices), market, schedule,
        initial_capital_gbp=1000., holding_sessions=horizon,
        cost_per_side_bps=cost, net_orders=True
    )


def reconcile(test, result, market, initial, cost):
    """Reconstruct quantities, cash and entitlements from the external ledger."""
    daily, orders, holdings = (result[key] for key in ("daily", "orders", "holdings"))
    fills = orders.loc[orders["status"].eq("assumed_fill")].copy()
    signs = fills["side"].map({"buy": 1., "sell": -1.})
    test.assertTrue(signs.notna().all())
    np.testing.assert_allclose(fills["notional_gbp"], fills["units"] * fills["price_gbp"])
    np.testing.assert_allclose(fills["cost_gbp"], fills["notional_gbp"] * cost / 10000)
    cash_changes = -signs * fills["notional_gbp"] - fills["cost_gbp"]
    np.testing.assert_allclose(fills["cash_change_gbp"], cash_changes)
    #sells must fund buys without borrowing even within an opening session.
    test.assertTrue((initial + orders["cash_change_gbp"].cumsum()).ge(-1e-6).all())
    cash = initial + orders.groupby("Date")["cash_change_gbp"].sum().reindex(daily.index, fill_value=0.).cumsum()
    np.testing.assert_allclose(daily["cash_gbp"], cash, atol=1e-6)
    test.assertTrue(daily["cash_gbp"].ge(0).all())
    fees = fills.groupby("Date")["cost_gbp"].sum().reindex(daily.index, fill_value=0.)
    np.testing.assert_allclose(daily["cost_gbp"], fees, atol=1e-8)
    fills["signed_units"] = signs * fills["units"]
    movements = fills.groupby(["Date", "ticker"])["signed_units"].sum().unstack("ticker", fill_value=0.)
    inventory = movements.reindex(daily.index, fill_value=0.).cumsum()
    actual = holdings["units"].unstack("ticker").reindex(index=daily.index, columns=inventory.columns).fillna(0.)
    np.testing.assert_allclose(actual, inventory, rtol=1e-10, atol=1e-6)
    overnight = actual.shift(1, fill_value=0.)
    dividends = market["dividend_gbp"].unstack("ticker").reindex(index=daily.index, columns=actual.columns)
    entitlements = (overnight * dividends).where(overnight.gt(0), 0.).sum(axis=1, skipna=False)
    np.testing.assert_allclose(daily["dividends_accrued_gbp"], entitlements, atol=1e-8)
    np.testing.assert_allclose(daily["dividend_reserve_gbp"], entitlements.cumsum(), atol=1e-8)
    values = holdings["value_gbp"].groupby(level="Date").sum().reindex(daily.index, fill_value=0.)
    np.testing.assert_allclose(daily["stock_value_gbp"], values)
    np.testing.assert_allclose(daily["equity_gbp"], cash + values + entitlements.cumsum(), atol=1e-6)
    test.assertTrue(fills.groupby(["Date", "ticker"])["side"].nunique().le(1).all())
    refs = market.reindex(pd.MultiIndex.from_frame(fills[["Date", "ticker"]]))
    test.assertFalse(refs["known_suspended"].any())
    np.testing.assert_allclose(fills["price_gbp"], refs["open_gbp"], rtol=0, atol=0)


class NettingArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.schedule, self.market = fixture()
        self.session, self.exit = self.schedule.index[[2, 4]]
        self.day = self.market.xs(self.session, level="Date").copy()
        self.positions = {"A.L": {"units": 100., "exit_session": self.session}}

    def execute(self, cash, budgets, cost=100., positions=None):
        return backtest.execute_open_orders(
            cash, self.positions if positions is None else positions,
            self.day, budgets, self.session, self.exit,
            cost_per_side_bps=cost, net_orders=True
        )

    def test_unchanged_holding_renews_without_turnover_or_fees(self):
        result = self.execute(0., {"A.L": 1000.})
        self.assertEqual(result["positions"]["A.L"], {"units": 100., "exit_session": self.exit})
        self.assertEqual(result["cash_gbp"], 0.)
        order = result["orders"].iloc[0]
        self.assertEqual((order["side"], order["status"]), ("hold", "retained"))
        self.assertTrue(order[["notional_gbp", "cost_gbp", "cash_change_gbp"]].eq(0).all())

    def test_partial_reduction_charges_only_the_sale(self):
        result = self.execute(0., {"A.L": 800.})
        self.assertEqual(result["positions"]["A.L"]["units"], 80.)
        self.assertEqual(result["cash_gbp"], 198.)
        self.assertEqual(len(result["orders"]), 1)
        sale = result["orders"].iloc[0]
        self.assertEqual((sale["side"], sale["notional_gbp"], sale["cost_gbp"]), ("sell", 200., 2.))

    def test_partial_addition_keeps_fee_inside_the_allocation(self):
        result = self.execute(100., {"A.L": 1100.})
        self.assertAlmostEqual(result["positions"]["A.L"]["units"], 100. + 10. / 1.01)
        self.assertAlmostEqual(result["cash_gbp"], 0.)
        self.assertEqual(len(result["orders"]), 1)
        buy = result["orders"].iloc[0]
        self.assertEqual(buy["side"], "buy")
        self.assertAlmostEqual(buy["notional_gbp"], 100. / 1.01)
        self.assertAlmostEqual(buy["cost_gbp"], 1. / 1.01)

    def test_cash_constrained_rebalance_matches_closed_form(self):
        result = self.execute(0., {"A.L": 1000., "B.L": 1000.})
        scale = 990. / 1990.
        self.assertAlmostEqual(result["budget_scale"], scale)
        self.assertAlmostEqual(result["positions"]["A.L"]["units"], 100. * scale)
        self.assertAlmostEqual(result["positions"]["B.L"]["units"], 100. * scale / 1.01)
        self.assertEqual(list(result["orders"]["side"]), ["sell", "buy"])
        self.assertTrue(result["orders"]["cash_change_gbp"].cumsum().ge(-1e-10).all())
        self.assertAlmostEqual(result["cash_gbp"], 0.)

    def test_full_exit_and_replacement_include_both_fees(self):
        result = self.execute(0., {"B.L": 1000.})
        self.assertNotIn("A.L", result["positions"])
        self.assertAlmostEqual(result["positions"]["B.L"]["units"], 99. / 1.01)
        np.testing.assert_allclose(result["orders"]["cost_gbp"], [10., 9.9 / 1.01])
        self.assertAlmostEqual(result["cash_gbp"], 0.)
        liquidated = self.execute(0., {})
        self.assertEqual(liquidated["positions"], {})
        self.assertEqual(liquidated["cash_gbp"], 990.)

    def test_cancelled_slot_is_not_redistributed(self):
        self.day.loc["A.L", "known_suspended"] = True
        result = self.execute(1000., {"A.L": 500., "B.L": 500.}, positions={})
        self.assertAlmostEqual(result["cash_gbp"], 500.)
        self.assertAlmostEqual(result["positions"]["B.L"]["units"], 50. / 1.01)
        self.assertEqual(result["orders"].iloc[0]["reason"], "suspended")

    def test_blocked_or_not_due_holding_is_not_renewed_or_mutated(self):
        for suspended in [True, False]:
            with self.subTest(suspended=suspended):
                self.day.loc["A.L", "known_suspended"] = suspended
                positions = copy.deepcopy(self.positions)
                if not suspended:
                    positions["A.L"]["exit_session"] = self.schedule.index[3]
                original = copy.deepcopy(positions)
                day_before = self.day.copy(deep=True)
                result = self.execute(100., {"A.L": 100., "B.L": 100.}, positions=positions)
                self.assertEqual(positions, original)
                pd.testing.assert_frame_equal(self.day, day_before)
                self.assertEqual(result["positions"]["A.L"], original["A.L"])
                self.assertEqual(result["cash_gbp"], 50.)
                self.assertAlmostEqual(result["positions"]["B.L"]["units"], 5. / 1.01)
                self.assertIn("already_held", result["orders"]["reason"].tolist())

    def test_missing_open_prevents_renewal_until_a_later_session(self):
        schedule, market = self.schedule, self.market
        market.loc[(self.session, "A.L"), "open_gbp"] = np.nan
        result = run_netted(schedule, market, [(0, {"A.L": 1.}), (1, {"A.L": 1.})])
        self.assertEqual(result["holdings"].loc[(self.session, "A.L"), "scheduled_exit"], self.session)
        sells = result["orders"].loc[lambda x: x["side"].eq("sell")]
        self.assertEqual(list(sells["status"]), ["blocked", "assumed_fill"])
        self.assertEqual(sells.iloc[-1]["Date"], schedule.index[3])

    def test_retention_and_partial_sale_refresh_marks_when_close_missing(self):
        for opening, expected_units in [(10., 100.), (20., 50.)]:
            with self.subTest(opening=opening):
                market = self.market.copy()
                market.loc[(self.session, "A.L"), ["open_gbp", "close_gbp"]] = [opening, np.nan]
                result = run_netted(self.schedule, market, [(0, {"A.L": 1.}), (1, {"A.L": 1.})])
                holding = result["holdings"].loc[(self.session, "A.L")]
                self.assertEqual(holding["mark_gbp"], opening)
                self.assertEqual(holding["units"], expected_units)
                self.assertTrue(holding["stale_mark"])
                self.assertEqual(holding["scheduled_exit"], self.schedule.index[3])
                sales = result["orders"].loc[lambda x: x["side"].eq("sell")]
                self.assertEqual(sales.iloc[-1]["Date"], self.schedule.index[3])

    def test_renewal_accrues_dividends_once_and_keeps_them_out_of_budget(self):
        market = self.market.copy()
        for pos, amount, price in [(2, 1., 9.), (3, .5, 8.5)]:
            market.loc[(self.schedule.index[pos], "A.L"),
                       ["dividend_gbp", "open_gbp", "close_gbp"]] = [amount, price, price]
        result = run_netted(self.schedule, market, [(0, {"A.L": 1.}), (1, {"A.L": 1.})])
        self.assertEqual(result["daily"]["dividend_reserve_gbp"].iloc[-1], 150.)
        self.assertEqual(result["daily"]["equity_gbp"].iloc[-1], 1000.)
        self.assertEqual(result["orders"].loc[lambda x: x["Date"].eq(self.session), "status"].tolist(), ["retained"])
        reconcile(self, result, market, 1000., 0.)

    def test_nonzero_cost_multisession_book_reconciles(self):
        market = self.market.copy()
        for pos, date in enumerate(self.schedule.index):
            market.loc[(date, "A.L"), ["open_gbp", "close_gbp"]] = [10. + pos, 10.5 + pos]
            market.loc[(date, "B.L"), ["open_gbp", "close_gbp"]] = [15. - .2 * pos, 15.2 - .2 * pos]
        market.loc[(self.schedule.index[3], "A.L"), "dividend_gbp"] = .25
        market.loc[(self.schedule.index[4], "B.L"), "known_suspended"] = True
        choices = [(0, {"A.L": .7, "B.L": .3}), (1, {"A.L": .4, "B.L": .6}),
                   (2, {"A.L": .8, "B.L": .2}), (3, {"A.L": .5, "B.L": .5})]
        result = run_netted(self.schedule, market, choices, cost=100.)
        reconcile(self, result, market, 1000., 100.)
        self.assertEqual(result["final_positions"], {})

    def test_future_prices_dividends_and_signals_do_not_change_past(self):
        choices = [(0, {"A.L": 1.}), (1, {"A.L": 1.}), (4, {"A.L": 1.})]
        baseline = run_netted(self.schedule, self.market, choices, cost=100.)
        market = self.market.copy()
        future = market.index.get_level_values("Date") > self.schedule.index[3]
        market.loc[future, ["open_gbp", "close_gbp"]] *= 2.
        market.loc[future, "dividend_gbp"] = .5
        altered = run_netted(self.schedule, market, choices[:2] + [(4, {"B.L": 1.})], cost=100.)
        cutoff = self.schedule.index[3]
        for key in ["daily", "holdings"]:
            pd.testing.assert_frame_equal(baseline[key].loc[:cutoff], altered[key].loc[:cutoff])
        pd.testing.assert_frame_equal(baseline["orders"].loc[lambda x: x["Date"] <= cutoff],
                                      altered["orders"].loc[lambda x: x["Date"] <= cutoff])


class UKNettedNotebookTests(unittest.TestCase):
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
            if defines(source, "uk_netted_gross_backtest"):
                break
        else:
            raise AssertionError("No saved UK netted-backtest cell found")
        protected = [NOTEBOOK, CACHE, PROJECT / "src" / "reversal_backtest.py"]
        cls.hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        cls.ns = {"__name__": "__main__", "display": lambda *a, **k: None}
        run_cells(cells, cls.ns)
        cls.result = cls.ns["uk_netted_gross_backtest"]
        cls.round_trip = cls.ns["uk_gross_backtest"]

    def test_fresh_replay_preserves_files_and_development_boundary(self):
        for path, digest in self.hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        expected = self.ns["uk_schedule"].loc[self.ns["uk_decision_dates"][0]:].index
        pd.testing.assert_index_equal(self.result["daily"].index, expected)
        self.assertTrue((expected < pd.Timestamp(self.ns["UK_PERIODS"]["holdout_start"])).all())

    def test_zero_cost_equivalence_covers_entire_daily_book(self):
        for key in ["daily", "holdings"]:
            pd.testing.assert_frame_equal(self.result[key], self.round_trip[key],
                                          check_exact=False, rtol=1e-10, atol=1e-6)
        self.assertEqual(self.result["final_positions"], self.round_trip["final_positions"])

    def test_turnover_equals_net_quantity_changes_in_round_trip_ledger(self):
        old = self.round_trip["orders"].loc[lambda x: x["status"].eq("assumed_fill")].copy()
        old["signed_notional"] = old["notional_gbp"] * old["side"].map({"buy": 1., "sell": -1.})
        expected = old.groupby(["Date", "ticker"])["signed_notional"].sum().abs()
        new = self.result["orders"].loc[lambda x: x["status"].eq("assumed_fill")]
        actual = new.groupby(["Date", "ticker"])["notional_gbp"].sum().reindex(expected.index, fill_value=0.)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-6)
        self.assertLess(actual.sum(), old["notional_gbp"].sum())

    def test_cash_inventory_dividends_and_wealth_reconcile(self):
        reconcile(self, self.result, self.ns["uk_execution_market"],
                  self.ns["UK_BASELINE"]["initial_capital_gbp"], 0.)
        retained = self.result["orders"].loc[lambda x: x["status"].eq("retained")]
        self.assertTrue(retained[["notional_gbp", "cost_gbp", "cash_change_gbp"]].eq(0).all().all())


if __name__ == "__main__":
    unittest.main()
