"""Check per-asset net costs in the reversal project's accounting engine."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import portfolio as engine


class NetTradeCostTests(unittest.TestCase):
    def test_unchanged_holdings_have_no_cost(self):
        _, _, trades, audit = engine.rebalance_gbp(
            {"A": 5.}, 500., {"A": 100.}, {"A": .5, "CASH": .5})
        self.assertAlmostEqual(audit["total_cost_gbp"], 0., places=12)
        self.assertAlmostEqual(trades["signed_notional_gbp"].abs().sum(), 0., places=10)

    def test_partial_reduction_costs_only_the_reduction(self):
        # Reduce A from GBP 1,000 to GBP 800. Exactly GBP 200 is sold.
        post_equity = 1000. - 200. * .00125
        units, cash, trades, audit = engine.rebalance_gbp(
            {"A": 10.}, 0., {"A": 100.},
            {"A": 800. / post_equity, "CASH": 199.75 / post_equity})
        self.assertAlmostEqual(units["A"], 8.)
        self.assertAlmostEqual(cash, 199.75)
        self.assertAlmostEqual(trades.loc["A", "signed_notional_gbp"], -200.)
        self.assertAlmostEqual(audit["total_cost_gbp"], .25)

    def test_switching_assets_costs_both_legs_not_cash_flow_difference(self):
        rate = .00125
        expected_purchase = 1000. * (1 - rate) / (1 + rate)
        _, _, trades, audit = engine.rebalance_gbp(
            {"A": 10.}, 0., {"A": 100., "B": 50.}, {"B": 1., "CASH": 0.})
        self.assertAlmostEqual(trades.loc["A", "signed_notional_gbp"], -1000.)
        self.assertAlmostEqual(trades.loc["B", "signed_notional_gbp"], expected_purchase)
        self.assertAlmostEqual(audit["total_cost_gbp"], rate * (1000. + expected_purchase))
        self.assertGreater(audit["total_cost_gbp"], rate * abs(trades["signed_notional_gbp"].sum()))

    def test_drift_adjustment_does_not_liquidate_and_repurchase(self):
        # GBP 600/400 -> equal weight. Sell GBP 100.125, buy GBP 99.875.
        _, _, trades, audit = engine.rebalance_gbp(
            {"A": 6., "B": 4.}, 0., {"A": 100., "B": 100.},
            {"A": .5, "B": .5, "CASH": 0.})
        self.assertAlmostEqual(trades.loc["A", "signed_notional_gbp"], -100.125)
        self.assertAlmostEqual(trades.loc["B", "signed_notional_gbp"], 99.875)
        self.assertAlmostEqual(audit["traded_notional_gbp"], 200.)
        self.assertAlmostEqual(audit["total_cost_gbp"], .25)

    def test_all_cash_target_sells_departing_assets_and_costs_the_sale(self):
        units, cash, trades, audit = engine.rebalance_gbp(
            {"A": 10.}, 0., {"A": 100.}, {"CASH": 1.})
        self.assertEqual(units["A"], 0.)
        self.assertAlmostEqual(cash, 998.75)
        self.assertAlmostEqual(trades.loc["A", "signed_notional_gbp"], -1000.)
        self.assertAlmostEqual(audit["total_cost_gbp"], 1.25)
        self.assertAlmostEqual(audit["accounting_error_gbp"], 0.)

    def test_missing_departing_or_targeted_prices_fail(self):
        for prices in ({"A": 100.}, {"B": 50.}):
            with self.subTest(prices=prices), self.assertRaisesRegex(ValueError, "valid GBP price"):
                engine.rebalance_gbp(
                    {"A": 10.}, 0., prices, {"B": 1., "CASH": 0.})

    def test_initial_buy_accounts_for_costs_before_allocating(self):
        units, cash, trades, audit = engine.rebalance_gbp(
            {}, 1000., {"A": 100., "B": 200.}, {"A": .5, "B": .5, "CASH": 0.})
        invested = 1000. / 1.00125
        self.assertAlmostEqual(units["A"], invested / 200.)
        self.assertAlmostEqual(units["B"], invested / 400.)
        self.assertAlmostEqual(cash, 0.)
        self.assertAlmostEqual(audit["total_cost_gbp"], 1000. - invested)
        self.assertAlmostEqual(audit["equity_after_gbp"], invested)


if __name__ == "__main__":
    unittest.main()
