"""Hand-calculated cost funding, frozen holdings and validation contracts."""
import numpy as np
import pandas as pd
import pytest
from foundation.portfolio.legacy_rebalance import rebalance_with_costs

def s(values):
    return pd.Series(values, index=['A', 'B'], dtype=float)

def test_entry_funds_its_own_costs():
    result = rebalance_with_costs(s([0, 0]), 1000, s([.5, .5]), 100)
    np.testing.assert_allclose(result['holdings_gbp'], [500 / 1.01] * 2)

    assert result['total_cost_gbp'] == pytest.approx(1000 / 101)
    assert result['cash_gbp'] == pytest.approx(0)

def test_switch_charges_sale_and_purchase_and_reconciles():
    result = rebalance_with_costs(s([1000, 0]), 0, s([0, 1]), 100)
    np.testing.assert_allclose(result['trades_gbp'], [-1000, 990 / 1.01])
    assert result['holdings_gbp'].sum() + result['cash_gbp'] + result['total_cost_gbp'] == pytest.approx(1000)

def test_blocked_purchase_stays_in_cash_and_inputs_are_unchanged():
    holdings, weights = s([0, 0]), s([.5, .5])
    result = rebalance_with_costs(holdings, 1000, weights, 0, pd.Series([True, False], index=holdings.index))
    np.testing.assert_array_equal(result['holdings_gbp'], [0, 500])

    assert result['cash_gbp'] == 500
    pd.testing.assert_series_equal(holdings, s([0, 0]))
    pd.testing.assert_series_equal(weights, s([.5, .5]))

def test_trapped_position_scales_tradable_allocation():
    result = rebalance_with_costs(s([800, 0]), 200, s([0, 1]), 100, pd.Series([True, False], index=['A', 'B']))
    assert result['holdings_gbp']['A'] == 800

    assert result['trades_gbp']['A'] == 0
    assert result['holdings_gbp']['B'] == pytest.approx(200 / 1.01)

@pytest.mark.parametrize('holdings,cash,weights,cost', [([-1, 0], 100, [0, 0], 0), ([0, 0], -1, [0, 0], 0),
    ([0, np.nan], 100, [0, 0], 0), ([0, 0], 100, [.6, .6], 0),
    ([0, 0], 100, [1, 0], -1), ([0, 0], 100, [1, 0], 10000)])

def test_invalid_accounting_inputs_fail(holdings, cash, weights, cost):
    with pytest.raises(ValueError):
        rebalance_with_costs(s(holdings), cash, s(weights), cost)

def test_misaligned_flags_fail():
    with pytest.raises(ValueError, match='Frozen'):
        rebalance_with_costs(s([0, 0]), 100, s([1, 0]), 0, pd.Series([True, False], index=['B', 'A']))
