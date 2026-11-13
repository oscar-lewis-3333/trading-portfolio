#compact information computed from live simulation outputs
import numpy as np
import pandas as pd
import reversal_validation


def concentration_diagnostics(inputs, result, ledgers):
    equity = result["equity"]
    returns = equity.pct_change(fill_method=None).iloc[1:]
    excess = result["net_excess"].iloc[:, 0]
    daily = returns.mul(10000).rename(columns={"Walk-forward": "strategy_return_bps", "Matched benchmark": "benchmark_return_bps"})
    daily["net_excess_bps"] = excess * 10000

    ranked = daily["net_excess_bps"].sort_values(ascending=False, kind="stable")
    rows = []
    for k in (1, 5, 10, 20):
        top, total = ranked.iloc[:k].sum(), ranked.sum()
        rows.append({"top_days": k, "share_of_total_net_excess_pct": 100 * top / total if total else np.nan,
                     "top_days_contribution_to_mean_bps": top / len(ranked),
                     "other_days_contribution_to_mean_bps": (total - top) / len(ranked)})
        
    contributions = {}
    for name, ledger in ledgers.items():
        contributions[name] = reversal_validation.attribute_stock_pnl(ledger, inputs["market"].loc[equity.index[0]:],
            inputs["spec"]["initial_capital_gbp"])["net_pnl_gbp"]
        
    stocks = pd.DataFrame(contributions).fillna(0.)
    stocks["wealth_difference_gbp"] = stocks["Walk-forward"] - stocks["Matched benchmark"]
    np.testing.assert_allclose(stocks["wealth_difference_gbp"].sum(),
        equity["Walk-forward"].iloc[-1] - equity["Matched benchmark"].iloc[-1], rtol=1e-10, atol=1e-6)
    
    holdings = ledgers["Walk-forward"]["holdings"][["value_gbp", "stale_mark"]].copy()
    dates = holdings.index.get_level_values("Date")
    holdings["weight_pct"] = 100 * holdings["value_gbp"] / equity["Walk-forward"].reindex(dates).to_numpy()
    largest = holdings["weight_pct"].groupby(level="Date").max().reindex(equity.index, fill_value=0.)

    return {"concentration": pd.DataFrame(rows).set_index("top_days"),
            "strongest_days": daily.nlargest(5, "net_excess_bps"),
            "weakest_days": daily.nsmallest(5, "net_excess_bps"),
            "stock_contributions": pd.concat([stocks.nlargest(5, "wealth_difference_gbp"), stocks.nsmallest(5, "wealth_difference_gbp")]),
            "largest_weights": holdings.nlargest(5, "weight_pct"),
            "weight_distribution": largest.describe(percentiles=[.5, .9, .95, .99])}
