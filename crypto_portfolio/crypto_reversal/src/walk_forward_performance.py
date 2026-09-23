import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reversal_sweep import summarise_development_ledger
from reversal_walk_forward import build_development_folds


def summarise_walk_forward(results, protocol, rules):
    #return paired metrics, daily net returns and valuation curves (GBP)
    folds = build_development_folds(protocol, rules)
    if set(results) != {"strategy", "benchmark"}:
        raise ValueError("Provide the strategy and its matched benchmark.")
    
    start = folds.test_start.iloc[0]
    end = folds.test_end_exclusive.iloc[-1]

    dates = pd.date_range(start, end, freq="D", inclusive="left")
    window = dict(protocol, development_start=start.strftime("%Y-%m-%d"), development_end_exclusive=end.strftime("%Y-%m-%d"))
    initial = float(rules["initial_test_capital_gbp"])
    metrics, returns, curves = {}, {}, {}
    for name in ("strategy", "benchmark"):
        ledger = results[name][0]
        if not ledger.index.equals(dates):
            raise ValueError("Both ledgers must cover every walk-forward day exactly once.")
        
        metrics[name] = summarise_development_ledger(ledger, window)
        reported = ledger["equity_close_gbp"].to_numpy(dtype=float)
        reconstructed = initial * (1. + ledger["net_return"]).cumprod().to_numpy()
        if (not np.isfinite(reported).all() or (reported <= 0).any() or not np.allclose(reported, reconstructed, rtol=1e-10, atol=1e-8)):
            raise ValueError("Net returns do not reconcile to equity and initial capital.")
        
        returns[name] = ledger["net_return"].copy()
        metrics[name]["final_equity_gbp"] = reported[-1]
        curves[name] = np.r_[initial, reported]
    returns = pd.DataFrame(returns)
    summary = pd.DataFrame(metrics).T
    summary["mean_daily_net_excess_bps"] = returns.sub(returns["benchmark"], axis=0).mean() * 10000.
    summary["sharpe_difference"] = summary["sharpe_zero_cash_rate"] - summary.loc["benchmark", "sharpe_zero_cash_rate"]

    #daily closes are valued at the FOLLOWING midnight, not candle start time.
    valuation_times = pd.date_range(start, end, freq="D", name="valuation_at")
    equity = pd.DataFrame(curves, index=valuation_times)
    return summary, returns, equity


def plot_walk_forward_equity(equity, rules, show=True):
    #plot equity curves
    if (set(equity.columns) != {"strategy", "benchmark"} or len(equity) < 2
            or not isinstance(equity.index, pd.DatetimeIndex)
            or str(equity.index.tz) != "UTC" or equity.index.has_duplicates
            or equity.index.hasnans or not equity.index.is_monotonic_increasing
            or not np.isfinite(equity.to_numpy()).all() or equity.le(0).any().any()):
        raise ValueError("Provide positive paired equity curves with unique ordered UTC valuations.")
    if not np.allclose(equity.iloc[0], rules["initial_test_capital_gbp"], rtol=0, atol=1e-8):
        raise ValueError("Include the initial capital valuation.")
    drawdown = equity.div(equity.cummax()).sub(1.)
    labels = {"strategy": "Reversal selection", "benchmark": "Matched benchmark"}
    round_trip_bps = 2 * (rules["fee_bps"] + rules["other_cost_bps"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    equity.rename(columns=labels).plot(ax=axes[0], logy=True)
    axes[0].set(title=f"Development walk-forward equity — {round_trip_bps:g} bp round-trip cost assumption", ylabel="Portfolio value (£, log scale)")
    drawdown.rename(columns=labels).mul(100).plot(ax=axes[1], legend=False)
    axes[1].set(title="Drawdown from the running peak", xlabel="Valuation date", ylabel="Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes, drawdown
