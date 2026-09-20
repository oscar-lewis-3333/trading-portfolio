"""Equity and drawdown plots for the continuous walk-forward portfolios."""
import pandas as pd
import matplotlib.pyplot as plt


def plot_walk_forward_equity(walk_forward_runs, protocol, show=True):
    """Plot GBP equity and drawdown, including the initial cash valuation."""
    equity = pd.DataFrame({
        name: ledger.set_index("valuation_at")["equity_close_gbp"]
        for name, (ledger, _, _) in walk_forward_runs.items()
    })
    initial = pd.DataFrame(
        protocol["initial_cash_gbp"],
        index=[pd.Timestamp(protocol["oos_start"], tz="UTC")],
        columns=equity.columns,
    )
    equity = pd.concat([initial, equity]).sort_index()
    drawdown = equity.div(equity.cummax()).sub(1)
    labels = {"walk_forward": "Momentum", "benchmark": "Benchmark"}
    round_trip_bps = 2 * sum(protocol["costs"][key] for key in ("fee_bps", "other_cost_bps"))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    equity.rename(columns=labels).plot(ax=axes[0], logy=True)
    axes[0].set(
        title=f"Walk-forward equity after {round_trip_bps:g}bp round-trip costs",
        ylabel="Portfolio value (£, log scale)",
    )
    drawdown.rename(columns=labels).mul(100).plot(ax=axes[1], legend=False)
    axes[1].set(
        title="Drawdown from the running peak",
        xlabel="Valuation date",
        ylabel="Drawdown (%)",
    )
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes, equity, drawdown
