import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter


def plot_equity_and_drawdown(equity, *, initial_capital_gbp, start_date, title="Portfolio value after modelled transaction costs"):

    #plot closing portfolio values and drawdowns (inc. initial capital)
    if equity.empty:
        raise ValueError("Provide nonempty equity curves.")

    start_date = pd.Timestamp(start_date)
    if (not equity.index.is_unique or not equity.index.is_monotonic_increasing or start_date >= equity.index[0]):
        raise ValueError("Equity dates must be ordered and unique, after start_date.")

    labels = {
        "trend": "Trend",
        "monthly_equal_weight": "Monthly equal weight",
        "buy_and_hold": "Buy and hold" 
    }

    #include capital before first trade when calculating running peaks
    starting_point = pd.DataFrame(initial_capital_gbp, index=pd.DatetimeIndex([start_date], name="Date"), columns=equity.columns)
    equity_plot = pd.concat([starting_point, equity]).rename(columns=labels)
    drawdown_plot = equity_plot / equity_plot.cummax() - 1
    colors = ["#0072B2", "#D55E00", "#009E73"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    equity_plot.plot(ax=axes[0], color=colors, linewidth=1.6)
    drawdown_plot.plot(ax=axes[1], color=colors, linewidth=1.3, legend=False)

    axes[0].set_title(title)
    axes[0].set_ylabel("Portfolio value")
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("£{x:,.0f}"))

    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[1].axhline(0, color="black", linewidth=0.7)

    for ax in axes:
        ax.grid(alpha=0.25)

    plt.show()
    return fig, axes