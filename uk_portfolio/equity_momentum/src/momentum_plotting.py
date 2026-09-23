import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter


def plot_equity_and_drawdown(equity, *, initial_capital_gbp, start_date, title="UK equity momentum"):
    #plot net portfolio values and drawdowns, inclusive of starting capital
    if (equity.empty or not np.isfinite(equity.to_numpy()).all() or equity.le(0).any().any()):
        raise ValueError("Provide positive, finite equity curves.")
    if not np.isfinite(initial_capital_gbp) or initial_capital_gbp <= 0:
        raise ValueError("Starting capital must be positive and finite.")

    start_date = pd.Timestamp(start_date)
    if (not isinstance(equity.index, pd.DatetimeIndex) or not equity.index.is_unique or not equity.index.is_monotonic_increasing or pd.isna(start_date) or start_date >= equity.index[0]):
        raise ValueError("Equity dates must be sorted and follow start_date.")

    starting_point = pd.DataFrame(initial_capital_gbp, index=pd.DatetimeIndex([start_date], name="Date"), columns=equity.columns)

    values = pd.concat([starting_point, equity])
    drawdowns = values.div(values.cummax()).sub(1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    for label in values.columns:
        line, = axes[0].plot(values.index, values[label], label=label, linewidth=1.6)
        axes[1].plot(drawdowns.index, drawdowns[label], color=line.get_color(), linewidth=1.3)

    axes[0].set_title(title)
    axes[0].set_ylabel("Portfolio value after costs")
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("£{x:,.0f}"))
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[1].axhline(0, color="black", linewidth=0.7)

    for ax in axes:
        ax.grid(alpha=0.25)

    plt.show()
    return fig, axes