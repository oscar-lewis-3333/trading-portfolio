import matplotlib.pyplot as plt
import numpy as np


def plot_equity_and_drawdown(equity, *, title="UK reversal baseline — development period", log_equity=False):
    #plot portfolio equity and drawdown for each scenario
    if equity.empty or not np.isfinite(equity.to_numpy()).all():
        raise ValueError("Equity must contain finite values with no missing dates.")
    if equity.le(0).any().any():
        raise ValueError("Equity must be positive.")
    if not equity.index.is_unique or not equity.index.is_monotonic_increasing:
        raise ValueError("Equity dates must be unique and sorted.")

    drawdown = equity.div(equity.cummax()).sub(1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for label in equity.columns:
        line, = axes[0].plot(equity.index, equity[label], label=label)
        axes[1].plot(drawdown.index, 100 * drawdown[label], color=line.get_color())

    axes[0].set_title(title)
    if log_equity:
        axes[0].set_yscale("log")
        
    axes[0].set_ylabel("Portfolio equity (£)")
    axes[0].legend()

    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig