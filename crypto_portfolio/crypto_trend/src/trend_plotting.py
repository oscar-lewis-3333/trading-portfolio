import matplotlib

import matplotlib.pyplot as plt
import pandas as pd


def plot_evaluation_equity(portfolios, start, end, path=None, initial=1000.0):
    #rebased equity and drawdown over the evaluation window, one line per portfolio
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, (_, ledger) in portfolios.items():
        returns = ledger.loc[(ledger.index >= start) & (ledger.index < end), "net_return"]
        equity = initial * (1 + returns).cumprod()
        top.plot(equity.index, equity, label=name, linewidth=1.3)
        bottom.plot(equity.index, equity / equity.cummax() - 1, linewidth=1.0)
    top.set_ylabel("Portfolio value (£)")
    top.set_title("Evaluation, rebased to £1,000, after modelled transaction costs")
    top.legend(frameon=False)
    bottom.set_ylabel("Drawdown")
    bottom.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for axis in (top, bottom):
        axis.grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
