import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# tables are private (not in the repo); set ISA_SHOWCASE_TABLES to point elsewhere
TABLES = Path(os.environ.get("ISA_SHOWCASE_TABLES",
                             Path(__file__).resolve().parents[4] / "trading_research_notes/ops/public_showcase/tables"))
PARTS = ["core", "defensive", "rebound", "overlay", "sleeves"]
COLORS = {"core": "#2a78d6", "defensive": "#eb6834", "rebound": "#1baf7a", "overlay": "#eda100", "sleeves": "#e87ba4",
          "book": "#0b0b0b"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.dpi": 110,
    })
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    pd.set_option("display.width", 200)


def table(name):
    return pd.read_csv(TABLES / name)


def monthly():
    d = pd.read_csv(TABLES / "parts_monthly.csv", index_col="month")
    d.index = pd.PeriodIndex(d.index, freq="M").to_timestamp("M")
    return d


def wealth(r, start=10_000):
    return start * (1 + r).cumprod()


def drawdown(r):
    w = (1 + r).cumprod()
    return 100 * (w / w.cummax() - 1)


def rolling_sharpe(r, cash, n=36):
    x = r - cash
    return x.rolling(n).mean() / x.rolling(n).std() * np.sqrt(12)


def _label_ends(ax, series, colors):
    for name, s in series.items():
        s = s.dropna()
        ax.annotate(name, (s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")


def curves(series, colors, title, log=True, path=None):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    w = {k: wealth(v) for k, v in series.items()}
    for k, v in w.items():
        a1.plot(v.index, v, color=colors[k], label=k)
        a2.plot(v.index, drawdown(series[k]), color=colors[k], linewidth=1.5)
    if log:
        a1.set_yscale("log")
    a1.set_title(title, loc="left")
    a1.set_ylabel("£10,000 grows to (£)")
    a2.set_ylabel("drawdown (%)")
    if len(series) > 1:
        a1.legend(loc="upper left")
    _label_ends(a1, w, colors)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    plt.show()


def lines(series, colors, title, ylabel, zero=True, path=None):
    fig, ax = plt.subplots(figsize=(11, 4))
    for k, v in series.items():
        ax.plot(v.index, v, color=colors[k], label=k)
    if zero:
        ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(loc="upper left")
    _label_ends(ax, series, colors)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    plt.show()


def year_bars(df, cols, title, ylabel, path=None):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(df))
    pos, neg = np.zeros(len(df)), np.zeros(len(df))
    for c in cols:
        v = df[c].to_numpy(float)
        base = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=base, color=COLORS.get(c, MUTED), label=c, width=0.8, edgecolor=SURFACE, linewidth=1)
        pos, neg = pos + np.where(v >= 0, v, 0), neg + np.where(v < 0, v, 0)
    if "book" in df:
        ax.plot(x, df["book"], "o", color=INK, markersize=4, label="book")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x[::2], df.index[::2], rotation=0)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    lo, hi = min(neg.min(), 0), max(pos.max(), 0)
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=min(6, len(cols) + 1))
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    plt.show()


def calendar_returns(d, cols):
    g = d[cols].groupby(d.index.year)
    out = 100 * (g.apply(lambda x: (1 + x).prod() - 1))
    out.insert(0, "months", g.size())
    return out
