import numpy as np
from foundation.reporting import metrics
from foundation.validate import sharpe

PPY = 12


def book(M, X, a, cost, rb=None):
    M = np.asarray(M, float)
    a = np.asarray(a, float)
    T, K = a.shape
    tot = M.copy()
    c = np.zeros(T)
    prev = np.vstack([np.zeros((1, K)), a[:-1]])
    for i in range(K):
        xi = np.where(a[:, i] != 0, np.nan_to_num(np.asarray(X, float)[:, i]) - M, 0.0)
        tot = tot + a[:, i] * xi
        c = c + cost[i] * np.abs(a[:, i] - prev[:, i])
    if rb is not None:
        tot = tot + np.asarray(rb, float)
    return tot - c, c


def stats(total, cash, cost=None, start=0, start_wealth=10_000):
    r = np.asarray(total, float)[start:]
    ca = np.asarray(cash, float)[start:]
    m = metrics(r, PPY)
    out = {"months": len(r), "final": start_wealth * m["terminal_wealth"], "cagr_pct": 100 * m["cagr"],
           "sharpe": sharpe(r - ca, periods_per_year=PPY), "mdd_pct": 100 * m["max_drawdown"],
           "vol_pct": 100 * float((r - ca).std(ddof=1) * np.sqrt(PPY))}
    if cost is not None:
        out["cost_pct_yr"] = 100 * float(np.asarray(cost)[start:].sum()) / (len(r) / PPY)
    return out
