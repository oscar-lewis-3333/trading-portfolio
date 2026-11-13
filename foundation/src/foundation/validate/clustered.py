
#uncertainty testing for a forward test results where observations on the same day are not independent

import numpy as np
import pandas as pd


def clustered_mean(values, dates):
    #mean, with standard error from daily means (observations on the same day count as one)

    values = np.asarray(values, dtype=float)
    daily = pd.Series(values).groupby(pd.Series(np.asarray(dates))).mean()
    n = len(daily)
    if n < 2:
        return {"mean": float(values.mean()) if len(values) else None, "se": None, "t": None, "n_days": n, "n": len(values)}
    
    se = float(daily.std(ddof=1) / np.sqrt(n))
    mean = float(daily.mean())
    return {"mean": mean, "se": se, "t": mean / se if se else None, "n_days": n, "n": len(values)}


def weekly_block_bootstrap(values, dates, n_boot=5000, seed=0, level=0.95):
    #percentile interval for the mean using block bootstrap

    frame = pd.DataFrame({"v": np.asarray(values, dtype=float), "week": pd.to_datetime(pd.Series(np.asarray(dates))).dt.to_period("W").astype(str)})
    weeks = frame.week.unique()
    if len(weeks) < 3:
        return None
    
    rng = np.random.default_rng(seed)
    groups = {w: frame.loc[frame.week == w, "v"].to_numpy() for w in weeks}
    stats = [np.concatenate([groups[w] for w in rng.choice(weeks, size=len(weeks), replace=True)]).mean() for _ in range(n_boot)]
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return [float(lo), float(hi)]
