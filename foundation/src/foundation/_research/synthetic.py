import numpy as np
import pandas as pd

from .contracts import validate_panel


def make_panel(assets=24, days=900, start="2022-01-03", seed=0, market_volatility=0.045, idiosyncratic_volatility=0.035, drift=0.0004, gap_probability=0.0):
    #synthetic prices with a common factors (tests, demos don't touch real data).
    for name, value in (("assets", assets), ("days", days)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer of at least two.")
    if not 0 <= gap_probability < 0.5:
        raise ValueError("gap_probability must lie in [0, 0.5).")

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D", tz="UTC")

    #do BTC(-GBP) first as it acts as the market proxy for other signals
    names = ["BTC-GBP"] + [f"A{index:02d}-GBP" for index in range(1, assets)]

    market = rng.normal(drift, market_volatility, size=days)
    betas = np.concatenate([[1.0], rng.uniform(0.6, 1.6, size=assets - 1)])
    idio = rng.normal(0.0, idiosyncratic_volatility, size=(days, assets))
    returns = market[:, None] * betas[None, :] + idio
    closes = 100.0 * np.exp(np.cumsum(returns, axis=0))

    frame = pd.DataFrame(closes, index=dates, columns=names)
    panel = frame.stack().rename("close").reset_index()
    panel.columns = ["timestamp", "product_id", "close"]

    #stagger the listings, leading to a non constant universe
    listed = {name: dates[0] if index < 2 else dates[min(days - 200, 20 * index)] for index, name in enumerate(names)}
    panel = panel.loc[panel.apply(lambda row: row["timestamp"] >= listed[row["product_id"]], axis=1)].copy()

    if gap_probability > 0:
        keep = rng.random(len(panel)) >= gap_probability
        panel = panel.loc[keep].copy()

    panel["eligible"] = True
    panel = panel.sort_values(["timestamp", "product_id"]).reset_index(drop=True)

    return validate_panel(panel[["product_id", "timestamp", "close", "eligible"]])


def make_returns(panel):
    #wide return matrix for the harness
    from .panel import daily_returns, pivot_closes

    return daily_returns(pivot_closes(panel))
