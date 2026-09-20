import numpy as np
import pandas as pd
from portfolio import rebalance_gbp


def run_daily_ledger(gbp_prices, execution_targets, initial_cash=1000.0, fee_bps=9.0, other_cost_bps=3.5):
    #simulate Tuesday rebalances and daily GBP closing valuations

    targets = execution_targets.copy()
    if targets.empty or not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Provide targets and positive initial cash.")
    if targets.columns.has_duplicates or "CASH" not in targets:
        raise ValueError("Expected unique target columns including CASH.")
    
    times = targets.index
    if (not isinstance(times, pd.DatetimeIndex) or str(times.tz) != "UTC" or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing):
        raise ValueError("Execution times must be unique, sorted UTC dates.")
    if not ((times.dayofweek == 1) & (times == times.normalize())).all():
        raise ValueError("Expected Tuesday execution at UTC midnight.")
    
    values = targets.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError("Invalid target weights.")
    if (not np.isfinite([fee_bps, other_cost_bps]).all() or min(fee_bps, other_cost_bps) < 0 or fee_bps + other_cost_bps >= 10000):
        raise ValueError("Invalid cost assumptions.")

    prices = gbp_prices.copy()
    required = {"product_id", "timestamp", "open_gbp", "close_gbp"}
    if prices.empty or not required.issubset(prices):
        raise ValueError("Missing GBP price history.")
    
    candle_times = prices["timestamp"]
    if (not isinstance(candle_times.dtype, pd.DatetimeTZDtype) or str(candle_times.dt.tz) != "UTC" or candle_times.isna().any() or candle_times.ne(candle_times.dt.normalize()).any()):
        raise ValueError("Expected UTC midnight candle timestamps.")
    if prices["product_id"].isna().any() or prices.duplicated(["product_id", "timestamp"]).any():
        raise ValueError("Missing asset IDs or duplicate candles.")
    if times.min() < candle_times.min() or times.max() > candle_times.max():
        raise ValueError("Execution targets fall outside price history.")
    
    expected = pd.date_range(times.min(), candle_times.max(), freq="W-TUE")
    if not times.equals(expected):
        raise ValueError("A Tuesday target is missing; retain cash decisions.")

    assets = targets.columns.drop("CASH").tolist()
    dates = pd.date_range(times.min(), candle_times.max(), freq="D")
    panels = {field: prices.pivot(index="timestamp", columns="product_id", values=field) .reindex(index=dates, columns=assets)
        for field in ("open_gbp", "close_gbp")}
    units = pd.Series(0., index=assets)
    cash = previous_equity = float(initial_cash)
    ledger_rows, trade_parts, position_rows = [], [], []

    def require_prices(row, needed, date, label):
        selected = row.reindex(needed)
        bad = ~np.isfinite(selected.to_numpy(dtype=float)) | selected.le(0).to_numpy()

        if bad.any():
            raise ValueError(f"{date}: invalid {label} prices for {selected.index[bad].tolist()}")
        return selected

    def mark(row, date, label):
        held = units.loc[units.gt(0)]
        p = require_prices(row, held.index, date, label)
        equity = float(cash + (held * p).sum())
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError(f"{date}: invalid portfolio equity.")
        return equity

    for date in dates:
        opening = panels["open_gbp"].loc[date]
        closing = panels["close_gbp"].loc[date]
        equity_open = mark(opening, date, "opening")
        fee = other = notional = 0.0
        scheduled = date in targets.index

        if scheduled:
            target = targets.loc[date]
            wanted = target.drop("CASH").loc[lambda x: x.gt(0)].index
            needed = units.loc[units.gt(0)].index.union(wanted)
            require_prices(opening, needed, date, "execution")
            units, cash, trades, audit = rebalance_gbp(units, cash, opening, target, fee_bps=fee_bps, other_cost_bps=other_cost_bps)
            fee = float(trades["fee_gbp"].sum())
            other = float(trades["other_cost_gbp"].sum())
            notional = float(audit["traded_notional_gbp"])
            actual = trades.loc[trades["signed_notional_gbp"].ne(0)].reset_index()
            if not actual.empty:
                actual["decision_at"] = date - pd.Timedelta(days=1)
                actual["execution_at"] = date
                trade_parts.append(actual)

        equity_after = mark(opening, date, "post-rebalance")
        equity_close = mark(closing, date, "closing")
        error = equity_after + fee + other - equity_open
        if abs(error) > 1e-10 * max(1., equity_open):
            raise RuntimeError(f"{date}: accounting does not reconcile.")

        ledger_rows.append({
            "candle_start": date,
            "valuation_at": date + pd.Timedelta(days=1),
            "scheduled_rebalance": scheduled,
            "equity_open_gbp": equity_open,
            "equity_after_rebalance_gbp": equity_after,
            "equity_close_gbp": equity_close,
            "cash_gbp": cash,
            "fee_gbp": fee,
            "other_cost_gbp": other,
            "traded_notional_gbp": notional,
            "net_return": equity_close / previous_equity - 1.,
            "crypto_weight_close": 1. - cash / equity_close,
            "accounting_error_gbp": error
        })
        position_rows.append(units.reindex(assets).copy())
        previous_equity = equity_close

    ledger = pd.DataFrame(ledger_rows).set_index("candle_start")
    positions = pd.DataFrame(position_rows, index=ledger.index).reindex(columns=assets)
    trade_columns = ["product_id", "reference_price_gbp", "units_before", "units_after",
                     "signed_notional_gbp", "fee_gbp", "other_cost_gbp", "decision_at", "execution_at"]
    
    trade_log = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame(columns=trade_columns)
    return ledger, trade_log, positions
