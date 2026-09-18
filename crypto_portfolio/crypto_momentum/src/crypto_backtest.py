import numpy as np
import pandas as pd


def rebalance_portfolio(units, cash, prices, target_weights, fee_bps=9.0, other_cost_bps=3.5):
    #rebalance at reference prices with proportional trading costs
    book = pd.DataFrame({
        "units": pd.Series(units, dtype=float),
        "price": pd.Series(prices, dtype=float),
        "weight": pd.Series(target_weights, dtype=float)
    })
    if book.empty or not book.index.is_unique:
        raise ValueError("Provide a nonempty, unique asset universe.")
    if not np.isfinite(book.to_numpy()).all():
        raise ValueError("Inputs must be finite and have matching assets.")
    if not np.isfinite([cash, fee_bps, other_cost_bps]).all():
        raise ValueError("Cash and cost assumptions must be finite.")

    if (cash < 0 or (book["units"] < 0).any() or (book["price"] <= 0).any() or (book["weight"] < 0).any() or book["weight"].sum() > 1):
        raise ValueError("Invalid holdings, prices or long-only weights.")

    fee_rate = fee_bps / 10_000
    other_rate = other_cost_bps / 10_000
    cost_rate = fee_rate + other_rate
    if fee_rate < 0 or other_rate < 0 or cost_rate >= 1:
        raise ValueError("Costs must be nonnegative and total less than 100%.")

    current_values = book["units"] * book["price"]
    equity_before = float(cash + current_values.sum())
    if equity_before <= 0:
        raise ValueError("Portfolio equity must be positive.")

    #find affordable post-cost equity by halving interval repeatedly
    lower = 0.0
    upper = equity_before

    for _ in range(60):
        candidate = (lower + upper) / 2
        target_values = book["weight"] * candidate
        turnover_value = (target_values - current_values).abs().sum()
        if candidate + cost_rate * turnover_value <= equity_before:
            lower = candidate
        else:
            upper = candidate

    target_values = book["weight"] * lower
    signed_notional = target_values - current_values
    traded_notional = signed_notional.abs()
    fees = fee_rate * traded_notional
    other_costs = other_rate * traded_notional
    new_units = target_values / book["price"]
    new_cash = float(cash - signed_notional.sum() - fees.sum() - other_costs.sum())
    if new_cash < -1e-10 * equity_before:
        raise RuntimeError("Rebalancing produced negative cash.")

    #neglect only a possible float negative residual
    new_cash = max(new_cash, 0.0)

    trades = pd.DataFrame({
        "reference_price": book["price"],
        "units_before": book["units"],
        "units_after": new_units,
        "signed_notional": signed_notional,
        "fee": fees,
        "other_cost": other_costs
    })
    trades.index.name = "symbol"

    return new_units, new_cash, trades

def run_daily_backtest(prices, targets, symbols, initial_cash=1_000.0, fee_bps=9.0, other_cost_bps=3.5):
    #simulating scheduled rebalances and daily closing portfolio values
    symbols = list(symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Provide a nonempty, unique asset universe.")
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite.")
    if targets.empty:
        raise ValueError("No target allocations were supplied.")
    if targets["execution_at"].duplicated().any():
        raise ValueError("Duplicate execution dates.")
    if (targets[["decision_at", "execution_at"]].isna().any().any() or (targets["decision_at"] >= targets["execution_at"]).any()):
        raise ValueError("Each decision must precede its execution.")

    allocations = targets[symbols + ["GBP_cash"]].astype(float)
    if (not np.isfinite(allocations.to_numpy()).all() or (allocations < 0).any().any() or not np.allclose(allocations.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("Target weights must be nonnegative and sum to 1.")
    
    open_prices = prices.pivot(index="timestamp", columns="symbol", values="open").reindex(columns=symbols).sort_index()
    close_prices = prices.pivot(index="timestamp", columns="symbol", values="close").reindex(columns=symbols).reindex(index=open_prices.index)

    for table in (open_prices, close_prices):
        values = table.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Missing or invalid opening/closing prices.")

    schedule = targets.set_index("execution_at").sort_index()
    if not schedule.index.isin(open_prices.index).all():
        raise ValueError("An execution date has no matching price observation.")

    dates = open_prices.index[open_prices.index >= schedule.index.min()]
    if not dates.equals(pd.date_range(dates.min(), dates.max(), freq="D")):
        raise ValueError("The simulation requires consecutive daily prices.")

    units = pd.Series(0.0, index=symbols)
    cash = float(initial_cash)
    previous_equity = float(initial_cash)

    ledger_rows = []
    trade_parts = []

    for date in dates:
        opening = open_prices.loc[date]
        closing = close_prices.loc[date]

        equity_open = float(cash + (units * opening).sum())
        fee = other_cost = traded_notional = 0.0
        scheduled = date in schedule.index

        if scheduled:
            target = schedule.loc[date]

            units, cash, trades = rebalance_portfolio(
                units=units,
                cash=cash,
                prices=opening,
                target_weights=target[symbols],
                fee_bps=fee_bps,
                other_cost_bps=other_cost_bps
            )

            fee = float(trades["fee"].sum())
            other_cost = float(trades["other_cost"].sum())
            traded_notional = float(trades["signed_notional"].abs().sum())
            trades = trades.reset_index()
            trades["decision_at"] = target["decision_at"]
            trades["execution_at"] = date
            trade_parts.append(trades)

        equity_after_rebalance = float(cash + (units * opening).sum())
        equity_close = float(cash + (units * closing).sum())

        ledger_rows.append({
            "candle_start": date,
            "valuation_at": date + pd.Timedelta(days=1),
            "scheduled_rebalance": scheduled,
            "equity_open": equity_open,
            "equity_after_rebalance": equity_after_rebalance,
            "equity_close": equity_close,
            "cash_gbp": cash,
            "fee": fee,
            "other_cost": other_cost,
            "traded_notional": traded_notional,
            "net_return": equity_close / previous_equity - 1,
            "rebalance_accounting_error": equity_after_rebalance + fee + other_cost - equity_open,
            **{f"units_{symbol}": units[symbol] for symbol in symbols}
        })

        previous_equity = equity_close

    ledger = pd.DataFrame(ledger_rows).set_index("candle_start")
    trade_log = pd.concat(trade_parts, ignore_index=True)

    return ledger, trade_log