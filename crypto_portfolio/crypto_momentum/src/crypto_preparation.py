import numpy as np
import pandas as pd


def audit_daily_prices(prices, symbols, start, end):
    #normalise a panel copy, and more importantly report data-quality problems
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}

    missing_columns = required.difference(prices.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")
    if not prices["symbol"].isin(symbols).all():
        raise ValueError("The panel contains unexpected or missing symbols.")

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Use timezone-aware start and end dates.")

    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")

    if start >= end:
        raise ValueError("start must be earlier than end.")

    if start != start.normalize() or end != end.normalize():
        raise ValueError("Use UTC midnight boundaries.")

    panel = prices.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    panel[numeric_columns] = panel[numeric_columns].apply(pd.to_numeric, errors="coerce")
    expected_dates = pd.date_range(start=start, end=end - pd.Timedelta(days=1), freq="D")

    report_rows = []

    for symbol in symbols:
        group = panel.loc[panel["symbol"].eq(symbol)].copy()
        timestamps = group["timestamp"]
        problems = pd.DataFrame(index=group.index)

        problems["invalid_timestamp_rows"] = timestamps.isna()
        problems["outside_range_rows"] = (timestamps < start) | (timestamps >= end)
        problems["off_midnight_rows"] = timestamps.notna() & timestamps.ne(timestamps.dt.normalize())
        problems["duplicate_rows"] = timestamps.duplicated(keep=False)
        problems["nonfinite_numeric_rows"] = ~np.isfinite(group[numeric_columns].to_numpy(dtype=float)).all(axis=1)
        problems["nonpositive_price_rows"] = (group[["open", "high", "low", "close"]] <= 0).any(axis=1)
        problems["invalid_ohlc_rows"] = ((group["low"] > group["high"]) | (group["open"] < group["low"]) | (group["open"] > group["high"])
            | (group["close"] < group["low"]) | (group["close"] > group["high"]))

        problems["negative_volume_rows"] = group["volume"] < 0
        missing_dates = expected_dates.difference(pd.DatetimeIndex(timestamps.dropna()))
        counts = problems.sum().astype(int).to_dict()
        report_rows.append({
            "symbol": symbol,
            "rows": len(group),
            "expected_days": len(expected_dates),
            "missing_days": len(missing_dates),
            **counts,
            "zero_volume_rows": int(group["volume"].eq(0).sum()),
            "passed": (len(missing_dates) == 0 and not problems.any().any()
            )
        })

    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    report = pd.DataFrame(report_rows).set_index("symbol")

    return panel, report