
#paired circular moving block-bootstrap for frozen sharpe comparison
import numpy as np
import pandas as pd
from weekly_sweep import _check_protocol


def _sharpe(values, axis, annualisation_days):
    std = values.std(axis=axis, ddof=1)
    if np.any(std <= 0) or np.any(np.ptp(values, axis=axis) == 0):
        raise ValueError("Zero-variance returns make the Sharpe comparison undefined.")
    result = values.mean(axis=axis) / std * np.sqrt(annualisation_days)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite Sharpe ratio.")
    return result


def paired_sharpe_bootstrap(strategy_returns, benchmark_returns, block_days=28, repetitions=10000, seed=20260919, confidence_level=0.95, annualisation_days=365, batch_size=250):
    #resample paired consecutive blocks, preserving their internal time order

    if not isinstance(strategy_returns, pd.Series) or not isinstance(benchmark_returns, pd.Series):
        raise ValueError("Supply indexed daily return Series.")
    dates = strategy_returns.index

    if (not isinstance(dates, pd.DatetimeIndex) or str(dates.tz) != "UTC"
            or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or len(dates) < 2 or not dates.equals(benchmark_returns.index)):
        raise ValueError("Returns must have identical, unique, sorted UTC date indices.")
    if (not (dates == dates.normalize()).all()
            or not dates.equals(pd.date_range(dates[0], periods=len(dates), freq="D"))):
        raise ValueError("Return indices must contain consecutive UTC calendar days.")
    for label, value in [("block_days", block_days), ("repetitions", repetitions), ("batch_size", batch_size)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer.")
    if repetitions < 2 or block_days > len(dates) // 2:
        raise ValueError("Need at least two repetitions and at least two blocks of observations.")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1).")
    if not np.isfinite(annualisation_days) or annualisation_days <= 0:
        raise ValueError("annualisation_days must be positive.")

    a = strategy_returns.to_numpy(dtype=float)
    b = benchmark_returns.to_numpy(dtype=float)

    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a <= -1).any() or (b <= -1).any():
        raise ValueError("Returns must be finite and greater than -100%.")
    sharpe_a = float(_sharpe(a, 0, annualisation_days))
    sharpe_b = float(_sharpe(b, 0, annualisation_days))
    n = len(a)
    blocks_per_sample = (n + block_days - 1) // block_days
    offsets = np.arange(block_days)
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)
    for first in range(0, repetitions, batch_size):
        count = min(batch_size, repetitions - first)
        starts = rng.integers(0, n, size=(count, blocks_per_sample))
        indices = ((starts[:, :, None] + offsets) % n).reshape(count, -1)[:, :n]
        differences[first:first + count] = _sharpe(a[indices], 1, annualisation_days) - _sharpe(b[indices], 1, annualisation_days)
    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(differences, [tail, 1 - tail])
    summary = pd.Series({
        "observations": n, "block_days": block_days, "repetitions": repetitions,
        "confidence_level": confidence_level,
        "strategy_sharpe": sharpe_a, "benchmark_sharpe": sharpe_b,
        "sharpe_difference": sharpe_a - sharpe_b,
        "ci_lower": float(lower), "ci_upper": float(upper)
    })
    return summary, differences


def test_walk_forward_edge(walk_forward_runs, protocol):
    #apply frozen primary block length and both sensitivity checks

    _check_protocol(protocol)
    start = pd.Timestamp(protocol["oos_start"], tz="UTC")
    end = pd.Timestamp(protocol["oos_end"], tz="UTC")
    expected = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    series = {}
    for name in ["walk_forward", "benchmark"]:
        ledger = walk_forward_runs[name][0]

        if not ledger.index.equals(expected):
            raise ValueError("Use the complete frozen evaluation period for both portfolios.")
        if not ledger["valuation_at"].eq(ledger.index + pd.Timedelta(days=1)).all():
            raise ValueError("Inconsistent daily valuation timestamps.")
        series[name] = ledger["net_return"]
    spec = protocol["inference"]
    blocks = [spec["primary_block_days"], *spec["sensitivity_block_days"]]
    summaries, draws = [], {}
    for block in blocks:
        summary, samples = paired_sharpe_bootstrap(series["walk_forward"], series["benchmark"], block_days=block, repetitions=spec["repetitions"], seed=spec["seed"], confidence_level=spec["confidence_level"], annualisation_days=protocol["annualisation_days"])
        summaries.append(summary)
        draws[block] = samples
    return pd.DataFrame(summaries).set_index("block_days"), draws