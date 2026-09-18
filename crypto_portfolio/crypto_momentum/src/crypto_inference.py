import numpy as np
import pandas as pd


def _circular_block_indices(n, block_days, rng):

    #sample n blocks, wrapping nicely around sample boundary
    number_of_blocks = (n + block_days - 1) // block_days
    starts = rng.integers(0, n, size=number_of_blocks)
    indices = (starts[:, None] + np.arange(block_days)[None, :]) % n

    return indices.ravel()[:n]


def _column_sharpes(values, annualisation_days):
#calculating sharpes seperately for each column
    volatility = values.std(axis=0, ddof=1)

    if (np.any(np.ptp(values, axis=0) == 0) or np.any(volatility <= 0) or not np.isfinite(volatility).all()):
        raise ValueError("Sharpe is undefined for a constant return series.")

    sharpes = (values.mean(axis=0) / volatility * np.sqrt(annualisation_days))
    if not np.isfinite(sharpes).all():
        raise ValueError("Nonfinite Sharpe estimate.")

    return sharpes


def paired_sharpe_bootstrap(strategy_returns, benchmark_returns, block_days=28, repetitions=10_000, seed=20260916, confidence_level=0.95, annualisation_days=365.25):

    if not isinstance(strategy_returns, pd.Series) or not isinstance(benchmark_returns, pd.Series):
        raise ValueError("Provide two pandas Series with matching dates.")
    if not strategy_returns.index.equals(benchmark_returns.index):
        raise ValueError("Strategy and benchmark dates must match exactly.")

    index = strategy_returns.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing or len(index) < 2):
        raise ValueError("Provide unique, ordered, timezone-aware daily dates.")

    utc_index = index.tz_convert("UTC")
    expected = pd.date_range(utc_index[0], utc_index[-1], freq="D")

    if (not utc_index.equals(expected) or not utc_index.equals(utc_index.normalize())):
        raise ValueError("Returns must cover consecutive UTC calendar days.")
    for name, value in (("block_days", block_days), ("repetitions", repetitions)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")
    if not 1 <= block_days <= len(index):
        raise ValueError("Block length must be between 1 and sample length.")
    if repetitions < 2:
        raise ValueError("At least two bootstrap repetitions are required.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must lie between 0 and 1.")
    if not np.isfinite(annualisation_days) or annualisation_days <= 0:
        raise ValueError("Annualisation days must be positive and finite.")

    values = np.column_stack([strategy_returns.to_numpy(dtype=float), benchmark_returns.to_numpy(dtype=float)])
    if not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("Returns must be finite and greater than -100%.")

    observed = _column_sharpes(values, annualisation_days)
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)

    for repetition in range(repetitions):
        indices = _circular_block_indices(len(values), block_days, rng)

        #both columns use identical sample data
        sample_sharpes = _column_sharpes(values[indices], annualisation_days)

        differences[repetition] = (sample_sharpes[0] - sample_sharpes[1])
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(differences, [alpha / 2, 1.0 - alpha / 2])

    summary = pd.Series({
        "observations": len(values),
        "block_days": block_days,
        "repetitions": repetitions,
        "strategy_sharpe": observed[0],
        "benchmark_sharpe": observed[1],
        "sharpe_difference": observed[0] - observed[1],
        "ci_lower": lower,
        "ci_upper": upper
    })

    draws = pd.Series(differences, name="bootstrap_sharpe_difference")
    return summary, draws