import numpy as np
import pandas as pd


def bootstrap_sweep_excess(excess_returns, *, block_length=63, n_bootstrap=10_000, alpha=0.05, seed=42):
    #test positive mean net excess returns across a config grid
    if (not isinstance(excess_returns, pd.DataFrame) or excess_returns.empty or not isinstance(excess_returns.index, pd.DatetimeIndex)
        or not excess_returns.index.is_unique or not excess_returns.index.is_monotonic_increasing or not excess_returns.columns.is_unique):
        raise ValueError("Expected a return matrix with unique columns and sorted, unique dates.")

    for value in (block_length, n_bootstrap):
        if (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError("Block length and bootstrap count must be positive integers.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one.")

    values = excess_returns.to_numpy(dtype=float)
    n_sessions, n_configs = values.shape
    if n_sessions < 2 * block_length:
        raise ValueError("Provide at least two blocks of observations.")
    if not np.isfinite(values).all():
        raise ValueError("Excess returns must be finite.")

    observed_mean = values.mean(axis=0)

    #impose zero mean. H0: mean excess <=0
    centred = values - observed_mean

    rng = np.random.default_rng(seed)
    n_blocks = (n_sessions + block_length - 1) // block_length
    offsets = np.arange(block_length)
    bootstrap_errors = np.empty((n_bootstrap, n_configs))

    for draw in range(n_bootstrap):
        starts = rng.integers(0, n_sessions, size=n_blocks)
        indices = ((starts[:, None] + offsets) % n_sessions).ravel()[:n_sessions]

        #use identical sampled dates for each config
        bootstrap_errors[draw] = centred[indices].mean(axis=0)

    lower_error, upper_error = np.quantile(bootstrap_errors, [alpha / 2, 1 - alpha / 2], axis=0)
    #one sided test, with H_1: positive mean excess return
    p_values = (1 + (bootstrap_errors >= observed_mean).sum(axis=0)) / (n_bootstrap + 1)

    #holm correction across every config
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(p_values[order] * (n_configs - np.arange(n_configs)))
    p_holm = np.empty(n_configs)
    p_holm[order] = np.minimum(adjusted_sorted, 1.0)

    return pd.DataFrame({
        "mean_daily_net_excess_bps": observed_mean * 10_000,
        "lower_ci_bps": (observed_mean - upper_error) * 10_000,
        "upper_ci_bps": (observed_mean - lower_error) * 10_000,
        "p_one_sided": p_values,
        "p_holm": p_holm,
        "reject_holm": (observed_mean > 0) & (p_holm <= alpha)
    }, index=excess_returns.columns).rename_axis("configuration_id")