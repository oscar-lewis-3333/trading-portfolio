import numpy as np
import pandas as pd

from .metrics import block_bootstrap_ir, information_ratio, shrunk_information_ratio
from .trials import record_trial

MAX_ABS_CORRELATION = 0.40
MIN_IR_LOWER_BOUND = -0.25
MAX_COST_SHARE = 0.25


def align_active(streams):
    #a frame of active returns, aligned to fixed, shared dates

    if not isinstance(streams, dict) or not streams:
        raise ValueError("Supply a non-empty dict of signal_id to active return Series.")

    frame = pd.DataFrame({key: value.astype(float) for key, value in streams.items()}).dropna()
    if frame.empty:
        raise ValueError("The active streams share no common dates.")
    if len(frame) < 60:
        raise ValueError("At least 60 shared observations are needed to estimate correlation.")

    return frame.sort_index(axis=1)


def correlation_matrix(streams):
    #correlation matrix for active returns

    return align_active(streams).corr()


def stressed_correlation(streams, benchmark_returns, tail=0.10):
    #correlations rise in crashes, so size for the stressed matrix, not the normal one
    if not 0 < tail < 0.5:
        raise ValueError("tail must lie in (0, 0.5).")

    frame = align_active(streams)
    benchmark = benchmark_returns.astype(float).reindex(frame.index).dropna()
    if len(benchmark) < 60:
        raise ValueError("The benchmark does not cover enough shared dates.")

    cutoff = benchmark.quantile(tail)
    worst = frame.loc[benchmark.index[benchmark <= cutoff]]
    if len(worst) < 20:
        raise ValueError("Too few stressed observations to estimate correlation.")

    return worst.corr()


def ledoit_wolf_intensity(frame):

    values = frame.to_numpy(dtype=float)
    t, n = values.shape
    centred = values - values.mean(axis=0)
    sample = centred.T @ centred / t
    std = np.sqrt(np.diag(sample))

    corr = sample / np.outer(std, std)
    rbar = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    target = rbar * np.outer(std, std)
    np.fill_diagonal(target, np.diag(sample))

    y = centred ** 2
    pi_mat = y.T @ y / t - sample ** 2
    pi_hat = pi_mat.sum()
    theta_ii = np.einsum("ti,tj->ij", centred ** 3, centred) / t - np.diag(sample)[:, None] * sample
    rho_off = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                rho_off += rbar / 2 * (np.sqrt(sample[j, j] / sample[i, i]) * theta_ii[i, j] + np.sqrt(sample[i, i] / sample[j, j]) * theta_ii[j, i])
    rho_hat = np.trace(pi_mat) + rho_off
    gamma_hat = ((sample - target) ** 2).sum()
    if gamma_hat <= 0:
        return 1.0
    kappa = (pi_hat - rho_hat) / gamma_hat

    return float(min(1.0, max(0.0, kappa / t)))


def shrunk_covariance(streams, shrinkage=None):
    #shrinkage towards a constant-correlation target, with the Ledoit-Wolf optimal intensity if no shrinkage.

    frame = align_active(streams)
    sample = frame.cov()
    names = sample.columns
    deviations = np.sqrt(np.diag(sample.to_numpy(dtype=float)))
    if not np.isfinite(deviations).all() or (deviations <= 0).any():
        raise ValueError("Every stream needs positive variance.")

    correlation = frame.corr().to_numpy(dtype=float)
    off_diagonal = correlation[~np.eye(len(names), dtype=bool)]
    average = float(off_diagonal.mean()) if off_diagonal.size else 0.0

    target_corr = np.full_like(correlation, average)
    np.fill_diagonal(target_corr, 1.0)
    target = target_corr * np.outer(deviations, deviations)

    if shrinkage is None:
        shrinkage = ledoit_wolf_intensity(frame) if len(names) > 1 else 0.0
    if isinstance(shrinkage, bool) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1].")

    blended = (1.0 - shrinkage) * sample.to_numpy(dtype=float) + shrinkage * target

    return pd.DataFrame(blended, index=names, columns=names)


def equal_risk_contribution(covariance, iterations=5000, tolerance=1e-10):
    #risk parity allocator, each portfolio slice contributes the same risk

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The covariance must be square.")
    if not np.isfinite(matrix).all():
        raise ValueError("The covariance must be finite.")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-10):
        raise ValueError("The covariance must be symmetric.")

    size = matrix.shape[0]
    weights = np.full(size, 1.0 / size)

    for _ in range(iterations):
        marginal = matrix @ weights
        if not np.isfinite(marginal).all() or (marginal <= 0).any():
            raise ValueError("Risk contributions must be positive; check the covariance.")

        contributions = weights * marginal
        target = contributions.sum() / size
        updated = weights * np.sqrt(target / contributions)
        updated = updated / updated.sum()

        if np.abs(updated - weights).max() < tolerance:
            weights = updated
            break
        weights = updated

    names = covariance.columns if isinstance(covariance, pd.DataFrame) else range(size)

    return pd.Series(weights, index=names, name="weight")


def portfolio_ir(streams, weights=None, periods_per_year=365):
    #information ratio of the (weighted, active) portfolio

    frame = align_active(streams)
    if weights is None:
        covariance = shrunk_covariance(streams)
        weights = equal_risk_contribution(covariance)

    aligned = pd.Series(weights).reindex(frame.columns)
    if aligned.isna().any():
        raise ValueError("Weights must cover every stream.")

    combined = (frame * aligned).sum(axis=1)

    return information_ratio(combined, periods_per_year)


def marginal_ir(streams, candidate_id, periods_per_year=365):
    #if we remove IR, what do we lose?
    if candidate_id not in streams:
        raise ValueError(f"'{candidate_id}' is not among the streams.")
    if len(streams) < 2:
        raise ValueError("Marginal contribution needs at least two streams.")

    without = {key: value for key, value in streams.items() if key != candidate_id}

    return portfolio_ir(streams, periods_per_year=periods_per_year) - portfolio_ir(without, periods_per_year=periods_per_year)


def screen_candidate(candidate_id, candidate_active, accepted, ledger_summary=None,  max_abs_correlation=MAX_ABS_CORRELATION, periods_per_year=365,
                     min_ir_lower_bound=MIN_IR_LOWER_BOUND, bootstrap_resamples=10000,
                     block_days=28, trial_log=None, trial_params=None, prior_sd=0.30):
    
    #apply four rules to a candidate's returns, if all pass, accepted for further research.
    reasons = []
    streams = dict(accepted)
    streams[candidate_id] = candidate_active

    own_ir = information_ratio(candidate_active, periods_per_year)
    if not np.isfinite(own_ir):
        reasons.append("The candidate's information ratio is undefined.")

    #1: the one-sided 95% block-bootstrap lower bound on IR
    ir_lower = None
    try:
        interval = block_bootstrap_ir(candidate_active, block_days=block_days, resamples=bootstrap_resamples, periods_per_year=periods_per_year)
        ir_lower = interval["lower"]
        if ir_lower <= min_ir_lower_bound:
            reasons.append(f"IR lower bound {ir_lower:+.2f} is not above {min_ir_lower_bound:+.2f}.")

    except ValueError as exc:
        reasons.append(f"IR lower bound could not be estimated: {exc}")

    active = candidate_active.dropna()
    years = (len(active) / periods_per_year) if len(active) else 0.0
    shrunk = shrunk_information_ratio(own_ir, years, prior_sd) if years else float("nan")

    correlations = {}
    if accepted:
        frame = align_active(streams)
        column = frame.corr()[candidate_id].drop(candidate_id)
        correlations = column.to_dict()
        worst = column.abs().max()
        if worst > max_abs_correlation:
            offender = column.abs().idxmax()
            reasons.append(f"Correlation {column[offender]:+.2f} with '{offender}' exceeds {max_abs_correlation:.2f}.")

    contribution = marginal_ir(streams, candidate_id, periods_per_year) if len(streams) > 1 else own_ir
    if contribution <= 0:
        reasons.append(f"Marginal information ratio contribution is {contribution:+.3f}.")

    cost_share = None
    if ledger_summary is not None:
        gross = float(ledger_summary.get("gross_active_return", np.nan))
        cost = float(ledger_summary.get("cost_fraction_total", np.nan))
        if np.isfinite(gross) and np.isfinite(cost) and gross > 0:
            cost_share = cost / gross
            if cost_share > MAX_COST_SHARE:
                reasons.append(f"Costs consume {cost_share:.0%} of gross active return, above {MAX_COST_SHARE:.0%}.")

    result = {"candidate_id": candidate_id,
        "information_ratio": float(own_ir),
        "ir_lower_bound": ir_lower,
        "shrunk_ir": shrunk,
        "correlations": correlations,
        "marginal_ir": float(contribution),
        "cost_share": cost_share,
        "accepted": not reasons,
        "reasons": reasons}
    
    if trial_log is not None:
        record_trial(trial_log, result, trial_params)

    return result
