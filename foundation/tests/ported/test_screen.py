import numpy as np
import pandas as pd
import pytest

from foundation._research.screen import correlation_matrix, equal_risk_contribution, marginal_ir, portfolio_ir, screen_candidate, shrunk_covariance, stressed_correlation


def _streams(count=3, days=500, correlation=0.0, seed=0):
    dates = pd.date_range("2022-01-03", periods=days, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.01, days)
    streams = {}
    for index in range(count):
        own = rng.normal(0.0003, 0.01, days)
        blended = correlation * common + np.sqrt(max(1.0 - correlation ** 2, 0.0)) * own
        streams[f"s{index}"] = pd.Series(blended, index=dates)

    return streams


def test_correlation_matrix_is_symmetric_with_unit_diagonal():
    matrix = correlation_matrix(_streams())
    assert np.allclose(np.diag(matrix.to_numpy()), 1.0)
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)


def test_equal_risk_contribution_equalises_contributions():
    covariance = pd.DataFrame([[0.04, 0.00, 0.00],
                               [0.00, 0.01, 0.00],
                               [0.00, 0.00, 0.09]], index=list("abc"), columns=list("abc"))
    weights = equal_risk_contribution(covariance)

    contributions = weights * (covariance.to_numpy() @ weights)
    assert np.allclose(contributions.to_numpy(), contributions.to_numpy()[0], rtol=1e-6)
    assert weights.sum() == pytest.approx(1.0)
    #least volatile source is b, hence should have the largest weight

    assert weights.idxmax() == "b"


def test_shrunk_covariance_stays_positive_definite():
    covariance = shrunk_covariance(_streams(count=5, days=200, seed=3))
    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())
    assert (eigenvalues > 0).all()


def test_uncorrelated_streams_beat_correlated_ones_on_portfolio_ir():
    independent = portfolio_ir(_streams(count=6, correlation=0.0, seed=1))
    correlated = portfolio_ir(_streams(count=6, correlation=0.9, seed=1))
    assert independent > correlated


def test_marginal_ir_is_defined_for_each_stream():
    streams = _streams(count=4, seed=2)
    for name in streams:
        assert np.isfinite(marginal_ir(streams, name))


def test_screen_rejects_a_near_duplicate():
    streams = _streams(count=2, seed=5)
    accepted = {"s0": streams["s0"]}
    duplicate = streams["s0"] * 1.01

    result = screen_candidate("copy", duplicate, accepted)
    assert not result["accepted"]
    assert any("Correlation" in reason for reason in result["reasons"])


def test_screen_accepts_an_uncorrelated_positive_stream():
    dates = pd.date_range("2022-01-03", periods=600, freq="D", tz="UTC")
    rng = np.random.default_rng(9)
    accepted = {"s0": pd.Series(rng.normal(0.0004, 0.01, 600), index=dates)}
    candidate = pd.Series(rng.normal(0.0006, 0.01, 600), index=dates)

    result = screen_candidate("candidate", candidate, accepted)
    assert result["accepted"], result["reasons"]
    assert abs(list(result["correlations"].values())[0]) < 0.4


def test_screen_flags_costly_signals():
    dates = pd.date_range("2022-01-03", periods=600, freq="D", tz="UTC")
    rng = np.random.default_rng(11)
    candidate = pd.Series(rng.normal(0.0005, 0.01, 600), index=dates)

    result = screen_candidate("candidate", candidate, {}, ledger_summary={"gross_active_return": 0.10, "cost_fraction_total": 0.08})
    assert not result["accepted"]
    assert any("Costs consume" in reason for reason in result["reasons"])


def test_stressed_correlation_uses_the_worst_benchmark_days():
    dates = pd.date_range("2022-01-03", periods=600, freq="D", tz="UTC")
    rng = np.random.default_rng(13)
    benchmark = pd.Series(rng.normal(0.0, 0.02, 600), index=dates)

    #independent streams should have low correlation, but same move on the worst benchmark days
    tail = benchmark <= benchmark.quantile(0.10)
    common = pd.Series(rng.normal(0.0, 0.01, 600), index=dates)
    streams = {"a": pd.Series(rng.normal(0.0, 0.01, 600), index=dates).where(~tail, common),
               "b": pd.Series(rng.normal(0.0, 0.01, 600), index=dates).where(~tail, common)}

    calm = correlation_matrix(streams).loc["a", "b"]
    stressed = stressed_correlation(streams, benchmark).loc["a", "b"]
    assert stressed > calm


def test_screen_rejects_a_stream_whose_ir_lower_bound_is_too_low():
    dates = pd.date_range("2022-01-03", periods=600, freq="D", tz="UTC")
    rng = np.random.default_rng(21)
    candidate = pd.Series(rng.normal(-0.0008, 0.01, 600), index=dates)

    result = screen_candidate("weak", candidate, {}, bootstrap_resamples=2000)
    assert result["ir_lower_bound"] < -0.25
    assert any("IR lower bound" in reason for reason in result["reasons"])


def test_ir_lower_bound_sits_below_the_estimate():
    from foundation._research.metrics import block_bootstrap_ir
    dates = pd.date_range("2022-01-03", periods=700, freq="D", tz="UTC")
    active = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, 700), index=dates)
    interval = block_bootstrap_ir(active, resamples=2000)
    assert interval["lower"] < interval["estimate"] < interval["upper"]


def test_shrunk_ir_is_pulled_towards_zero_more_for_short_samples():
    from foundation._research.metrics import shrunk_information_ratio
    two_years = shrunk_information_ratio(1.0, 2.0)
    ten_years = shrunk_information_ratio(1.0, 10.0)
    assert 0 < two_years < ten_years < 1.0
    assert two_years == pytest.approx(0.09 / (0.09 + 0.5))


def test_ledoit_wolf_intensity_falls_as_observations_grow():
    from foundation._research.screen import ledoit_wolf_intensity
    rng = np.random.default_rng(4)
    loadings = rng.uniform(0.2, 0.9, 6)
    def draw(t):
        common = rng.normal(size=(t, 1))
        return pd.DataFrame(common * loadings + rng.normal(size=(t, 6)) * np.linspace(0.5, 2.0, 6))
    short, long = ledoit_wolf_intensity(draw(80)), ledoit_wolf_intensity(draw(4000))
    assert 0 <= long < short <= 1


def test_every_screen_is_logged_when_asked(tmp_path):
    from foundation._research.trials import load_trials, trial_sharpes
    streams = _streams(count=2, seed=6)
    log = tmp_path / "trials.jsonl"
    screen_candidate("a", streams["s0"], {}, bootstrap_resamples=500, trial_log=log, trial_params={"lookback": 30})
    screen_candidate("a", streams["s1"], {}, bootstrap_resamples=500, trial_log=log, trial_params={"lookback": 60})
    trials = load_trials(log)
    assert [t["params"]["lookback"] for t in trials] == [30, 60]
    assert len(trial_sharpes(log)) == 2


def test_deflated_sharpe_falls_as_trials_accumulate():
    from foundation._research.metrics import deflated_sharpe_ratio
    dates = pd.date_range("2022-01-03", periods=730, freq="D", tz="UTC")
    active = pd.Series(np.random.default_rng(8).normal(0.0006, 0.01, 730), index=dates)
    few = deflated_sharpe_ratio(active, [1.0, 0.2, -0.3])
    many = deflated_sharpe_ratio(active, list(np.random.default_rng(1).normal(0, 0.6, 200)))
    assert 0 <= many["deflated_sharpe"] < few["deflated_sharpe"] <= 1
