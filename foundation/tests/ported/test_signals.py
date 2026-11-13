import numpy as np
import pandas as pd
import pytest

from foundation._research.benchmark import equal_weight_targets, universe_breadth
from foundation._research.contracts import CASH, validate_targets
from foundation._research.panel import weekly_schedule
from foundation._research.registry import available, build_targets, check_no_lookahead
from foundation._research.signals.overlay import apply_exposure
from foundation._research.synthetic import make_panel

IMPLEMENTED = "trend_absolute_30d", "trend_absolute_fast", "trend_absolute_slow", "trend_vol_scaled", "trend_agreement", "trend_agreement_vol", "xs_momentum", "xs_reversal", "xs_residual_momentum", "vol_target"


@pytest.fixture(scope="module")
def panel():
    return make_panel(assets=24, days=500, seed=11)


@pytest.fixture(scope="module")
def schedule(panel):
    return weekly_schedule(panel)


def test_all_expected_signals_are_registered():
    assert set(IMPLEMENTED).issubset(set(available()))


@pytest.mark.parametrize("signal_id", IMPLEMENTED)
def test_every_signal_satisfies_the_output_contract(signal_id, panel, schedule):
    targets = build_targets(signal_id, panel, schedule)
    validate_targets(targets, schedule)
    assert (targets.to_numpy(dtype=float) >= 0).all()
    assert np.allclose(targets.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


@pytest.mark.parametrize("signal_id", IMPLEMENTED)
def test_no_signal_looks_ahead(signal_id, panel, schedule):
    result = check_no_lookahead(signal_id, panel, schedule, samples=3, seed=5)
    assert result["passed"], f"{signal_id} leaked at {result['failures']}"


def test_lookahead_check_catches_a_planted_leak(panel, schedule):
    from foundation._research.contracts import normalise_targets
    from foundation._research.panel import pivot_closes
    from foundation._research.registry import register

    @register("leaky_test_signal")
    def leaky(panel, schedule):
        #deliberately use the future to hold the asset which rises the most after the decision

        closes = pivot_closes(panel)
        assets = sorted(panel["product_id"].unique())
        weights = pd.DataFrame(0.0, index=schedule, columns=assets)
        for stamp in schedule:
            future = closes.loc[closes.index >= stamp]
            if len(future) < 8:
                continue

            ahead = (future.iloc[7] / future.iloc[0]) - 1.0
            best = ahead.dropna()
            if not best.empty:
                weights.loc[stamp, best.idxmax()] = 1.0

        return normalise_targets(weights, schedule, assets)

    result = check_no_lookahead("leaky_test_signal", panel, schedule, samples=3, seed=5)
    assert not result["passed"]


def test_absolute_momentum_holds_cash_in_a_falling_market():
    panel = make_panel(assets=22, days=400, seed=2, drift=-0.004, market_volatility=0.01, idiosyncratic_volatility=0.002)
    schedule = weekly_schedule(panel)
    targets = build_targets("trend_absolute_30d", panel, schedule)

    settled = targets.iloc[len(targets) // 2:]
    assert settled[CASH].mean() > 0.8


def test_cross_sectional_momentum_is_fully_invested_when_the_universe_is_active(panel, schedule):
    targets = build_targets("xs_momentum", panel, schedule)
    breadth = universe_breadth(panel, schedule)
    #with any non-empty universe traded, the first weeks are active before a 90-day return exists

    warmed = panel["timestamp"].min() + pd.Timedelta(days=91)
    active = breadth.index[breadth["active"] & (breadth.index >= warmed)]

    invested = 1.0 - targets.loc[active, CASH]
    assert (invested > 0.99).all()


def test_cross_sectional_momentum_holds_the_expected_count(panel, schedule):
    targets = build_targets("xs_momentum", panel, schedule, {"top_fraction": 0.20})
    breadth = universe_breadth(panel, schedule)
    stamp = breadth.index[breadth["active"]][-1]

    held = (targets.loc[stamp].drop(CASH) > 0).sum()
    assert held == int(np.ceil(0.20 * breadth.loc[stamp, "eligible_assets"]))


def test_reversal_and_momentum_disagree(panel, schedule):
    momentum = build_targets("xs_momentum", panel, schedule, {"lookback_days": 7, "top_fraction": 0.20})
    reversal = build_targets("xs_reversal", panel, schedule, {"lookback_days": 7, "top_fraction": 0.20})

    overlap = (momentum.drop(columns=CASH) > 0) & (reversal.drop(columns=CASH) > 0)
    assert overlap.to_numpy().sum() == 0


def test_benchmark_is_equal_weight_over_eligible_members(panel, schedule):
    targets = equal_weight_targets(panel, schedule)
    breadth = universe_breadth(panel, schedule)
    stamp = breadth.index[breadth["active"]][-1]

    held = targets.loc[stamp].drop(CASH)
    positive = held[held > 0]
    assert len(positive) == breadth.loc[stamp, "eligible_assets"]
    assert positive.std(ddof=0) == pytest.approx(0.0, abs=1e-12)


def test_apply_exposure_scales_without_breaking_the_contract(panel, schedule):
    targets = build_targets("xs_momentum", panel, schedule)
    exposure = pd.Series(0.5, index=schedule)
    scaled = apply_exposure(targets, exposure)

    validate_targets(scaled, schedule)
    assert (scaled[CASH] >= targets[CASH] - 1e-12).all()


def test_vol_target_reduces_exposure_when_volatility_rises():
    calm = make_panel(assets=22, days=400, seed=4, market_volatility=0.01, idiosyncratic_volatility=0.005)
    wild = make_panel(assets=22, days=400, seed=4, market_volatility=0.06, idiosyncratic_volatility=0.03)

    calm_targets = build_targets("vol_target", calm, weekly_schedule(calm))
    wild_targets = build_targets("vol_target", wild, weekly_schedule(wild))

    assert calm_targets[CASH].mean() < wild_targets[CASH].mean()


def _two_asset_panel(days=400, drift=0.002, seed=3, volatility=0.03):
    #any non-empty universe must trade
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=days, freq="D", tz="UTC")
    rows = []
    for name in ("BTC-GBP", "ETH-GBP"):
        closes = 100.0 * np.exp(np.cumsum(rng.normal(drift, volatility, size=days)))
        rows.append(pd.DataFrame({"product_id": name, "timestamp": dates, "close": closes, "eligible": True}))
    return pd.concat(rows, ignore_index=True)


def test_two_asset_universe_is_traded():
    panel = _two_asset_panel()
    targets = build_targets("trend_absolute_30d", panel, weekly_schedule(panel))
    assert (targets[CASH] < 1.0).any()
    assert targets.drop(columns=CASH).max().max() <= 0.5 + 1e-12


def test_agreement_vol_is_capped_and_scaled():
    #steady uptrend with ~10% annualised volatility, so every lookback agrees
    panel = _two_asset_panel(drift=0.01, volatility=0.005)
    schedule = weekly_schedule(panel)
    full = build_targets("trend_agreement_vol", panel, schedule, {"target_volatility": 10.0})
    scaled = build_targets("trend_agreement_vol", panel, schedule, {"target_volatility": 0.05})
    late = schedule[-10:]

    #steady uptrend, every lookback agrees, so unconstrained slot is 1/N
    assert np.allclose(full.loc[late].drop(columns=CASH).to_numpy(), 0.5)
    assert (scaled.loc[late].drop(columns=CASH).to_numpy() < 0.5).all()


def test_agreement_vol_is_flat_in_a_downtrend():
    panel = _two_asset_panel(drift=-0.01, volatility=0.005)
    schedule = weekly_schedule(panel)
    targets = build_targets("trend_agreement_vol", panel, schedule)
    assert (targets.loc[schedule[-10:], CASH] == 1.0).all()


def test_no_trade_band_reduces_turnover():
    panel = _two_asset_panel(drift=0.0005, volatility=0.04)
    schedule = weekly_schedule(panel)
    loose = build_targets("trend_agreement_vol", panel, schedule, {"no_trade_band": 0.0})
    banded = build_targets("trend_agreement_vol", panel, schedule, {"no_trade_band": 0.10})
    turnover = lambda frame: frame.drop(columns=CASH).diff().abs().sum().sum()
    assert turnover(banded) < turnover(loose)
    assert check_no_lookahead("trend_agreement_vol", panel, schedule, {"no_trade_band": 0.10}, samples=3)["passed"]


def test_universe_cap_is_never_applied_alphabetically():
    import pytest
    from foundation._research.benchmark import eligible_universe
    from foundation._research.synthetic import make_panel
    panel = make_panel(assets=35, days=300)
    stamp = panel["timestamp"].max()
    with pytest.raises(ValueError, match="liquidity"):
        eligible_universe(panel, stamp, max_assets=30)
