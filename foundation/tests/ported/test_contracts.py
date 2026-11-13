import numpy as np
import pandas as pd
import pytest

from foundation._research.contracts import CASH, all_cash, normalise_targets, validate_panel, validate_schedule, validate_targets
from foundation._research.synthetic import make_panel


def test_valid_panel_passes():
    panel = make_panel(assets=6, days=120)
    assert validate_panel(panel) is panel


@pytest.mark.parametrize("mutation", ["naive_timestamp", "duplicate", "negative_close", "cash_product"])
def test_panel_defects_are_rejected(mutation):
    panel = make_panel(assets=6, days=120).copy()
    if mutation == "naive_timestamp":
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(None)
    elif mutation == "duplicate":
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    elif mutation == "negative_close":
        panel.loc[0, "close"] = -1.0
    else:
        panel.loc[0, "product_id"] = CASH

    with pytest.raises(ValueError):
        validate_panel(panel)


def test_schedule_must_be_sorted_unique_utc_midnights():
    good = pd.date_range("2022-01-03", periods=10, freq="W-MON", tz="UTC")
    assert validate_schedule(good) is good

    with pytest.raises(ValueError):
        validate_schedule(good[::-1])
    with pytest.raises(ValueError):
        validate_schedule(good.tz_convert(None).tz_localize("Europe/London"))
    with pytest.raises(ValueError):
        validate_schedule(good + pd.Timedelta(hours=5))


def test_targets_must_be_long_only_and_sum_to_one():
    schedule = pd.date_range("2022-01-03", periods=4, freq="W-MON", tz="UTC")
    targets = all_cash(schedule, ["BTC-GBP", "ETH-GBP"])
    assert np.allclose(targets.sum(axis=1).to_numpy(), 1.0)

    short = targets.copy()
    short.loc[schedule[0], "BTC-GBP"] = -0.1
    with pytest.raises(ValueError):
        validate_targets(short)

    unbalanced = targets.copy()
    unbalanced.loc[schedule[0], "BTC-GBP"] = 0.5
    with pytest.raises(ValueError):
        validate_targets(unbalanced)


def test_normalise_targets_closes_the_cash_residual():
    schedule = pd.date_range("2022-01-03", periods=3, freq="W-MON", tz="UTC")
    raw = pd.DataFrame(0.0, index=schedule, columns=["BTC-GBP", "ETH-GBP"])
    raw.loc[schedule[1], "BTC-GBP"] = 0.4

    targets = normalise_targets(raw, schedule, ["BTC-GBP", "ETH-GBP"])
    assert targets.loc[schedule[1], CASH] == pytest.approx(0.6)
    assert targets.loc[schedule[0], CASH] == pytest.approx(1.0)


def test_normalise_targets_rejects_leverage():
    schedule = pd.date_range("2022-01-03", periods=2, freq="W-MON", tz="UTC")
    raw = pd.DataFrame(0.8, index=schedule, columns=["BTC-GBP", "ETH-GBP"])

    with pytest.raises(ValueError):
        normalise_targets(raw, schedule, ["BTC-GBP", "ETH-GBP"])
