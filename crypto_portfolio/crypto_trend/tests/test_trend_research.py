import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[2] / "signal_factory"))
import trend_research  # noqa: E402

PROTOCOL = ROOT / "research" / "protocol_v1.json"


def test_frozen_protocol_loads():
    protocol = trend_research.load_protocol(PROTOCOL)
    assert protocol["universe"] == ["BTC-GBP", "ETH-GBP"]


def test_edited_protocol_is_refused(tmp_path):
    record = json.loads(PROTOCOL.read_text())
    record["protocol"]["cash"]["primary"] = 0.05
    edited = tmp_path / "protocol.json"
    edited.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="edited"):
        trend_research.load_protocol(edited)


def test_grid_is_the_frozen_eighteen_in_order():
    configs = trend_research.sweep_configurations(trend_research.load_protocol(PROTOCOL))
    assert list(configs.index) == [f"C{n:02d}" for n in range(1, 19)]
    first = configs.loc["C01"]
    assert first["lookbacks"] == (7, 14, 30, 60, 90, 180)
    assert first["target_volatility"] == 0.40 and first["no_trade_band"] == 0.0
    assert configs.loc["C05", "target_volatility"] == trend_research.NO_CAP_TARGET


def _fake_run(daily_return, turnover, days):
    import pandas as pd
    index = pd.date_range("2019-07-01", periods=days, freq="D", tz="UTC")
    rng = __import__("numpy").random.default_rng(int(daily_return * 1e6) % 1000)
    ledger = pd.DataFrame({"net_return": daily_return + rng.normal(0, 0.01, days),
                           "one_way_turnover": turnover}, index=index)
    targets = pd.DataFrame({"BTC-GBP": 0.5, "CASH": 0.5}, index=index[index.dayofweek == 0])
    return targets, ledger


def test_annual_selection_uses_only_earlier_data_and_breaks_ties_on_turnover():
    protocol = trend_research.load_protocol(PROTOCOL)
    protocol = dict(protocol, selection=dict(protocol["selection"], years_selected=[2021]))
    good = _fake_run(0.004, 0.02, 900)
    runs = {"C01": _fake_run(0.0005, 0.01, 900), "C02": good, "C03": (good[0], good[1].assign(one_way_turnover=0.01))}
    #C02 and C03 have identical returns; C03 trades less, so it wins the tie
    choices = trend_research.select_annual(protocol, runs)
    assert choices.loc[2021, "chosen"] == "C03"

    #changing returns after the training end must not change the choice
    late = runs["C01"][1].copy()
    late.loc[late.index >= "2021-01-01", "net_return"] = 0.5
    runs["C01"] = (runs["C01"][0], late)
    assert trend_research.select_annual(protocol, runs).loc[2021, "chosen"] == "C03"


def test_walk_forward_takes_each_years_rows_from_its_choice():
    import pandas as pd
    a, b = _fake_run(0.001, 0.0, 900), _fake_run(0.001, 0.0, 900)
    b = (b[0].assign(**{"BTC-GBP": 0.2, "CASH": 0.8}), b[1])
    choices = pd.DataFrame({"chosen": ["C01", "C02"]}, index=pd.Index([2020, 2021], name="year"))
    combined = trend_research.walk_forward_targets(choices, {"C01": a, "C02": b})
    assert combined.index.year.min() == 2020
    assert (combined.loc[combined.index.year == 2020, "BTC-GBP"] == 0.5).all()
    assert (combined.loc[combined.index.year == 2021, "BTC-GBP"] == 0.2).all()
