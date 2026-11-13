import pandas as pd
import pytest

import foundation._research
from foundation._research.catalogue import CATALOGUE, build_queue, catalogue_frame, cluster_of, entry, validate_catalogue
from foundation._research.registry import available, build_targets, defaults_for
from foundation._research.synthetic import make_panel


def test_catalogue_is_well_formed():
    assert validate_catalogue() is CATALOGUE
    assert len(CATALOGUE) == 49
    assert len({record["id"] for record in CATALOGUE}) == 49


def test_every_registered_signal_is_catalogued():
    catalogued = {record["id"] for record in CATALOGUE}
    unknown = set(available()).difference(catalogued)
    assert not unknown, f"registered but not catalogued: {sorted(unknown)}"


def test_catalogue_frame_reports_status():
    frame = catalogue_frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc["vol_target", "status"] == "registered"
    assert frame.loc["micro_book_imbalance", "status"] == "planned"


def test_build_queue_is_the_first_tranche():
    queue = build_queue()
    #ten unique build priorities, eleven rows in the queue, so the two cross-sectional factor signals share a slot
    
    assert queue["build_priority"].nunique() == 10
    assert len(queue) == 11
    assert queue.index[0] == "vol_target"
    assert queue["build_priority"].is_monotonic_increasing


def test_cluster_lookup():
    assert cluster_of("xs_momentum") == "XS"
    assert cluster_of("xs_residual_momentum") == "XS_RES"
    with pytest.raises(ValueError):
        cluster_of("not_a_signal")


def test_entry_returns_a_copy():
    record = entry("vol_target")
    record["name"] = "mutated"
    assert entry("vol_target")["name"] != "mutated"


def test_defaults_are_frozen_parameters():
    assert defaults_for("trend_absolute_30d")["lookback_days"] == 30
    assert defaults_for("trend_absolute_fast")["lookback_days"] == 14
    assert defaults_for("trend_absolute_slow")["lookback_days"] == 90


def test_unknown_signal_is_rejected():
    panel = make_panel(assets=6, days=120)
    schedule = foundation._research.weekly_schedule(panel)
    with pytest.raises(ValueError):
        build_targets("not_a_signal", panel, schedule)
