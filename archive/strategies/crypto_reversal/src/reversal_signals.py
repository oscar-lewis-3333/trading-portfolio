"""Rank eligible assets by their completed one-day price return."""
import numpy as np
import pandas as pd

from preparation import add_history_segments


def rank_reversal_candidates(eligibility_panel, daily_universe, protocol):
    #rank frozen eligible members using completed past candles only
    
    days = protocol["signal"]["formation_days"]
    fraction = protocol["signal"]["selection_fraction"]
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError("formation_days must be a positive integer.")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("selection_fraction must be in (0, 1].")
    if protocol["signal"]["negative_return_required"]:
        raise ValueError("This version does not apply a negative-return filter.")

    history = add_history_segments(eligibility_panel)
    if not np.isfinite(history["close"]).all() or history["close"].le(0).any():
        raise ValueError("Close prices must be finite and positive.")
    
    history["formation_reference_close"] = history.groupby(["product_id", "segment_id"])["close"].shift(days)
    history["formation_return"] = history["close"] / history["formation_reference_close"] - 1
    history["decision_at"] = history["signal_available_at"]

    keys = ["product_id", "decision_at"]
    if daily_universe.duplicated(keys).any():
        raise ValueError("Duplicate decision members.")
    
    members = daily_universe[keys + ["timestamp", "close", "liquidity_rank"]].copy()
    ranked = members.merge(history[keys + ["formation_reference_close", "formation_return", "close"]], on=keys, how="left", validate="one_to_one", suffixes=("", "_history"))
    if not ranked["timestamp"].add(pd.Timedelta(days=1)).eq(ranked["decision_at"]).all():
        raise ValueError("Member candles must complete at the decision time.")
    if not ranked["close"].eq(ranked["close_history"]).all():
        raise ValueError("Membership prices do not match the source history.")
    if not np.isfinite(ranked["formation_return"]).all():
        raise ValueError("A member has insufficient contiguous formation history.")
    
    ranked = ranked.drop(columns="close_history")
    ranked["reversal_score"] = -ranked["formation_return"]
    ranked = ranked.sort_values(["decision_at", "reversal_score", "product_id"], ascending=[True, False, True]).reset_index(drop=True)
    ranked["reversal_rank"] = ranked.groupby("decision_at").cumcount() + 1
    ranked["universe_size"] = ranked.groupby("decision_at")["product_id"].transform("size")

    ranked["selection_count"] = np.ceil(fraction * ranked["universe_size"]).astype("int64")
    ranked["selected"] = ranked["reversal_rank"].le(ranked["selection_count"])
    ranked["reversal_weight"] = ranked["selected"].astype(float) / ranked["selection_count"]
    ranked["benchmark_weight"] = 1.0 / ranked["universe_size"]
    return ranked
