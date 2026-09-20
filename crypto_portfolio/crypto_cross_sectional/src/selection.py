import numpy as np
import pandas as pd


def build_weekly_universe(policy_panel, max_assets=30, min_assets=20):
    #select a liquidity-ranked universe using monday data
    for name, value in (("max_assets", max_assets), ("min_assets", min_assets)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if min_assets > max_assets:
        raise ValueError("min_assets cannot exceed max_assets.")

    required = {
        "product_id",
        "signal_available_at",
        "universe_candidate",
        "median_dollar_volume"
    }

    missing = required.difference(policy_panel.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = policy_panel.copy()
    if frame.empty:
        raise ValueError("The policy panel is empty.")
    
    times = frame["signal_available_at"]
    if not isinstance(times.dtype, pd.DatetimeTZDtype):
        raise ValueError("signal_available_at must be timezone-aware.")

    frame["decision_at"] = times.dt.tz_convert("UTC")
    decisions = frame["decision_at"]
    if decisions.isna().any() or decisions.ne(decisions.dt.normalize()).any():
        raise ValueError("Decision timestamps must be valid UTC midnights.")
    if frame["product_id"].isna().any():
        raise ValueError("Missing product IDs.")
    if frame.duplicated(["product_id", "decision_at"]).any():
        raise ValueError("Duplicate product/decision observations.")
    
    eligible = frame["universe_candidate"]
    if not pd.api.types.is_bool_dtype(eligible) or eligible.isna().any():
        raise ValueError("universe_candidate must contain non-null booleans.")

    #preserve every monday, including weeks with no eligible assets
    schedule = pd.date_range(decisions.min(), decisions.max(), freq="W-MON", name="decision_at")

    candidates = frame.loc[decisions.dt.dayofweek.eq(0) & eligible].copy()
    liquidity = candidates["median_dollar_volume"].to_numpy(dtype=float)
    if not np.isfinite(liquidity).all() or (liquidity <= 0).any():
        raise ValueError("Eligible assets need finite, positive liquidity.")

    candidates = candidates.sort_values(["decision_at", "median_dollar_volume", "product_id"], ascending=[True, False, True])
    candidates["liquidity_rank"] = candidates.groupby("decision_at").cumcount() + 1
    summary = candidates.groupby("decision_at").size().reindex(schedule, fill_value=0).rename("eligible_assets").to_frame()
    summary["breadth_ok"] = summary["eligible_assets"].ge(min_assets)
    summary["universe_size"] = summary["eligible_assets"].clip(upper=max_assets).where(summary["breadth_ok"], 0)
    candidates["breadth_ok"] = candidates["decision_at"].map(summary["breadth_ok"])
    members = candidates.loc[candidates["breadth_ok"] & candidates["liquidity_rank"].le(max_assets)].copy()

    return members.reset_index(drop=True), summary

def build_momentum_targets(weekly_universe, weekly_summary, top_fraction=0.20):
    #build decision-time weights for momentum (and matched benchmark)
    if (isinstance(top_fraction, bool) or not np.isfinite(top_fraction) or not 0 < top_fraction <= 1):
        raise ValueError("top_fraction must be in (0, 1].")

    frame = weekly_universe.copy()
    schedule = weekly_summary.index

    if schedule.has_duplicates or not schedule.is_monotonic_increasing:
        raise ValueError("The decision schedule must be unique and sorted.")
    if frame[["decision_at", "product_id"]].isna().any().any():
        raise ValueError("Missing decision times or product IDs.")
    if frame.duplicated(["decision_at", "product_id"]).any():
        raise ValueError("Duplicate assets within a weekly universe.")
    if not frame["decision_at"].isin(schedule).all():
        raise ValueError("Universe contains decisions outside the schedule.")

    actual_sizes = frame.groupby("decision_at").size().reindex(schedule, fill_value=0)
    if not actual_sizes.eq(weekly_summary["universe_size"]).all():
        raise ValueError("Universe membership does not match the summary.")
    momentum = frame["momentum_return"].to_numpy(dtype=float)
    if not np.isfinite(momentum).all():
        raise ValueError("Every universe member needs finite momentum.")

    #product ID breaks ties
    frame = frame.sort_values(["decision_at", "momentum_return", "product_id"], ascending=[True, False, True])
    grouped = frame.groupby("decision_at")
    frame["momentum_rank"] = grouped.cumcount() + 1
    frame["universe_size"] = grouped["product_id"].transform("size")
    frame["selection_count"] = np.ceil(top_fraction * frame["universe_size"]).astype(int)

    #we do not require positive momentum
    frame["selected"] = (frame["momentum_rank"] <= frame["selection_count"])
    frame["momentum_weight"] = np.where(frame["selected"], 1.0 / frame["selection_count"], 0.0)
    frame["benchmark_weight"] = 1.0 / frame["universe_size"]

    assets = sorted(frame["product_id"].unique())

    def to_weight_matrix(weight_column):
        weights = (frame.pivot(index="decision_at",
            columns="product_id",
            values=weight_column
            ).reindex(index=schedule, columns=assets).fillna(0.0))
        
        #explicit cash decisions preverse weeks without a valid universe
        weights["CASH"] = actual_sizes.eq(0).astype(float)
        weights.columns.name = None
        if not np.allclose(weights.sum(axis=1).to_numpy(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("Target weights must sum to one.")

        return weights

    return frame.reset_index(drop=True), to_weight_matrix("momentum_weight"), to_weight_matrix("benchmark_weight")