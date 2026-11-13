#stages for non-ML notebook, to unclog space.

from pathlib import Path
import hashlib
import json
import pickle
from importlib.metadata import version

import numpy as np
import pandas as pd

import reversal_research_data as data
import reversal_backtest as backtest
import reversal_features as features
import reversal_validation as validation
import reversal_walk_forward as walk_forward
from reversal_preparation import prepare_uk_extension

INFERENCE_COLUMNS = ["periods", "mean_bps", "hac_se_bps", "ci_lower_bps", "ci_upper_bps", "p_one_sided", "p_holm", "reject_holm"]
CHOICE_COLUMNS = ["selection_date", "test_start", "test_end", "configuration_id", "formation_sessions", "top_frac", "holding_sessions"]


def _hash_frames(*frames):
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        digest.update(str(getattr(frame, "columns", frame.name if isinstance(frame, pd.Series) else None)).encode())
    return digest.hexdigest()


def cached_stage(inputs, name, compute, *, use_cache=True, parameters=None):
    #cache by exact inputs, parameters, runtime and all research code
    project = inputs["project"]
    sources = sorted(path for path in (project / "src").rglob("*.py") if not path.name.startswith(("ml_", "reversal_ml_")))
    identity = {"inputs": inputs["fingerprint"], "parameters": parameters,
                "pandas": pd.__version__, "numpy": np.__version__,
                "statsmodels": version("statsmodels"), "scipy": version("scipy"),
                "sources": {str(p.relative_to(project)): data.sha256(p) for p in sources}}
    
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = project / "data/notebook_replay_v1" / key
    path, manifest = directory / f"{name}.pkl", directory / f"{name}.json"

    if use_cache and path.is_file() and manifest.is_file():
        record = json.loads(manifest.read_text())
        if record["identity"] != identity or record["sha256"] != data.sha256(path):
            raise ValueError(f"Replay cache changed: {path}")
        
        return pd.read_pickle(path)
    result = compute()
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")

    with temporary.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)

    temporary.replace(path)
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps({"identity": identity, "sha256": data.sha256(path)}, indent=2))
    temporary.replace(manifest)
    return result


def build_features(prices, schedule, spec, scale=None, windows=None):
    #use past liquidity and complete (common) formation window. gaps preserved

    windows = sorted(windows or {c["formation_sessions"] for c in spec["configuration_grid"]})
    rules = spec["eligibility"]
    activity = prices[["ticker", "Close", "Volume"]].copy()
    activity["traded_value_proxy_gbp"] = activity["Close"] * activity["Volume"]

    if scale is not None:
        activity["traded_value_proxy_gbp"] *= activity["ticker"].map(scale)

    activity["reported_zero_volume"] = activity["Volume"].eq(0).astype(float).where(activity["Volume"].notna())
    liquidity = features.build_liquidity_features(activity, schedule, lookback=rules["liquidity_lookback_sessions"]).set_index("ticker", append=True).sort_index()

    keyed = prices.set_index("ticker", append=True).sort_index()
    valid_close = np.isfinite(keyed["adj_close"]) & keyed["adj_close"].gt(0)
    window = max(windows) + 1

    complete = valid_close.groupby(level="ticker", sort=False).transform(lambda values: values.rolling(window, min_periods=window).sum().eq(window))
    eligible = (liquidity["history_ready"] & liquidity["median_traded_value_gbp"].ge(rules["min_median_traded_value_gbp"]) & liquidity["zero_volume_fraction"].le(rules["max_zero_volume_fraction"]))

    if rules["require_complete_formation_window"]:
        eligible &= complete
    if rules["require_positive_signal_day_volume"]:
        eligible &= np.isfinite(keyed["Volume"]) & keyed["Volume"].gt(0)

    panels = {}
    for lookback in windows:
        panel = features.build_raw_reversal_features(prices, schedule, lookback=int(lookback))
        panel = panel.set_index("ticker", append=True).sort_index()
        pd.testing.assert_index_equal(panel.index, eligible.index)
        eligible &= np.isfinite(panel["raw_reversal_score"])
        panels[lookback] = panel

    for panel in panels.values():
        panel["eligible"] = eligible
    return panels


def decision_dates(schedule, spec):
    positions = np.arange(len(schedule))
    result = {}
    for holding in sorted({c["holding_sessions"] for c in spec["configuration_grid"]}):
        calendar = validation.build_rebalance_calendar(schedule, anchor=pd.Timestamp(spec["rebalance_anchor"]), rebalance_sessions=int(holding))
        result[int(holding)] = schedule.index[calendar["is_rebalance"].to_numpy() & (positions + 1 + holding < len(schedule))]

    return result


def _complete_inputs(prepared):
    inputs = prepared.copy()
    spec = inputs["spec"]
    inputs["grid"] = pd.DataFrame(spec["configuration_grid"]).set_index("configuration_id").sort_index()
    inputs["features"] = build_features(inputs["prices"], inputs["schedule"], spec, inputs.get("scale"))
    inputs["decision_dates"] = decision_dates(inputs["schedule"], spec)

    signature = _hash_frames(inputs["market"], inputs["schedule"], inputs["grid"], *inputs["features"].values())
    inputs["fingerprint"] = hashlib.sha256((signature + json.dumps(spec, sort_keys=True)).encode()).hexdigest()
    return inputs


def prepare_development(project):
    inputs = _complete_inputs(data.prepare_development(project))
    inputs["baseline_features"] = build_features(inputs["prices"], inputs["schedule"], inputs["spec"], inputs["scale"], windows=[5])[5]
    return inputs


def prepare_validation(project, development):
    inputs = _complete_inputs(data.prepare_validation(project, development))
    for lookback, panel in development["features"].items():
        pd.testing.assert_frame_equal(inputs["features"][lookback].loc[panel.index], panel)

    return inputs


def prepare_extension(project):
    prepared = prepare_uk_extension(project)
    inputs = _complete_inputs({"project": Path(project), "prices": prepared["prices"], "market": prepared["market"], "schedule": prepared["schedule"],
        "spec": prepared["manifest"]["frozen_strategy"], "vintage": "corrected-extension"})
    inputs["prepared_manifest"] = prepared["manifest"]
    inputs["corrections"] = prepared["corrections"]
    return inputs


def coverage(inputs):
    eligible = next(iter(inputs["features"].values()))["eligible"].groupby(level="Date").sum()
    return eligible.groupby(eligible.index.year).agg(["min", "median", "max"]).rename_axis("year")


def summary(equity, results=None):
    table = pd.DataFrame({"final_equity_gbp": equity.iloc[-1],
                          "total_return_pct": (equity.iloc[-1] / equity.iloc[0] - 1) * 100,
                          "max_drawdown_pct": (equity / equity.cummax() - 1).min() * 100})
    if results:
        table["costs_gbp"] = pd.Series({k: v["daily"]["cost_gbp"].sum() for k, v in results.items()})
    return table


def annual_returns(equity):
    returns = equity.pct_change(fill_method=None).iloc[1:]
    return ((1 + returns).groupby(returns.index.year).prod() - 1).mul(100).rename_axis("year")


def matched_benchmark(decisions):
    targets = decisions[["signal_time", "eligible"]].copy()
    counts = targets["eligible"].groupby(level="Date").transform("sum")
    exposure = decisions["target_weight"].groupby(level="Date").transform("sum")
    targets["target_weight"] = targets["eligible"].astype(float).mul(exposure).div(counts.where(counts.gt(0))).fillna(0.)
    return targets


def _configuration(inputs, row, feature_panel=None):
    spec = inputs["spec"]
    return validation.evaluate_execution_configuration(
        feature_panel=inputs["features"][int(row.formation_sessions)] if feature_panel is None else feature_panel,
        decision_dates=inputs["decision_dates"][int(row.holding_sessions)],
        market=inputs["market"], schedule=inputs["schedule"], top_frac=float(row.top_frac),
        holding_sessions=int(row.holding_sessions), min_eligible=spec["minimum_eligible"],
        initial_capital_gbp=spec["initial_capital_gbp"], cost_per_side_bps=spec["walk_forward"]["cost_per_side_bps"])


def run_baseline(inputs, *, use_cache=True):
    def calculate():
        config = inputs["grid"].loc[inputs["grid"]["is_baseline"]].iloc[0]
        run = _configuration(inputs, config, inputs["baseline_features"])
        first = inputs["decision_dates"][5][0]
        initial_targets = run["benchmark_decisions"].xs(first, level="Date", drop_level=False)

        horizon = len(inputs["schedule"]) - inputs["schedule"].index.get_loc(first) - 2
        buy_hold = backtest.run_execution_backtest(initial_targets, inputs["market"], inputs["schedule"],
            initial_capital_gbp=inputs["spec"]["initial_capital_gbp"], holding_sessions=horizon,
            cost_per_side_bps=inputs["spec"]["walk_forward"]["cost_per_side_bps"], net_orders=True)
        
        results = {"Reversal": run["strategy"], "Rebalanced universe": run["benchmark"], "Buy and hold": buy_hold}
        equity = pd.DataFrame({k: v["daily"]["equity_gbp"] for k, v in results.items()})
        return {"equity": equity, "summary": summary(equity, results), "annual": annual_returns(equity)}
    
    return cached_stage(inputs, "baseline", calculate, use_cache=use_cache)


def _candidate_returns(inputs, *, progress=False):
    curves, excess = {}, {}
    expected = inputs["schedule"].loc[inputs["spec"]["rebalance_anchor"]:].index

    for row in inputs["grid"].itertuples():
        if progress:
            print(f"Calculating configuration {row.Index + 1}/{len(inputs['grid'])}", flush=True)

        result = _configuration(inputs, row)
        equity = pd.DataFrame({k: result[k]["daily"]["equity_gbp"] for k in ("strategy", "benchmark")})
        pd.testing.assert_index_equal(equity.index, expected)

        returns = equity.pct_change(fill_method=None).iloc[1:]
        excess[row.Index] = returns["strategy"] - returns["benchmark"]
        curves[row.Index] = equity

    panel = pd.DataFrame(excess).rename_axis(columns="configuration_id")
    if not np.isfinite(panel.to_numpy()).all():
        raise ValueError("Unknown candidate returns")
    
    return {"net_excess": panel, "equity": curves}


def run_sweep(inputs, *, use_cache=True, progress=False):
    def calculate():
        result = _candidate_returns(inputs, progress=progress)
        inference = validation.infer_sweep_means(result["net_excess"], inputs["grid"], inputs["spec"]["primary_inference"])
        result["table"] = inputs["grid"].join(inference)
        return result
    
    return cached_stage(inputs, "development_sweep", calculate, use_cache=use_cache)


def load_candidate_returns(inputs):
    #read input key match, never choose arbitrary cache

    project, spec = inputs["project"], inputs["spec"]
    if inputs["vintage"] == "original-validation":
        digest = hashlib.sha256((project / "data/uk_holdout_2021_2023_v1/frozen_spec.json").read_bytes())
        keyed = inputs["prices"].set_index("ticker", append=True).sort_index()

        for values in [inputs["market"], keyed["adj_close"], inputs["schedule"]]:
            digest.update(pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes())

        directory = project / "data/uk_holdout_2021_2023_v1/candidate_runs" / digest.hexdigest()[:16]
    else:
        digest = hashlib.sha256(json.dumps({"spec": spec, "prepared_inputs": inputs["prepared_manifest"]["input_sha256"],
            "pandas_version": pd.__version__, "cache_version": 1}, sort_keys=True).encode())
        
        for frame in [inputs["market"], inputs["schedule"], inputs["grid"], *inputs["features"].values()]:
            digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        for holding, dates in sorted(inputs["decision_dates"].items()):
            digest.update(str(holding).encode()); digest.update(dates.asi8.tobytes())

        directory = project / "data/uk_extension_2023_2025_v1/candidate_returns" / digest.hexdigest()[:20]
    expected = inputs["schedule"].loc[spec["rebalance_anchor"]:].index[1:]
    values = {}

    for configuration in inputs["grid"].index:
        path = directory / f"configuration_{configuration:02d}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing input-matched candidate cache: {path}. Use rebuild_candidates=True to calculate locally.")
        
        value = pd.read_pickle(path)
        pd.testing.assert_index_equal(value.index, expected)
        if not np.isfinite(value.to_numpy()).all():
            raise ValueError(f"Unknown returns: {path}")
        
        values[configuration] = value
    return pd.DataFrame(values).rename_axis(columns="configuration_id")


def select_configurations(inputs, returns):
    rules = inputs["spec"]["walk_forward"]
    folds = walk_forward.build_walk_forward_folds(returns, inputs["schedule"], initial_train_sessions=rules["initial_train_sessions"], test_sessions=rules["test_sessions"])
    return walk_forward.select_walk_forward_configurations(returns, folds, inputs["grid"], rules)


def selected_decisions(inputs, choices, start=None):
    if start is not None:
        start = pd.Timestamp(start)
        choices = choices.loc[choices["test_end"] >= start]
        cash_date = inputs["schedule"].index[inputs["schedule"].index < start][-1]

    decisions = walk_forward.build_walk_forward_decisions(choices, inputs["features"],
        inputs["decision_dates"], min_eligible=inputs["spec"]["minimum_eligible"])
    
    if start is not None:
        decisions = decisions.loc[cash_date:].copy()
        if cash_date not in decisions.index.get_level_values("Date"):
            panel = inputs["features"][min(inputs["features"])]
            marker = panel.xs(cash_date, level="Date", drop_level=False)[["signal_time", "eligible"]].copy()
            marker = marker.assign(eligible=False, selected=False, target_weight=0., holding_sessions=1)
            decisions = pd.concat([marker, decisions]).sort_index()

    return decisions


def _evaluate_selected(inputs, choices, start=None, cost=None):
    spec = inputs["spec"]
    decisions = selected_decisions(inputs, choices, start)
    lengths = decisions.groupby(level="Date")["holding_sessions"].first().astype("int64")
    first = decisions.index.get_level_values("Date").min()

    schedule, market = inputs["schedule"].loc[first:], inputs["market"].loc[first:]
    results = {}
    for name, targets in {"Walk-forward": decisions, "Matched benchmark": matched_benchmark(decisions)}.items():
        result = backtest.run_execution_backtest(targets, market, schedule,
            initial_capital_gbp=spec["initial_capital_gbp"],
            cost_per_side_bps=spec["walk_forward"]["cost_per_side_bps"] if cost is None else cost,
            net_orders=True, holding_sessions_by_date=lengths)
        pd.testing.assert_index_equal(result["daily"].index, schedule.index)
        results[name] = result

    equity = pd.DataFrame({k: v["daily"]["equity_gbp"] for k, v in results.items()})
    returns = equity.pct_change(fill_method=None).iloc[1:]
    excess = (returns["Walk-forward"] - returns["Matched benchmark"]).to_frame("walk_forward").rename_axis(columns="configuration_id")

    inference = validation.infer_sweep_means(excess, pd.DataFrame(index=excess.columns), spec["primary_inference"])
    return {"equity": equity, "summary": summary(equity, results), "net_excess": excess,
            "inference": inference, "choices": choices, "annual": annual_returns(equity)}, results


def run_walk_forward(inputs, sweep, *, use_cache=True):
    def calculate():
        choices = select_configurations(inputs, sweep["net_excess"])
        return _evaluate_selected(inputs, choices)[0]
    
    return cached_stage(inputs, "development_walk_forward", calculate, use_cache=use_cache, parameters={"candidate_returns": _hash_frames(sweep["net_excess"])})


def run_cost_sensitivity(inputs, development, costs=(5, 10, 25, 40, 50), *, use_cache=True):
    def calculate():
        rows = []
        for cost in costs:
            result = development if cost == inputs["spec"]["walk_forward"]["cost_per_side_bps"] else _evaluate_selected(inputs, development["choices"], cost=cost)[0]
            rows.append({"cost_per_side_bps": cost,
                "strategy_final_gbp": result["equity"]["Walk-forward"].iloc[-1],
                "benchmark_final_gbp": result["equity"]["Matched benchmark"].iloc[-1],
                "mean_daily_net_excess_bps": result["net_excess"].iloc[:, 0].mean() * 10000})
        return pd.DataFrame(rows).set_index("cost_per_side_bps")
    
    return cached_stage(inputs, "cost_sensitivity", calculate, use_cache=use_cache, parameters={"costs": list(costs), "choices": _hash_frames(development["choices"])})



def run_holdout(inputs, start, *, use_cache=True, rebuild_candidates=False):
    #rebuilding uses same engines and writes only the separate replay cache

    returns = (_candidate_returns(inputs, progress=True)["net_excess"] if rebuild_candidates else load_candidate_returns(inputs))
    def calculate():
        choices = select_configurations(inputs, returns)
        result, ledgers = _evaluate_selected(inputs, choices, start=start)

        if inputs["vintage"] == "corrected-extension":
            from reversal_research_reporting import concentration_diagnostics
            result["diagnostics"] = concentration_diagnostics(inputs, result, ledgers)

        result["choices"] = choices.loc[choices["test_end"] >= pd.Timestamp(start)]
        return result
    
    return cached_stage(inputs, "holdout", calculate, use_cache=use_cache,
        parameters={"start": start, "candidate_returns": _hash_frames(returns)})


def run_risk_comparison(inputs, extension, *, cap=.05, soft_limit=-.10, hard_limit=-.25, cash_aer=.038, use_cache=True):
    from risk_limits import drawdown_trading_exposure
    from reversal_risk.execution import run_cash_interest_backtest
    from reversal_risk.reporting import build_final_report

    def calculate():
        decisions = selected_decisions(inputs, extension["choices"], start="2024-01-01")
        lengths = decisions.groupby(level="Date")["holding_sessions"].first().astype("int64")
        if not lengths.eq(1).all():
            raise ValueError("Drawdown experiment requires one-session holdings")
        
        capped = decisions.copy()
        capped["target_weight"] = capped["target_weight"].clip(upper=cap)
        reference = extension["equity"]["Walk-forward"]
        exposure = pd.Series(drawdown_trading_exposure(reference / reference.cummax() - 1,
                            soft_limit=soft_limit, hard_limit=hard_limit), index=reference.index)
        scale = exposure.reindex(decisions.index.get_level_values("Date"))

        if scale.isna().any():
            raise ValueError("Missing drawdown exposure")
        
        drawdown = decisions.copy()
        drawdown["target_weight"] *= scale.to_numpy()
        first = reference.index[0]
        results = {}

        for name, targets in {"Original": decisions, "Capped": capped, "Drawdown": drawdown}.items():
            for label, weights in {name: targets, f"{name} benchmark": matched_benchmark(targets)}.items():
                results[label] = run_cash_interest_backtest(weights, inputs["market"].loc[first:], inputs["schedule"].loc[first:],
                    initial_capital_gbp=inputs["spec"]["initial_capital_gbp"],
                    cost_per_side_bps=inputs["spec"]["walk_forward"]["cost_per_side_bps"], net_orders=True,
                    holding_sessions_by_date=lengths, cash_aer=cash_aer)
        return build_final_report(results, rules=inputs["spec"]["primary_inference"], cash_aer=cash_aer)
    
    return cached_stage(inputs, "risk", calculate, use_cache=use_cache, parameters={"cap": cap, "soft_limit": soft_limit, "hard_limit": hard_limit,
            "cash_aer": cash_aer, "choices": _hash_frames(extension["choices"]), "reference": _hash_frames(extension["equity"])})

