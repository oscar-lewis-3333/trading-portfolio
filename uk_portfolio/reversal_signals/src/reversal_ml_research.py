import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import copy

import pandas as pd
import numpy as np

from reversal_ml_data import prepare_inputs, SNAPSHOT, SPEC, FEATURE_COLUMNS, MAX_TARGET_WEIGHT
from reversal_ml_annual import run_annual_cases, select_finalists, link_fixed_configuration, WEIGHT_COLUMNS, PRIMARY_VARIANTS
from reversal_ml_walk_forward import run_walk_forward, CASH_AER, evaluate_selected
from ml_trade_filter import XGB_SETTINGS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ResearchCache:
    #local trusted cache.

    def __init__(self, project):
        self.project = Path(project).resolve()
        self.provenance = {"inputs": {path: sha256(self.project / path) for path in (SNAPSHOT, SPEC)},
            "sources": {str(path.relative_to(self.project)): sha256(path) for path in sorted((self.project / "src").rglob("*.py"))},
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "pandas-market-calendars")}}
        encoded = json.dumps(self.provenance, sort_keys=True).encode()
        self.fingerprint = hashlib.sha256(encoded).hexdigest()
        self.directory = self.project / "data/ml_extension_v1/cache" / self.fingerprint

    def get(self, stage, compute, *, use_cache=True):
        if stage not in {"development", "assessment", "continuation", "walk_forward"}:
            raise ValueError(f"Unknown research stage: {stage}")
        
        path = self.directory / f"{stage}.pkl"
        manifest = path.with_suffix(".json")
        if use_cache and path.is_file() and manifest.is_file():
            record = json.loads(manifest.read_text())
            if record["fingerprint"] == self.fingerprint and sha256(path) == record["sha256"]:
                print(f"Loaded verified local cache: {stage}.")
                return pd.read_pickle(path)
            
            raise ValueError(f"Changed or damaged ML cache: {path}. Recompute with use_cache=False.")
        print(f"Rebuilding {stage} from frozen inputs.", flush=True)
        result = compute()
        self.save(stage, result)
        return result

    def save(self, stage, result):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{stage}.pkl"
        temporary = path.with_suffix(".tmp")
        pd.to_pickle(result, temporary)
        temporary.replace(path)
        path.with_suffix(".json").write_text(json.dumps({"fingerprint": self.fingerprint, "sha256": sha256(path)}, indent=2) + "\n")


def protocol(inputs):
    return {"features": FEATURE_COLUMNS, "target": "planned open-to-open gross forward return including dividends",
            "filter": "retain prediction > 0; retain base trade if features are missing",
            "maximum_target_weight": MAX_TARGET_WEIGHT, "ridge_alpha": 1.0,
            "xgboost": XGB_SETTINGS, "initial_capital_gbp": inputs["spec"]["initial_capital_gbp"],
            "cost_per_side_bps": inputs["spec"]["walk_forward"]["cost_per_side_bps"],
            "annual_development_years": [2017, 2018, 2019], "annual_assessment_years": [2020, 2021],
            "annual_continuation_years": [2022, 2023, 2024, 2025],
            "annual_cash_aer": 0.0, "continuous_cash_aer": CASH_AER,
            "continuous_rules": inputs["spec"]["walk_forward"],
            "continuous_selection": "uncapped candidate net excess, zero cash interest",
            "status": "both configuration 28 and adaptive walk-forward kept for further testing."}


def run_development(inputs, cache, *, use_cache=True, progress=False):
    cases = [(int(config), variant) for config in inputs["grid"].index for variant in WEIGHT_COLUMNS]
    return cache.get("development", lambda: run_annual_cases(
        inputs, (2017, 2018, 2019), cases, progress=progress), use_cache=use_cache)


def run_assessment(inputs, development, cache, *, use_cache=True, progress=False):
    _, _, cases = select_finalists(development, inputs["grid"])
    return cache.get("assessment", lambda: run_annual_cases(inputs, (2020, 2021), cases, progress=progress), use_cache=use_cache)


def run_continuation(inputs, cache, *, use_cache=True, progress=False):
    cases = [(config, variant) for config in (28, 12) for variant in ("Unfiltered", "XGBoost", "XGBoost control")]
    return cache.get("continuation", lambda: run_annual_cases(inputs, (2022, 2023, 2024, 2025), cases, progress=progress), use_cache=use_cache)


def run_continuous(inputs, cache, *, use_cache=True, progress=False):
    return cache.get("walk_forward", lambda: run_walk_forward(inputs, progress=progress), use_cache=use_cache)

COST_SCENARIOS_BPS = (0.0, 10.0, 20.0, 30.0)


def with_cost(inputs, cost_per_side_bps):
    #transaction cost sensitivity
    cost = float(cost_per_side_bps)
    if not 0 <= cost < 10_000:
        raise ValueError("Cost per side must be between 0 and 10,000 bps.")

    spec = copy.deepcopy(inputs["spec"])
    spec["walk_forward"]["cost_per_side_bps"] = cost
    return {**inputs, "spec": spec}

def replay_continuous(inputs, continuous, cost_per_side_bps):
    #replay saved walk-forward targets at a fixed cost per side
    scenario = with_cost(inputs, cost_per_side_bps)
    decisions = continuous["selected_decisions"]
    lengths = {variant: frame["holding_sessions"].groupby(level="Date").first().astype("int64") for variant, frame in decisions.items()}

    return evaluate_selected(scenario, continuous["choice_folds"], decisions, lengths)

def continuous_cost_sensitivity(inputs, continuous, costs=COST_SCENARIOS_BPS):
    #now test transaction fee sensitivity on the walk-forward. this runs the walk-forward portfolios at a variety of transaction costs.
    base = float(inputs["spec"]["walk_forward"]["cost_per_side_bps"])
    rows = []

    for cost in costs:
        daily, summary = ((continuous["daily"], continuous["summary"]) if float(cost) == base
                          else replay_continuous(inputs, continuous, cost))

        for variant in summary.index.get_level_values("variant").unique():
            strategy = daily[(variant, "Strategy")]["equity_gbp"].pct_change().dropna()
            benchmark = daily[(variant, "Benchmark")]["equity_gbp"].pct_change().dropna()
            excess = (strategy - benchmark).dropna()

            for portfolio, returns in (("Strategy", strategy), ("Benchmark", benchmark)):
                row = summary.loc[(variant, portfolio)].to_dict()
                row.update(cost_per_side_bps=float(cost), variant=variant, portfolio=portfolio,
                           net_sharpe=np.sqrt(252) * returns.mean() / returns.std())
                if portfolio == "Strategy":
                    row["mean_daily_net_excess_bps"] = 10_000 * excess.mean()
                rows.append(row)

    return pd.DataFrame(rows).set_index(["cost_per_side_bps", "variant", "portfolio"]).sort_index()

FIXED_VARIANTS = ("Unfiltered", "XGBoost", "XGBoost control")


def replay_fixed(inputs, cost_per_side_bps, *, configuration_id=28, progress=False):
    #rerun a fixed configs annual portfolio at different transaction costs.
    scenario = with_cost(inputs, cost_per_side_bps)
    cases = [(configuration_id, variant) for variant in FIXED_VARIANTS]
    stage = run_annual_cases(scenario, tuple(range(2017, 2026)), cases, progress=progress)

    return link_fixed_configuration([stage], inputs["schedule"], configuration_id=configuration_id)

def fixed_cost_sensitivity(inputs, fixed_28, costs=COST_SCENARIOS_BPS, *, configuration_id=28, progress=False):
    #replay config 28 at each wanted transaction fee.
    base = float(inputs["spec"]["walk_forward"]["cost_per_side_bps"])
    tables = {}

    for cost in costs:
        fixed = (fixed_28 if float(cost) == base
                 else replay_fixed(inputs, cost, configuration_id=configuration_id, progress=progress))
        later = fixed["returns"].loc["2022-01-01":]
        table = fixed["summary"].copy()
        table["return_2022_2025_pct"] = 100 * ((1 + later).prod() - 1)
        tables[float(cost)] = table

    return pd.concat(tables, names=["cost_per_side_bps", "variant"])