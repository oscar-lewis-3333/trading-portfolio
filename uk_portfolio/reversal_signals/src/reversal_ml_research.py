import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pandas as pd

from reversal_ml_data import prepare_inputs, SNAPSHOT, SPEC, FEATURE_COLUMNS, MAX_TARGET_WEIGHT
from reversal_ml_annual import run_annual_cases, select_finalists, link_fixed_configuration, WEIGHT_COLUMNS, PRIMARY_VARIANTS
from reversal_ml_walk_forward import run_walk_forward, CASH_AER
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
