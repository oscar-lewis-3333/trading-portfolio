import numpy as np
import pandas as pd

from .contracts import validate_panel, validate_schedule, validate_targets

_REGISTRY = {}


def register(signal_id, defaults=None):
    #build function attached to each catalogue entry, with ids matching catalogue.py else rejected
    if not isinstance(signal_id, str) or not signal_id:
        raise ValueError("signal_id must be a non-empty string.")
    if defaults is not None and not isinstance(defaults, dict):
        raise ValueError("defaults must be a dict or None.")

    def decorator(build):
        if signal_id in _REGISTRY:
            raise ValueError(f"Signal '{signal_id}' is already registered.")
        _REGISTRY[signal_id] = {"build": build, "defaults": dict(defaults or {})}

        return build

    return decorator


def available():
    #sorted list of signal ids.

    return sorted(_REGISTRY)


def defaults_for(signal_id):
    #freeze parameters chosen on development data to avoid overfitting

    if signal_id not in _REGISTRY:
        raise ValueError(f"Signal '{signal_id}' is not registered.")

    return dict(_REGISTRY[signal_id]["defaults"])


def build_targets(signal_id, panel, schedule, params=None):
    #only way to run a signal, validated in, validated out

    if signal_id not in _REGISTRY:
        raise ValueError(f"Signal '{signal_id}' is not registered.")
    if params is not None and not isinstance(params, dict):
        raise ValueError("params must be a dict or None.")

    validate_panel(panel)
    validate_schedule(schedule)

    settings = defaults_for(signal_id)
    settings.update(params or {})
    targets = _REGISTRY[signal_id]["build"](panel, schedule, **settings)

    return validate_targets(targets, schedule)


def check_no_lookahead(signal_id, panel, schedule, params=None, samples=5, seed=0):
    #recompute early decisions from shortened panel, with any difference being leakage

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer.")

    full = build_targets(signal_id, panel, schedule, params)
    if len(schedule) < 2:
        raise ValueError("A look-ahead check needs at least two decisions.")

    rng = np.random.default_rng(seed)

    #leave all decisions cash during the warm up
    candidates = schedule[max(1, len(schedule) // 4):]
    chosen = pd.DatetimeIndex(sorted(rng.choice(candidates, size=min(samples, len(candidates)), replace=False)))

    failures = []
    for stamp in chosen:
        truncated = panel.loc[panel["timestamp"] < stamp]
        if truncated.empty:
            continue
        
        partial_schedule = validate_schedule(schedule[schedule <= stamp])
        replayed = build_targets(signal_id, truncated, partial_schedule, params)

        expected = full.loc[stamp].reindex(replayed.columns).fillna(0.0).to_numpy(dtype=float)
        actual = replayed.loc[stamp].to_numpy(dtype=float)
        if not np.allclose(expected, actual, rtol=0, atol=1e-9):
            failures.append(stamp)

    return {"signal_id": signal_id, "checked": list(chosen), "failures": failures, "passed": not failures}
