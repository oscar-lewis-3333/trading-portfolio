
#log of each screened trial.

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def record_trial(path, result, params=None, note=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = params or {}
    entry = {"recorded_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": result["candidate_id"],
        "params": params,
        "params_sha256": hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest(),
        "information_ratio": result["information_ratio"],
        "ir_lower_bound": result.get("ir_lower_bound"),
        "accepted": result["accepted"],
        "reasons": result["reasons"],
        "note": note}
    with path.open("a") as sink:
        sink.write(json.dumps(entry, default=str) + "\n")

    return entry


def load_trials(path):
    path = Path(path)
    if not path.exists():
        return []

    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trial_sharpes(path):
    #each recorded run

    return [t["information_ratio"] for t in load_trials(path) if t["information_ratio"] is not None]
