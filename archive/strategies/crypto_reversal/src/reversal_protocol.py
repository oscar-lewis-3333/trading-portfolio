import json
from pathlib import Path


def research_protocol():
    #return independent settings on each call
    return {
        "version": "crypto_reversal_research",
        "stage": "Development study design; rebalance sweep not yet specified",
        "hypothesis": "Temporary selling pressure can be followed by relative price recovery",
        "history_snapshot": "../crypto_cross_sectional/data/raw/coinbase_coverage_20260918T183827_526442Z",
        "history_start": "2019-01-01",
        "history_end_exclusive": "2026-09-01",
        "quote_currency": "USD",
        "universe_basis": "Saved current Revolut UK GBP candidates with Coinbase USD history; survivorship remains",
        "meme_policy_version": "meme_exclusions_v1",
        "excluded_products": [
            "BONK-USD",
            "DOGE-USD",
            "FLOKI-USD",
            "PENGU-USD",
            "PEPE-USD",
            "SHIB-USD",
            "SPX-USD",
            "TRUMP-USD",
            "WIF-USD"
        ],
        "eligibility": {
            "history_days": 180,
            "liquidity_days": 30,
            "min_median_dollar_volume": 1000000.0
        },
        "universe": {
            "max_assets": 30,
            "min_assets": 1,
            "liquidity_order": "Descending median dollar volume, then ascending product_id",
            "empty_universe": "Target 100% cash when no assets pass eligibility; required exits need valid execution prices and incur costs on net trades"
        },
        "signal": {
            "decision_frequency": "Daily at 00:00 UTC",
            "formation_days": 1,
            "score": "Negative trailing calendar-day close-to-close return",
            "selection_fraction": 0.2,
            "selection_count": "ceil(selection_fraction * eligible_universe_size)",
            "ties": "Ascending product_id",
            "negative_return_required": False,
            "weights": "Equal weight among selected assets",
            "availability": "Only candles completed by decision time; never bridge history gaps"
        },
        "development_start": "2022-01-01",
        "development_end_exclusive": "2025-01-01",
        "window_rule": "Start inclusive, end exclusive; development outcomes must be available no later than the development end",
        "combined_portfolio_evaluation": "Owned by the future bot notebook; preserve versions for every strategy change",
        "holdout_start": "2025-01-01",
        "holdout_end_exclusive": "2026-09-01",
        "warmup_rule": "Earlier prices may supply historical features but are not scored as evaluation returns",
        "holdout_status": "Previously examined in other crypto projects; reserved for this strategy, not a pristine research holdout",
        "bootstrap_phase": "Holdout only, after strategy and inference rules are frozen",
        "holdout_release_rule": "Freeze the strategy, selection rule, benchmark, execution, costs and inference before evaluating holdout returns",
        "transaction_cost_basis": "Sum of per-asset costs on absolute net traded notional at execution; do not offset different assets or charge unchanged units",
        "next_step": "Specify the development sweep and selection criterion before running backtests"
    }


def freeze_research_protocol(project_root):
    path = Path(project_root) / "research" / "research_protocol_v2.json"
    expected = research_protocol()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError:
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("Existing research protocol differs. Changes need a new version.")
    return expected
