
#frozen adapter and fixed config runners

import copy
import numpy as np
import pandas as pd
from .reversal_ml_annual import run_annual_cases, link_fixed_configuration
from .reversal_ml_walk_forward import run_walk_forward, evaluate_selected
COST_SCENARIOS_BPS = (0.0, 10.0, 20.0, 30.0)
FIXED_VARIANTS = ("Unfiltered", "XGBoost", "XGBoost control")

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
