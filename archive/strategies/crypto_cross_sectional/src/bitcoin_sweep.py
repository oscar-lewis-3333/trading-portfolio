import pandas as pd

from bitcoin_filter import build_bitcoin_filter, apply_bitcoin_filter
from features import add_momentum_features
from selection import build_weekly_universe, build_momentum_targets
from intraday_backtest import run_intraday_ledger
from weekly_sweep import _check_protocol, summarise_training_ledger


BITCOIN_LOOKBACKS = (7, 14, 30, 60, 90, 120, 180)


def run_bitcoin_training_sweep(policy_panel, gbp_prices, execution_schedule, execution_references, base_protocol, training_end="2024-01-01"):
    #compare BTC filters using only data before training_end
    _check_protocol(base_protocol)

    start = pd.Timestamp(base_protocol["simulation_start"], tz="UTC")
    end = pd.Timestamp(training_end, tz="UTC")
    score_start = base_protocol["training_score_start"]

    #preserve earlier feature history
    history = policy_panel.loc[policy_panel["timestamp"].lt(end)].copy()

    prices = gbp_prices.loc[gbp_prices["timestamp"].ge(start) & gbp_prices["timestamp"].lt(end)].copy()

    schedule = execution_schedule.loc[execution_schedule["day"].ge(start) & execution_schedule["day"].lt(end)].copy().sort_values("day")
    decisions = pd.DatetimeIndex(schedule["day"], name="decision_at")
    expected = pd.date_range(start, end - pd.Timedelta(days=1), freq="W-MON")

    if decisions.empty or not decisions.equals(expected):
        raise ValueError("The training schedule must contain every Monday.")
    if prices.empty or prices["timestamp"].max() != end - pd.Timedelta(days=1):
        raise ValueError("Prices do not reach the complete training cutoff.")

    references = execution_references.loc[execution_references["day"].isin(decisions)].copy()

    #fix the underlying strategy across all candidates
    featured = add_momentum_features(history, lookback_days=90)
    members, universe_summary = build_weekly_universe(featured, **base_protocol["universe"])
    if not members["momentum_ready"].all():
        raise ValueError("A universe member lacks momentum history.")

    _, momentum, equal_weight = build_momentum_targets(members, universe_summary, top_fraction=0.10)
    momentum = momentum.loc[decisions]
    equal_weight = equal_weight.loc[decisions]

    rows, runs, filter_audits = [], {}, {}

    #zero denotes the unfiltered control
    for lookback in (0, *BITCOIN_LOOKBACKS):
        label = "unfiltered" if lookback == 0 else f"btc_{lookback}d"

        if lookback == 0:
            strategy_targets = momentum
            benchmark_targets = equal_weight
        else:
            audit = build_bitcoin_filter(history, decisions, lookback)
            filter_audits[label] = audit
            strategy_targets = apply_bitcoin_filter(momentum, audit)
            benchmark_targets = apply_bitcoin_filter(equal_weight, audit)

        metrics = {}

        for name, targets in (("momentum", strategy_targets), ("equal_weight", benchmark_targets)):
            run = run_intraday_ledger(prices, targets, schedule, references, initial_cash=base_protocol["initial_cash_gbp"], **base_protocol["costs"])
            runs[(label, name)] = run

            ledger = run[0]
            metrics[name] = summarise_training_ledger(ledger, score_start, training_end)
            window = ledger.loc[(ledger.index >= pd.Timestamp(score_start, tz="UTC")) & (ledger.index < end)]
            metrics[name]["cash_day_fraction"] = window["crypto_weight_close"].abs().le(1e-12).mean()

        strategy = metrics["momentum"]
        benchmark = metrics["equal_weight"]

        rows.append({
            "filter": label,
            "btc_lookback_days": lookback,
            **strategy,
            "benchmark_total_return": benchmark["total_return"],
            "benchmark_sharpe": benchmark["sharpe_zero_cash_rate"],
            "benchmark_max_drawdown": benchmark["max_drawdown"],
            "sharpe_difference": strategy["sharpe_zero_cash_rate"]- benchmark["sharpe_zero_cash_rate"]
            })

        print(f"Completed {label}", flush=True)

    return pd.DataFrame(rows).set_index("filter"), runs, filter_audits