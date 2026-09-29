"""Protocol checks, the C1 configuration grid and the development sweep.

Every function takes its dates from the frozen protocol; nothing here uses "today".
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import signal_factory as sf
from signal_factory.contracts import CASH
from signal_factory.metrics import max_drawdown, sharpe

NO_CAP_TARGET = 10.0   # a 1,000% volatility target never binds; the protocol's "no cap"
PERIODS_PER_YEAR = 365


def load_protocol(path):
    #read the frozen protocol and refuse it if it has been edited since freezing
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(record["protocol"], sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != record["protocol_sha256"]:
        raise ValueError("The frozen protocol has been edited; write a new version instead.")
    return record["protocol"]


def _files_digest(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def check_saved_data(protocol, raw_dir):
    #the evaluation must use exactly the files fingerprinted at the freeze
    data = protocol["data"]
    end = pd.Timestamp(data["daily_end_exclusive"])
    #the exact 250-day blocks load_daily_history requests for the frozen window
    blocks, block = [], pd.Timestamp(data["daily_start"])
    while block < end:
        block_end = min(block + pd.Timedelta(days=250), end)
        blocks.append(f"{block:%Y%m%dT%H%M}_{block_end:%Y%m%dT%H%M}.json")
        block = block_end
    daily = sorted(p for p in (Path(raw_dir) / "daily").glob("*.json") if p.name.split("_", 2)[2] in blocks)
    minute = sorted((Path(raw_dir) / "minute").glob("*.json"))
    minute = [p for p in minute if pd.Timestamp(p.name.split("_")[2][:8]) < end]
    checks = {"daily_files": len(daily) == data["daily_files"],
              "daily_sha256": _files_digest(daily) == data["daily_sha256"],
              "minute_files": len(minute) == data["minute_files"],
              "minute_sha256": _files_digest(minute) == data["minute_sha256"]}
    if not all(checks.values()):
        raise ValueError(f"Saved data differs from the frozen fingerprints: {checks}")
    return checks


def sweep_configurations(protocol):
    #the 18 frozen settings, in sweep order (the protocol's tie-break uses this order)
    grid = protocol["sweep"]
    rows = []
    for lookbacks, target, band in itertools.product(grid["lookbacks"], grid["target_volatility"], grid["no_trade_band"]):
        rows.append({"lookbacks": tuple(lookbacks),
                     "target_volatility": NO_CAP_TARGET if target is None else float(target),
                     "vol_cap": "none" if target is None else f"{target:.0%}",
                     "no_trade_band": float(band),
                     "vol_window_days": protocol["signal"]["vol_window_days"]})
    configs = pd.DataFrame(rows)
    configs.index = [f"C{number:02d}" for number in range(1, len(configs) + 1)]
    if len(configs) != grid["configurations"]:
        raise ValueError("The grid does not match the frozen configuration count.")
    return configs


def signal_params(config):
    return {"lookbacks": tuple(config["lookbacks"]), "vol_window_days": int(config["vol_window_days"]),
            "target_volatility": float(config["target_volatility"]), "no_trade_band": float(config["no_trade_band"])}


def run_strategy(signal_id, params, panel, schedule, returns, executions, cost_mode="pessimistic", cash_rate=0.0):
    #targets and the daily ledger for one strategy, trading at the protocol's fills
    targets = sf.build_targets(signal_id, panel, schedule, params)
    ledger = sf.run_portfolio(targets, returns, cost_mode, cash_rate=cash_rate,
                              executions=executions.loc[schedule])
    return targets, ledger


def summarise(targets, ledger, start, end):
    #one row of results over [start, end); the portfolio itself runs continuously from its first decision
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    window = ledger.loc[(ledger.index >= start) & (ledger.index < end)]
    returns = window["net_return"]
    decisions = targets.loc[(targets.index >= start) & (targets.index < end)]
    if returns.empty:
        #a portfolio that does not exist yet in this window (e.g. buy-and-hold before 2024)
        return {key: np.nan for key in ("total_return", "cagr", "volatility", "sharpe", "max_drawdown",
                                        "annual_turnover", "average_cash", "weeks_traded")}
    years = len(returns) / PERIODS_PER_YEAR
    total = float((1 + returns).prod() - 1)
    return {"total_return": total,
            "cagr": (1 + total) ** (1 / years) - 1,
            "volatility": float(returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)),
            "sharpe": sharpe(returns, 0.0, PERIODS_PER_YEAR),
            "max_drawdown": max_drawdown(returns),
            "annual_turnover": float(window["one_way_turnover"].sum() / years),
            "average_cash": float(decisions[CASH].mean()),
            "weeks_traded": int(window["one_way_turnover"].gt(1e-12).sum())}


def development_sweep(protocol, panel, schedule, returns, executions):
    #all 18 settings plus the frozen 30d rule, development period only; nothing is selected here
    start = protocol["periods"]["first_decision"]
    end = protocol["periods"]["development_end_exclusive"]
    end_stamp = pd.Timestamp(end, tz="UTC")
    development = schedule[(schedule >= pd.Timestamp(start, tz="UTC")) & (schedule < end_stamp)]
    development_returns = returns.loc[returns.index < end_stamp]

    configs = sweep_configurations(protocol)
    rows = {}
    for name, config in configs.iterrows():
        targets, ledger = run_strategy("trend_agreement_vol", signal_params(config), panel, development,
                                       development_returns, executions)
        rows[name] = summarise(targets, ledger, start, end)

    targets, ledger = run_strategy("trend_absolute_30d", {"lookback_days": 30}, panel, development,
                                   development_returns, executions)
    rows["30d momentum"] = summarise(targets, ledger, start, end)

    table = pd.DataFrame(rows).T
    labels = configs[["lookbacks", "vol_cap", "no_trade_band"]].copy()
    labels["lookbacks"] = labels["lookbacks"].map(lambda windows: ",".join(map(str, windows)))
    labels.loc["30d momentum"] = ["30", "none", 0.0]
    return labels.join(table)


def _stamp(date):
    return pd.Timestamp(date, tz="UTC")


def run_all_configurations(protocol, panel, schedule, returns, executions, cost_mode="pessimistic", cash_rate=0.0):
    #every setting run once, continuously from the first decision to the evaluation end.
    #signals only use completed candles before each decision, so a later end cannot change earlier results
    start, end = _stamp(protocol["periods"]["first_decision"]), _stamp(protocol["periods"]["evaluation_end_exclusive"])
    decisions = schedule[(schedule >= start) & (schedule < end)]
    window_returns = returns.loc[returns.index < end]
    runs = {}
    for name, config in sweep_configurations(protocol).iterrows():
        runs[name] = run_strategy("trend_agreement_vol", signal_params(config), panel, decisions,
                                  window_returns, executions, cost_mode, cash_rate)
    return runs


def select_annual(protocol, runs):
    #each January: best net Sharpe from the first decision up to that January, rounded to 3 dp;
    #ties go to lower turnover, then to the earlier setting in sweep order
    start = protocol["periods"]["first_decision"]
    order = list(runs)
    rows = []
    for year in protocol["selection"]["years_selected"]:
        scores = pd.DataFrame({name: summarise(targets, ledger, start, f"{year}-01-01")
                               for name, (targets, ledger) in runs.items()}).T
        scores["score"] = scores["sharpe"].astype(float).round(3)
        scores["order"] = [order.index(name) for name in scores.index]
        best = scores.sort_values(["score", "annual_turnover", "order"], ascending=[False, True, True]).index[0]
        rows.append({"year": year, "chosen": best, "training_end": f"{year}-01-01",
                     "training_sharpe": float(scores.loc[best, "sharpe"]),
                     "runner_up": scores.sort_values(["score", "annual_turnover", "order"],
                                                     ascending=[False, True, True]).index[1]})
    return pd.DataFrame(rows).set_index("year")


def walk_forward_targets(choices, runs):
    #each year's decisions come from that year's chosen setting; the portfolio starts in cash in the first selected year
    first_year = int(choices.index.min())
    any_targets = next(iter(runs.values()))[0]
    decisions = any_targets.index[any_targets.index.year >= first_year]
    rows = [runs[choices.loc[stamp.year, "chosen"]][0].loc[stamp] for stamp in decisions]
    return pd.DataFrame(rows, index=decisions)


def constant_mix_targets(schedule, symbols, crypto_weight):
    #weekly rebalance to a fixed crypto share split equally, the rest in cash
    frame = pd.DataFrame(crypto_weight / len(symbols), index=schedule, columns=list(symbols))
    frame[CASH] = 1.0 - crypto_weight
    return frame


def evaluation_portfolios(protocol, runs, choices, panel, schedule, returns, executions, cost_mode="pessimistic", cash_rate=0.0):
    #the annually selected multi-horizon portfolio and the frozen benchmarks, each as (targets, ledger)
    periods = protocol["periods"]
    start, end = _stamp(periods["first_decision"]), _stamp(periods["evaluation_end_exclusive"])
    symbols = protocol["universe"]
    decisions = schedule[(schedule >= start) & (schedule < end)]
    window_returns = returns.loc[returns.index < end]
    fills = executions.loc[:, ["execution_day", *symbols]]

    portfolios = {}
    c1 = walk_forward_targets(choices, runs)
    portfolios["Multi-horizon trend"] = (c1, sf.run_portfolio(c1, window_returns, cost_mode, cash_rate=cash_rate,
                                                          executions=fills.loc[c1.index]))
    portfolios["30d momentum"] = run_strategy("trend_absolute_30d", {"lookback_days": 30}, panel, decisions,
                                          window_returns, executions, cost_mode, cash_rate)
    mix = constant_mix_targets(decisions, symbols, 0.5)
    portfolios["25/25/50 mix"] = (mix, sf.run_portfolio(mix, window_returns, cost_mode, cash_rate=cash_rate,
                                                        executions=fills.loc[mix.index]))
    first_evaluation = decisions[decisions >= _stamp(periods["evaluation_start"])][:1]
    hold = constant_mix_targets(first_evaluation, symbols, 1.0)
    portfolios["50/50 buy-and-hold"] = (hold, sf.run_portfolio(hold, window_returns, cost_mode, cash_rate=cash_rate,
                                                               executions=fills.loc[hold.index]))
    return portfolios


def results_table(portfolios, start, end):
    return pd.DataFrame({name: summarise(targets, ledger, start, end) for name, (targets, ledger) in portfolios.items()}).T


def yearly_returns(portfolios, start, end):
    start, end = _stamp(start), _stamp(end)
    table = {}
    for name, (_, ledger) in portfolios.items():
        returns = ledger.loc[(ledger.index >= start) & (ledger.index < end), "net_return"]
        table[name] = (1 + returns).groupby(returns.index.year).prod() - 1
    return pd.DataFrame(table)


def sharpe_difference_tests(protocol, portfolios, strategy="Multi-horizon trend", benchmark="30d momentum"):
    #the frozen primary test plus its block-length sensitivities
    inference = protocol["inference"]
    start, end = _stamp(protocol["periods"]["evaluation_start"]), _stamp(protocol["periods"]["evaluation_end_exclusive"])
    window = lambda ledger: ledger.loc[(ledger.index >= start) & (ledger.index < end), "net_return"]
    seed = int(inference["method"].split("seed ")[1])
    rows = []
    for block in [inference["block_days"], *inference["sensitivity_block_days"]]:
        result = sf.block_bootstrap_difference(window(portfolios[strategy][1]), window(portfolios[benchmark][1]),
                                               block_days=block, resamples=10000, seed=seed, periods_per_year=PERIODS_PER_YEAR)
        result["primary"] = block == inference["block_days"]
        rows.append(result)
    return pd.DataFrame(rows).set_index("block_days")
