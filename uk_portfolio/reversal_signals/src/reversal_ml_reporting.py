#tables for the ML notebook

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


def annual_returns(stage):
    return stage["summary"]["net_return_pct"].unstack("year")


def stock_exposure(daily):
    #daily closing stock valie / equity (%), excluding initial cash

    values = daily[["stock_value_gbp", "equity_gbp"]].iloc[1:]
    if (values.empty or not np.isfinite(values).all().all()or not values["equity_gbp"].gt(0).all()):
        raise ValueError("Exposure requires observed stock values and positive equity.")
    
    return 100 * values["stock_value_gbp"] / values["equity_gbp"]


def _net_sharpe(returns):
    #annualised daily net Sharpe

    if returns.empty or not np.isfinite(returns).all():
        raise ValueError("Sharpe requires observed, finite daily returns.")
    
    volatility = returns.std(ddof=1)
    return np.sqrt(252) * returns.mean() / volatility if volatility > 0 else np.nan


def annual_comparison(stage):
    #annual returns and exposure table, labelled by year

    metrics = stage["summary"][["net_return_pct", "mean_stock_exposure_pct"]]
    metrics = metrics.rename(columns={"net_return_pct": "Return (%)", "mean_stock_exposure_pct": "Avg stock exposure (%)"})
    return metrics.unstack("year").swaplevel(0, 1, axis=1).sort_index(axis=1, level=0, sort_remaining=False)


def fixed_comparison_summary(fixed, stages, *, configuration_id=28):
    #average exposure over the same sessions as corresponding annual returns
    means = {}
    for variant in fixed["summary"].index:
        parts = [stock_exposure(daily) for stage in stages for (year, config_id, name), daily in stage["daily"].items()
                 if config_id == configuration_id and name == variant]
        
        exposure = pd.concat(parts).sort_index()
        pd.testing.assert_index_equal(exposure.index, fixed["returns"].index)
        means[variant] = exposure.mean()

    return fixed["summary"].assign(mean_stock_exposure_pct=pd.Series(means))


def continuous_comparison_summary(continuous):
    #add realised exposure for each strategy and its corresponding benchmark
    summary = continuous["summary"].copy()
    dates = continuous["daily"][summary.index[0]].index
    for key in summary.index:
        pd.testing.assert_index_equal(continuous["daily"][key].index, dates)

    summary["mean_stock_exposure_pct"] = [stock_exposure(continuous["daily"][key]).mean() for key in summary.index]
    summary["net_sharpe"] = [_net_sharpe(continuous["daily"][key]["equity_return"].iloc[1:])for key in summary.index]
    return summary


def continuous_annual_returns(continuous):
    returns = pd.DataFrame({variant: daily["equity_return"].iloc[1:] for (variant, role), daily in continuous["daily"].items() if role == "Strategy"})
    return ((1 + returns).groupby(returns.index.year).prod() - 1).mul(100).rename_axis("year")


def continuous_annual_comparison(continuous):
    histories = {variant: stock_exposure(daily) for (variant, role), daily in continuous["daily"].items() if role == "Strategy"}
    dates = next(iter(histories.values())).index

    for history in histories.values():
        pd.testing.assert_index_equal(history.index, dates)

    exposure = pd.DataFrame(histories)
    returns = continuous_annual_returns(continuous)
    mean_exposure = exposure.groupby(exposure.index.year).mean().rename_axis("year")

    daily_returns = pd.DataFrame({variant: daily["equity_return"].iloc[1:] for (variant, role), daily in continuous["daily"].items() if role == "Strategy"})
    sharpe = daily_returns.groupby(daily_returns.index.year).agg(_net_sharpe).rename_axis("year")
    metrics = pd.concat({"Return (%)": returns,
                         "Avg stock exposure (%)": mean_exposure,
                         "Net Sharpe": sharpe}, axis=1)
    
    return metrics.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0, sort_remaining=False)


def prediction_skill_summary(development):
    scores = development["prediction_skill"]

    return pd.DataFrame({name: {"configurations": len(scores),
        "positive_MSE_skill": int(scores[f"{family}_mse_skill_vs_mean"].gt(0).sum()),
        "median_MSE_skill": scores[f"{family}_mse_skill_vs_mean"].median()} for name, family in (("Ridge", "ridge"), ("XGBoost", "xgb"))}).T


def plot_equity_and_drawdown(equity, *, title, subtitle, boundaries=(), log_scale=False):
    #plot complete equity history including initial cash
    drawdown = 100 * (equity / equity.cummax() - 1)
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    colors = {"Unfiltered": "#475569", "XGBoost": "#0072B2", "XGBoost control": "#D55E00"}
    for name in equity:
        style = {"color": colors.get(name), "linewidth": 2,"linestyle": "--" if "control" in name else "-"}
        axes[0].plot(equity.index, equity[name], label=name, **style)
        axes[1].plot(drawdown.index, drawdown[name], **style)

    if log_scale:
        axes[0].set_yscale("log")
        axes[0].yaxis.set_major_formatter(ScalarFormatter())

    axes[0].set_ylabel("Net return index (start = 100" + (", log scale)" if log_scale else ")"))
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")

    for axis in axes:
        axis.grid(alpha=0.2)
        for date in boundaries:
            axis.axvline(pd.Timestamp(date), color="grey", linestyle=":", linewidth=1)

    figure.suptitle(title, fontsize=14)
    figure.supxlabel(subtitle, fontsize=9)
    return figure, axes


def plot_fixed_comparison(fixed):
    return plot_equity_and_drawdown(fixed["equity"], title="Configuration 28: unfiltered versus XGBoost",
        subtitle="Linked annual simulations, 2017–2025; costs included; zero cash interest.",
        boundaries=("2020-01-01", "2022-01-01"), log_scale=True)


def plot_continuous_comparison(continuous):
    equity = pd.DataFrame({variant: daily["equity_gbp"] / daily["equity_gbp"].iloc[0] * 100 for (variant, role), daily in continuous["daily"].items() if role == "Strategy"})
    return plot_equity_and_drawdown(equity, title="Continuous configuration selection: raw reversal versus XGBoost",
        subtitle="22 March 2019–31 December 2025; 5% target cap; 10 bp/side; fixed 3.8% cash AER.",
        boundaries=("2020-01-01", "2022-01-01"))


def export_results(project, inputs, cache, development, assessment, continuation, fixed, continuous):
    #exporting calculated tables and recording input, output hashes
    from reversal_ml_research import protocol, sha256

    project = Path(project)
    results = project / "results/ml_extension"
    images = project / "images"
    results.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    tables = {"development_prediction_skill": development["prediction_skill"],
        "development_net_sharpe": development["statistics"]["net_sharpe"].unstack("variant"),
        "assessment_annual_returns": annual_returns(assessment),
        "assessment_annual_comparison": annual_comparison(assessment),
        "continuation_annual_returns": annual_returns(continuation),
        "continuation_annual_comparison": annual_comparison(continuation),
        "continuation_metrics": continuation["summary"],
        "fixed_28_daily_returns": fixed["returns"],
        "fixed_28_summary": fixed_comparison_summary(fixed, [development, assessment, continuation]),
        "continuous_summary": continuous_comparison_summary(continuous),
        "continuous_choices": continuous["choice_comparison"],
        "continuous_annual_returns": continuous_annual_returns(continuous),
        "continuous_annual_comparison": continuous_annual_comparison(continuous),
        "continuous_daily_returns": pd.DataFrame({f"{variant} / {role}": daily["equity_return"].iloc[1:] for (variant, role), daily in continuous["daily"].items()})}
    outputs = []
    for name, table in tables.items():
        path = results / f"{name}.csv"
        table.to_csv(path)
        outputs.append(path)
    for name, figure in (("ml_fixed_28_equity_drawdown.png", plot_fixed_comparison(fixed)[0]),
                          ("ml_walk_forward_equity_drawdown.png", plot_continuous_comparison(continuous)[0])):
        path = images / name
        figure.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        plt.close(figure)
        outputs.append(path)
    manifest = {"fingerprint": cache.fingerprint, "provenance": cache.provenance,
                "protocol": protocol(inputs),
                "outputs": {str(path.relative_to(project)): sha256(path) for path in outputs}}
    manifest_path = results / "reproduction.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return pd.Series({"tables": len(tables), "figures": 2, "manifest": str(manifest_path.relative_to(project))}, name="Retained outputs")
