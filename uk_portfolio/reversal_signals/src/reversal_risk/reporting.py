

#file for building the final report of the notebook
import numpy as np
import pandas as pd

from reversal_validation import infer_sweep_means


def build_final_report(results, rules, cash_aer=0.038):
    variants = ("Original", "Capped", "Drawdown")
    expected = {label for name in variants for label in (name, f"{name} benchmark")}
    if set(results) != expected:
        raise ValueError("Supply all three variants and their matched benchmarks.")
    cash_aer = float(cash_aer)
    if not np.isfinite(cash_aer) or cash_aer < 0:
        raise ValueError("Cash AER must be finite and nonnegative.")

    first_daily = results["Original"]["daily"]
    dates = first_daily.index
    if (not isinstance(dates, pd.DatetimeIndex) or len(dates) < 3
            or dates.hasnans or not dates.is_unique
            or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize())):
        raise ValueError("Provide sorted, unique daily session dates.")
    initial = float(first_daily.equity_gbp.iloc[0])
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError("Initial equity must be positive.")
    for result in results.values():
        daily = result["daily"]
        if not daily.index.equals(dates):
            raise ValueError("All portfolios must use exactly the same dates.")
        if not np.isclose(daily.equity_gbp.iloc[0], initial, rtol=0, atol=1e-8):
            raise ValueError("All portfolios must start with the same capital.")

    equity = pd.DataFrame({name: result["daily"].equity_gbp
                           for name, result in results.items()})
    if not np.isfinite(equity.to_numpy()).all() or equity.le(0).to_numpy().any():
        raise ValueError("Resolve invalid equity observations before inference.")
    returns = equity.pct_change(fill_method=None).iloc[1:]
    elapsed = pd.Series(dates, index=dates).diff().dt.days.iloc[1:]
    cash_returns = np.expm1(np.log1p(cash_aer) * elapsed / 365.25).rename("Cash only")
    cash_equity = initial * np.exp(
        np.log1p(cash_aer) * (dates - dates[0]).days.to_numpy() / 365.25
    )
    equity["Cash only"] = cash_equity

    contrasts, metadata = {}, []
    for variant in variants:
        comparators = {
            "matched benchmark": returns[f"{variant} benchmark"],
            "cash": cash_returns,
            "zero": pd.Series(0.0, index=returns.index),
        }
        for comparator, comparator_return in comparators.items():
            label = f"{variant} vs {comparator}"
            contrasts[label] = returns[variant] - comparator_return
            metadata.append({
                "configuration_id": label,
                "variant": variant,
                "comparator": comparator,
                "role": "primary" if comparator == "matched benchmark" else "secondary",
            })
    contrast_returns = pd.DataFrame(contrasts).rename_axis(columns="configuration_id")
    grid = pd.DataFrame(metadata).set_index("configuration_id")
    inference = grid.join(infer_sweep_means(contrast_returns, grid, rules))
    inference = inference.rename(columns={"p_holm": "p_holm_family"})

    summary = pd.DataFrame({
        "final_equity_gbp": equity.iloc[-1],
        "total_return_pct": 100 * (equity.iloc[-1] / initial - 1),
        "max_drawdown_pct": 100 * (equity / equity.cummax() - 1).min(),
        "costs_gbp": pd.Series({name: result["daily"].cost_gbp.sum()
                                for name, result in results.items()}),
        "cash_interest_gbp": pd.Series({name: result["daily"].cash_interest_gbp.sum()
                                        for name, result in results.items()}),
        "final_stale_value_gbp": pd.Series({name: result["daily"].stale_value_gbp.iloc[-1]
                                            for name, result in results.items()}),
        "remaining_positions": pd.Series({name: len(result["final_positions"])
                                           for name, result in results.items()}),
    })
    summary.loc["Cash only", ["costs_gbp", "final_stale_value_gbp", "remaining_positions"]] = 0
    summary.loc["Cash only", "cash_interest_gbp"] = cash_equity[-1] - initial
    return {"equity": equity, "summary": summary, "returns": returns,
            "cash_returns": cash_returns, "contrast_returns": contrast_returns,
            "inference": inference}
