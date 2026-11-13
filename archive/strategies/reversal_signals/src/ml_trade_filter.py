
#reusable return forecasting filters, which give explicit observation-time boundaries. knowledge about reversal scores, exchanges and execution is none here.

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


XGB_SETTINGS = {
    "objective": "reg:squarederror", "n_estimators": 100,
    "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 50,
    "reg_lambda": 10.0, "subsample": 1.0, "colsample_bytree": 1.0,
    "tree_method": "hist", "random_state": 42, "n_jobs": 2,
}


def observable_split(data, start, end, *, include_exit_at_start=False, require_exit_before_end=False):
    #separate completed training labels from future candidate decisions
    
    signal, exit_time = data["signal_time"], data["exit_time"]
    usable = data["features_ready"] & data["outcome_observed"]
    known = exit_time.le(start) if include_exit_at_start else exit_time.lt(start)
    training = usable & signal.lt(start) & known
    candidates = signal.ge(start) & signal.lt(end)
    if require_exit_before_end:
        candidates &= exit_time.lt(end)
    if (training & candidates).any():
        raise ValueError("Training and prediction rows overlap.")
    return {"train_mask": training, "candidate_mask": candidates, "score_mask": candidates & usable}


def fit_predict(training, candidates, features, *, family, target="forward_return", ridge_alpha=1.0, xgb_settings=None):
    #fit from scratch. return forecasts without dropping 'unready' candidate assets. unready rows recieve NaN for caller's fallback
    
    features = list(features)
    if training.empty or not np.isfinite(training[features + [target]]).all().all():
        raise ValueError("Training requires finite features and observed targets.")
    
    mean = float(training[target].mean())
    if family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
    elif family == "xgb":
        settings = dict(XGB_SETTINGS if xgb_settings is None else xgb_settings)
        settings["base_score"] = mean
        model = XGBRegressor(**settings)
    else:
        raise ValueError(f"Unknown model family: {family}")
    
    model.fit(training[features], training[target])
    ready = candidates["features_ready"]
    predictable = candidates.loc[ready, features]

    forecast = pd.Series(np.nan, index=candidates.index, name=f"{family}_prediction")
    if not predictable.empty:
        if not np.isfinite(predictable).all().all():
            raise ValueError("A ready prediction row has non-finite features.")
        if family == "ridge":
            #stable, row-wise dot product
            scaled = model.named_steps["standardscaler"].transform(predictable)
            ridge = model.named_steps["ridge"]
            values = np.sum(scaled * ridge.coef_, axis=1) + ridge.intercept_
        else:
            values = model.predict(predictable)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {family} forecasts.")
        
        forecast.loc[predictable.index] = values
    return forecast, {"training_rows": len(training), "prediction_rows": len(predictable), "training_mean": mean}


def filter_weights(base, prediction, ready, *, threshold=0.0):
    #keep positive forecasts. negative forecasts become cash
    if not base.index.equals(prediction.index) or not base.index.equals(ready.index):
        raise ValueError("Weights, forecasts and readiness must have identical indices.")
    if not np.isfinite(base).all() or base.lt(0).any():
        raise ValueError("Base weights must be finite and non-negative.")
    if not np.isfinite(prediction.loc[ready]).all():
        raise ValueError("Missing forecast for a candidate with complete features.")
    
    return base.where(prediction.gt(threshold) | ~ready, 0.0)


def exposure_matched_control(base, filtered, *, group_levels):

    #scale every base holding equally to match each filtered target exposure
    if not base.index.equals(filtered.index):
        raise ValueError("Base and filtered weights must have identical indices.")
    if not np.isfinite(filtered).all() or filtered.lt(0).any() or filtered.gt(base).any():
        raise ValueError("Filtered weights must lie between zero and their base weights.")
    
    total = base.groupby(level=group_levels).transform("sum")
    retained = filtered.groupby(level=group_levels).transform("sum")
    control = base * retained.div(total.where(total.gt(0))).fillna(0.0)
    np.testing.assert_allclose(control.groupby(level=group_levels).sum(), filtered.groupby(level=group_levels).sum(), rtol=1e-12, atol=1e-12)
    return control
