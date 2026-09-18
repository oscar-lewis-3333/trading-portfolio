import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve, roc_curve, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve

def cross_validate_model(model, X, y, k=5, scoring='r2'):
    #we do a k-fold cross validation, we split data in k folds, train on k-1 folds, and test on the remaining. Repeat for all orderings of k

    #return individual fold scores, means, standard deviation. Standard deviation gives how much performance estimate varies based on which split chosen.

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    return {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
    }
def cv_complexity_sweep(X, y, max_degree=4, k=5):
    #repeat poly sweep but cross-validated scores not single split, see how things change for polynomials when split isnt fixed. effectively combine polynomial example with above function
    rows=[]
    for degree in range(1, max_degree+1):
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), LinearRegression())
        cv = cross_validate_model(model, X, y, k=k)
        rows.append({
            'degree': degree,
            'cv_mean_r2': round(cv['mean'], 4),
            'cv_std': round(cv['std'], 4),
            'worst_fold': round(cv['scores'].min(), 4),
            'best_fold': round(cv['scores'].max(), 4),
        })

    return pd.DataFrame(rows).set_index('degree')

def tune_hyperparameters(model, param_grid, X, y, k=5, scoring='r2'):
    #we exhaustively search over the parameter grid to find the best ones using cross-validation. return best parameters and full results.

    #notice this is biased upwards. by trying enough parameters and selecting the max over certain conditions, we are validation folds to influence the decision, making it no longer clean
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    search = GridSearchCV(model, param_grid, cv=kf, scoring=scoring, n_jobs=-1)
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].sort_values('rank_test_score')

    return search.best_params_, search.best_score_, results

#we now improve on that. We hold a test, tune on remainder by cross-validation, then evaluate once on untouched set. Problem above is that given enough parameters, every split of the dataset has been tested on in cross-validation.
#the difference between the scores of the function above and below is the difference made by the inflated estimate.

def tune_and_evaluate(model, param_grid, X, y, test_size=0.2, k=5):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X_tune, X_holdout, y_tune, y_holdout = train_test_split(X, y, test_size=test_size, random_state=42)
    best_params, best_cv, results = tune_hyperparameters(model, param_grid, X_tune, y_tune, k=k)

    best_model = model.set_params(**best_params).fit(X_tune, y_tune)
    holdout_score = r2_score(y_holdout, best_model.predict(X_holdout))

    print(f"Best params:      {best_params}")
    print(f"Best CV score:    {best_cv:.4f}")
    print(f"Holdout score:    {holdout_score:.4f}")
    print(f"Optimism:         {best_cv - holdout_score:.4f}")

    return best_model, results

def threshold_sweep(model, X_test, y_test, thresholds=None):
    #we show how precision and recall trade off as the decision threshold moves.
    #0.5 is default, right threshold depends on relative costs of false-positives vs missed positives

    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            'threshold': round(t, 2),
            'n_signals': int(pred.sum()),
            'precision': round(precision_score(y_test, pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, pred, zero_division=0), 4),
            'f1': round(f1_score(y_test, pred, zero_division=0), 4),
        })
    return pd.DataFrame(rows).set_index('threshold')

def calibration_check(model, X_test, y_test, n_bins=10):
    #we check predicted probablities do what they say. For example a well-calibrated model that is '70% confident' should be right about 70% of the time

    proba = model.predict_proba(X_test)[:, 1]
    true_freq, pred_freq = calibration_curve(y_test, proba, n_bins=n_bins, strategy='quantile')

    return pd.DataFrame({
        'predicted_prob': np.round(pred_freq, 4),
        'actual_freq': np.round(true_freq, 4),
        'error': np.round(true_freq - pred_freq, 4),
    })

def regularisation_sweep(X, y, alphas=None, k=5):
    #sweep regularisation strength for Ridge, Lasso to find where bias^2 + variance minimal. alpha=0 is LS regression.
    from sklearn.linear_model import Ridge, Lasso

    if alphas is None:
        alphas = np.logspace(-4, 3, 20) #values from 10^-4 to 10^3 spaced logarithmically

    rows =[]
    for alpha in alphas:
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        lasso = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=5000))

        ridge_cv = cross_validate_model(ridge, X, y, k=k)
        lasso_cv = cross_validate_model(lasso, X, y, k=k)

        lasso_fitted = lasso.fit(X, y)
        n_nonzero = np.sum(lasso_fitted.named_steps['lasso'].coef_ != 0)

        rows.append({
            'alpha': round(alpha, 5),
            'ridge_r2': round(ridge_cv['mean'], 4),
            'lasso_r2': round(lasso_cv['mean'], 4),
            'lasso_features_kept': int(n_nonzero),
        })

    return pd.DataFrame(rows).set_index('alpha')
