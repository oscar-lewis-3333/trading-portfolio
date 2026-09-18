import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller #for the ADF test
from statsmodels.tsa.arima.model import ARIMA #for the model
import itertools
import warnings
warnings.filterwarnings('ignore') #clean up useless errors which clutter space


def test_stationary(series, label="series"):
#run the Augmented dickey-fuller test to determine whether a given series of values are time-dependent or not.

    result = adfuller(series.dropna()) #returns a tuple (test_statistic, p-value) and more information we do not require as of now

    print(f"ADF test - {label}")
    print(f"Test statistic - {result[0]:.4f}")
    print(f"p-value - {result[1]:.4f}")
    print(f"Stationary? {'Yes' if result[1] < 0.05 else 'No'}")

    return result[1] < 0.05

#we fit an arima model to a time series, for example a series of ticker returns/close prices.
#the model seeks to find coefficients (p, d, q), where p describes the autoregressivity (how many previous days todays price depends upon)
#d: how many time we need to difference to make data stationary, usually 0 or 1 in finance, and q: how many previous days errors contribute to todays value

#this is done in 2 functions, one to find 'optimal' (p, d, q), one to fit these to the model.

def find_best_arima_model(series, max_p=3, max_d=1, max_q=3):
    #we search to find best (p,d,q) using AIC, which balances model fit against model complexity (less complex = better). We choose the least of these values

    series = series.dropna()
    series = series.asfreq('B') #business days. removes warnings later on
    best_aic = np.inf
    best_order = None

    for p, d, q in itertools.product(range(max_p+1), range(max_d+1), range(max_q+1)):
        try: #cycle through all combinations  to see which one has the lowest AIC. choose that one.
            model = ARIMA(series, order=(p,d,q))
            fitted=model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except Exception: #may have some convergence issues, doesn't matter, continue.
            continue 
    print(f"Best order: {best_order} (AIC = {best_aic:.2f})")
    return best_order

def fit_arima(series, order=None, max_p=3, max_d=1, max_q=3):
    series = series.dropna()
    series = series.asfreq('B')

    if order is None:
        order = find_best_arima_model(series, max_p, max_d, max_q)

    model = ARIMA(series.dropna(), order=order) #use (p, d, q) to fit the model to our series
    fitted = model.fit()
    return fitted
