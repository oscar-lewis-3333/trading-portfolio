import warnings
warnings.filterwarnings('ignore') #remove harmless warnings

import numpy as np
from arch import arch_model 
import pandas as pd

#the idea of GARCH is the volatility version of ARIMA. We model the squared volatility at a point as being a sum of:
#a constant, long term volatility 
#a constant scaling of the day before's square error (more shock recently means more volatility next day)
#a constant scaling of the day before's square volatility (volatility is in same ballpark as yesterday)

def fit_garch(returns, p=1, q=1):
    #we choose p,q=1 to only look at the day before, but any choice of p,q can be chosen in principle. p=q=1 is standard.

    #arch_model expects returns in percentage, so we scale internally by 100 (numerical accuracy). we scale back down when interpreting.
    returns_pct = returns.dropna() * 100
    model = arch_model(returns_pct, vol='Garch', p=p, q=q, dist='normal')
    fitted = model.fit(disp='off') #cleans up warnings/unnecessary outputs
    return fitted

def forecast_volatility(fitted_model, horizon=10):
    #we use the above model to look forwards in time and predict the volatility. Horizon is the number of days we look forward.

    forecast = fitted_model.forecast(horizon=horizon, reindex=False) #projects GARCH recursion forwards using last values of omega, beta, alpha
    #above has units percentage ^2, so we convert back, then annualise.
    daily_variance = forecast.variance.values[-1] / 100**2 
    daily_vol = np.sqrt(daily_variance)
    annualised_vol = daily_vol * np.sqrt(252) # ~252 trading days per year

    return annualised_vol

#we now aim to backtest, that is, check if our model is accurate at predicting the past events.

def backtest_garch(returns, train_size=500, horizon=1, refit_every=20, actual_window=21):
    #we use our forecast function at points back in time to forecast forwards, and compare to actual values. After the timestep is done, we repeat

    #train_size: minimum observations until first forecast, refit_every: days until we refit model (slow process)

    returns = returns.dropna()
    results = []
    fitted = None

    for i in range(train_size, len(returns) - actual_window):
        if fitted is None or (i - train_size) % refit_every == 0: #if we haven't fitted, or refit_every days after a refit
            train = returns.iloc[:i]
            fitted = fit_garch(train)

        forecast = fitted.forecast(horizon=horizon, reindex=False)
        pred_var = forecast.variance.values[-1][horizon-1] / 100**2
        pred_vol = np.sqrt(pred_var) * np.sqrt(252) #annualised predicted volatility

        future_returns = returns.iloc[i:i + actual_window]
        actual_vol = future_returns.std() * np.sqrt(252) 

        results.append({
            'date': returns.index[i + horizon - 1],
            'predicted_vol': pred_vol,
            'actual_vol': actual_vol
        })

    return pd.DataFrame(results).set_index('date')
#we now link to the black-scholes project. we compare the forecast above to the implied volatility of a ticker using black-scholes inverse formula (see that project)
#these two methods are independent in principle, despite both being ways of looking the volatility of a ticker in the future
def compare_garch_vs_implied(ticker, period="5y", expiry_index=5, r_val=0.05):
    import sys
    sys.path.append('../../past_market_analysis/src')
    sys.path.append('../../options_pricing/src')

    from data_loader import fetch_price_data
    from analysis import compute_returns
    from black_scholes import full_iv_analysis

    #start by forecasting using GARCH
    df = fetch_price_data(ticker, period=period)
    df = compute_returns(df)
    fitted = fit_garch(df['Return'])
    garch_vol = forecast_volatility(fitted, horizon=1)[0]

    #now compute implied volatility
    iv_result = full_iv_analysis(ticker, expiry_index=expiry_index, r_val=r_val)

    if iv_result is None:
        print(f"Could not retrieve implied volatility for {ticker}")
        return None

    #we also look at the realised volatility to see how both compare
    implied_vol = iv_result['iv']
    realised_vol = iv_result['realised_vol']

    print(f"\n--- {ticker} volatility comparison ---")
    print(f"Realised (historical):  {realised_vol:.4f} ({realised_vol*100:.2f}%)")
    print(f"GARCH forecast:         {garch_vol:.4f} ({garch_vol*100:.2f}%)")
    print(f"Implied (market):       {implied_vol:.4f} ({implied_vol*100:.2f}%)")
    print(f"Implied - GARCH:        {(implied_vol - garch_vol)*100:.2f} pp")

    return {
        'ticker': ticker,
        'realised_vol': realised_vol,
        'garch_vol': garch_vol,
        'implied_vol': implied_vol,
        'gap': implied_vol - garch_vol
    }

#we end the new statistics by looking at GARCH variants, specifically GJR-GARCH and EGARCH. In GJR-GARCH, we add a term which only counts to negative shocks (if /epsilon_{t-1} < 0)
#In EGARCH we look at log variances without needing parameter constraints

def fit_asymmetric_garch(returns, model_type='GJR', p=1, q=1):
    returns_pct = returns.dropna() * 100

    if model_type == 'GJR':
        model = arch_model(returns_pct, vol='Garch', p=p, o=1, q=q, dist='normal')
    elif model_type == 'EGARCH':
        model = arch_model(returns_pct, vol='EGARCH', p=p, o=1, q=q, dist='normal') #o=1 is the asymmetry order
    else:
        raise ValueError("model_type must be 'GJR' or 'EGARCH")
    
    fitted = model.fit(disp='off')
    return fitted
#to conclude, we compare across different tickers to see differences between values of (p,d,q) chosen for ARIMA, and the quality difference of different GARCH models across a variety of tickers

def cross_ticker_validation(tickers, period="5y"):

    import sys
    sys.path.append('../../past_market_analysis')
    from data_loader import fetch_price_data
    from analysis import compute_returns
    from arima_model import find_best_arima_model

    results = []
    for ticker in tickers:
        print(f"\nProcessing {ticker}")
        try: 
            df = fetch_price_data(ticker, period=period)
            df = compute_returns(df)
            returns = df['Return']

            arima_order = find_best_arima_model(returns)
            plain = fit_garch(returns)
            gjr = fit_asymmetric_garch(returns, model_type='GJR')

            results.append({
                'Ticker': ticker,
                'ARIMA_order': str(arima_order),
                'GARCH_alpha_beta': round(plain.params['alpha[1]'] + plain.params['beta[1]'], 4),
                'GJR_gamma': round(gjr.params['gamma[1]'], 4),
                'GJR_gamma_pval': round(gjr.pvalues['gamma[1]'], 4),
                'AIC_improvement': round(plain.aic - gjr.aic, 2)
            })
        except Exception as e:
            print(f"Failed for {ticker}: {e}")
    return pd.DataFrame(results).set_index('Ticker')

