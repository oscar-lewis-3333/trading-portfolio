import numpy as np
import pandas as pd
from scipy.optimize import minimize
from risk_metrics import shrunk_covariance

def portfolio_stats(weights, mu, cov, risk_free=0.0):
    #we compute expected return, volatility and sharpe ratio for a given set of portfolio weights
    weights = np.array(weights)
    ret = weights @ mu
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - risk_free) / vol
    return ret, vol, sharpe

def min_variance_portfolio(mu, cov, target_return=None, allow_shorting=False, max_weight=None):
    #we aim to find the weights which minimize portfolio variance
    #it a target return is given, we contrain the expected return to that value
    #if allow_shorting is False, we constrain weights to be non-negative

    n = len(mu)

    def objective(weights):
        return weights @ cov @ weights  #variance of our portfolio

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  #weights must sum to 1

    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda w: w @ mu - target_return})  #expected return constraint
    if allow_shorting:
        bounds = None
    else:
        upper = max_weight if max_weight is not None else 1
        bounds = [(0, upper)] * n 

    w0 = np.repeat(1/n, n) #guess all weights are equal
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints) #'SLSQP' is a sequential least squares programming optimization method, which is one of the few scipy optimisers that handles bounds and constraints simultaenously.
    return result.x

def max_sharpe_portfolio(mu, cov, risk_free=0.0, allow_shorting=False, max_weight=None):
    #find weights which maximise sharpe ratio. This is called the tangency portfolio.
    #This is equivalent to minimising the negative sharpe ratio, the process we emplore

    n = len(mu)

    def negative_sharpe(weights):
        ret = weights @ mu
        vol = np.sqrt(weights @ cov @ weights)
        return -(ret - risk_free) / vol 
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  
    if allow_shorting:
        bounds = None
    else:
        upper = max_weight if max_weight is not None else 1
        bounds = [(0, upper)] * n 

    w0 = np.repeat(1/n, n)
    result = minimize(negative_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def efficient_frontier(mu, cov, num_points=100, allow_shorting=False):
    #we trace the efficient frontier by miniming variance at each of n_points target return levels

    min_ret, max_ret = mu.min(), mu.max()
    targets = np.linspace(min_ret, max_ret, num_points)
    vols, rets, all_weights = [], [], []
    for target in targets:
        try: 
            weights = min_variance_portfolio(mu, cov, target_return=target, allow_shorting=allow_shorting)
            ret, vol, _ = portfolio_stats(weights, mu, cov)
            vols.append(vol)
            rets.append(ret)
            all_weights.append(weights)
        except Exception:
            continue #if a target return isn't possible (i.e not in (min_ret, max_ret)), we skip it

    return np.array(vols), np.array(rets), np.array(all_weights)

def stability_test(returns_df, n_periods=4, allow_shorting=False):
    #split returns history into n_periods different periods, optimise on each, and compare resulting weights.
    #If these weights (Markowitz) is robust, these weights should be similar across periods. Else, it suggests that we are fitting noise.

    splits = np.array_split(returns_df, n_periods)
    all_weights = []
    labels = []

    for i, chunk in enumerate(splits):
        mu_chunk = chunk.mean().values * 252
        cov_chunk = chunk.cov().values * 252

        weights = max_sharpe_portfolio(mu_chunk, cov_chunk, allow_shorting=allow_shorting)
        all_weights.append(weights)

        start = chunk.index[0].strftime('%Y-%m-%d')
        end = chunk.index[-1].strftime('%Y-%m-%d')
        labels.append(f"{start} to {end}")
    weights_df = pd.DataFrame(all_weights, columns=returns_df.columns, index=labels)
    return weights_df
#we see in the notebook that this approach is not stable, so we seek an alternative using Ledoit-Wolf shinkage. We now see if it works

def stability_test_shrunk(returns_df, n_periods=4, allow_shorting=False):
    #we repeat the test above but using the shrunk covariance matrix

    splits = np.array_split(returns_df, n_periods)
    all_weights = []
    labels = []

    for chunk in splits:
        mu_chunk = chunk.mean().values * 252
        cov_chunk, _ = shrunk_covariance(chunk)

        weights = max_sharpe_portfolio(mu_chunk, cov_chunk, allow_shorting=allow_shorting)
        all_weights.append(weights)
        labels.append(f"{chunk.index[0].strftime('%Y-%m')} to {chunk.index[-1].strftime('%Y-%m')}")

    return pd.DataFrame(all_weights, columns=returns_df.columns, index=labels)

def risk_parity_portfolio(cov):
    #we find weights where each asset contributes equally to portfolio risk. No returns used, so avoids stability issues.
    #problem is not convex, so numerical solver necessary.

    n = cov.shape[0]
    target = 1/n

    def objective(weights):
        port_vol = np.sqrt(weights @ cov @ weights)
        marginal = (cov @ weights) / port_vol
        contribution_pct = marginal * weights / port_vol
        return np.sum((contribution_pct - target)**2)
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    bounds = [(0.001, 1)]*n
    w0 = np.repeat(1/n, n)

    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints) 
    return result.x 
    
def out_of_sample_test(returns_df, train_frac=0.6, risk_free=0.0):
    #we aim to optimise on the first train_frac of the data, and evaluate resulting fixed weights on rest of the period.
    #compares max sharpe, min-variance, risk parity and equal weight against each other

    from risk_metrics import shrunk_covariance
    split = int(len(returns_df) * train_frac)
    train = returns_df.iloc[:split]
    test = returns_df.iloc[split:] #test is post split, train before

    mu_train = train.mean().values * 252
    cov_train = train.cov().values * 252
    cov_train_shrunk, _ = shrunk_covariance(train)

    n = returns_df.shape[1]
    strategies = {
        'Max Sharpe':          max_sharpe_portfolio(mu_train, cov_train),
        'Max Sharpe (shrunk)': max_sharpe_portfolio(mu_train, cov_train_shrunk),
        'Min variance':        min_variance_portfolio(mu_train, cov_train),
        'Risk parity':         risk_parity_portfolio(cov_train),
        'Equal weight':        np.repeat(1/n, n),
    }

    mu_test = test.mean().values * 252
    cov_test = test.cov().values * 252
    rows = []
    for name, weights in strategies.items():
        ret_is, vol_is, sharpe_is = portfolio_stats(weights, mu_train, cov_train, risk_free) #in sample (train data)
        ret_oos, vol_oos, sharpe_oos = portfolio_stats(weights, mu_test, cov_test, risk_free) #out of sample (test data)
        rows.append({
            'Strategy': name,
            'IS_return': round(ret_is*100, 2),
            'IS_vol': round(vol_is*100, 2),
            'IS_Sharpe': round(sharpe_is, 3),
            'OOS_return': round(ret_oos*100, 2),
            'OOS_vol': round(vol_oos*100, 2),
            'OOS_Sharpe': round(sharpe_oos, 3),
        })
    
    results = pd.DataFrame(rows).set_index('Strategy')
    weights_df = pd.DataFrame(strategies, index=returns_df.columns).T

    print(f"Train: {train.index[0].date()} to {train.index[-1].date()}  ({len(train)} days)")
    print(f"Test:  {test.index[0].date()} to {test.index[-1].date()}  ({len(test)} days)\n")

    return results, weights_df
#we consider what positions shorting would allow (i.e allowing negative weights, but still requiring them to sum to 1)
#we see what a long only portfolio costs and what risks we take on with allowing shorts

def compare_shorting(mu, cov, tickers):
    results = {}
    for label, allow in [('Long only', False), ('Shorting allowed', True)]:
        weights = max_sharpe_portfolio(mu, cov, allow_shorting=allow)
        ret, vol, sharpe = portfolio_stats(weights, mu, cov)
        results[label] = {
            'weights': weights,
            'return': ret * 100,
            'vol': vol * 100,
            'sharpe': sharpe,
            'gross_exposure': np.sum(np.abs(weights)) * 100,
            'largest_short': min(weights.min(), 0) * 100,
        }
    rows = []
    for label, v in results.items():
        rows.append({
            'Strategy': label,
            'return': round(v['return'], 2),
            'vol': round(v['vol'], 2),
            'sharpe': round(v['sharpe'], 3),
            'gross_exposure': round(v['gross_exposure'], 2),
            'largest_short': round(v['largest_short'], 2),
        })
    summary = pd.DataFrame(rows).set_index('Strategy')
    weights = weights = pd.DataFrame({k: v['weights'] * 100 for k, v in results.items()},index=tickers).T.round(1) #tickers are the index, so transpose to get strategy as the index

    return summary, weights

def portfolio_turnover(w_old, w_new):
    #turnover between two allocation, the fraction of the portfolio that must be traded to move from one asset to another
    #defined as half the sum of absolute weight changes, so that fully liquidating gives one portfolio and buying another one gives 1.0 not 2.0

    return np.sum(np.abs(np.array(w_new) - np.array(w_old)))/2

def turnover_analysis(returns_df, n_periods=4, cost_bps=10):
    #we measure how much each trading strategy requires when re-optimised each period, and what that costs
    #cost_bps: roundtrip transaction cost in basis points (10bps = 0.1%)

    splits = np.array_split(returns_df, n_periods)
    n = returns_df.shape[1]

    strategies = {'Max Sharpe': [], 'Min variance': [], 'Risk parity': [], 'Equal weight': []}
    #calculate weights for each strategy per chunk
    for chunk in splits:
        mu_c = chunk.mean().values * 252
        cov_c = chunk.cov().values * 252
        strategies['Max Sharpe'].append(max_sharpe_portfolio(mu_c, cov_c))
        strategies['Min variance'].append(min_variance_portfolio(mu_c, cov_c))
        strategies['Risk parity'].append(risk_parity_portfolio(cov_c))
        strategies['Equal weight'].append(np.repeat(1/n, n))
    #calculate turnovers per trade, then use to return useful statistics
    rows = []
    for name, weight_list in strategies.items():
        turnovers = [portfolio_turnover(weight_list[i], weight_list[i+1]) for i in range(len(weight_list) - 1)]
        avg_turnover = np.mean(turnovers)
        rows.append({
            'Strategy': name,
            'Avg turnover': round(avg_turnover * 100, 1),
            'Max turnover': round(max(turnovers) * 100, 1),
            'Cost per rebalance (%)': round(avg_turnover * cost_bps / 100, 3),
        })

    return pd.DataFrame(rows).set_index('Strategy')





