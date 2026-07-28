#we start by recalling the solutions to the black-scholes equations in sympy
import numpy as np
import sympy as sp
from sympy.stats import Normal, cdf
from scipy.optimize import brentq #for implied volatility\
import yfinance as yf #for importing historical option prices

S, K, T, r, sigma = sp.symbols('S K T r sigma', positive=True) #all terms in black-scholes are positive

d1 = (sp.ln(S/K) + (r + sigma**2/2)*T) / (sigma*sp.sqrt(T))
d2 = d1 - sigma*sp.sqrt(T)

Z = Normal('Z', 0, 1)
def N(x):
    return cdf(Z)(x)

call_price = S * N(d1) - K * sp.exp(-r*T) * N(d2)
put_price  = K * sp.exp(-r*T) * N(-d2) - S * N(-d1)

def black_scholes_call(S_val, K_val, T_val, r_val, sigma_val):

    #S_val: current stock price
    #K_val: strike price
    #T_val: time to expiry in years
    #r_val: risk-free interest rate
    #sigma_val: annualised volatility

    price = call_price.subs({S: S_val, K: K_val, T: T_val, r:r_val, sigma:sigma_val}) #subsitute given values into equation above
    return float(price.evalf()) #return evaluated number as a float

def black_scholes_put(S_val, K_val, T_val, r_val, sigma_val):
    #analogous to above
    price = put_price.subs({S: S_val, K: K_val, T: T_val, r:r_val, sigma:sigma_val})
    return float(price.evalf())

#just as a side note, all options work on a per stock basis, so we add a function for convinience
def contract_price(option_price, no_of_shares=100):
    return option_price * no_of_shares


#all of the above is readily available online, now we see why it is useful, for the greeks. These are the set of partial derivatives of partial derivatives of the call_price equation w.r.t each variable
#all of these partial derivatives are labelled by different greek letters

delta_expr = sp.diff(call_price, S)
gamma_expr = sp.diff(call_price, S, 2)
vega_expr = sp.diff(call_price, sigma)
theta_expr = - sp.diff(call_price, T)
rho_expr = sp.diff(call_price, r)

def greeks(S_val, K_val, T_val, r_val, sigma_val):
    #we evaluate the greeks at the set of values

    subs_dict = {S: S_val, K: K_val, T: T_val, r: r_val, sigma: sigma_val}
    return {
        'delta': float(delta_expr.subs(subs_dict).evalf()),
        'gamma': float(gamma_expr.subs(subs_dict).evalf()),
        'vega':  float(vega_expr.subs(subs_dict).evalf()),
        'theta': float(theta_expr.subs(subs_dict).evalf()),
        'rho':   float(rho_expr.subs(subs_dict).evalf()),
    }

#the black-scholes equation gives the call/put prices based on the 5 variables, but what if we know this value already, and seek to find the volatility?
#we seek to work backwords and find whats called the implied volatility

def implied_volatility(market_price, S_val, K_val, T_val, r_val, vol_bounds=(0.001, 5.0)):
    #to use brentq, we need a function that crosses zero. we define such function

    def price_difference(sigma_guess):
        model_price = black_scholes_call(S_val, K_val, T_val, r_val, sigma_guess)
        return model_price - market_price
    return brentq(price_difference, vol_bounds[0], vol_bounds[1])

#to use a real life example, we need to import historical option prices, which we do here

def get_option_chain(ticker, expiry_index=0, min_volume=1):
    #fetch live option chain for ticker from yfinance, expiry_index gives which expiry dates to use

    stock = yf.Ticker(ticker)
    expiries = stock.options #gives list of expiry dates
    expiry_date = expiries[expiry_index] #gives the date

    chain = stock.option_chain(expiry_date) #returns real calls and puts for that stock expiring at that date (including traded prices, volume)
    calls = chain.calls #split chain into call options and put options
    puts = chain.puts
    calls = calls[calls['volume'].fillna(0) >= min_volume]
    puts = puts[puts['volume'].fillna(0) >= min_volume]


    current_price = stock.history(period='1d')['Close'].iloc[-1] 
    return calls, puts, expiry_date, current_price
#we realise that this isnt too plausible due to issues with yfinances tracking of options (more detail in the readme)

def compare_implied_vs_realised(ticker, realised_vol, r_val=0.05, min_strike_distance=15, expiry_index=5):
    #we compare implied volatilty from last market price to the realised volatility from historical data
    #note, we cannot take live bids/asks due to limitations with yfinance. so last price reflects the last completed trade, which could be minutes, hours, days stale. just a limitation of yfinance

    calls, puts, expiry, current_price = get_option_chain(ticker, expiry_index=expiry_index)

    from datetime import datetime
    expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
    days_to_expiry = (expiry_date - datetime.now()).total_seconds() / (24 * 3600)
    T_val = days_to_expiry / 365 #all times are in years

    calls = calls.copy()
    calls['distance'] = np.abs(calls['strike'] - current_price)

    candidate = calls[calls['distance'] < min_strike_distance].sort_values('distance').head(1)

    if len(candidate) == 0 or T_val <= 0:
        print("No suitable near-the-money contract found for this expiry")
        return None
    K_val = candidate['strike'].values[0]
    last_price = candidate['lastPrice'].values[0]

    iv = implied_volatility(last_price, current_price, K_val, T_val, r_val=r_val)

    print(f"Ticker:              {ticker}")
    print(f"Current price:       ${current_price:.2f}")
    print(f"Strike used:         ${K_val:.2f}")
    print(f"Expiry:              {expiry} (T={T_val:.4f} years)")
    print(f"Option lastPrice:    ${last_price:.2f}")
    print(f"Implied volatility:  {iv:.4f} ({iv*100:.2f}%)")
    print(f"Realised volatility: {realised_vol:.4f} ({realised_vol*100:.2f}%)")
    print(f"Difference:          {(iv-realised_vol)*100:.2f} percentage points")

    return iv

#we now put all of this together to get a full automated pipeline, ie given the ticker: 
#1: pick up a near-the-money option
#2: compute implied volatility
#3: find real market price
#4: compute realised volatility using historical data
#5: return everything neccessary to plot and compare in options.ipynb

def full_iv_analysis(ticker, period="1y", expiry_index=5, r_val=0.05, min_strike_distance=15):
    #we need to import things from past_market_analysis for 3,4
    import sys
    sys.path.append('../../past_market_analysis/src')
    from data_loader import fetch_price_data
    from analysis import compute_returns, summary_statistics
    from datetime import datetime

    #compute realised volatility from past data

    hist_df = fetch_price_data(ticker, period=period)
    hist_df = compute_returns(hist_df)
    stats = summary_statistics(hist_df)
    realised_vol = stats['annualised_vol']

    #choose live option identically to compare_implied_vs_realised
    calls, puts, expiry, current_price = get_option_chain(ticker, expiry_index=expiry_index)

    expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
    days_to_expiry = (expiry_date - datetime.now()).total_seconds() / (24 * 3600)
    T_val = days_to_expiry / 365 

    calls = calls.copy()
    calls['distance'] = np.abs(calls['strike'] - current_price)
    candidate = calls[calls['distance'] < min_strike_distance].sort_values('distance').head(1)
    if len(candidate) == 0:
        print(f"No contract found within ${min_strike_distance} of current price for {ticker}")
        return None

    K_val = candidate['strike'].values[0]
    last_price = candidate['lastPrice'].values[0]

    iv = implied_volatility(last_price, current_price, K_val, T_val, r_val)

    #create dictionary of results:
    result = {
        'ticker': ticker,
        'S_val': current_price,
        'K_val': K_val,
        'T_val': T_val,
        'r_val': r_val,
        'expiry': expiry,
        'option_price': last_price,
        'iv': iv,
        'realised_vol': realised_vol,
    }
    #printing all values
    print(f"Ticker:              {ticker}")
    print(f"Current price:       ${current_price:.2f}")
    print(f"Strike used:         ${K_val:.2f}")
    print(f"Expiry:              {expiry} (T={T_val:.4f} years)")
    print(f"Option lastPrice:    ${last_price:.2f}  (not a live quote, see above markdown comment)")
    print(f"Implied volatility:  {iv:.4f} ({iv*100:.2f}%)")
    print(f"Realised volatility: {realised_vol:.4f} ({realised_vol*100:.2f}%)")
    print(f"Difference:          {(iv-realised_vol)*100:.2f} percentage points")

    return result






    

