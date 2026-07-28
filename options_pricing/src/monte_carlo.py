#here we build tools which will be neccessary not just here, but in the future too.

#for black-scholes, we assume the 'market' follows the SDE: dS = mu *S *dt + sigma*S*dW(t). this has no analytical solution,
#but we can numerically approximate the solution by discretising the timestep, for which we have a solution at each time step.
#we define a function to simulate this (called GBM after geometric brownian motion because that the SDE)
import numpy as np

def simulate_gbm_path(S0, mu, sigma, T, n_steps):
    #S0: intital stock price
    #mu: expected annual return (drift)
    #sigma: annualised volatility
    #T: time in years
    #n_steps: number of discretised steps

    dt = T / n_steps
    prices = np.zeros(n_steps+1)
    prices[0] = S0

    for t in range(1, n_steps+1):
        Z = np.random.normal(0, 1)
        prices[t] = prices[t-1] * np.exp((mu - (sigma**2)/2) * dt + sigma *np.sqrt(dt)*Z) #closed form analytic solution for GBM after discretising. Normal distribution term adds 'randomness' to the result

    return prices
#this creates one possible price path, but each one is very different to the previous, due to the randomness from the normal variable
#we now simulate many many paths (n_paths many paths)

def simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths):
    dt = T/n_steps
    Z = np.random.normal(0, 1, size=(n_paths, n_steps))

    daily_returns = ((mu - (sigma**2)/2) * dt + sigma *np.sqrt(dt)*Z) #take the exponential part of closed form solution for each time step

    log_paths = np.cumsum(daily_returns, axis=1) #gives the cumulative sum of all paths, which is log (multiplied by a scalar of our real pathd)

    paths = S0*np.exp(log_paths)
    paths = np.hstack([np.full((n_paths, 1),S0),paths]) #sticks S0 on the front each path since cumsum starts from the first change, not initial price

    return paths

#we use this to price a call option and compare to black-scholes. We only consider european call options, where you can only call at the end of the contract, not in between

def monte_carlo_call_price(S0, K, T, r, sigma, n_steps=252, n_paths=100000):
    #notice that r is the risk free rate, not the expected rate (mu)
    paths = simulate_gbm_paths(S0, mu=r, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths)

    final_prices = paths[:, -1] #we look at the last column, which is price at expiry

    payoffs = np.maximum(final_prices - K, 0)

    discounted_price = np.exp(-r*T) * np.mean(payoffs)
    return discounted_price
#notice we use the risk-free result here as you simulate the world as if it were risk-neutral. its the mathematically correct measure under which discounted option prices are true expectations, called Girsanov's theorem.





    
