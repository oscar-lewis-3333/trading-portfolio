# OPTIONS PRICING,  STOCHASTIC MODELLING and IMPLIED VOLATILITY ANALYSIS

## OVERVIEW
A project dedicated to symbolic analytic solutions to Black-Scholes and Monte Carlo numerical approximations of such solutions. Also includes derivations of the greeks, computation of implied volatility and comparison with realised volatility.

## FEATURES
- Pricing call/put options using Black-Scholes in Sympy
- Derives Greeks using Sympy, plots them as a function of stock price
- Monte Carlo option pricing via simulated Geometric Brownian Motion price paths, averages many simulated paths, and plots the convergence of this to the analytic Black-Scholes solution
- Implied volatility solver that inverts the Black-Scholes pricing formula given a real market option price
- Automated pipeline which, given a ticker, imports a live option chain, its historical price data and computes the realised volatility from past data, implied volatility from the option and compares the two

## KEY DESIGN DECISIONS
- Both symbolic and numerical methods were used as a 'check' of one another. 
- Monte Carlo simulation uses risk-free interest rate instead of the assets expected return rate. This is due to it being the mathematically correct measure for which discounted options are true expectations, an idea known as Girsanov's theorem. Intuitively, it can be thought of simulating each asset as it were risk-neutral.
- Implied volatility is calculated using lastPrice, which may not be a live option price and could be a stale trade. This was due to limitations with options data (as will be discussed in limitations). We chose to only look at options with expiry date of at least a week for this reason, to avoid zero-DTE, which would cause wildly high implied volatility. The changes brought the implied volatility back into line with empirical estimates

## RESULTS
- Showed convergence of Monte Carlo method to the Black-Scholes symbolic solution, with the difference getting smaller as the number of paths simulated increases. For example, simulations often return < $0.10 difference on a call price of $26.81 using 500000 paths
- Using a real-life option chain, showed implied volatility was often greater than realised volatility when tested over a variety of tickers (SPY, AMZN, AAPL, NVDA). This is called a volatility risk premium and is a key metric used in quantitative trading.

## LIMITATIONS
- Issues with free options data through Yahoo finance. Forced to use past options with expiry dates further in the future which have a large potential to be stale. If the time to expiry is too small, we return results with massively inflated implied volatilities, a concept called zero-DTE. Can only be solved with accurate, live options data.
- Not all tickers have listed options, for example, the implied volatility section does not work for cryptocurrencies or some commodities.
- There are limitations to Black-Scholes itself. Real-world trading fees are ignored (commissions, overnight costs), constant volatility is assumed, which obviously does not hold and only holds for European style options (option can only be activated at the expiry date and at no point in between).

## LIBRARIES USED
- sympy — symbolic Black-Scholes formula and Greeks derivation
- numpy — numerical computation of Black-Scholes approximation
- scipy — root-finding for implied volatility 
- yfinance — importing option chains and past market data
- matplotlib — visualisation of plot

## PROJECT STRUCTURE
    src/
        black_scholes.py — symbolic pricing, Greeks, implied volatility, live option chain fetching
        monte_carlo.py   — use of monte carlo methods to approximate black-scholes call prices
        plotting.py      — Greeks plots, convergence plots, implied volatility comparison plots
    notebooks/
        options.ipynb    — main showcasing/analysis

## USAGE
```python
from black_scholes import black_scholes_call, greeks, full_iv_analysis
from monte_carlo import monte_carlo_call_price
from plotting import plot_greeks, plot_iv_vs_realised

#pricing of call options
price = black_scholes_call(S_val=220, K_val=200, T_val=0.25, r_val=0.05, sigma_val=0.30)

#numerical approximation of call option pricing
mc_price = monte_carlo_call_price(S0=220, K=200, T=0.25, r=0.05, sigma=0.30)

#symbolic calculation and plotting of greeks
g = greeks(S_val=220, K_val=200, T_val=0.25, r_val=0.05, sigma_val=0.3)
plot_greeks(greeks, K_val=200, T_val=0.25, r_val=0.05, sigma_val=0.3)

#statistical analysis and plotting of implied volatility vs realised volatility
result = full_iv_analysis("AAPL", expiry_index=5)
plot_iv_vs_realised(ticker=result['ticker'], iv=result['iv'], realised_vol=result['realised_vol'], S_val=result['S_val'], K_val=result['K_val'], T_val=result['T_val'], r_val=result['r_val'] )
```