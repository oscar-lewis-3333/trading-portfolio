import numpy as np
import matplotlib.pyplot as plt

def plot_greeks(greeks_func, K_val, T_val, r_val, sigma_val,
                 S_range=(100, 300), n_points=100):

    #we plot the greeks as a function of stock price, keeping strike price, time, volatility constant and rate of interest all fixed.
    S_values = np.linspace(S_range[0], S_range[1], n_points) #range of (realistic) stock prices 

    delta_vals, gamma_vals, vega_vals, theta_vals, rho_vals = [], [], [], [], []

    for S_val in S_values:
        g = greeks_func(S_val, K_val, T_val, r_val, sigma_val) #calculate all greeks together
        delta_vals.append(g['delta']) #put them into their respective list
        gamma_vals.append(g['gamma'])
        vega_vals.append(g['vega'])
        theta_vals.append(g['theta'])
        rho_vals.append(g['rho'])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    greek_data = [ #place all values above into a list of tuples
        ('Delta', delta_vals, 'steelblue'),
        ('Gamma', gamma_vals, 'coral'),
        ('Vega',  vega_vals,  'green'),
        ('Theta', theta_vals, 'purple'),
        ('Rho',   rho_vals,   'orange'),
    ]

    for i, (name, vals, color) in enumerate(greek_data): #enumarate over the tuples, and plot them one by one
        axes[i].plot(S_values, vals, color=color, linewidth=1.5)
        axes[i].axvline(K_val, color='black', linestyle='--',
                        linewidth=0.8, alpha=0.6, label=f'Strike (K={K_val})')
        axes[i].set_title(name)
        axes[i].set_xlabel('Stock Price')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)

    axes[5].axis('off')   # 2x3 array causes 6 panels, but we only have 5 greeks, so remove the empty one

    plt.suptitle('Option Greeks vs Stock Price', fontsize=14)
    plt.tight_layout()
    plt.show()

#we create a plotting function directly for showing the convergence of the monte carlo method to the analytic solution to black-scholes

def plot_convergence(mc_price_func, bs_price, path_counts=None, **kwargs):
    #mc_price_func is our monte carlo function, bs_price is analytic sol to black-scholes, path_counts is the number of paths to test over
    #**kwargs is to pass variables through mc_price_func (S0, K, T, r, sigma)

    if path_counts is None:
        path_counts = [100, 1000, 10000, 50000, 100000, 500000]

    #we make two plots, one about the raw numbers and how they get closer and another about the absolute values of errors
    mc_prices = [mc_price_func(**kwargs, n_paths=n) for n in path_counts]
    errors = [abs(p - bs_price) for p in mc_prices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    #panel dedicated to the raw numbers and seeing the proximity of them
    axes[0].plot(path_counts, mc_prices, 'o-', color='steelblue',
                 label='Monte Carlo estimate')
    axes[0].axhline(bs_price, color='red', linestyle='--',
                    label='Black-Scholes exact price')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Number of simulated paths')
    axes[0].set_ylabel('Call option price')
    axes[0].set_title('Price convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    #panel dedicated to the absolute error of the monte carlo method. plotted on a log-log scale
    axes[1].plot(path_counts, errors, 'o-', color='coral')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Number of simulated paths')
    axes[1].set_ylabel('Absolute error ($)')
    axes[1].set_title('Convergence error (log-log)')
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()

def plot_iv_vs_realised(ticker, iv, realised_vol, S_val, K_val, T_val, r_val):
    from black_scholes import black_scholes_call

    price_at_iv = black_scholes_call(S_val, K_val, T_val, r_val, iv)
    price_at_realised = black_scholes_call(S_val, K_val, T_val, r_val, realised_vol)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #the first plot compares the two volatilities
    axes[0].bar(['Realised\nvolatility', 'Implied\nvolatility'],
                [realised_vol*100, iv*100], #want the percentages not the decimals
                color=['steelblue', 'coral'])
    axes[0].set_ylabel('Annualised volatility (%)')
    axes[0].set_title(f'{ticker} — Realised vs Implied Vol')
    axes[0].grid(True, alpha=0.3, axis='y')

    #second plot dedicated to the different pricing according to these volatilities

    axes[1].bar(['Priced at\nrealised vol', 'Priced at\nimplied vol'],
                [price_at_realised, price_at_iv],
                color=['steelblue', 'coral'])
    axes[1].set_ylabel('Call option price ($)')
    axes[1].set_title('Resulting Option Price')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print(f"Volatility risk premium: {(iv-realised_vol)*100:.2f} percentage points")
    print(f"Price difference:       ${price_at_iv - price_at_realised:.2f}")

