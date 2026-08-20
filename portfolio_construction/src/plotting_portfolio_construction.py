import numpy as np
import matplotlib.pyplot as plt

def plot_efficient_frontier(mu, cov, tickers, vols, rets,
                             portfolios=None, n_random=3000, title_suffix=""):
    #we plot efficient frontier alongside individual assets, the optimal portfolios marked, and a scatter of random portfolios for comparison. 
    #we also plot our risk parity portfolio to compare to efficient frontier
    #portfolios is a dict input: {label: (weights, color, marker)}
    from optimisation import portfolio_stats

    fig, ax = plt.subplots(figsize=(11, 7))

    n = len(mu)
    rand_vols, rand_rets, rand_sharpes = [], [], []
    for _ in range(n_random):
        w = np.random.random(n)
        w /= w.sum()
        r, v, s = portfolio_stats(w, mu, cov)
        rand_vols.append(v); rand_rets.append(r); rand_sharpes.append(s)

    sc = ax.scatter(np.array(rand_vols)*100, np.array(rand_rets)*100,
                    c=rand_sharpes, cmap='viridis', s=6, alpha=0.4)
    plt.colorbar(sc, ax=ax, label='Sharpe ratio')

    ax.plot(vols*100, rets*100, color='black', linewidth=2,
            label='Efficient frontier')

    for i, ticker in enumerate(tickers):
        asset_vol = np.sqrt(cov[i, i]) * 100
        ax.scatter(asset_vol, mu[i]*100, marker='D', s=70,
                   edgecolor='black', linewidth=0.8, zorder=5)
        ax.annotate(ticker, (asset_vol, mu[i]*100),
                    xytext=(6, 6), textcoords='offset points', fontsize=10)

    if portfolios:
        for label, (w, colour, marker) in portfolios.items():
            r, v, s = portfolio_stats(w, mu, cov)
            ax.scatter(v*100, r*100, marker=marker, s=350, color=colour,
                       edgecolor='black', zorder=6,
                       label=f'{label} ({s:.3f})')

    ax.set_xlabel('Annualised volatility (%)')
    ax.set_ylabel('Annualised expected return (%)')
    ax.set_title(f'Efficient Frontier{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_shorting_comparison(mu, cov, tickers, n_points=50):
    #we compare the long-only efficient frontier to the shorting allowed efficent frontier to shown what positions shorting allows and at what cost

    from optimisation import efficient_frontier
    #different volatilities and returns for long, short positions
    vols_long, rets_long, _ = efficient_frontier(mu, cov, num_points=n_points, allow_shorting=False)
    vols_short, rets_short, _ = efficient_frontier(mu, cov, num_points=n_points, allow_shorting=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(vols_long*100, rets_long*100, color='steelblue', linewidth=2, label='Long only')
    ax.plot(vols_short*100, rets_short*100, color='coral', linewidth=2, linestyle='--', label='Shorting allowed')

    for i, ticker in enumerate(tickers):
        asset_vol = np.sqrt(cov[i, i]) * 100
        ax.scatter(asset_vol, mu[i]*100, marker='D', s=60, edgecolor='black', linewidth=0.8, zorder=5)
        ax.annotate(ticker, (asset_vol, mu[i]*100), xytext=(6, 6), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Annualised volatility (%)')
    ax.set_ylabel('Annualised expected return (%)')
    ax.set_title('Efficient frontier — long only vs shorting allowed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()