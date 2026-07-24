import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

def plot_price_history(df, ticker):
    """Plot price and volume history."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={'height_ratios': [3, 1]}) 

    axes[0].plot(df.index, df['Close'], color='steelblue', linewidth=1.5)
    axes[0].set_title(f'{ticker} Price History', fontsize=14)
    axes[0].set_ylabel('Price (USD)')
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(df.index, df['Volume'], color='steelblue', alpha=0.5)
    axes[1].set_ylabel('Volume')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_returns_analysis(df, ticker):
    """4-panel returns analysis."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig) #we split into 4 panels

    #1: daily returns over the interval
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df['Return'], color='steelblue',
             linewidth=0.8, alpha=0.8)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_title('Daily Returns')
    ax1.set_ylabel('Return')
    ax1.grid(True, alpha=0.3)

    #2: Distrubution of returns with normal pdf overlay
    ax2 = fig.add_subplot(gs[0, 1])
    r = df['Return'].dropna()
    ax2.hist(r, bins=50, density=True, color='steelblue',
             alpha=0.7, edgecolor='white', linewidth=0.3)
    x = np.linspace(r.min(), r.max(), 200)
    ax2.plot(x, stats.norm.pdf(x, r.mean(), r.std()),
             'r-', linewidth=2, label='Normal fit')
    ax2.set_title('Return Distribution')
    ax2.set_xlabel('Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    #3: cumaltive returns
    ax3 = fig.add_subplot(gs[1, 0])
    cumulative = (1 + df['Return']).cumprod() #if we invest $1, how does our investment look over time
    ax3.plot(df.index, cumulative, color='green', linewidth=1.5)
    ax3.axhline(1, color='black', linewidth=0.8, linestyle='--')
    ax3.set_title('Cumulative Returns')
    ax3.set_ylabel('Growth of $1')
    ax3.grid(True, alpha=0.3)

    #4: cumalative volatility
    ax4 = fig.add_subplot(gs[1, 1])
    col = [c for c in df.columns if 'Rolling_Vol' in c]
    if col:
        ax4.plot(df.index, df[col[0]], color='coral', linewidth=1.5)
        ax4.set_title(f'Rolling Annualised Volatility')
        ax4.set_ylabel('Volatility')
        ax4.grid(True, alpha=0.3)

    fig.suptitle(f'{ticker} Returns Analysis', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()

def plot_qq(df, ticker):
    """QQ plot to check normality of returns."""
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(df['Return'].dropna(), dist="norm", plot=ax)
    ax.set_title(f'{ticker} — QQ Plot of Returns')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


#The above is all as needed for an individual ticker, we look to compare as in analysis.py

def plot_asset_comparison(ranking):
    #we start by recalling the neccessary metrics (from summary statistics) into a dictionary (and making them look nicer for presentation reasons):
    
    metrics = {
        'sharpe_ratio':      'Sharpe Ratio',
        'annualised_return': 'Annualised Return',
        'annualised_vol':    'Annualised Volatility',
        'max_drawdown':      'Max Drawdown',
        'skewness':          'Skewness',
        'kurtosis':          'Kurtosis'
    }
    #now plotting bar charts of key metrics:

    fig, axes = plt.subplots(2,3, figsize=(16,8))
    axes = axes.flatten()

    for i, (col, label) in enumerate(metrics.items()):
        vals = ranking[col].astype(float)  #extract values from columns
        colours = ['green' if v > 0 else 'coral' for v in vals] #green is good, else not
        axes[i].bar(ranking.index, vals, color=colours, edgecolor='blue')
        axes[i].set_title(label, fontsize=12)
        axes[i].axhline(0, color='black')
        axes[i].set_xticklabels(ranking.index, rotation =45)
        axes[i].grid(True, axis='y')

        
    plt.suptitle('Asset Comparison')
    plt.tight_layout()
    plt.show()

def plot_cumulative_returns_comparison(tickers, period="2y"):
    from data_loader import fetch_price_data
    from analysis import compute_returns
    
    fig, ax = plt.subplots(figsize=(12,6))
    colours = ['blue', 'coral', 'green', 'purple', 'orange', 'teal']
    for i, ticker in enumerate(tickers):
        try:
            df =fetch_price_data(ticker, period=period)
            df = compute_returns(df)
            cumulative = 1 + df['Return'].cumprod()
            ax.plot(df.index, cumulative, label=ticker, color=colours[i % len(colours)])
        except Exception as e:
            print(f"Failed for {ticker}:{e}")

    ax.axhline(1, color='black')
    ax.set_title('Cumulative Returns Comparison')
    ax.set_ylabel('Growth of $1')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    

